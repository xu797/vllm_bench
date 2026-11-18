#!/usr/bin/env python3
"""Run a bounded, offline, single-GPU study with the installed vLLM CLI."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / "vllm-lab/bin/vllm"
HUB = Path.home() / ".cache/huggingface/hub"
MODELS = {
    "fp16": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "awq": ("Qwen/Qwen2.5-7B-Instruct-AWQ", "b25037543e9394b818fdfca67ab2a00ecc7dd641"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def capture(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return {"command": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def stop_process(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


class Monitor:
    def __init__(self, directory, url):
        self.directory, self.url = directory, url
        self.done = threading.Event()
        self.phase = "startup"
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.directory / "telemetry.jsonl").open("w") as out:
            while not self.done.is_set():
                row = {"unix_time": time.time(), "phase": self.phase}
                try:
                    row["prometheus"] = get(self.url + "/metrics")
                except Exception as error:
                    row["metrics_error"] = str(error)
                row["gpu"] = capture([
                    "nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits"])
                out.write(json.dumps(row) + "\n")
                out.flush()
                self.done.wait(2)


def cases(config, rates, repetitions, rate_only):
    items = []
    for rep, seed in enumerate([17, 29, 43, 59, 71, 89, 101, 127, 149, 173][:repetitions], 1):
        block = []
        if config != "awq_batch512" and rates:
            for rate in rates:
                block.append(dict(family="rate", input_len=1024, output_len=128, rate=rate, concurrency=None, n=192))
            if not rate_only:
                for inp, out in [(256, 128), (2048, 128), (128, 512)]:
                    block.append(dict(family="shape", input_len=inp, output_len=out, rate="inf", concurrency=32, n=128))
        elif config == "awq_batch512":
            # One-factor ablation: only the batched-token budget changes.
            for rate in ([8, 16] if rates else []):
                block.append(dict(family="rate", input_len=1024, output_len=128, rate=rate, concurrency=None, n=192))
            if not rate_only:
                block.append(dict(family="shape", input_len=2048, output_len=128, rate="inf", concurrency=32, n=128))
        random.Random(seed).shuffle(block)
        for case in block:
            case.update(rep=rep, seed=seed)
            items.append(case)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=48)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=["fp16", "awq", "awq_batch512"], choices=["fp16", "awq", "awq_batch512"])
    parser.add_argument("--rates", nargs="*", type=float, default=[2, 8, 16, 32])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--rate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.minutes * 60
    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    overrides = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0",
                 "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_LOG_STATS_INTERVAL": "2",
                 "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": "/usr/local/cuda-12.8"}
    env.update(overrides)
    manifest = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": sys.argv,
                "environment_overrides": overrides, "models": MODELS,
                "slo_ms": {"ttft": 1000, "tpot": 50}, "time_budget_minutes": args.minutes,
                "status": "running", "runs": []}
    for name, cmd in {
        "gpu": ["nvidia-smi", "-q"], "cpu": ["lscpu"], "system": ["uname", "-a"],
        "git": ["git", "rev-parse", "HEAD"], "git_status": ["git", "status", "--short"],
        "packages": [sys.executable, "-c", "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},indent=2))"],
    }.items():
        write_json(args.output / f"environment_{name}.json", capture(cmd))
    shutil.copy2(__file__, args.output / "run_study.py")
    write_json(args.output / "manifest.json", manifest)
    try:
        for config in args.configs:
            if time.monotonic() >= deadline:
                raise TimeoutError("Study time budget reached")
            directory = args.output / config
            directory.mkdir()
            kind = "fp16" if config == "fp16" else "awq"
            model_id, revision = MODELS[kind]
            model = HUB / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
            if not model.is_dir():
                raise FileNotFoundError(model)
            for filename in ["config.json", "generation_config.json", "tokenizer_config.json"]:
                shutil.copy2(model / filename, directory / filename)
            manifest.setdefault("tokenizer_sha256", {})[config] = hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest()
            server_cmd = [str(VLLM), "serve", str(model), "--served-model-name", "study-model",
                          "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "half",
                          "--gpu-memory-utilization", "0.90", "--max-model-len", "4096",
                          "--max-num-seqs", "64", "--max-num-batched-tokens", "512" if config == "awq_batch512" else "2048",
                          "--enable-chunked-prefill", "--no-enable-prefix-caching", "--seed", "0",
                          "--generation-config", "vllm"]
            if kind == "awq":
                server_cmd += ["--quantization", "awq_marlin"]
            write_json(directory / "server_command.json", {"argv": server_cmd, "env": overrides})
            monitor = Monitor(directory, url)
            with (directory / "server.log").open("w") as server_log:
                server = subprocess.Popen(server_cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                monitor.thread.start()
                try:
                    ready_deadline = min(deadline, time.monotonic() + 600)
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"Server exited: {config}; see {directory}/server.log")
                        try:
                            get(url + "/health")
                            break
                        except Exception:
                            if time.monotonic() > ready_deadline:
                                raise TimeoutError("Server startup exceeded budget")
                            time.sleep(2)
                    print(f"READY {config}", flush=True)
                    warmed = set()
                    for case in cases(config, args.rates, args.repetitions, args.rate_only):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Study time budget reached")
                        tag = f"{case['family']}_i{case['input_len']}_o{case['output_len']}_r{case['rate']}_rep{case['rep']}"
                        base_cmd = [str(VLLM), "bench", "serve", "--backend", "vllm", "--base-url", url,
                                    "--model", "study-model", "--tokenizer", str(model), "--dataset-name", "random",
                                    "--random-input-len", str(case["input_len"]), "--random-output-len", str(case["output_len"]),
                                    "--random-range-ratio", "0.0", "--random-prefix-len", "0", "--seed", str(case["seed"]),
                                    "--ignore-eos", "--temperature", "0", "--disable-tqdm"]
                        shape = (case["input_len"], case["output_len"])
                        if shape not in warmed:
                            warm_cmd = base_cmd + ["--num-prompts", "16", "--request-rate", "inf", "--max-concurrency", "16"]
                            monitor.phase = "warmup_" + tag
                            with (directory / ("warmup_" + tag + ".log")).open("w") as log:
                                warm = subprocess.Popen(warm_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                                try:
                                    if warm.wait(timeout=max(1, min(180, deadline-time.monotonic()))) != 0:
                                        raise RuntimeError(f"Warmup failed: {config}/{tag}")
                                finally:
                                    stop_process(warm)
                            warmed.add(shape)
                        cmd = base_cmd + ["--num-prompts", str(case["n"]), "--request-rate", str(case["rate"]),
                                          "--save-result", "--save-detailed", "--result-dir", str(directory.resolve()),
                                          "--result-filename", tag + ".json", "--percentile-metrics", "ttft,tpot,itl,e2el",
                                          "--metric-percentiles", "50,95,99", "--goodput", "ttft:1000", "tpot:50"]
                        if case["concurrency"] is not None:
                            cmd += ["--max-concurrency", str(case["concurrency"])]
                        write_json(directory / (tag + ".command.json"), {"argv": cmd, "case": case})
                        (directory / (tag + ".before.prom")).write_text(get(url + "/metrics"))
                        monitor.phase = tag
                        record = {"config": config, "tag": tag, **case, "started_unix": time.time()}
                        print(f"RUN {config}/{tag}", flush=True)
                        with (directory / (tag + ".log")).open("w") as log:
                            bench = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                record["returncode"] = bench.wait(timeout=max(1, min(240, deadline-time.monotonic())))
                            finally:
                                stop_process(bench)
                        record["finished_unix"] = time.time()
                        # Flush vLLM's periodic metrics before the after snapshot.
                        monitor.phase = "settle_" + tag
                        time.sleep(3)
                        (directory / (tag + ".after.prom")).write_text(get(url + "/metrics"))
                        manifest["runs"].append(record)
                        write_json(args.output / "manifest.json", manifest)
                        if record["returncode"] != 0:
                            raise RuntimeError(f"Benchmark failed: {config}/{tag}")
                        result = json.loads((directory / (tag + ".json")).read_text())
                        print(f"DONE {config}/{tag}: {result['output_throughput']:.1f} tok/s, failed={result['failed']}, p95TTFT={result['p95_ttft_ms']:.0f}ms", flush=True)
                finally:
                    monitor.done.set()
                    monitor.thread.join(timeout=35)
                    stop_process(server)
            time.sleep(3)
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
