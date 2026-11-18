#!/usr/bin/env python3
"""Summarize preserved vLLM JSON results without third-party dependencies."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [ROOT / "results/study_20261118_rerun", ROOT / "results/study_20261118_extension",
                 ROOT / "results/study_20261118_knee"]


def metric_values(text, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
    return [float(value) for value in pattern.findall(text)]


def read_prom(path, metric):
    if not path.exists():
        return []
    return metric_values(path.read_text(errors="replace"), metric)


def run_row(root, config, result_path):
    data = json.loads(result_path.read_text())
    sidecar = result_path.with_name(result_path.stem + ".command.json")
    case = json.loads(sidecar.read_text()).get("case", {}) if sidecar.exists() else {}
    tag = result_path.stem
    monitor_path = root / config / "telemetry.jsonl"
    kv_peak = waiting_peak = preemptions_peak = None
    if monitor_path.exists():
        samples = []
        for line in monitor_path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get("phase") != tag or "prometheus" not in sample:
                continue
            prom = sample["prometheus"]
            samples.append((metric_values(prom, "vllm:kv_cache_usage_perc"),
                            metric_values(prom, "vllm:num_requests_waiting"),
                            metric_values(prom, "vllm:num_preemptions_total")))
        kv = [v for sample in samples for v in sample[0]]
        waiting = [v for sample in samples for v in sample[1]]
        preemptions = [v for sample in samples for v in sample[2]]
        kv_peak = max(kv) if kv else None
        waiting_peak = max(waiting) if waiting else None
        preemptions_peak = max(preemptions) if preemptions else None
    before = read_prom(result_path.with_name(tag + ".before.prom"), "vllm:num_preemptions_total")
    after = read_prom(result_path.with_name(tag + ".after.prom"), "vllm:num_preemptions_total")
    preemption_delta = sum(after) - sum(before) if before and after else None
    return {
        "study": root.name,
        "model": config,
        "tag": tag,
        "family": case.get("family"),
        "rep": case.get("rep"),
        "seed": case.get("seed"),
        "input_tokens": case.get("input_len"),
        "output_tokens": case.get("output_len"),
        "target_rps": case.get("rate"),
        "max_concurrency": case.get("concurrency"),
        "requested": data.get("num_prompts"),
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "failure_rate": (data.get("failed", 0) / data["num_prompts"] if data.get("num_prompts") else None),
        "duration_s": data.get("duration"),
        "request_throughput_rps": data.get("request_throughput"),
        "output_throughput_tok_s": data.get("output_throughput"),
        "goodput_rps": data.get("request_goodput"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "p95_ttft_ms": data.get("p95_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "p95_tpot_ms": data.get("p95_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "p95_e2el_ms": data.get("p95_e2el_ms"),
        "p99_e2el_ms": data.get("p99_e2el_ms"),
        "max_client_concurrency": data.get("max_concurrent_requests"),
        "kv_cache_peak_fraction": kv_peak,
        "waiting_peak": waiting_peak,
        "preemptions_total_peak": preemptions_peak,
        "preemptions_delta": preemption_delta,
        "source": str(result_path.relative_to(ROOT)),
    }


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="Study result directories to scan")
    parser.add_argument("--output", type=Path, default=ROOT / "results/study_20261118",
                        help="Directory for runs.csv, aggregate.csv, and SUMMARY.md")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.roots] or DEFAULT_ROOTS
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for config in ("fp16", "awq", "awq_batch512"):
            folder = root / config
            if not folder.exists():
                continue
            for result_path in sorted(folder.glob("*.json")):
                if result_path.stem in {"config", "generation_config", "tokenizer_config"}:
                    continue
                try:
                    data = json.loads(result_path.read_text())
                    if "output_throughput" in data and "failed" in data:
                        rows.append(run_row(root, config, result_path))
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
    rows.sort(key=lambda row: (row["model"], str(row["family"]), str(row["input_tokens"]),
                               str(row["output_tokens"]), str(row["target_rps"]), row["study"], row["tag"]))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)

    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], row["family"], row["input_tokens"], row["output_tokens"],
               row["target_rps"], row["max_concurrency"])
        groups[key].append(row)
    aggregates = []
    for key, samples in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        agg = dict(zip(["model", "family", "input_tokens", "output_tokens", "target_rps", "max_concurrency"], key))
        agg["n_runs"] = len(samples)
        agg["requests"] = sum(row["requested"] or 0 for row in samples)
        agg["completed"] = sum(row["completed"] or 0 for row in samples)
        agg["failed"] = sum(row["failed"] or 0 for row in samples)
        agg["failure_rate"] = agg["failed"] / agg["requests"] if agg["requests"] else None
        for metric in ("output_throughput_tok_s", "request_throughput_rps", "goodput_rps",
                       "p95_ttft_ms", "p99_ttft_ms", "p95_tpot_ms", "p99_tpot_ms", "p95_e2el_ms"):
            values = [row[metric] for row in samples if row[metric] is not None]
            agg[metric + "_median"] = statistics.median(values) if values else None
            agg[metric + "_min"] = min(values) if values else None
            agg[metric + "_max"] = max(values) if values else None
        aggregates.append(agg)
    write_csv(out / "aggregate.csv", aggregates)

    lines = ["# RTX 3090 vLLM Study", "", f"Completed benchmark JSON files: {len(rows)}", "",
             "Each row summarizes repeated runs. Failure rates use all requested prompts; throughput is reported only for successful output tokens by vLLM.", "",
             "| Model | Workload | Repeats | Completed / requested | Failed | Median output tok/s | Median P95 TTFT ms | Median P99 TTFT ms | Goodput req/s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        workload = (f"{row['input_tokens']}/{row['output_tokens']} tokens, "
                    f"RPS {row['target_rps']}" if row["family"] == "rate" else
                    f"{row['input_tokens']}/{row['output_tokens']} tokens, concurrency {row['max_concurrency']}")
        def fmt(value, places=1):
            return "n/a" if value is None else f"{value:.{places}f}"
        lines.append(f"| {row['model']} | {workload} | {row['n_runs']} | {row['completed']} / {row['requests']} | "
                     f"{row['failed']} ({fmt(100 * row['failure_rate'])}%) | "
                     f"{fmt(row['output_throughput_tok_s_median'])} | {fmt(row['p95_ttft_ms_median'])} | "
                     f"{fmt(row['p99_ttft_ms_median'])} | {fmt(row['goodput_rps_median'])} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} runs and {len(aggregates)} aggregates to {out}")


if __name__ == "__main__":
    main()
