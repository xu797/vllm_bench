#!/usr/bin/env python3
"""Plot the RTX 3090 Qwen2.5-7B serving study from the per-run CSV."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results/study_20261118/runs.csv"
DEFAULT_OUTPUT = ROOT / "results/study_20261118/overview.png"
COLORS = {"fp16": "#2878B5", "awq": "#E07A24"}
LABELS = {"fp16": "FP16", "awq": "AWQ"}


def load_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def groups(rows, model, family, metric, input_tokens=None, output_tokens=None):
    selected = [r for r in rows if r["model"] == model and r["family"] == family
                and (input_tokens is None or r["input_tokens"] == str(input_tokens))
                and (output_tokens is None or r["output_tokens"] == str(output_tokens))]
    by_x = defaultdict(list)
    for row in selected:
        x = float(row["target_rps"]) if family == "rate" else (row["input_tokens"], row["output_tokens"])
        value = row.get(metric, "")
        if value not in ("", None):
            by_x[x].append(float(value))
    return by_x


def summary(ax, rows, model, metric, family, x_values, **filters):
    values = groups(rows, model, family, metric, **filters)
    med, low, high = [], [], []
    for x in x_values:
        sample = values.get(x, [])
        if not sample:
            med.append(np.nan)
            low.append(0)
            high.append(0)
            continue
        center = float(np.median(sample))
        med.append(center)
        low.append(center - min(sample))
        high.append(max(sample) - center)
    return np.array(med), np.array(low), np.array(high)


def add_range_line(ax, rows, model, metric, x_values, family="rate", **filters):
    med, low, high = summary(ax, rows, model, metric, family, x_values, **filters)
    ax.errorbar(x_values, med, yerr=np.vstack([low, high]), marker="o", markersize=5,
                linewidth=2.2, capsize=3, color=COLORS[model], label=LABELS[model])
    return med


def server_memory(root):
    result = {}
    for model, folder in (("fp16", "fp16"), ("awq", "awq")):
        candidates = sorted(root.rglob(f"{folder}/server.log"))
        preferred = [path for path in candidates if "study_20261118_knee" in str(path)]
        candidates = preferred or candidates
        if not candidates:
            continue
        text = candidates[0].read_text(errors="replace")
        weight = re.search(r"Model loading took ([0-9.]+) GiB memory", text)
        kv = re.search(r"Available KV cache memory: ([0-9.]+) GiB", text)
        if weight and kv:
            result[model] = (float(weight.group(1)), float(kv.group(1)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Root containing the study's fp16/awq server logs")
    args = parser.parse_args()
    rows = load_rows(args.data)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9E0E6",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    fig.suptitle("Qwen2.5-7B on RTX 3090 | FP16 vs AWQ", fontsize=16, fontweight="bold")
    rates = [2, 3, 4, 8, 16, 32]

    ax = axes[0, 0]
    for model in ("fp16", "awq"):
        add_range_line(ax, rows, model, "output_throughput_tok_s", rates,
                       input_tokens=1024, output_tokens=128)
    ax.set(title="固定负载吞吐（输入 1024 / 输出 128）", xlabel="目标请求速率 (req/s)", ylabel="输出吞吐 (tok/s)",
           xticks=rates)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[0, 1]
    for model in ("fp16", "awq"):
        add_range_line(ax, rows, model, "p95_ttft_ms", rates,
                       input_tokens=1024, output_tokens=128)
    ax.axhline(1000, color="#404B55", linestyle="--", linewidth=1.2, label="TTFT SLO 1 s")
    ax.set(title="首 token 延迟：高压时排队快速恶化", xlabel="目标请求速率 (req/s)",
           ylabel="P95 TTFT (ms，对数轴)", xticks=rates, yscale="log")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    ax.text(0.02, 0.05, "16–32 RPS 吞吐接近平台，P95 TTFT 已约 32–39 s",
            transform=ax.transAxes, va="bottom", color="#52616B", fontsize=8)

    ax = axes[1, 0]
    shape = [(128, 512), (256, 128), (2048, 128)]
    positions = np.arange(len(shape))
    width = 0.34
    for offset, model in ((-width / 2, "fp16"), (width / 2, "awq")):
        vals = []
        for inp, out in shape:
            samples = groups(rows, model, "shape", "output_throughput_tok_s", inp, out).get((str(inp), str(out)), [])
            vals.append(np.median(samples) if samples else np.nan)
        bars = ax.bar(positions + offset, vals, width, color=COLORS[model], label=LABELS[model], zorder=3)
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 30, f"{value:,.0f}",
                        ha="center", va="bottom", fontsize=8, color="#29343B")
    ax.set(title="不同请求形状下的吞吐（并发 32）", xlabel="输入 token / 输出 token",
           ylabel="输出吞吐 (tok/s)", xticks=positions,
           xticklabels=["128 / 512", "256 / 128", "2048 / 128"])
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    memory = server_memory((args.results_root or args.data.parent.parent).resolve())
    x = np.arange(2)
    width = 0.34
    for model, index in (("fp16", 0), ("awq", 1)):
        weight, kv = memory[model]
        bar1 = ax.bar(x[index] - width / 2, weight, width, color="#9BBBD3",
                      label="模型加载显存" if index == 0 else None, zorder=3)
        bar2 = ax.bar(x[index] + width / 2, kv, width, color="#405B73",
                      label="可用 KV Cache" if index == 0 else None, zorder=3)
        for bar, value in ((bar1[0], weight), (bar2[0], kv)):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8, color="#29343B")
    ax.set(title="显存预算：AWQ 为 KV Cache 留出更多空间", ylabel="显存 (GiB)",
           xticks=x, xticklabels=["FP16", "AWQ"], ylim=(0, 18))
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    fig.text(0.5, -0.015,
             "单卡 RTX 3090 24 GB · vLLM 0.33.0 · SLO: TTFT ≤ 1 s, TPOT ≤ 50 ms · 重复轮次范围见误差线",
             ha="center", fontsize=8, color="#52616B")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {args.output} and {pdf}")


if __name__ == "__main__":
    main()
