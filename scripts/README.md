# Experiment Scripts

See the repository [README](../README.md) for the experiment design, measured results, and reproduction commands.

- run_study.py runs vLLM serve benchmarks and captures GPU/Prometheus telemetry.
- summarize_study.py combines raw vLLM result JSON into per-run and aggregate tables.
- plot_results.py renders throughput, latency, workload-shape, and memory charts.
