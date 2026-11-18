# RTX 3090 vLLM Study

Completed benchmark JSON files: 64

Each row summarizes repeated runs. Failure rates use all requested prompts; throughput is reported only for successful output tokens by vLLM.

| Model | Workload | Repeats | Completed / requested | Failed | Median output tok/s | Median P95 TTFT ms | Median P99 TTFT ms | Goodput req/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| awq | 1024/128 tokens, RPS 16 | 4 | 768 / 768 | 0 (0.0%) | 515.0 | 31817.4 | 33081.2 | 0.0 |
| awq | 1024/128 tokens, RPS 2 | 4 | 768 / 768 | 0 (0.0%) | 252.9 | 547.8 | 703.5 | 2.0 |
| awq | 1024/128 tokens, RPS 3.0 | 3 | 576 / 576 | 0 (0.0%) | 375.2 | 841.8 | 1004.9 | 2.8 |
| awq | 1024/128 tokens, RPS 32 | 5 | 960 / 960 | 0 (0.0%) | 515.3 | 37544.9 | 39008.8 | 0.0 |
| awq | 1024/128 tokens, RPS 4.0 | 3 | 574 / 576 | 2 (0.3%) | 480.2 | 2159.1 | 2835.6 | 1.0 |
| awq | 1024/128 tokens, RPS 8 | 4 | 761 / 768 | 7 (0.9%) | 513.4 | 20287.7 | 21098.8 | 0.0 |
| awq | 128/512 tokens, concurrency 32 | 5 | 640 / 640 | 0 (0.0%) | 2596.8 | 782.7 | 808.5 | 5.1 |
| awq | 2048/128 tokens, concurrency 32 | 5 | 639 / 640 | 1 (0.2%) | 269.2 | 10854.4 | 13003.6 | 0.0 |
| awq | 256/128 tokens, concurrency 32 | 4 | 512 / 512 | 0 (0.0%) | 1405.2 | 1580.4 | 1580.9 | 5.5 |
| fp16 | 1024/128 tokens, RPS 16 | 3 | 576 / 576 | 0 (0.0%) | 500.2 | 32658.2 | 33792.4 | 0.0 |
| fp16 | 1024/128 tokens, RPS 2 | 3 | 576 / 576 | 0 (0.0%) | 248.3 | 630.1 | 833.9 | 1.9 |
| fp16 | 1024/128 tokens, RPS 3.0 | 3 | 571 / 576 | 5 (0.9%) | 365.7 | 884.4 | 1070.3 | 1.6 |
| fp16 | 1024/128 tokens, RPS 32 | 3 | 576 / 576 | 0 (0.0%) | 502.2 | 38088.9 | 39599.2 | 0.0 |
| fp16 | 1024/128 tokens, RPS 4.0 | 3 | 575 / 576 | 1 (0.2%) | 462.7 | 3697.1 | 3985.3 | 0.0 |
| fp16 | 1024/128 tokens, RPS 8 | 3 | 560 / 576 | 16 (2.8%) | 500.4 | 20811.4 | 21762.7 | 0.0 |
| fp16 | 128/512 tokens, concurrency 32 | 3 | 384 / 384 | 0 (0.0%) | 1444.0 | 833.9 | 834.9 | 2.8 |
| fp16 | 2048/128 tokens, concurrency 32 | 3 | 381 / 384 | 3 (0.8%) | 253.8 | 10774.0 | 12905.7 | 0.0 |
| fp16 | 256/128 tokens, concurrency 32 | 3 | 375 / 384 | 9 (2.3%) | 976.3 | 1598.7 | 1598.9 | 3.8 |
