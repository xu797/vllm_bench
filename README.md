# 🚀 vLLM Serving Benchmark Report: Qwen2.5-7B

本项目记录了使用 `vllm bench serve` 对 **Qwen2.5-7B-Instruct** 模型进行压力测试的结果。旨在评估该模型在单机/单卡 vLLM 部署下的吞吐量上限、延迟表现及并发瓶颈。

## 🖥️ 实验环境 (Test Environment)

*   **模型 (Model):** Qwen/Qwen2.5-7B-Instruct
*   **推理框架 (Framework):** vLLM (version 0.33.0)
*   **GPU:** *[NVIDIA RTX 3090 24GB]*
*   **CUDA Version:** 12.8
*   **PyTorch Version:** 2.14.0+cu130
*   **测试工具:** `vllm bench serve` (默认输入长度 1024，输出长度 128)

### ⚙️ 服务启动命令
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64
```

## 📊 压测数据汇总 (Benchmark Results)

我们使用了不同的请求速率 (Request Rate) 进行了 4 组测试，所有测试均为 **0 失败 (Failed requests: 0)**，服务表现出极高的稳定性。

![vLLM Benchmark on RTX 3090](./bench/vllm_benchmark_rtx3090.png)

### 1. 综合性能指标 (Overall Metrics)

| 测试组 | 目标速率<br>(req/s) | 成功请求数 | 实际吞吐率<br>(req/s) | 输出吞吐量<br>(tok/s) | 峰值并发数 | 平均首字延迟 (TTFT) | 平均生成延迟 (TPOT) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rate 1** | 1.00 | 20 | 0.88 | 112.72 | 7 | **327.16 ms** | **23.35 ms** |
| **Rate 5** | 5.00 | 50 | 3.49 | 446.24 | 37 | **587.16 ms** | **40.67 ms** |
| **Rate 10** | 10.00 | 100 | 4.82 | 617.38 | 79 | **1815.21 ms** | **64.53 ms** |
| **Rate 20** | 20.00 | 200 | 5.45 | 697.15 | 155 | **6824.76 ms** | **82.02 ms** |

### 2. 延迟分位数详情 (Latency Percentiles)

| 测试组 | TTFT P99 | TPOT P99 | ITL 均值 (Mean) |
| :--- | :--- | :--- | :--- |
| **Rate 1** | 741.38 ms | 27.38 ms | 23.35 ms |
| **Rate 5** | 1622.57 ms | 66.58 ms | 40.67 ms |
| **Rate 10** | 7628.66 ms | 105.54 ms | 64.53 ms |
| **Rate 20** | 23193.78 ms | 135.07 ms | 82.02 ms |

> *注：TTFT = Time to First Token (首字延迟), TPOT = Time per Output Token (单字生成延迟), ITL = Inter-token Latency (Token间延迟)*


## 🔍 深度性能分析 (Analysis)

### 1. 吞吐量天花板与算力瓶颈

*   随着请求速率从 1 提升至 20，系统的**输出吞吐量**从 `112 tok/s` 跃升至 `697 tok/s`。这证明了 vLLM 的 `Continuous Batching`（连续批处理）技术极大地提升了 GPU 利用率。
*   **性能拐点出现在 Rate 10 到 Rate 20 之间。** 当速率翻倍时，吞吐量仅从 `617 tok/s` 提升至 `697 tok/s` (仅提升约 13%)。这表明 GPU 的算力或显存带宽已经基本饱和，达到了该硬件配置下的吞吐量极限。

### 2. 延迟的恶化与长尾效应

*   **首字延迟 (TTFT) 急剧上升：** 在低并发 (Rate 1) 下，TTFT 仅为 `327ms`，体验极佳。但到了 Rate 20，平均用户需要等待 `6.8秒` 才能看到第一个字，P99 更是高达 `23秒`。这对于实时交互（Chatbot）场景是不可接受的。
*   **P99 数据的长尾现象：** 观察 Rate 10 的数据，TTFT 的中位数 (Median) 仅为 `194.75ms`，但均值 (Mean) 却高达 `1815.21ms`。这说明在中等并发下，绝大多数请求很快，但有极少数请求遭遇了严重的排队阻塞。

### 3. 并发队列与资源抢占

*   服务启动时限制了 `--max-num-seqs 64`。但在 Rate 20 的测试中，**峰值并发数达到了 155**。
*   这说明请求的到达速度（20 req/s）远超 GPU 的处理速度。多出来的请求堆积在 API 层的等待队列中，导致系统整体的排队延迟远大于计算延迟。