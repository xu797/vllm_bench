# vllm_bench
vllm_bench
```
uv venv vllm-lab --python 3.11
source vllm-lab/bin/activate
uv pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import vllm; print(vllm.__version__)"

```

```
vllm-bench/
├── scripts/
│   ├── serve.sh          # 启动推理服务
│   └── run_bench.sh      # 执行压测
├── bench/
│   ├── client.py         # 单请求客户端示例
│   ├── load_test.py      # 并发压测脚本
│   └── metrics.py        # 指标聚合与输出
├── results/
│   └── baseline.csv      # 实验结果留存
└── README.md             # 记录环境、命令与结论

```

https://blog.csdn.net/weixin_42651281/article/details/170393803

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64
