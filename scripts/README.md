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

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


conda activate vllm-lab

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 20 --request-rate 1 \
    --save-result --result-filename rate1.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 50 --request-rate 5 \
    --save-result --result-filename rate5.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 100 --request-rate 10 \
    --save-result --result-filename rate10.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 200 --request-rate 20 \
    --save-result --result-filename rate20.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 300 --request-rate 30 \
    --save-result --result-filename rate30.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 300 --request-rate 50 \
    --save-result --result-filename rate50.json

Qwen/Qwen2.5-7B-Instruct-AWQ


vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64

# 1. AWQ - 10 RPS
vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --num-prompts 100 --request-rate 10 \
    --save-result --result-filename awq_rate10.json

# 2. AWQ - 20 RPS
vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --num-prompts 200 --request-rate 20 \
    --save-result --result-filename awq_rate20.json

# 3. AWQ - 30 RPS
vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --num-prompts 300 --request-rate 30 \
    --save-result --result-filename awq_rate30.json

vllm bench serve --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --num-prompts 300 --request-rate 50 \
    --save-result --result-filename awq_rate50.json