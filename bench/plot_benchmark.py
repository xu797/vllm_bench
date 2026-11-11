import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. 实验数据 
# ---------------------------------------------------------
x_labels = [1, 5, 10, 20, 30, 50]

# 综合性能指标
req_throughput = [0.88, 3.49, 4.82, 5.45, 3.95, 3.91]
output_token_throughput = [112.72, 446.24, 617.38, 697.15, 505.96, 500.51]
peak_concurrency = [7, 37, 79, 155, 300, 300]

# 延迟分位数
ttft_mean = [327.16, 587.16, 1815.21, 6824.76, 31109.60, 33461.78]
ttft_p99 = [741.38, 1622.57, 7628.66, 23193.78, 62701.83, 67581.16]

# ---------------------------------------------------------
# 2. 全局样式设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.style.use('seaborn-v0_8-whitegrid') 
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('vLLM Serving Benchmark — Qwen2.5-7B-Instruct on RTX 3090', 
             fontsize=18, fontweight='bold', y=0.98)

# ---------------------------------------------------------
# 图1：Request Throughput vs Concurrency (左上)
# ---------------------------------------------------------
ax1 = axs[0, 0]
ax1.plot(x_labels, req_throughput, marker='o', color='#2b5cff', linewidth=2.5, markersize=7)
ax1.set_title('Request Throughput vs Concurrency', fontsize=14, pad=10)
ax1.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax1.set_ylabel('Request Throughput (req/s)', fontsize=12)
ax1.grid(True, linestyle='-', alpha=0.7)

# ---------------------------------------------------------
# 图2：Output Token Throughput vs Concurrency (右上) - 核心撞墙图
# ---------------------------------------------------------
ax2 = axs[0, 1]
ax2.plot(x_labels, output_token_throughput, marker='o', color='#20a060', linewidth=2.5, markersize=7)
# 标注最高点
ax2.annotate('Peak Throughput\n(697 tok/s)', xy=(20, 697.15), xytext=(25, 650),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)
ax2.set_title('Output Token Throughput vs Concurrency', fontsize=14, pad=10)
ax2.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax2.set_ylabel('Output Token Throughput (tok/s)', fontsize=12)
ax2.grid(True, linestyle='-', alpha=0.7)

# ---------------------------------------------------------
# 图3：Time to First Token (Mean vs P99) (左下)
# ---------------------------------------------------------
ax3 = axs[1, 0]
ax3.plot(x_labels, ttft_mean, marker='o', color='#ff9800', linewidth=2.5, markersize=7, label='Mean TTFT')
ax3.plot(x_labels, ttft_p99, marker='s', color='#d32f2f', linewidth=2.5, markersize=7, label='P99 TTFT')
# 使用对数坐标轴(可选)，因为 30 和 50 RPS 的延迟太夸张了，线性坐标下前面的点会被压缩得很厉害
# 如果不想用对数轴，删掉下面这行即可，视觉上会很震撼（直冲云霄）
# ax3.set_yscale('log') 
ax3.set_title('Time to First Token (Mean vs P99)', fontsize=14, pad=10)
ax3.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax3.set_ylabel('TTFT (ms)', fontsize=12)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, linestyle='-', alpha=0.7)

# ---------------------------------------------------------
# 图4：Peak Concurrency vs Theoretical Limit (右下)
# ---------------------------------------------------------
ax4 = axs[1, 1]
bar_positions = np.arange(len(x_labels))
ax4.bar(bar_positions, peak_concurrency, color='#7b2cbf', width=0.5, label='Peak Concurrent Requests')
# 这里的理论限制线可以去掉，或者改成 Max Num Seqs 限制
ax4.axhline(y=64, color='#d32f2f', linestyle='--', linewidth=2, 
            label='max-num-seqs Limit (64)')
ax4.set_title('Peak Concurrency vs Server Limit', fontsize=14, pad=10)
ax4.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax4.set_ylabel('Peak Concurrent Requests', fontsize=12)
ax4.set_xticks(bar_positions)
ax4.set_xticklabels(x_labels)
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, linestyle='-', alpha=0.7, axis='y')

# ---------------------------------------------------------
# 3. 调整布局并保存/显示
# ---------------------------------------------------------
plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.92) 

plt.savefig('vllm_benchmark_rtx3090.png', dpi=300, bbox_inches='tight')
plt.show()