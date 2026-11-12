import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. 实验数据
# ---------------------------------------------------------
x_labels = [10, 20, 30, 50] # 截取 10-50 的区间做对比最明显

# FP16 数据
fp16_throughput = [617.38, 697.15, 505.96, 500.51]
fp16_ttft = [1815.21, 6824.76, 31109.60, 33461.78]

# AWQ 数据
awq_throughput = [526.72, 785.71, 979.12, 523.26]
awq_ttft = [5496.82, 5512.25, 6012.94, 32372.15]

# ---------------------------------------------------------
# 2. 样式设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.style.use('seaborn-v0_8-whitegrid') 
fig, axs = plt.subplots(1, 2, figsize=(16, 6)) # 1行2列
fig.suptitle('Performance Comparison: FP16 vs AWQ (Qwen2.5-7B on RTX 3090)', 
             fontsize=18, fontweight='bold', y=1.05)

# ---------------------------------------------------------
# 图1：输出吞吐量对比 (左图)
# ---------------------------------------------------------
ax1 = axs[0]
ax1.plot(x_labels, fp16_throughput, marker='o', color='#d32f2f', linewidth=2.5, markersize=8, label='FP16')
ax1.plot(x_labels, awq_throughput, marker='s', color='#20a060', linewidth=2.5, markersize=8, label='AWQ')
# 标注最高点
ax1.annotate('AWQ Peak\n(979 tok/s)', xy=(30, 979.12), xytext=(35, 900),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)
ax1.set_title('Output Token Throughput', fontsize=15, pad=10)
ax1.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax1.set_ylabel('Throughput (tok/s)', fontsize=12)
ax1.set_xticks(x_labels)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='-', alpha=0.7)

# ---------------------------------------------------------
# 图2：首字延迟对比 (右图)
# ---------------------------------------------------------
ax2 = axs[1]
ax2.plot(x_labels, fp16_ttft, marker='o', color='#d32f2f', linewidth=2.5, markersize=8, label='FP16')
ax2.plot(x_labels, awq_ttft, marker='s', color='#20a060', linewidth=2.5, markersize=8, label='AWQ')
ax2.set_title('Mean Time to First Token (TTFT)', fontsize=15, pad=10)
ax2.set_xlabel('Configured Request Rate (RPS)', fontsize=12)
ax2.set_ylabel('TTFT (ms)', fontsize=12)
ax2.set_xticks(x_labels)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='-', alpha=0.7)

# 调整布局并保存
plt.tight_layout()
plt.savefig('fp16_vs_awq_comparison.png', dpi=300, bbox_inches='tight')
plt.show()