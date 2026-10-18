"""
Step 3: Long-Tail Case Study 可视化

生成Figure: 
(a) HR@10 by Item Popularity
(b) NDCG@10 by Item Popularity

用法:
    python visualize_long_tail.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ==================== 配置 ====================
DATA_NAME = "Beauty"  # 当前：Beauty | 原来：Video
OUTPUT_DIR = "./long_tail_results/"

# ==================== 加载结果 ====================
print("="*60)
print("加载评估结果...")
print("="*60)

result_file = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_long_tail_results.pkl")
with open(result_file, 'rb') as f:
    results = pickle.load(f)

print("✓ 结果加载完成")

# ==================== 准备数据 ====================
buckets = ['Head', 'Mid', 'Tail']
strategies = ['Single-Intent', 'Multi-Intent']
colors = ['orange', 'green']

hr_data = {s: [] for s in strategies}
ndcg_data = {s: [] for s in strategies}

for strategy in strategies:
    for bucket in buckets:
        if bucket in results[strategy]:
            hr_data[strategy].append(results[strategy][bucket]['hr'])
            ndcg_data[strategy].append(results[strategy][bucket]['ndcg'])
        else:
            hr_data[strategy].append(0)
            ndcg_data[strategy].append(0)

# ==================== 可视化 ====================
print("\n生成可视化...")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(buckets))
width = 0.35

# 子图(a): HR@10
ax1 = axes[0]
for i, (strategy, color) in enumerate(zip(strategies, colors)):
    offset = (i - 0.5) * width  # 两个柱：-0.5 和 +0.5
    bars = ax1.bar(x + offset, hr_data[strategy], width, 
                    label=strategy, alpha=0.8, color=color)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Item Popularity', fontsize=12, fontweight='bold')
ax1.set_ylabel('HR@20', fontsize=12, fontweight='bold')
ax1.set_title('(a) Hit Rate@20 by Item Popularity', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(buckets, fontsize=11)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max([max(hr_data[s]) for s in strategies]) * 1.15)

# 子图(b): NDCG@10
ax2 = axes[1]
for i, (strategy, color) in enumerate(zip(strategies, colors)):
    offset = (i - 0.5) * width  # 两个柱：-0.5 和 +0.5
    bars = ax2.bar(x + offset, ndcg_data[strategy], width,
                    label=strategy, alpha=0.8, color=color)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Item Popularity', fontsize=12, fontweight='bold')
ax2.set_ylabel('NDCG@20', fontsize=12, fontweight='bold')
ax2.set_title('(b) NDCG@20 by Item Popularity', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(buckets, fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max([max(ndcg_data[s]) for s in strategies]) * 1.15)

plt.tight_layout()

# 保存
output_path = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_long_tail_case_study.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ PNG保存到: {output_path}")

pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ PDF保存到: {pdf_path}")

plt.show()

# ==================== 分析增益 ====================
print("\n" + "="*60)
print("性能增益分析（Multi-Intent vs Single-Intent）")
print("="*60)

for bucket in buckets:
    if bucket in results['Multi-Intent'] and bucket in results['Single-Intent']:
        hr_multi = results['Multi-Intent'][bucket]['hr']
        hr_single = results['Single-Intent'][bucket]['hr']
        ndcg_multi = results['Multi-Intent'][bucket]['ndcg']
        ndcg_single = results['Single-Intent'][bucket]['ndcg']
        
        hr_gain = (hr_multi - hr_single) / hr_single * 100 if hr_single > 0 else 0
        ndcg_gain = (ndcg_multi - ndcg_single) / ndcg_single * 100 if ndcg_single > 0 else 0
        
        print(f"{bucket:5s}: HR@20 {hr_gain:+.1f}%, NDCG@20 {ndcg_gain:+.1f}%")

print("\n关键发现:")
print("  → 如果Tail区间的增益最大")
print("  → 证明多意图训练让seq_model对长尾物品更敏感")

print("\n" + "="*60)

