"""
Step 1: 统计物品热度并分桶

分析训练集中每个物品作为目标的频次，划分为：
- Head: Top 20%（热门物品）
- Mid: 20%-50%（中等热度）
- Tail: Bottom 50%（长尾物品）

用法:
    python analyze_item_popularity.py
"""

import numpy as np
import pickle
import os

# ==================== 配置 ====================
DATA_NAME = "Beauty"  # 当前：Beauty | 原来：Video, Toys_and_Games, Sports_and_Outdoors
DATA_DIR = "./datasets/"
OUTPUT_DIR = "./popularity_analysis/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 读取训练数据 ====================
print("="*60)
print("统计物品热度...")
print("="*60)

segmented_file = DATA_DIR + DATA_NAME + "_s.txt"

print(f"读取训练数据: {segmented_file}")

item_target_count = {}  # 统计每个物品作为目标的次数

with open(segmented_file, 'r') as f:
    for line in f:
        items = line.strip().split()
        if len(items) < 4:  # 至少需要4个物品（user_id + 至少3个items）
            continue
        
        # 格式：user_id item1 item2 ... itemN
        # 目标物品 = 倒数第3个（因为数据集划分：训练用前N-3个预测第N-2个）
        if len(items) >= 4:
            target_item = int(items[-3])  # 倒数第3个作为训练目标
            item_target_count[target_item] = item_target_count.get(target_item, 0) + 1

print(f"✓ 统计完成")
print(f"  总物品数（作为目标）: {len(item_target_count)}")
print(f"  总出现次数: {sum(item_target_count.values())}")

# ==================== 按热度排序 ====================
sorted_items = sorted(item_target_count.items(), key=lambda x: x[1], reverse=True)

# 统计信息
frequencies = [count for _, count in sorted_items]
print(f"\n物品频次统计:")
print(f"  最高频: {frequencies[0]:,}")
print(f"  中位数: {int(np.median(frequencies)):,}")
print(f"  最低频: {frequencies[-1]:,}")
print(f"  平均值: {np.mean(frequencies):.2f}")

# ==================== 分桶 ====================
num_items = len(sorted_items)

# Head: Top 20%
head_cutoff = int(num_items * 0.2)
head_items = set([item for item, _ in sorted_items[:head_cutoff]])

# Mid: 20%-50%
mid_cutoff = int(num_items * 0.5)
mid_items = set([item for item, _ in sorted_items[head_cutoff:mid_cutoff]])

# Tail: Bottom 50%
tail_items = set([item for item, _ in sorted_items[mid_cutoff:]])

print(f"\n分桶结果:")
print(f"  Head (Top 20%):    {len(head_items):,} 个物品")
print(f"  Mid (20%-50%):     {len(mid_items):,} 个物品")
print(f"  Tail (Bottom 50%): {len(tail_items):,} 个物品")

# 统计每个桶的频次范围
head_freqs = [count for item, count in sorted_items[:head_cutoff]]
mid_freqs = [count for item, count in sorted_items[head_cutoff:mid_cutoff]]
tail_freqs = [count for item, count in sorted_items[mid_cutoff:]]

print(f"\n频次范围:")
print(f"  Head: {min(head_freqs):,} - {max(head_freqs):,}")
print(f"  Mid:  {min(mid_freqs):,} - {max(mid_freqs):,}")
print(f"  Tail: {min(tail_freqs):,} - {max(tail_freqs):,}")

# ==================== 保存分桶结果 ====================
popularity_buckets = {
    'head_items': head_items,
    'mid_items': mid_items,
    'tail_items': tail_items,
    'item_frequency': item_target_count,
    'sorted_items': sorted_items
}

output_file = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_popularity_buckets.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(popularity_buckets, f)

print(f"\n✓ 分桶结果已保存到: {output_file}")

# ==================== 可视化频次分布 ====================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(len(frequencies)), frequencies, linewidth=1.5)
plt.axvline(x=head_cutoff, color='red', linestyle='--', label='Head (20%)', linewidth=2)
plt.axvline(x=mid_cutoff, color='orange', linestyle='--', label='Mid (50%)', linewidth=2)
plt.xlabel('Item Rank', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'{DATA_NAME} - Item Popularity Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.yscale('log')  # 对数坐标更清楚
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_file = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_popularity_distribution.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ 分布图保存到: {plot_file}")

plt.close()

print("\n" + "="*60)
print("✓ 物品热度分析完成！")
print("="*60)
print(f"\n下一步:")
print(f"  运行: python long_tail_evaluation.py")
print("="*60)

