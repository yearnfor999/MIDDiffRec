"""
Step 4: 可视化具体Tail物品案例（Figure c）

展示一个具体样本：
- 目标物品 ∈ Tail
- Multi-Intent排名更高
- Top-10推荐列表对照

用法:
    python visualize_case_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ==================== 配置 ====================
DATA_NAME = "Video"  # 当前：Beauty | 原来：Video
OUTPUT_DIR = "./long_tail_results/"

# ==================== 加载最佳案例 ====================
print("="*60)
print("加载最佳Tail案例...")
print("="*60)

case_file = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_best_tail_case.pkl")
with open(case_file, 'rb') as f:
    data = pickle.load(f)

best_case = data['best_case']
popularity_data = data['popularity_buckets']

head_items = popularity_data['head_items']
mid_items = popularity_data['mid_items']
tail_items = popularity_data['tail_items']

print("✓ 案例加载完成")
print(f"\n最佳案例:")
print(f"  用户ID: {best_case['real_user_id']}")
print(f"  用户索引: {best_case['user_index']}")
print(f"  序列长度: {best_case['seq_length']}")
print(f"  输入序列: {best_case['input_sequence'][:10]}... (前10个物品)")
print(f"  输入序列完整: {best_case['input_sequence']}")
print(f"")
print(f"  目标物品: {best_case['target_item']} (Tail)")
print(f"  Single-Intent排名: #{best_case['rank_single']}")
print(f"  Multi-Intent排名: #{best_case['rank_multi']}")
print(f"  排名提升: {best_case['improvement']} 位")

# ==================== 准备数据 ====================

def get_item_bucket(item_id):
    """判断物品属于哪个桶"""
    if item_id in head_items:
        return 'Head'
    elif item_id in mid_items:
        return 'Mid'
    elif item_id in tail_items:
        return 'Tail'
    else:
        return 'Unknown'

# Top-20推荐列表（带标签）
single_top20 = best_case['pred_single_top20']
multi_top20 = best_case['pred_multi_top20']
target = best_case['target_item']

# ==================== 可视化（表格形式）==================== 
print("\n生成可视化...")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9  # 缩小字体

fig = plt.figure(figsize=(14, 14))  # 增加高度以容纳20行

# 创建网格布局：上面显示输入序列，下面是推荐表格
gs = fig.add_gridspec(2, 1, height_ratios=[0.6, 4.4], hspace=0)  # 完全消除中间空白

# ==================== 上半部分：输入序列 ====================
ax_seq = fig.add_subplot(gs[0])
ax_seq.axis('off')

input_seq = best_case['input_sequence']
seq_text = "Input Sequence (User Interaction History):\n"
seq_text += "  " + " → ".join([f"Item {item}" for item in input_seq])
seq_text += f"  →  Item {target} (Target, Tail)"

ax_seq.text(0.5, 0.3, seq_text,  # 降低垂直位置
           ha='center', va='center',
           fontsize=11, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),  # 减小padding
           wrap=True)

# ==================== 下半部分：推荐表格 ====================
ax = fig.add_subplot(gs[1])
ax.axis('tight')
ax.axis('off')

# 构建表格数据（Top-20）
table_data = []
for rank in range(20):
    single_item = single_top20[rank]
    multi_item = multi_top20[rank]
    
    single_bucket = get_item_bucket(single_item)
    multi_bucket = get_item_bucket(multi_item)
    
    # 标记目标物品
    single_mark = ' ✓ GT' if single_item == target else ''
    multi_mark = ' ✓ GT' if multi_item == target else ''
    
    table_data.append([
        f'{rank+1}',
        f'{single_item}',
        f'({single_bucket}){single_mark}',
        f'{multi_item}',
        f'({multi_bucket}){multi_mark}'
    ])

# 创建表格
table = ax.table(cellText=table_data,
                colLabels=['Rank', 'Single Intent\nItem ID', 'Bucket', 
                          'Multi Intent Distribution\nItem ID', 'Bucket'],
                cellLoc='center',
                loc='center',
                colWidths=[0.10, 0.225, 0.225, 0.225, 0.225])

table.auto_set_font_size(False)
table.set_fontsize(9)  # 缩小字体
table.scale(1, 1.8)  # 缩小行高（原来2.5，现在1.8）

# 设置表头样式
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# 高亮目标物品所在行
for i, row in enumerate(table_data, 1):
    if '✓ GT' in row[2]:  # Single列有目标
        for j in [1, 2]:
            table[(i, j)].set_facecolor('#FFE699')
    if '✓ GT' in row[4]:  # Multi列有目标
        for j in [3, 4]:
            table[(i, j)].set_facecolor('#C6E0B4')

# 将标题放在最上面（使用fig.suptitle）
title_text = f'(c) Top-20 Recommendation Lists for a Tail Item\n'
title_text += f'User {best_case["real_user_id"]} (Sequence Length: {best_case["seq_length"]})  |  '
title_text += f'Target: Item {target} (Tail)  |  '
title_text += f'Single Intent: Rank #{best_case["rank_single"]}  |  Multi Intent Distribution: Rank #{best_case["rank_multi"]}  '
title_text += f'(Improvement: {best_case["improvement"]} positions)'

fig.suptitle(title_text, fontsize=11, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存
output_path = os.path.join(OUTPUT_DIR, f"{DATA_NAME}_case_example.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ PNG保存到: {output_path}")

pdf_path = output_path.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ PDF保存到: {pdf_path}")

plt.show()

print("\n" + "="*60)
print("✓ 案例可视化完成！")
print("="*60)
print(f"\nFigure (c): 单个Tail样本的Top-20推荐对照")
print(f"  用户ID: {best_case['real_user_id']}")
print(f"  序列长度: {best_case['seq_length']}")
print(f"  输入序列: {best_case['input_sequence']}")
print(f"")
print(f"  目标物品(Tail): Item {target}")
print(f"  Single-Intent: 排名 #{best_case['rank_single']}")
print(f"  Multi-Intent: 排名 #{best_case['rank_multi']} (提升 {best_case['improvement']} 位)")
print(f"\n完整Case Study:")
print(f"  Figure (a): HR@20 by popularity (群体定量分析)")
print(f"  Figure (b): NDCG@20 by popularity (群体定量分析)")
print(f"  Figure (c): Top-20 qualitative example (个体定性分析)")
print("="*60)

