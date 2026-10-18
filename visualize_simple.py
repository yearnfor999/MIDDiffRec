"""
简化版意图可视化脚本 - 只处理单个模型文件

用法:
    python visualize_simple.py

修改配置区的参数即可
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from models import InDiRec
from datasets import get_seqs_and_matrixes, DatasetForInDiRec
from trainer import InDiRecTrainer
from utils import set_seed

# ==================== 配置区（修改这里）====================

# 模型文件路径
CHECKPOINT_PATH = "output/InDiRec-Sports_and_Outdoors-110.pt" # InDiRec-Sports_and_Outdoors-159.pt    InDiRec-Toys_and_Games-29.pt

# 数据集名称
DATA_NAME = "Sports_and_Outdoors"  # Beauty, Sports_and_Outdoors, Video, Toys_and_Games

# 模型类型标签（用于文件命名和图标题）
MODEL_TYPE = "multi_intent"  # multi_intent, single_intent, no_intent

# 可视化样本数（设为很大值表示使用全部样本）
NUM_SAMPLES = 1000000 # 22363   19412

# 意图数量（需与训练时一致）
INTENT_NUM = 1024

# GPU设备（-1表示使用CPU）
CUDA = 0

# t-SNE参数
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# 输出路径（自动根据模型类型命名）
OUTPUT_PATH = f"./visualizations/{DATA_NAME}_{MODEL_TYPE}.png"

# ==========================================================

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
USE_CHINESE = False  # 使用英文

# 固定随机种子
set_seed(42)

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA) if CUDA >= 0 else ""
device = torch.device("cuda:0" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
print(f"使用设备: {device}")

# ==================== 加载数据 ====================
print(f"\n{'='*60}")
print("加载数据...")
print(f"{'='*60}")

data_dir = "./datasets/"
data_file = data_dir + DATA_NAME + ".txt"
segmented_file = data_dir + DATA_NAME + "_s.txt"

# 训练数据（用于聚类）
train_user_ids, train_seq = get_seqs_and_matrixes("training", segmented_file)

# 测试数据（用于可视化）
eval_user_ids, user_seq, max_item, _, test_rating_matrix = get_seqs_and_matrixes("rating", data_file)

# 用户ID映射
all_user_ids = set(train_user_ids + eval_user_ids)
sorted_user_ids = sorted(all_user_ids)
user_id_map = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
num_users = len(sorted_user_ids)

train_user_indices = [user_id_map[uid] for uid in train_user_ids]
eval_user_indices = [user_id_map[uid] for uid in eval_user_ids]

print(f"数据集: {DATA_NAME}")
print(f"用户数: {num_users}, 物品数: {max_item}")

# ==================== 创建模型参数 ====================
class Args:
    def __init__(self):
        # 数据相关
        self.num_users = num_users
        self.item_size = max_item + 2
        self.mask_id = max_item + 1
        self.train_matrix = test_rating_matrix
        
        # 模型结构
        self.hidden_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.hidden_act = "gelu"
        self.attention_probs_dropout_prob = 0.5
        self.hidden_dropout_prob = 0.5
        self.max_seq_length = 50
        
        # 意图相关
        self.intent_num = INTENT_NUM
        
        # 扩散相关
        self.timesteps = 200
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.diffuser_type = "mlp1"
        self.beta_sche = "exp"
        self.w = 2.0
        self.p = 0.1
        
        # BSARec相关
        self.alpha = 0.4
        self.c = 10
        
        # 训练相关（可视化时不需要，但trainer初始化需要）
        self.batch_size = 256
        self.lr = 0.001
        self.weight_decay = 0.0
        self.initializer_range = 0.02
        
        # 损失权重
        self.rec_weight = 1.0
        self.diff_weight = 1.0
        self.diffusion_rec_weight = 0.05
        self.cl_weight = 0.2
        self.tau = -1
        
        # 对比学习相关
        self.sim = "dot"
        self.temperature = 1.0
        
        # 日志相关
        self.log_file = "visualization.log"
        self.print_log_freq = 1
        
        # 鲁棒性实验相关
        self.noise_ratio = 0.0

args = Args()

# ==================== 创建数据加载器 ====================
cluster_dataset = DatasetForInDiRec(args, train_seq, user_ids=train_user_indices, data_type="train")
cluster_dataloader = DataLoader(cluster_dataset, sampler=SequentialSampler(cluster_dataset), 
                               batch_size=args.batch_size, drop_last=False)

test_dataset = DatasetForInDiRec(args, user_seq, user_ids=eval_user_indices, data_type="test")
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), 
                            batch_size=args.batch_size, drop_last=False)

# ==================== 加载模型 ====================
print(f"\n{'='*60}")
print(f"加载模型: {CHECKPOINT_PATH}")
print(f"{'='*60}")

model = InDiRec(device, args=args)
trainer = InDiRecTrainer(model, cluster_dataloader, cluster_dataloader, 
                        test_dataloader, test_dataloader, device, args)
trainer.load(CHECKPOINT_PATH)
print("✓ 模型加载成功")

# ==================== 重建聚类 ====================
print(f"\n{'='*60}")
print("重建意图聚类...")
print(f"{'='*60}")

trainer.model.eval()
kmeans_training_data = []

with torch.no_grad():
    for rec_batch in tqdm(cluster_dataloader, desc="提取聚类特征"):
        rec_batch = tuple(t.to(device) for t in rec_batch)
        _, subsequence, _, _, _ = rec_batch
        
        sequence_output = trainer.seq_model(subsequence)[:, -1, :]
        kmeans_training_data.append(sequence_output.detach().cpu().numpy())

kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
sequence_ids = np.arange(len(kmeans_training_data))

# 训练聚类器
for clusters in trainer.clusters_t:
    for j, cluster in enumerate(clusters):
        cluster.train(kmeans_training_data, sequence_ids)

# 构建用户意图分布
trainer.build_user_intent_distribution(cluster_dataloader)
print(f"✓ 聚类完成，意图数: {INTENT_NUM}")

# ==================== 提取表示 ====================
print(f"\n{'='*60}")
print(f"提取表示（样本数: {NUM_SAMPLES}，类型: {MODEL_TYPE}）...")
print(f"{'='*60}")

trainer.model.eval()

representations = []
intent_labels = []
sample_count = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="提取表示"):
        if sample_count >= NUM_SAMPLES:
            break
        
        batch = tuple(t.to(device) for t in batch)
        user_ids, input_seq, _, _, _ = batch
        
        # 获取序列表示
        seq_output = trainer.seq_model(input_seq)[:, -1, :]
        seq_repr = seq_output.detach().cpu().numpy()
        
        # 获取意图标签（用于着色）
        batch_intent_ids, _ = trainer.clusters[0].query(seq_repr)
        batch_intent_ids = batch_intent_ids.detach().cpu().numpy()
        
        # 直接使用当前模型训练得到的表示（序列最后一步）
        batch_repr = seq_repr
        
        representations.append(batch_repr)
        intent_labels.append(batch_intent_ids)
        sample_count += len(batch_repr)

representations = np.concatenate(representations)[:NUM_SAMPLES]
intent_labels = np.concatenate(intent_labels)[:NUM_SAMPLES]

print(f"✓ 提取完成: {len(representations)} 个样本")

# ==================== t-SNE 可视化 ====================
print(f"\n{'='*60}")
print("t-SNE 降维...")
print(f"{'='*60}")

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, 
            random_state=42, verbose=0)
embeddings_2d = tsne.fit_transform(representations)

print(f"✓ 降维完成: {embeddings_2d.shape}")

# 活跃簇数量
n_active = len(np.unique(intent_labels))
print(f"活跃意图数: {n_active}/{INTENT_NUM}")

# ==================== 绘图 ====================
print(f"\n{'='*60}")
print("生成可视化图...")
print(f"{'='*60}")

# 模型类型标题
title_names = {
    'multi_intent': 'Multi-Intent Distribution' if not USE_CHINESE else '多意图分布',
    'single_intent': 'Single Intent (Hard Assignment)' if not USE_CHINESE else '单一意图（硬分配）',
    'no_intent': 'No Intent (w=-1)' if not USE_CHINESE else '无意图 (w=-1)'
}

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 绘制散点图（全部蓝色小点）
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                    c='blue', s=8, alpha=0.5, edgecolors='none')

# 设置标题和标签
title = title_names.get(MODEL_TYPE, MODEL_TYPE)
ax.set_title(f'{title} - {DATA_NAME}', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('t-SNE Dim 1', fontsize=11)
ax.set_ylabel('t-SNE Dim 2', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# 添加样本数量注释
ax.text(0.02, 0.98, f'Samples: {len(representations)}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存
os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ PNG保存到: {OUTPUT_PATH}")

# 保存PDF
pdf_path = OUTPUT_PATH.replace('.png', '.pdf')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"✓ PDF保存到: {pdf_path}")

plt.close()

print(f"\n{'='*60}")
print(f"✓ 可视化完成！模型类型: {MODEL_TYPE}")
print(f"{'='*60}")

