#!/usr/bin/env python3
"""
玩具例子：单意图BSARec系统完整流程演示
按照张哥的超小例子，手工设定所有数据，验证流程正确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设备和随机种子
device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("🎯 玩具例子：单意图BSARec系统流程演示")
print("=" * 60)

# ============================================================================
# 场景设定
# ============================================================================
L = 5  # 序列最大长度
D = 4  # 隐藏维度
K = 3  # 聚类簇数
num_items = 9  # 物品数量 (1-9)
num_users = 3  # 用户数量 (u1, u2, u3)

print(f"\n📊 场景设定:")
print(f"  序列长度: {L}, 隐藏维度: {D}")
print(f"  聚类簇数: {K}, 物品数量: {num_items}")
print(f"  用户数量: {num_users}")

# 子序列数据（按张哥的例子）
sequences_data = [
    {"seq_id": 0, "user": 0, "sequence": [1, 2, 3]},      # u1
    {"seq_id": 1, "user": 0, "sequence": [1, 2, 3, 4]},   # u1  
    {"seq_id": 2, "user": 1, "sequence": [5, 6]},         # u2
    {"seq_id": 3, "user": 1, "sequence": [5, 6, 7]},      # u2
    {"seq_id": 4, "user": 2, "sequence": [2, 3, 4]},      # u3
    {"seq_id": 5, "user": 2, "sequence": [8, 9]},         # u3
]

print(f"\n📋 子序列数据:")
for data in sequences_data:
    print(f"  seq_{data['seq_id']}: u{data['user']+1} -> {data['sequence']}")

# ============================================================================
# Step 1: 提特征（最后一步表示）
# ============================================================================
print(f"\n🧠 Step 1: 提取序列特征（最后一步表示）")

# 手工设定的最后一步表示（按张哥的数据）
sequence_features = torch.tensor([
    [0.9, 0.8, 0.7, 0.7],  # h_0 (seq 0)
    [0.8, 0.7, 0.8, 0.6],  # h_1 (seq 1)
    [0.1, 0.2, 0.1, 0.2],  # h_2 (seq 2)
    [0.2, 0.2, 0.1, 0.1],  # h_3 (seq 3)
    [0.7, 0.6, 0.6, 0.7],  # h_4 (seq 4)
    [0.2, 0.1, 0.2, 0.2],  # h_5 (seq 5)
], dtype=torch.float32)

print(f"  特征矩阵 H: {sequence_features.shape}")
for i, feat in enumerate(sequence_features):
    print(f"    h_{i} = {feat.tolist()}")

# ============================================================================
# Step 2: KMeans聚类
# ============================================================================
print(f"\n🔄 Step 2: KMeans聚类 (K={K})")

# 手工设定的簇中心（按张哥的例子）
cluster_centers = torch.tensor([
    [0.65, 0.62, 0.64, 0.66],  # c_0
    [0.18, 0.18, 0.15, 0.17],  # c_1  
    [0.80, 0.78, 0.76, 0.74],  # c_2
], dtype=torch.float32)

print(f"  簇中心:")
for i, center in enumerate(cluster_centers):
    print(f"    c_{i} = {center.tolist()}")

# 计算每个序列到各簇中心的距离，找最近簇
distances = torch.cdist(sequence_features, cluster_centers)  # [6, 3]
seq2cluster = distances.argmin(dim=1)  # [6]

print(f"\n  序列到簇的分配:")
print(f"    距离矩阵: {distances.shape}")
for i, cluster_id in enumerate(seq2cluster):
    dist = distances[i, cluster_id].item()
    print(f"    seq_{i} -> 簇_{cluster_id.item()} (距离: {dist:.3f})")

# 构建 cluster2sequences 映射
cluster2sequences = {i: [] for i in range(K)}
for seq_id, cluster_id in enumerate(seq2cluster):
    cluster2sequences[cluster_id.item()].append(seq_id)

print(f"\n  簇到序列的映射:")
for cluster_id, seq_ids in cluster2sequences.items():
    print(f"    簇_{cluster_id}: {seq_ids}")

# ============================================================================
# Step 3: 构建"用户 → 意图分布"
# ============================================================================
print(f"\n📈 Step 3: 构建用户意图分布")

# 统计每个用户的序列落在哪些簇
user_cluster_counts = torch.zeros(num_users, K)  # [3, 3]

for data in sequences_data:
    user_id = data['user']
    seq_id = data['seq_id']
    cluster_id = seq2cluster[seq_id].item()
    user_cluster_counts[user_id, cluster_id] += 1

print(f"  用户-簇计数矩阵:")
for i in range(num_users):
    print(f"    u{i+1}: {user_cluster_counts[i].tolist()}")

# 行归一化得到概率分布
user_intent_hist = user_cluster_counts / (user_cluster_counts.sum(dim=1, keepdim=True) + 1e-8)

print(f"\n  用户意图分布 (归一化后):")
for i in range(num_users):
    print(f"    u{i+1}: {user_intent_hist[i].tolist()}")

# ============================================================================
# Step 4: 单意图向量构造
# ============================================================================
print(f"\n🎯 Step 4: 单意图向量构造")

# 简化的 intent_mlp (线性层 + GELU)
class SimpleIntentMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.linear(x))

intent_mlp = SimpleIntentMLP(K, D)

# 为每个用户生成意图向量
intent_vectors = intent_mlp(user_intent_hist)  # [3, 4]

print(f"  Intent MLP: {K} -> {D}")
print(f"  用户意图向量:")
for i in range(num_users):
    print(f"    u{i+1}: {intent_vectors[i].detach().numpy()}")

# ============================================================================
# Step 5: 训练时的单意图流程演示
# ============================================================================
print(f"\n🏃 Step 5: 训练流程演示 (以u3为例)")

# 模拟BSARec双分支
class ToyBSARec(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 简化：用随机初始化的线性层模拟双分支
        self.freq_branch = nn.Linear(hidden_size, hidden_size)  # 频域分支
        self.attn_branch = nn.Linear(hidden_size, hidden_size)  # 注意力分支
        
    def get_short_term_repr(self, seq_repr):
        """频域分支 - 短期兴趣"""
        return torch.tanh(self.freq_branch(seq_repr))
    
    def get_long_term_repr(self, seq_repr):  
        """注意力分支 - 长期兴趣"""
        return torch.tanh(self.attn_branch(seq_repr))

# 物品嵌入表
item_embeddings = nn.Embedding(num_items + 1, D)  # +1 for padding

# BSARec模型
bsarec = ToyBSARec(D)

# 以u3的一个样本为例: input_seq=[2,3,4], target=7
user_id = 2  # u3
input_seq = torch.tensor([2, 3, 4, 0, 0])  # padding到长度5
target = 7

print(f"  用户: u{user_id+1}")
print(f"  输入序列: {input_seq.tolist()}")
print(f"  目标物品: {target}")

# 1. 获取序列表示（简化：用最后一个非零位置）
seq_repr = sequence_features[4]  # 对应seq_4的特征
print(f"  序列表示: {seq_repr.tolist()}")

# 2. BSARec双分支提取
short_term = bsarec.get_short_term_repr(seq_repr)  # [D]
long_term = bsarec.get_long_term_repr(seq_repr)    # [D]
intent_vec = intent_vectors[user_id]               # [D]

print(f"  短期兴趣: {short_term.detach().numpy()}")
print(f"  长期兴趣: {long_term.detach().numpy()}")  
print(f"  意图向量: {intent_vec.detach().numpy()}")

# 3. 扩散目标
x_0 = item_embeddings(torch.tensor(target))  # [D]
print(f"  目标嵌入 x_0: {x_0.detach().numpy()}")

# 4. 模拟扩散过程
timesteps = 32
t = torch.randint(0, timesteps, (1,)).item()
print(f"  随机时间步 t: {t}")

# 简化的前向加噪
alpha_bar = 0.5  # 简化的噪声调度
noise = torch.randn_like(x_0)
x_noisy = (alpha_bar ** 0.5) * x_0 + ((1 - alpha_bar) ** 0.5) * noise

print(f"  噪声: {noise.detach().numpy()}")
print(f"  加噪后 x_noisy: {x_noisy.detach().numpy()}")

# 5. 五路融合 + 去噪器
class ToyDiffuser(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, fused_input):
        return self.net(fused_input)

diffuser = ToyDiffuser(5 * D, D)  # 5路融合

# 时间嵌入（简化）
time_emb = torch.sin(torch.tensor(t, dtype=torch.float32)).repeat(D)

# CFG条件随机置空（训练时，概率p=0.1）
p = 0.1
short_term_original = short_term.clone()
intent_vec_original = intent_vec.clone()

if torch.rand(1) < p:
    short_term = torch.zeros_like(short_term)
    print(f"  [CFG] 短期兴趣被置空")
if torch.rand(1) < p:
    intent_vec = torch.zeros_like(intent_vec)
    print(f"  [CFG] 意图向量被置空")

# 五路融合
fused_input = torch.cat([x_noisy, short_term, intent_vec, long_term, time_emb])
print(f"  融合输入: {fused_input.shape} (5 * {D} = {5*D})")

# 去噪预测
predicted_x_0 = diffuser(fused_input)
print(f"  预测 x_0: {predicted_x_0.detach().numpy()}")

# 6. 损失计算演示
diff_loss = F.mse_loss(predicted_x_0, x_0)
print(f"  扩散损失: {diff_loss.item():.4f}")

# ============================================================================
# Step 6: 推理演示
# ============================================================================
print(f"\n🔮 Step 6: 推理演示 (CFG采样)")

def simple_cfg_sampling(user_id, seq_repr, timesteps=10, w=2.0):
    """简化的CFG采样"""
    # 获取条件
    short_term = bsarec.get_short_term_repr(seq_repr)
    long_term = bsarec.get_long_term_repr(seq_repr)
    intent_vec = intent_vectors[user_id]
    
    # 从纯噪声开始
    x_t = torch.randn(D)
    
    for t in reversed(range(timesteps)):
        # 条件分支
        time_emb = torch.sin(torch.tensor(t, dtype=torch.float32)).repeat(D)
        fused_cond = torch.cat([x_t, short_term, intent_vec, long_term, time_emb])
        x_0_cond = diffuser(fused_cond)
        
        # 无条件分支（短期和意图置零）
        fused_uncond = torch.cat([x_t, torch.zeros_like(short_term), 
                                torch.zeros_like(intent_vec), long_term, time_emb])
        x_0_uncond = diffuser(fused_uncond)
        
        # CFG组合
        x_0_guided = (1 + w) * x_0_cond - w * x_0_uncond
        
        # 简化的DDPM采样（直接用预测的x_0）
        if t > 0:
            noise_scale = 0.1
            x_t = x_0_guided + noise_scale * torch.randn_like(x_0_guided)
        else:
            x_t = x_0_guided
    
    return x_t

# 对u3进行推理
final_x_0 = simple_cfg_sampling(user_id=2, seq_repr=sequence_features[4])
print(f"  最终采样结果: {final_x_0.detach().numpy()}")

# 计算推荐分数
all_item_embeddings = item_embeddings.weight[1:]  # 排除padding
scores = torch.matmul(final_x_0, all_item_embeddings.T)  # [num_items]
top_k_items = torch.topk(scores, k=5).indices + 1  # +1因为物品ID从1开始

print(f"  推荐分数: {scores.detach().numpy()}")
print(f"  Top-5推荐: {top_k_items.tolist()}")

# ============================================================================
# Step 7: 总结
# ============================================================================
print(f"\n🎉 Step 7: 流程总结")
print(f"  ✅ 聚类: 6个子序列 -> 3个簇")
print(f"  ✅ 用户意图分布: 每个用户的簇频次归一化")  
print(f"  ✅ 单意图向量: MLP映射分布到固定维度")
print(f"  ✅ BSARec双分支: 频域(短期) + 注意力(长期)")
print(f"  ✅ 五路融合: 噪声+短期+意图+长期+时间")
print(f"  ✅ CFG训练: 条件随机置空")
print(f"  ✅ CFG推理: 条件/无条件双分支指导")

print(f"\n💡 关键理解:")
print(f"  • 单意图 = 用户意图分布通过MLP映射为单个向量")
print(f"  • BSARec = 双分支架构捕捉短期(频域)和长期(注意力)兴趣")
print(f"  • 扩散 = 直接预测x₀，不是预测噪声")
print(f"  • CFG = 训练时条件dropout，推理时双分支指导")
print(f"  • 五路融合 = 噪声状态+短期+意图+长期+时间的完整条件")
