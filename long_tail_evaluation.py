"""
Step 2: Long-Tail Case Study - 对比两种训练方式

对比：
1. Single-Intent: 用单意图训练的模型（原生推理）
2. Multi-Intent: 用多意图训练的模型（原生推理）

按物品热度分桶（Head/Mid/Tail）对比HR@20和NDCG@20

用法:
    python long_tail_evaluation.py
"""

import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from models import InDiRec
from datasets import get_seqs_and_matrixes, DatasetForInDiRec
from utils import set_seed

# ==================== 配置 ====================
DATA_NAME = "Beauty"  # 当前：Beauty | 原来：Video
# 模型路径
MULTI_INTENT_MODEL = "output/InDiRec-Beauty-91.pt"   # 多意图训练的模型
SINGLE_INTENT_MODEL = "output/InDiRec-Beauty-160.pt"  # 单意图训练的模型
# 原来Video: MULTI=InDiRec-Video-114.pt, SINGLE=InDiRec-Video-163.pt
INTENT_NUM = 32  # Beauty: 32 | Video: 128
CUDA = 0
TOP_K = 20  # HR@20, NDCG@20

# ========================================================

set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA) if CUDA >= 0 else ""
device = torch.device("cuda:0" if torch.cuda.is_available() and CUDA >= 0 else "cpu")

print(f"使用设备: {device}")

# ==================== 加载物品热度分桶 ====================
print("\n" + "="*60)
print("1. 加载物品热度分桶...")
print("="*60)

popularity_file = f"./popularity_analysis/{DATA_NAME}_popularity_buckets.pkl"
with open(popularity_file, 'rb') as f:
    popularity_data = pickle.load(f)

head_items = popularity_data['head_items']
mid_items = popularity_data['mid_items']
tail_items = popularity_data['tail_items']

print(f"Head物品数: {len(head_items)}")
print(f"Mid物品数: {len(mid_items)}")
print(f"Tail物品数: {len(tail_items)}")

# ==================== 加载数据 ====================
print("\n" + "="*60)
print("2. 加载数据集...")
print("="*60)

data_dir = "./datasets/"
data_file = data_dir + DATA_NAME + ".txt"
segmented_file = data_dir + DATA_NAME + "_s.txt"

eval_user_ids, user_seq, max_item, _, test_rating_matrix = get_seqs_and_matrixes("rating", data_file)
train_user_ids, train_seq = get_seqs_and_matrixes("training", segmented_file)

sorted_user_ids = sorted(set(eval_user_ids))
user_id_map = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
eval_user_indices = [user_id_map[uid] for uid in eval_user_ids]
train_user_indices = [user_id_map[uid] for uid in train_user_ids if uid in user_id_map]

print(f"数据集: {DATA_NAME}")
print(f"用户数: {len(sorted_user_ids)}, 物品数: {max_item}")

# ==================== 创建模型参数 ====================
class Args:
    def __init__(self):
        self.num_users = len(sorted_user_ids)
        self.item_size = max_item + 2
        self.mask_id = max_item + 1
        self.train_matrix = test_rating_matrix
        self.hidden_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.hidden_act = "gelu"
        self.attention_probs_dropout_prob = 0.5
        self.hidden_dropout_prob = 0.5
        self.max_seq_length = 50
        self.intent_num = INTENT_NUM
        self.timesteps = 200
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.diffuser_type = "mlp1"
        self.beta_sche = "exp"
        self.w = 2.0
        self.p = 0.1
        self.alpha = 0.4
        self.c = 10
        self.batch_size = 256
        self.lr = 0.001
        self.weight_decay = 0.0
        self.initializer_range = 0.02
        self.noise_ratio = 0.0
        self.sim = "dot"
        self.temperature = 1.0

args = Args()

# ==================== 加载模型 ====================
print("\n" + "="*60)
print("3. 加载两个模型...")
print("="*60)

# 多意图模型
model_multi = InDiRec(device, args=args)
model_multi.load_state_dict(torch.load(MULTI_INTENT_MODEL, map_location=device))
model_multi.to(device)
model_multi.eval()
print(f"✓ 多意图模型加载完成: {MULTI_INTENT_MODEL}")

# 单意图模型
model_single = InDiRec(device, args=args)
model_single.load_state_dict(torch.load(SINGLE_INTENT_MODEL, map_location=device))
model_single.to(device)
model_single.eval()
print(f"✓ 单意图模型加载完成: {SINGLE_INTENT_MODEL}")

# ==================== 创建测试数据加载器 ====================
test_dataset = DatasetForInDiRec(args, user_seq, user_ids=eval_user_indices, data_type="test")
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                            batch_size=args.batch_size, drop_last=False)

# ==================== 评估两个模型（都用原生推理）====================
print("\n" + "="*60)
print("4. 评估两个模型...")
print("="*60)

def evaluate_model(model, model_name):
    """
    评估单个模型（原生推理：seq_output @ item_emb）
    
    Args:
        model: 加载好的模型
        model_name: 模型名称
    """
    print(f"\n评估模型: {model_name}")
    print("-"*40)
    
    all_predictions = []
    all_answers = []
    all_user_ids = []  # 记录用户ID
    all_input_seqs = []  # 记录输入序列
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=model_name):
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, input_seq_len, target_pos, answers = batch
            
            # 原生推理：只用seq_model
            rec_output = model.seq_model(input_ids)[:, -1, :]
            scores = torch.matmul(rec_output, model.item_embeddings.weight.T)
            
            # 转numpy并过滤历史
            scores = scores.detach().cpu().numpy()
            batch_user_index = user_ids.detach().cpu().numpy()
            scores[args.train_matrix[batch_user_index].toarray() > 0] = 0
            
            # Top-K
            ind = np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
            arr_ind = scores[np.arange(len(scores))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind, axis=1)[:, ::-1]
            batch_pred_list = ind[np.arange(len(scores))[:, None], arr_ind_argsort]
            
            all_predictions.append(batch_pred_list)
            all_answers.append(answers.detach().cpu().numpy())
            all_user_ids.append(user_ids.detach().cpu().numpy())
            all_input_seqs.append(input_ids.detach().cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    answers = np.concatenate(all_answers, axis=0)
    user_ids_arr = np.concatenate(all_user_ids, axis=0)
    input_seqs_arr = np.concatenate(all_input_seqs, axis=0)
    
    print(f"✓ {model_name} 评估完成: {len(predictions)} 个样本")
    
    return predictions, answers, user_ids_arr, input_seqs_arr

# 评估两个模型
pred_single, ans_single, user_ids_single, input_seqs_single = evaluate_model(model_single, "Single-Intent")
pred_multi, ans_multi, user_ids_multi, input_seqs_multi = evaluate_model(model_multi, "Multi-Intent")

# 确保答案一致
assert np.array_equal(ans_single, ans_multi), "测试集不一致"
assert np.array_equal(user_ids_single, user_ids_multi), "用户顺序不一致"
answers = ans_single
user_ids_arr = user_ids_single
input_seqs_arr = input_seqs_single

# ==================== 按物品热度分桶计算指标 ====================
print("\n" + "="*60)
print("6. 按物品热度分桶计算指标...")
print("="*60)

def calculate_metrics_by_bucket(predictions, answers, bucket_items, bucket_name):
    """计算特定桶内的HR和NDCG"""
    # 找到目标物品在该桶内的样本
    # answers可能是1D或2D，需要展平
    answers_flat = answers.flatten() if answers.ndim > 1 else answers
    mask = np.array([int(ans) in bucket_items for ans in answers_flat])
    
    if mask.sum() == 0:
        print(f"  {bucket_name}: 无样本")
        return None
    
    bucket_preds = predictions[mask]
    bucket_answers = answers_flat[mask]
    
    # 计算HR和NDCG
    hits = 0
    ndcg_sum = 0.0
    
    for i in range(len(bucket_answers)):
        pred = bucket_preds[i]
        answer = int(bucket_answers[i])  # 确保是标量
        
        if answer in pred:
            hits += 1
            rank = np.where(pred == answer)[0][0] + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)
    
    hr = hits / len(bucket_answers)
    ndcg = ndcg_sum / len(bucket_answers)
    
    print(f"  {bucket_name}: N={len(bucket_answers):,}, HR@{TOP_K}={hr:.4f}, NDCG@{TOP_K}={ndcg:.4f}")
    
    return {
        'bucket': bucket_name,
        'count': len(bucket_answers),
        'hr': hr,
        'ndcg': ndcg
    }

results = {
    'Single-Intent': {},
    'Multi-Intent': {}
}

for strategy_name, preds in [
    ('Single-Intent', pred_single),
    ('Multi-Intent', pred_multi)
]:
    print(f"\n{strategy_name}:")
    for bucket_name, bucket_items in [
        ('Head', head_items),
        ('Mid', mid_items),
        ('Tail', tail_items)
    ]:
        result = calculate_metrics_by_bucket(preds, answers, bucket_items, bucket_name)
        if result:
            results[strategy_name][bucket_name] = result

# ==================== 筛选最佳Tail案例（用于Figure c）====================
print("\n" + "="*60)
print("7. 筛选最佳Tail案例...")
print("="*60)

best_cases = []

for i in range(len(answers)):
    target = int(answers[i])
    
    # 只看Tail物品
    if target not in tail_items:
        continue
    
    # 找目标物品在两个模型推荐列表中的排名
    pred_single_i = pred_single[i]
    pred_multi_i = pred_multi[i]
    
    rank_single_list = np.where(pred_single_i == target)[0]
    rank_multi_list = np.where(pred_multi_i == target)[0]
    
    rank_single = rank_single_list[0] + 1 if len(rank_single_list) > 0 else 999
    rank_multi = rank_multi_list[0] + 1 if len(rank_multi_list) > 0 else 999
    
    # 筛选条件：Multi命中Top-10 且 (Single未命中 或 排名更低)
    if rank_multi <= TOP_K and rank_multi < rank_single:
        improvement = rank_single - rank_multi
        
        # 获取用户ID和输入序列
        user_idx = user_ids_arr[i]
        real_user_id = eval_user_ids[user_idx]
        input_seq = input_seqs_arr[i]
        # 去除padding（0）
        input_seq_clean = input_seq[input_seq > 0].tolist()
        
        best_cases.append({
            'sample_index': i,
            'user_index': int(user_idx),
            'real_user_id': int(real_user_id),
            'input_sequence': input_seq_clean,
            'seq_length': len(input_seq_clean),
            'target_item': target,
            'rank_single': rank_single,
            'rank_multi': rank_multi,
            'improvement': improvement,
            'pred_single_top20': pred_single_i[:20].tolist(),
            'pred_multi_top20': pred_multi_i[:20].tolist()
        })

# 按排名提升和序列长度综合排序
# 优先：序列长度较短（≤20），然后按排名提升排序
short_cases = [c for c in best_cases if c['seq_length'] <= 20]
medium_cases = [c for c in best_cases if 20 < c['seq_length'] <= 30]
long_cases = [c for c in best_cases if c['seq_length'] > 30]

# 分别排序
short_cases_sorted = sorted(short_cases, key=lambda x: x['improvement'], reverse=True)
medium_cases_sorted = sorted(medium_cases, key=lambda x: x['improvement'], reverse=True)
long_cases_sorted = sorted(long_cases, key=lambda x: x['improvement'], reverse=True)

# 优先选择短序列
best_cases_sorted = short_cases_sorted + medium_cases_sorted + long_cases_sorted

print(f"找到 {len(best_cases)} 个符合条件的Tail案例")
print(f"  短序列(≤20): {len(short_cases)}")
print(f"  中等序列(20-30): {len(medium_cases)}")
print(f"  长序列(>30): {len(long_cases)}")

if len(best_cases_sorted) > 0:
    best = best_cases_sorted[0]
    print(f"\n最佳案例:")
    print(f"  样本索引: {best['sample_index']}")
    print(f"  用户索引: {best['user_index']}")
    print(f"  真实用户ID: {best['real_user_id']}")
    print(f"  序列长度: {best['seq_length']}")
    print(f"  输入序列: {best['input_sequence'][:10]}... (前10个物品)")
    print(f"  目标物品: {best['target_item']} (Tail)")
    print(f"")
    print(f"  Single-Intent排名: #{best['rank_single']}")
    print(f"  Multi-Intent排名: #{best['rank_multi']}")
    print(f"  排名提升: {best['improvement']} 位")
    print(f"\n  Single-Intent Top-20: {best['pred_single_top20']}")
    print(f"  Multi-Intent Top-20: {best['pred_multi_top20']}")
else:
    print("⚠️  未找到符合条件的案例")

# ==================== 保存结果 ====================
output_dir = "./long_tail_results/"
os.makedirs(output_dir, exist_ok=True)

result_file = os.path.join(output_dir, f"{DATA_NAME}_long_tail_results.pkl")
with open(result_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\n✓ 结果已保存到: {result_file}")

# 保存最佳案例
if len(best_cases_sorted) > 0:
    case_file = os.path.join(output_dir, f"{DATA_NAME}_best_tail_case.pkl")
    with open(case_file, 'wb') as f:
        pickle.dump({'best_case': best_cases_sorted[0], 
                     'all_cases': best_cases_sorted[:10],  # 保存Top-10案例
                     'popularity_buckets': popularity_data}, f)
    print(f"✓ 最佳案例已保存到: {case_file}")

# ==================== 打印对比 ====================
print("\n" + "="*60)
print("结果对比")
print("="*60)

import pandas as pd

summary = []
for strategy in ['Single-Intent', 'Multi-Intent']:
    for bucket in ['Head', 'Mid', 'Tail']:
        if bucket in results[strategy]:
            r = results[strategy][bucket]
            summary.append({
                'Strategy': strategy,
                'Bucket': bucket,
                'Count': r['count'],
                'HR@20': f"{r['hr']:.4f}",
                'NDCG@20': f"{r['ndcg']:.4f}"
            })

df = pd.DataFrame(summary)
print(df.to_string(index=False))

# 保存CSV
csv_file = os.path.join(output_dir, f"{DATA_NAME}_long_tail_summary.csv")
df.to_csv(csv_file, index=False)
print(f"\n✓ CSV保存到: {csv_file}")

print("\n" + "="*60)
print("✓ Long-Tail评估完成！")
print("="*60)
print(f"\n下一步:")
print(f"  运行: python visualize_long_tail.py")
print("="*60)

