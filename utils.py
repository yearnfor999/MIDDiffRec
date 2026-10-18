import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def show_args_info(args):
    print(f"=======================Configure Info:=======================")
    # vars(args) 会把 Namespace 转换成一个字典，key 是参数名，value 是参数值
    for arg in vars(args):
        # getattr(args, arg) → 取出参数名对应的值
        value = getattr(args, arg)
        # 特殊处理字典类型，避免格式化错误
        if isinstance(value, dict):
            if arg == 'user_id_map':
                # 对于用户ID映射，只显示映射数量而不显示完整字典
                print(f"{arg:<30} : {f'<dict with {len(value)} mappings>':>35}")
            else:
                print(f"{arg:<30} : {str(value):>35}")
        else:
            print(f"{arg:<30} : {value:>35}")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")

# 没有用到
def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

# 只要把“按步的系数表 [T]”变成“按 batch 逐样本取值并能和 x 广播”的张量，就需要 extract()
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    # a 就是存放 \sqrt{\bar\alpha_t}或 \sqrt{1 - \bar\alpha_t} 的表
    # extract(a, t, x_shape) 就是从这个「累乘表」里，拿出当前 batch 每个样本对应的步数 𝑡 的系数，并 reshape 成可以和 x_start 广播的形状
    # 有一张全局的系数表 a（长度 T，比如存的是 \sqrt{\bar\alpha_t}
    # 你有一个 batch 的时间步索引 t（形状 [B]），比如 [15, 27, 400, …]
    # 目的：按 batch 每个样本自己的 t[i]，从表里取对应的系数 a[t[i]]

    # 举例
    # a = torch.tensor([10, 20, 30, 40, 50])   # shape [5]
    # t = torch.tensor([0, 2, 4])              # shape [3]
    # out = a.gather(-1, t)   # 等价于 [a[0], a[2], a[4]]
    # print(out)  # [10, 30, 50]
    out = a.gather(-1, t.cpu())
    # reshape：把 [B] 变成 [B, 1, 1, ..., 1]，长度和 x_shape 一样，只在第 0 维是 B，其余都是 1
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device) 

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: 
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

# 举例：
# 设 batch_size=3, padded_seq_len=6, input_seq_len=[2,4,3], max_item=10
# 可能生成（随机示例）：
# item_ids(0): [3,7]          → padded: [0,0,0,0,3,7]
# item_ids(1): [5,4,8,2]      → padded: [0,0,5,4,8,2]
# item_ids(2): [6,1,3]        → padded: [0,0,0,6,1,3]

# generate_padded_sequences_tensor：随机生成一批右对齐、用 0 填充的序列，并转成 torch.long 张量
def generate_padded_sequences_tensor(batch_size, padded_seq_len, max_item, input_seq_len):
    # 定义函数，参数含义：
    # batch_size：批大小 B
    # padded_seq_len：补齐后的统一长度 L
    # max_item：物品 ID 的上界
    # input_seq_len：记录每条样本的实际长度（未补齐）， 比如：input_seq_len = [2, 4, 3]，样本0：真实长度 2 ，样本1：真实长度 4 ，样本2：真实长度 3

    # 装每条序列（先放 numpy 数组，最后再堆成 2D）
    sequences = []

    # 循环 B 次，逐个样本生成
    for index in range(batch_size):
        # 取出第 index 条样本的实际长度 actual_length
        actual_length = input_seq_len[index]
        # 生成一条随机物品序列，长度 = actual_length
        # 取值范围是 [1, max_item-2)（上界不含）：
        # 这样就避开 0（留给 PAD），也避开最后两个 ID（多半留给特殊符号，如 [MASK]/[CLS]/[EOS] 等）
        item_ids = np.random.randint(1, max_item - 2, size=actual_length)
        # 先建一条全 0 的序列，长度 L。0 即 PAD token
        padded_sequence = np.zeros(padded_seq_len, dtype=int)
        # 右对齐地放入真实 token：把最后 actual_length 个位置替换为 item_ids。左边（开头）的部分仍为 0（PAD）
        padded_sequence[-actual_length:] = item_ids
        # 放进列表
        sequences.append(padded_sequence)

    # 把列表堆成二维数组，形状 [batch_size, padded_seq_len]
    sequences_array = np.array(sequences)
    # 转为 PyTorch 张量，类型 long（整数），便于后续embedding 查表等操作（nn.Embedding 要求 long）
    # 返回形状 [B, L] 的 LongTensor，内容是右对齐 + 左侧 0-PAD 的随机序列
    sequences_tensor = torch.tensor(sequences_array, dtype=torch.long)  
    
    return sequences_tensor


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False # 只要有任意一个指标“明显变好”，就算有改进
        return True # 两个指标都没有超过 best+delta → 视为“没有改进”

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
