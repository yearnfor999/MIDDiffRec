# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Swish 和 SiLU 是同一个激活函数，只是命名不同
def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # 输入 x 形状 [..., H]（典型是 [B, S, H]），B（batch_size）：批大小，S（seq_len）：序列长度（推荐序列里的物品数），H（hidden_size）：每个序列位置的向量维度，也就是 embedding 的长度、Transformer 中的隐藏层大小
        # u = x.mean(-1, keepdim=True)：按最后一维求均值 μ,按最后一维做均值：形状 [*, H] -> [*, 1]
        # keepdim=True 让维度不丢失，方便后续广播
        u = x.mean(-1, keepdim=True)
        # 先中心化 (x - u)，再平方、再按最后一维取均值，得到方差 σ²
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # 标准化
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        # LayerNorm =
        # 1️⃣ 归一化：对每个样本、每个时间步的 H 维向量做零均值、单位方差的标准化
        # 2️⃣ 仿射变换：乘 γ、加 β，让模型有能力调整归一化后的特征分布

        # 仿射变换（可学习的缩放和平移），是 LayerNorm 的第二个关键步骤
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        # 物品嵌入表，形状 (item_size, hidden_size)；padding_idx=0 表示 id=0 的向量不会被更新（专用 padding）
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # 位置编码 (max_seq_length, hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        # input_ids：形状 [B, S] 的长整型张量，元素是物品 ID（含 0=PAD）
        seq_length = input_ids.size(1) # 当前序列长度 S
        # 先构造 [0,1,...,L-1]（形状 [L]） 得到一维向量 [0, 1, 2, ..., S-1]，放在与输入相同的设备上（CPU/GPU）
        # 形状 [S]，dtype long（embedding 查表要求 long）
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # (S,)先 unsqueeze(0) → [1, S]，再 expand_as(sequence) → [B, S]
        # 这样每个 batch 的每个位置都有相同的位置索引
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # 用物品 ID 查 nn.Embedding(item_size, hidden_size)
        # 形状从 [B, S] 变成 [B, S, D] ,其中 D 为嵌入维度
        # 若某位置是 0（padding），由于 padding_idx=0，返回的是固定的 padding 向量（通常是 0 且不更新）
        items_embeddings = self.item_embeddings(input_ids)
        # 用位置 ID 查 nn.Embedding(max_seq_length, hidden_size)
        # 形状同样从 [B, S] 变成 [B, S, D]
        position_embeddings = self.position_embeddings(position_ids)
        # 物品语义 + 位置信息 → 模型就既知道“这是哪个物品”，也知道“它在序列中的位置”
        embeddings = items_embeddings + position_embeddings # 这里是 逐元素数值相加（element-wise addition），所以维度不会变 (256,50,64)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]
        
        :return: A 3d tensor with shape of (N, T_q, C)
        
        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res
        

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads # 注意力头数
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads) # 单头维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 所有头合并后的维度与 hidden_size(分头之前的) 维度相同

        self.query = nn.Linear(args.hidden_size, self.all_head_size) # 查询权重矩阵 WQ
        self.key = nn.Linear(args.hidden_size, self.all_head_size) # 键权重矩阵 WK
        self.value = nn.Linear(args.hidden_size, self.all_head_size) # 值权重矩阵 WV

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size) # Linear(in_features=64, out_features=64, bias=True)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    # 拆头
    # x.size()[:-1]：去除最后一维外的形状（这里是 [B,S]）
    # + (h, d)：拼上“注意力头数”和“每头维度”，得到新形状 [B,S,h,d]
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 把 [B,S,H] 重排为 [B,S,h,d]，前提是 H==h*d（所以要整除）
        # 例子：[2,4,8] → [2,4,2,4]
        x = x.view(*new_x_shape)
        # .permute(0, 2, 1, 3)：[B,S,h,d] → [B,h,S,d]
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor) # 查询矩阵 Q (256,50,64)
        mixed_key_layer = self.key(input_tensor) # 键矩阵 K (256,50,64)
        mixed_value_layer = self.value(input_tensor) # 值矩阵 V (256,50,64)

        # [batch_size heads seq_len head_size]
        # transpose_for_scores （拆头）: [B,S,H] → [B,S,h,d]  其中：h和d分别为“注意力头数”和“每头维度
        # .permute(0, 2, 1, 3)：[B,S,h,d] → [B,h,S,d]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 每个查询位置 t 对所有键位置 k 的相似度分数
        # key_layer.transpose(-1, -2): [B,h,S,d] → [B, h, d, S]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 缩放（Scaled Dot-Product）
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # 加 mask：在 softmax 之前，把不该看的分数变成很大的负数（比如 -1e4），这样 softmax 后它们的概率几乎为 0
        # 加掩码（padding+因果）  具体解释见 Typora
        # attention_scores: [B,h,S,S], attention_mask: [B,1,1,S]    # attention_mask 形状 [B,1,1,S]，广播到 [B,h,S,S] 按“列”相加
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # 在最后一维（key 轴 S）做 softmax，得到注意力概率 [B,h,S,S]，每个查询位置 t 对所有 key 的权重和为 1
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # 得到注意力权重矩阵 (256, 1, 50, 50)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        # 对概率做 Dropout
        attention_probs = self.attn_dropout(attention_probs)
        # [B,h,S,S] @ [B,h,S,d] → [B,h,S,d]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [B,h,S,d] → [B,S,h,d]  .contiguous() 以便后续 view  .contiguous() = 把张量的内存重新排成一块连续的区域，方便 view 按新的形状“平铺”解释  .permute() 改了维度顺序 → 内存布局变成非连续  .contiguous() 保证内存连续
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 计算合并形状：context_layer.size()[:-2] 是 [B,S]，self.all_head_size=h*d=H；
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # (256,50,64)
        # 合并头：[B,S,h,d] → [B,S,H]
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # 返回 [B,S,H]，可直接接 FFN 子层或下一层注意力
        return hidden_states


# FeedForward
class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        # 第一层线性映射：Linear(in_features=64, out_features=256, bias=True)
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        # 若传的是字符串（如 "gelu", "relu", "swish"），就从字典 ACT2FN 里取对应函数
        if isinstance(args.hidden_act, str):
            # gelu
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else: # 否则就直接用传入的可调用对象（可以是 nn.GELU()、F.gelu 等）
            self.intermediate_act_fn = args.hidden_act

        # 第二层线性映射：Linear(in_features=256, out_features=64, bias=True)。把维度降回隐藏维度，便于与残差相加
        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        # 自定义 LayerNorm
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        #  总结流程：
        # 升维 (H→4H) → 激活 → 降维 (4H→H) → Dropout → 残差 + LayerNorm

        # [B,S,H]→[B,S,4H]
        hidden_states = self.dense_1(input_tensor)
        # gelu,非线性激活  形状不变[B,S,4H]→[B,S,4H]
        hidden_states = self.intermediate_act_fn(hidden_states)

        # [B,S,4H]→[B,S,H]
        hidden_states = self.dense_2(hidden_states)
        # 形状不变[B,S,H]→[B,S,H]
        hidden_states = self.dropout(hidden_states)
        # 残差连接 + LayerNorm  形状不变[B,S,4H]→[B,S,4H]
        # 残差连接：hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args) # 实例化，只调用 init 部分
        self.intermediate = Intermediate(args) # 实例化，只调用 init 部分

    def forward(self, hidden_states, attention_mask):
        # hidden_states ：[B, S, H]，attention_mask ： [B,1,1,S]
        # self.layer(...) 做多头注意力：
        # 内部流程：Q/K/V → 点积打分/缩放 → +mask → softmax → @V → 拼头 → 线性 → dropout → Post-LN
        # 形状保持 [B, S, H] (256,50,64)
        attention_output = self.attention(hidden_states, attention_mask) # 调用 MultiHeadAttention 的 forward()
        # self.feed_forward(...)：
        # 内部流程：Linear(H→4H) → 激活 → Linear(4H→H) → dropout → Post-LN
        # 形状仍是 [B, S, H] (256,50,64)
        # 返回这一层的输出，供下一层用
        intermediate_output = self.intermediate(attention_output) # 调用 FeedForward 的 forward()
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args) # 实例化，只调用 Layer 类的 init 部分
        # 建一个长度为 num_hidden_layers 的层列表，每个元素都是 深拷贝 的独立层（参数互不共享）
        # 深拷贝的意义：如果不深拷贝而复用同一个 block 对象，会造成参数共享（所有层是同一套权重），通常不是我们想要的
        # 用 nn.ModuleList 的原因：让 PyTorch 正确注册这些子模块的参数（普通 Python list 不会被自动注册）
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)]) # num_hidden_layers 为 2, 两层

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # 这边没有做，把输入也存进去
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask) # 调用 Layer 的 forward()
            # 如果要求“输出所有层”，每跑完一层就把该层输出追加进列表
            # 列表顺序 = [第1层输出, 第2层输出, …, 第N层输出]
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        # 如果不需要所有中间层，那只在最后把最终一层的输出追加进去
        # 列表顺序 = [第N层输出]
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

# 把「步数 t」变成一条 sin/cos 多频率 的向量（Transformer 同款），再交给 MLP 做可学习变换
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 64

    def forward(self, time):
        device = time.device
        # 将维度一分为二：前 D/2 维存 sin，后 D/2 维存 cos
        half_dim = self.dim // 2   # 32
        # 计算对数频率步长： Δ=ln(10000)/(half_dim−1)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 广播相乘：把每个样本的时间步乘上每个频率
        # 形状：time[:, None] 是 [B, 1]，embeddings[None, :] 是 [1, half_dim]，相乘得到 [B, half_dim]
        # 语义：为 batch 中每个 t 生成一组角度 t⋅freq
        embeddings = time[:, None] * embeddings[None, :] # (256,32)
        # 对这组角度分别取 sin 和 cos，再在最后一维拼接
        # 形状：[B, half_dim] 拼接后得到 [B, 2*half_dim]，如果 dim 是偶数，则正好 [B, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (256,64)
        # 返回 [B, D] 的步嵌入，后续与 x（当前状态）和 s（条件向量）拼接喂入去噪头
        return embeddings