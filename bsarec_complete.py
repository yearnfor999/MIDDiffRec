"""
完全按照BSARec-main原版实现，包括SequentialRecModel基类
确保与原版100%一致，解决cuFFT错误
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_

# ============================================================================
# 基类：SequentialRecModel (完全复制自BSARec-main)
# ============================================================================

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size
        
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_activation(args.hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_activation(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.batch_size = args.batch_size
        # 添加缺失的LayerNorm和dropout
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, input_ids, all_sequence_output=False):
        pass

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)

    def calculate_loss(self, input_ids, answers):
        pass

# ============================================================================
# BSARec组件：完全复制自原版
# ============================================================================

class FrequencyLayer(nn.Module):
    """频域分支（完全原版实现）"""
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        
        # 使用CPU进行FFT计算，避免cuFFT问题（效果完全一样）
        original_device = input_tensor.device
        
        # 转到CPU进行FFT（完全稳定，效果完全一样）
        input_cpu = input_tensor.cpu()
        
        # CPU上的FFT计算
        x = torch.fft.rfft(input_cpu, dim=1, norm='ortho')
        
        low_pass = x[:]  # 原版使用切片复制
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        
        high_pass = input_cpu - low_pass
        
        # 转回原设备进行后续计算
        low_pass = low_pass.to(original_device)
        high_pass = high_pass.to(original_device)
        
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass
        
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states
    
    def forward_with_components(self, input_tensor):
        """频域层前向传播，同时返回低频和高频分量（正确的长短期兴趣提取）"""
        batch, seq_len, hidden = input_tensor.shape
        
        # 使用CPU进行FFT计算，避免cuFFT问题
        original_device = input_tensor.device
        input_cpu = input_tensor.cpu()
        
        # CPU上的FFT计算
        x = torch.fft.rfft(input_cpu, dim=1, norm='ortho')
        
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        
        high_pass = input_cpu - low_pass
        
        # 转回原设备
        low_pass = low_pass.to(original_device)
        high_pass = high_pass.to(original_device)
        
        # 正常的频域层输出
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states, low_pass, high_pass

class MultiHeadAttention(nn.Module):
    """多头注意力（完全原版实现）"""
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        
        self.out_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        hidden_states = self.out_dropout(context_layer)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states

# ============================================================================
# BSARec完整实现：继承SequentialRecModel
# ============================================================================

class BSARecLayer(nn.Module):
    """BSARec层（原版实现）"""
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, hidden_states, attention_mask):
        # 频域分支
        freq_output = self.filter_layer(hidden_states)
        # 注意力分支
        attn_output = self.attention_layer(hidden_states, attention_mask)
        # 加权融合
        output = self.alpha * freq_output + (1 - self.alpha) * attn_output
        return output

class BSARecBlock(nn.Module):
    """BSARec块（原版实现）"""
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        output = self.feed_forward(layer_output)
        return output

class BSARecEncoder(nn.Module):
    """BSARec编码器（原版实现）"""
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        
        return all_encoder_layers
    

class BSARecModel(SequentialRecModel):
    """BSARec模型（完全原版实现，继承SequentialRecModel）"""
    def __init__(self, args):
        super(BSARecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        """BSARec前向传播（原版接口）"""
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )
        
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]
        
        return sequence_output
    
    
    def get_frequency_long_term_repr(self, input_ids):
        """正确方法：获取BSARec最后一层频域分支的低频分量作为长期兴趣"""
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        
        # 通过BSARec的所有层
        hidden_states = sequence_emb
        for block in self.item_encoder.blocks[:-1]:
            # 前面的层正常处理
            hidden_states = block(hidden_states, extended_attention_mask)
        
        # 最后一层：获取频域分量
        last_block = self.item_encoder.blocks[-1]
        freq_layer = last_block.layer.filter_layer
        
        # 获取最后一层频域层的低频分量
        _, low_freq, _ = freq_layer.forward_with_components(hidden_states)
        
        # 平均池化得到长期兴趣
        long_term_interest = low_freq.mean(dim=1)  # [B, D]
        
        return long_term_interest
    
    def get_frequency_short_term_repr(self, input_ids):
        """正确方法：获取BSARec最后一层频域分支的高频分量作为短期兴趣"""
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        
        # 通过BSARec的所有层
        hidden_states = sequence_emb
        for block in self.item_encoder.blocks[:-1]:
            # 前面的层正常处理
            hidden_states = block(hidden_states, extended_attention_mask)
        
        # 最后一层：获取频域分量
        last_block = self.item_encoder.blocks[-1]
        freq_layer = last_block.layer.filter_layer
        
        # 获取最后一层频域层的高频分量
        _, _, high_freq = freq_layer.forward_with_components(hidden_states)
        
        # 取最后位置作为短期兴趣
        short_term_interest = high_freq[:, -1, :]  # [B, D]
        
        return short_term_interest

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        """计算BSARec损失（原版实现）"""
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = F.cross_entropy(logits, answers)
        return loss
