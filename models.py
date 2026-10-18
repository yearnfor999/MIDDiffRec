from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
from modules import *
from utils import extract
from bsarec_complete import BSARecModel



class Diffusion():
    def __init__(self, args):
        self.timesteps = args.timesteps # 扩散步数 T 50
        self.beta_start = args.beta_start # 线性 β 调度用到的起止值 0.0001
        self.beta_end = args.beta_end # 0.02
        self.args = args

        if self.args.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.args.beta_sche == 'exp':
            self.betas = self.exp_beta_schedule(self.timesteps) # (50,)
        elif self.args.beta_sche =='cosine':
            self.betas = self.cosine_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='sqrt':
            self.betas = torch.tensor(self.betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas（timesteps, ）
        # α_t = 1 - β_t ，形状 [T]   α_t 逐渐减小，β_t逐渐增大
        self.alphas = 1. - self.betas
        # torch.cumprod(self.alphas, axis=0)：在第 0 维上做累积连乘，得到一个长度为 timesteps 的向量
        # 详细解释见 Typora
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # torch.sqrt：对张量逐元素开平方
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # posterior_mean_coef1 为均值的 x_0 前面的系数
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # posterior_mean_coef2 为均值的 x_t 前面的系数
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 方差
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        # 线性 β：在 [beta_start, beta_end] 间均匀取 T 个点
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # 生成“指数型”的 β 序列
    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        # 生成长度为 T 的 1D 向量 x，从 1 均匀递增到 2T+1（含端点）
        x = torch.linspace(1, 2 * timesteps + 1, timesteps) # 50
        # β=1−exp(−βmin/T − xt⋅(βmax−βmin)/(2T^2))  β 随 t 增大而增大
        betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)) # 50
        return betas

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


class InDiRec(nn.Module):

    def __init__(self, device, args, num_heads=1):
        super(InDiRec, self).__init__()
        self.state_size = args.max_seq_length # 50
        self.hidden_size = args.hidden_size # 64
        self.item_num = args.item_size # 12103
        self.dropout = nn.Dropout(args.hidden_dropout_prob) # 0.5
        self.diffuser_type = args.diffuser_type # mlp1
        self.device = device # cuda:0
        self.w = args.w # CFG 指导强度 2.0
        self.p = args.p # 条件随机置空概率 0.1

        # 序列编码器：统一使用BSARec
        self.seq_model = BSARecModel(args)
        # 直接使用BSARec的item_embeddings
        self.item_embeddings = self.seq_model.item_embeddings
        
        # BSARec模式：完全基于序列，不使用静态用户嵌入
        self.user_embeddings = None
        self.long_term_mlp = None
        
        # 新增：分布加权原型→非线性变换的两步处理
        # 第一步：分布加权融合（无参数，纯矩阵乘法）
        # 第二步：非线性变换MLP
        self.intent_mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(args.hidden_size, args.hidden_size),  # 输入维度改为hidden_size（原型融合后的维度）
            nn.GELU()  # 改为GELU，避免Sigmoid压缩到[0,1]
        )
        
        # 意图原型中心存储（在build_user_intent_distribution时设置）
        self.intent_centers = None

        self.diff = Diffusion(args) # 实例化 Diffusion

        # 无条件占位向量（给 CFG/条件drop 用） (1,64)
        # 建一个只含 1 个条目的可学习向量表，维度为 D
        # 语义：当你想“无条件”（或把条件 s 随机遮蔽）时，用这个向量当作“空条件 h”
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        # 用较小方差初始化，避免量级远大于item/seq表征导致训练不稳
        # 使用标准的初始化尺度：std=0.02 或 Xavier缩放 std=1/√D
        nn.init.normal_(self.none_embedding.weight, mean=0.0, std=0.02)


        # 把「步数 t」变成一条 sin/cos 多频率 的向量（Transformer 同款），再交给 MLP 做可学习变换
        # 作为“时间条件”参与 diffuser 的预测（和 x、s 一起拼接）
        self.step_mlp = nn.Sequential(
            # 扩散的去噪过程在每个时间步 t 的噪声强度都不同，网络必须“知道现在是第几步”。但直接把一个标量 t 喂进 MLP 不够表达力，因此要先把 t 映射成一个多频率的连续向量
            # 把时间步 t（形如 [B] 的 long / float）映射成 [B, D] 的正弦-余弦位置编码向量（半维 sin、半维 cos，拼起来得 D）
            # SinusoidalPositionEmbeddings 没有可学习权重
            SinusoidalPositionEmbeddings(self.hidden_size), # 实例化   [B] -> [B, D]
            # nn.Linear(D, 2D)
            nn.Linear(self.hidden_size, self.hidden_size*2), # [B, D] -> [B, 2D]
            nn.GELU(),
            # nn.Linear(2D, D)
            nn.Linear(self.hidden_size*2, self.hidden_size), # [B, 2D] -> [B, D]
        )

        # 没有用到
        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(D, 2D)
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        # 没有用到
        self.diff_mlp = nn.Sequential(
            # nn.Linear(3D, 2D)
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            # nn.Linear(2D, D)
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        # 改进的扩散去噪器：根据模式选择输入维度
        # 传统模式：5路融合（x + long_term + short_term + intent + time）
        # BSARec模式：4路融合（x + BSARec_short + intent + BSARec_long + time）
        if hasattr(args, 'use_bsarec') and args.use_bsarec:
            input_dim = self.hidden_size * 5  # BSARec模式：去掉静态用户偏好，仍是5D
        else:
            input_dim = self.hidden_size * 5  # 传统模式：5D
        
        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(input_dim, self.hidden_size)
            )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(input_dim, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )

    def process_intent_distribution(self, user_intent_dist):
        """
        分布加权原型→非线性变换的两步处理
        Args:
            user_intent_dist: [B, intent_num] 用户意图分布
        Returns:
            intent_vectors: [B, hidden_size] 最终的意图条件向量
        """
        # 第一步：分布加权的原型融合
        # user_intent_dist: [B, K], self.intent_centers: [K, D]
        # 矩阵乘法: [B, K] @ [K, D] = [B, D]
        tilde_s = user_intent_dist @ self.intent_centers  # [B, hidden_size]  (256,32)x(32,64)=(256,64)
        
        # 第二步：非线性变换
        intent_vectors = self.intent_mlp(tilde_s)  # [B, hidden_size] (256,64)
        
        return intent_vectors


    # ⚠️ 已废弃：原始的forward函数（3D输入，与diffuser的5D期望不匹配）
    # def forward(self, x, s, step):
    #     #  x:(batch, dim) , h:(batch, dim), t:(batch, )
    #     # x：当前被噪的向量 𝑥𝑡， 形状 [B, D]
    #     # s: 条件向量，形状 [B, D]。"来自 SASRec 的最后一步隐藏态"，也可以换成论文里的同簇原型/邻居聚合向量 𝑆𝑒（意图对齐信号）
    #     # step: 时间步索引 𝑡，形状 [B,]
    # 
    #     # self.step_mlp(step) 告诉去噪器"现在噪到第几步"
    #     t = self.step_mlp(step) # t:(batch, dim)  (256,64)
    # 
    #     if self.diffuser_type == 'mlp1':
    #         res = self.diffuser(torch.cat((x, s, t), dim=1)) # (batch, 3 x dim) -->diffuser-->(batch, dim)  (256,64)
    #     elif self.diffuser_type == 'mlp2':
    #         res = self.diffuser(torch.cat((x, s, t), dim=1))
    #     return res # (batch, dim)

    # 新增：增强的forward函数，融合长期兴趣、短期兴趣、意图信号
    def forward_enhanced(self, x, input_seq, user_ids, intent_centers, step, training=True):
        """
        单意图扩散去噪，融合三种兴趣信息（频域分支版本）
        Args:
            x: [B, D] 当前噪声状态
            input_seq: [B, S] 输入序列（用于长短期兴趣）
            user_ids: [B] 用户ID
            intent_centers: [B, D] 意图中心向量
            step: [B] 时间步
            training: bool 是否在训练阶段（用于条件随机置空和梯度阻断）
        Returns:
            res: [B, D] 去噪结果
        """
        B = x.size(0)

        # 1) 先取长/短期兴趣
        long_term  = self.get_long_term_interest(input_seq)   # [B, D]
        short_term = self.get_short_term_interest(input_seq)  # [B, D]

        # 2) 训练时：阻断 seq_model/意图分支梯度 + 做CFG式条件dropout
        if training:
            # 只让 diff_loss 训练去噪器/时间嵌入等，不去更新 BSARec
            long_term  = long_term.detach()
            short_term = short_term.detach()

            # 条件随机置空（dropout）
            none_vec = self.none_embedding.weight[0].unsqueeze(0)  # [1, D] - 优化：避免重复创建张量
            
            # 对短期兴趣进行条件随机置空（概率为p）
            short_term_mask = (torch.rand(B, device=x.device) <= self.p).float().view(B, 1)
            short_term = (1 - short_term_mask) * short_term + short_term_mask * none_vec
            
            # 对意图信号进行条件随机置空（概率为p）
            intent_mask = (torch.rand(B, device=x.device) <= self.p).float().view(B, 1)
            intent_centers = (1 - intent_mask) * intent_centers + intent_mask * none_vec

        # 3) 时间步嵌入
        time_emb = self.step_mlp(step)  # [B, D]

        # 4) 五路融合并预测
        fused = torch.cat([x, short_term, intent_centers, long_term, time_emb], dim=1)  # [B, 5D]
        return self.diffuser(fused)

    # ⚠️ 已废弃：classifier-free guidance 的无条件分支（3D输入，与diffuser的5D期望不匹配）
    # def forward_uncon(self, x, step):
    #     # x：当前被噪的向量 𝑥𝑡， 形状 [B, D]
    #     # step：时间步索引，形状 [B] 或 [B,1]
    # 
    #     # self.none_embedding 是你之前定义的 Embedding(1, D)
    #     # 从 Embedding(1, D) 里取唯一的索引 0，得到一个的"无条件向量"，形状 [1, D]
    #     # 当条件（同簇意图向量𝑆𝑒）被丢弃时，用这个可学习的"空条件"来占位，让网络学到uncond 分支
    #     h = self.none_embedding(torch.tensor([0]).to(self.device)) # (1,64)
    #     # h 本来是 [1,64]（从 self.none_embedding 取出来的）
    #     # h.view(1,64) → 显式 reshape，保证形状 [1,64]
    #     # [h.view(1,64)] * x.shape[0] → 把这个张量放到一个 list 里，然后复制 B 次
    #     # 比如 x.shape[0]=3，结果是 [h,h,h]
    #     # torch.cat(..., dim=0) → 沿着第 0 维拼接，得到 [3,64]
    #     # 作用：把单个"无条件向量"扩展成一个 batch，保证它能和 x（形状 [B,64]）拼接使用
    #     # 本质就是广播，把"1 行"复制成 "B 行"
    #     h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0) # 把单个 [1,64] 的 h 复制 B 次，拼成 [B,64]，以便和批内每一行对齐
    # 
    #     # 把时间步 step 输入 step_mlp（Sinusoidal + Linear+GELU+Linear），输出 [B,D] 的 时间步嵌入 t
    #     t = self.step_mlp(step)
    # 
    #     if self.diffuser_type == 'mlp1':
    #         # (B,3D) -> (B,D)  去噪器：self.diffuser
    #         # x 给出"当前画面"，h 表示"无条件（空条件）"，t 表示"时间/噪声程度"；三者合起来，网络给出一步去噪的预测（可能是 ^ ϵ 或 ^x0，取决于你训练目标）
    #         res = self.diffuser(torch.cat((x, h, t), dim=1))
    #     elif self.diffuser_type == 'mlp2':
    #         res = self.diffuser(torch.cat((x, h, t), dim=1))
    #     # 输出：unconditional 前向的预测，用于CFG 的 uncond 分支或无条件训练样本的损失
    #     return res

    # 新增：增强的无条件forward函数
    def forward_uncon_enhanced(self, x, step):
        """
        无条件扩散去噪，所有条件都用none_embedding替代
        Args:
            x: [B, D] 当前噪声状态
            step: [B] 时间步
        Returns:
            res: [B, D] 去噪结果
        """
        batch_size = x.shape[0]
        
        # 所有条件都用无条件向量替代
        none_vec = self.none_embedding.weight[0].unsqueeze(0)  # [1, D] - 优化：避免重复创建张量
        none_condition = none_vec.expand(batch_size, -1)  # [B, D]
        
        # 时间步嵌入
        time_emb = self.step_mlp(step)  # [B, D]
        
        # 五路信息融合（全部用无条件向量）
        # ⚠️ 关键修复：保持与forward_enhanced完全一致的拼接顺序
        # forward_enhanced: [x, short_term, intent_centers, long_term, time_emb]
        fused_input = torch.cat([
            x,                      # 当前噪声状态
            none_condition,         # 无条件短期兴趣 (对应short_term位置)
            none_condition,         # 无条件意图信号 (对应intent_centers位置)
            none_condition,         # 无条件长期兴趣 (对应long_term位置)   这边应该保留长期兴趣
            time_emb               # 时间步嵌入
        ], dim=1)  # [B, 5D]
        
        # 扩散去噪
        res = self.diffuser(fused_input)  # [B, 5D] -> [B, D]
        
        return res

    # 把"干净样本"的物品ID 映射到连续向量
    def calculate_x0(self, x0):
        return self.item_embeddings(x0)


    # 构造正常的有条件和无条件占位向量，并按概率 p 做“条件丢弃”→ CFG 的关键
    def calculate_s(self, input_seq, p):

        # input_seq 意图一样的序列
        seq_out = self.seq_model(input_seq) # (B,S,D)  (256,50,64)
        s = seq_out[:,-1,:] # B x Dim  (B,D) (256,64)

        B, D = s.shape[0], s.shape[1] # B = batch_size，D = embedding_dim
        # 为每个样本采一个二值开关（0/1）
        # 1. torch.rand(B)
        # 生成一个长度为 B 的张量，每个元素是 [0,1) 均匀分布的随机数。
        # 例如 B=5，可能得到：
        # [0.12, 0.83, 0.45, 0.05, 0.67]

        # 2. torch.rand(B) - p
        # 从每个随机数中减去丢弃概率 p
        # 设p=0.3，减完后：
        # [ -0.18, 0.53, 0.15, -0.25, 0.37 ]

        # 3. torch.sign(...)
        # 对每个元素取符号：
        # 负数 → -1， 0 → 0， 正数 → +1

        # 上面的例子：
        # [-1, +1, +1, -1, +1]

        # 4. ( ... + 1 ) / 2
        # 得到结果： [0, 1, 1, 0, 1]

        # 5. 含义
        # mask1d[i] = 1 ⇔ rand[i] > p，即保留条件
        # mask1d[i] = 0 ⇔ rand[i] ≤ p，即丢弃条件
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2 # (B,) (256,)
        # 把 1D 向量变成二维列向量，方便后续在特征维复制/广播
        maske1d = mask1d.view(B, 1) # (B, 1) (256,1)
        # 把列向量在特征维复制 D 次，得到 mask 的形状 (B, D)，从而能与 s 逐元素相乘
        mask = torch.cat([maske1d] * D, dim=1) # (B, D) (256,64)
        mask = mask.to(self.device)

        # 条件丢弃（CFG 的“uncond”样本）
        # 当 mask=1（保留）：s = s * 1 + none * 0 = s（正常的有条件）
        # 当 mask=0（丢弃）：s = s * 0 + none * 1 = none（改用无条件占位向量）
        # self.none_embedding 是 nn.Embedding(1, D)：none_embedding([0]) 形状 [1, D]，与 (1 - mask)（[B, D]）相乘时会广播成 [B, D]，从而为该 batch 里被丢弃的样本提供相同的可学习“空条件”
        s = s * mask + self.none_embedding(torch.tensor([0], device=self.device)) * (1 - mask) # (256,64)

        return s

    # 新增：直接对簇中心向量应用CFG dropout
    def calculate_s_from_intent_centers(self, intent_centers, p):
        """
        直接对簇中心向量应用CFG的条件丢弃
        intent_centers: [B, D] 簇中心向量
        p: dropout概率
        """
        B, D = intent_centers.shape[0], intent_centers.shape[1]
        
        # 生成CFG dropout掩码
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2  # [B]
        mask1d = mask1d.view(B, 1)  # [B, 1]
        mask = torch.cat([mask1d] * D, dim=1)  # [B, D]
        mask = mask.to(self.device)
        
        # 条件丢弃：保留簇中心或替换为无条件向量
        s = intent_centers * mask + self.none_embedding(torch.tensor([0], device=self.device)) * (1 - mask)
        
        return s

    # 新增：获取长期兴趣（BSARec完全替代静态用户偏好）
    def get_long_term_interest(self, input_seq):
        """
        使用BSARec频域分支的低频部分提取长期兴趣
        Args:
            input_seq: [B, S] 输入序列
        Returns:
            long_term_interest: [B, D] 长期兴趣向量（BSARec频域分支低频）
        """
        # 使用BSARec频域分支的低频部分（长期兴趣）
        long_term_interest = self.seq_model.get_frequency_long_term_repr(input_seq)  # [B, D]
        
        return long_term_interest
    
    def get_short_term_interest(self, input_seq):
        """
        使用BSARec频域分支的高频部分提取短期兴趣
        Args:
            input_seq: [B, S] 输入序列
        Returns:
            short_term_interest: [B, D] 短期兴趣向量（BSARec频域分支高频）
        """
        # 使用BSARec频域分支的高频部分（短期兴趣）
        short_term_interest = self.seq_model.get_frequency_short_term_repr(input_seq)  # [B, D]
        
        return short_term_interest

    # 正向加噪 q(x_t | x_0)
    def forward_process(self, x_start, t, noise=None):
        # 给定干净样本表示 x_start（即 𝑥0）和时间步 t，生成对应噪声水平的 𝑥𝑡
        if noise is None:
            # 若未传入 noise，就按 x_start 的形状采样标准高斯噪声 𝜖
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.diff.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.diff.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        # 逐元素线性组合出 𝑥𝑡 = √ˉ𝛼𝑡.x0 + √(1-ˉ𝛼𝑡)ϵ
        # 第一项：把干净信号 𝑥0 衰减到当前步的能量 √ˉ𝛼𝑡
        # 第二项：匹配方差的噪声项 √(1-ˉ𝛼𝑡)ϵ
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # ⚠️ 已废弃：3D版本的forward方法（避免维度不匹配）
    # def forward(self, x, s, step):
    #     """
    #     简单的条件扩散预测（用于CFG的条件分支）
    #     Args:
    #         x: [B, D] 当前噪声状态
    #         s: [B, D] 条件向量
    #         step: [B] 时间步
    #     Returns:
    #         res: [B, D] 预测结果
    #     """
    #     # 时间步嵌入
    #     t = self.step_mlp(step)
    #     
    #     # 三路融合（简化版，对应修改原版的做法）
    #     if self.diffuser_type == 'mlp1':
    #         res = self.diffuser(torch.cat((x, s, t), dim=1))
    #     elif self.diffuser_type == 'mlp2':
    #         res = self.diffuser(torch.cat((x, s, t), dim=1))
    #     
    #     return res


    # ⚠️ 已废弃：反向采样（使用3D的简化路径，与diffuser的5D期望不匹配）
    # @torch.no_grad()
    # def sample_from_reverse_process(self, s):
    #     # s 是条件向量（比如你的 𝑆𝑒或最后一步隐藏态），形状 [B, D]
    #     # 这个函数会一次性生成一个 batch（大小 = B）的样本
    # 
    #     # 初始化当前状态为纯高斯噪声 𝑥𝑇
    #     # 形状与 s 相同（[B, D]）
    #     x = torch.randn_like(s) # 纯高斯噪声 𝑥𝑇
    #     # 时间步从 T−1 → 0 逐步往回走
    #     # self.diff.timesteps 是 𝑇
    #     # reversed(range(0, T)) 会依次给出 n = T-1, T-2, ..., 0
    #     for n in reversed(range(0, self.diff.timesteps)):
    #         # 创建一个张量，形状是 (B,)，其中 B = s.shape[0]（批大小）， 每个位置的值都填成 n（当前时间步的编号）
    #         # 举例：
    #         # 批大小 B=4，当前循环 n=300，则得到 t = [300, 300, 300, 300]
    #         # dtype=torch.long：后面会用 t 去 gather 日程系数（必须是整型索引）
    #         t = torch.full((s.shape[0], ), n, device=self.device, dtype=torch.long)
    #         # x:(batch, dim), s:(batch, dim), t:(batch, ), n:1~timesteps
    #         # 调用"一步采样"函数：给当前的 x、条件 s、步索引 t，用 CFG 做一次从
    #         # 𝑥𝑛 → 𝑥𝑛 − 1 的更新
    #         # 传入两个前向：
    #         # self.forward = cond 分支 𝑓𝜃(𝑥𝑡,𝑠,𝑡)
    #         # self.forward_uncon = uncond 分支 𝑓𝜃(𝑥𝑡,∅, 𝑡)
    # 
    #         # | 循环 n | 构造的 t (每个样本) | 进入本轮的 x   | 本轮输出   |
    #         # | ---- |     -----------    | --------      | --------- |
    #         # | T-1  |     [T-1,…]        | x_T           | x_{T-1}   |
    #         # | T-2  |     [T-2,…]        | x_{T-1}       | x_{T-2}   |
    #         # | …    |       …            | …             | …         |
    #         # | 0    |     [0,…]          | x_1           | x_0       |
    #         x = self.p_sample_with_guidance(self.forward, self.forward_uncon, x, s, t, n)
    # 
    #     # 循环结束后返回最终的 𝑥0
    #     return x

    # 新增：增强的反向采样，支持长短期兴趣和意图信号
    # 扩散的逆过程不需要梯度更新
    @torch.no_grad()
    def sample_from_reverse_process_enhanced(self, input_seq, user_ids, intent_centers,
                                             use_ddim=False, ddim_steps=20):
        """
        增强的反向扩散采样：内部强制 eval()，确保老师路径全静态。
        """
        was_training = self.training
        self.eval()
        try:
            batch_size = intent_centers.shape[0]
            x = torch.randn(batch_size, self.hidden_size, device=self.device)

            if use_ddim:
                step_size = max(1, self.diff.timesteps // max(1, int(ddim_steps)))
                sampling_timesteps = list(range(0, self.diff.timesteps, step_size))
                if sampling_timesteps[-1] != self.diff.timesteps - 1:
                    sampling_timesteps.append(self.diff.timesteps - 1)
                for i, n in enumerate(reversed(sampling_timesteps)):
                    t = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
                    prev_t = sampling_timesteps[len(sampling_timesteps) - i - 2] if i < len(sampling_timesteps) - 1 else 0
                    x = self.p_sample_ddim_enhanced(x, input_seq, user_ids, intent_centers, t, n, prev_t)
            else:
                for n in reversed(range(0, self.diff.timesteps)):
                    t = torch.full((batch_size,), n, device=self.device, dtype=torch.long)
                    x = self.p_sample_with_guidance_enhanced(x, input_seq, user_ids, intent_centers, t, n)

            return x
        finally:
            if was_training:
                self.train()

    # ⚠️ 已废弃：p_sample_with_guidance（使用3D的简化路径，与diffuser的5D期望不匹配）
    # @torch.no_grad()
    # def p_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, t_index):
    #     # self.forward = cond 分支 𝑓𝜃(𝑥𝑡,𝑠,𝑡)
    #     # self.forward_uncon = uncond 分支 𝑓𝜃(𝑥𝑡,∅, 𝑡)
    #     # x 是纯高斯噪声 𝑥𝑇
    #     # s 是条件向量（比如你的 𝑆𝑒或最后一步隐藏态），形状 [B, D]
    #     # t 是 [B] 的步索引  "步索引"就是离散时间步 t∈{0,…,T−1} 的编号，用来指明当前所处的噪声强度（对应 𝛼𝑡, 𝛽𝑡,~𝛽𝑡等表里的第 𝑡个元素）
    #     # t_index 是当前的标量步号（与 t 值相同）
    # 
    #     # guidance, (batch, dim)
    #     # CFG 线性组合：做 两次前向（cond/uncond），用 guidance scale self.w = ω 放大条件的影响：
    #     # 𝑓^ =(1 + 𝜔)𝑓cond − 𝜔𝑓uncond
    #     # 这里变量名写成 x_start，说明你的网络输出的是 ^𝑥0（x0-预测）。若你的网络输出的是 ^ϵ，这里应改名并在后面用 ^ϵ 版本的后验公式
    #     x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)
    #     x_t = x
    # 
    #     # model_mean = (batch,1) x (batch, dim) + (batch,1) x (batch, dim) = (batch, dim)
    #     # 后验均值（DDPM, x0 预测版）
    #     model_mean = (
    #         # x0 前面的系数
    #         extract(self.diff.posterior_mean_coef1, t, x_t.shape) * x_start +
    #         # xt 前面的系数
    #         extract(self.diff.posterior_mean_coef2, t, x_t.shape) * x_t
    #     )
    # 
    #     if t_index == 0:
    #         # 最后一步（t=0）：通常不再加噪声（𝜎0 = 0），直接输出均值更清晰、少抖动
    #         return model_mean
    #     else:
    #         # 否则（t>0）：按 DDPM 的后验方差 ~βt 采样
    # 
    #         # self.diff.posterior_variance 是一个长度为 T 的表，提前算好 ~βt
    #         # extract(...) 根据 batch 的 t 索引，把每个样本对应步的系数取出来，并 reshape 成 [B,1,...]，可以和 x 做逐元素运算
    #         # posterior_variance 是 ~βt 的表；提取后是 [B,1,...]
    #         posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
    #         noise = torch.randn_like(x) # 噪声 z
    # 
    #         # xt−1=μt+√(~βt)z,z∼N(0,I)
    #         return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # 新增：增强版CFG采样，支持长短期兴趣和意图信号
    @torch.no_grad()
    def p_sample_with_guidance_enhanced(self, x, input_seq, user_ids, intent_centers, t, t_index):
        """
        增强版CFG采样，融合长短期兴趣和意图信号
        Args:
            x: [B, D] 当前噪声状态
            input_seq: [B, S] 输入序列
            user_ids: [B] 用户ID
            intent_centers: [B, D] 意图中心向量
            t: [B] 时间步张量
            t_index: 标量时间步
        Returns:
            x: [B, D] 下一步状态
        """
        # CFG线性组合：有条件预测 + 无条件预测（推理时不进行条件置空）
        cond_pred = self.forward_enhanced(x, input_seq, user_ids, intent_centers, t, training=False)
        uncond_pred = self.forward_uncon_enhanced(x, t)
        
        # CFG引导：x_start = (1 + w) * cond - w * uncond
        x_start = (1 + self.w) * cond_pred - self.w * uncond_pred
        
        # 后验采样（与原版相同）
        model_mean = (
            extract(self.diff.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.diff.posterior_mean_coef2, t, x.shape) * x
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # 新增：DDIM采样方法（确定性，跳步）
    @torch.no_grad()
    def p_sample_ddim_enhanced(self, x, input_seq, user_ids, intent_centers, t, t_index, prev_t=0):
        """
        DDIM采样：确定性跳步采样，大幅减少采样步数
        Args:
            x: [B, D] 当前噪声状态
            input_seq: [B, S] 输入序列
            user_ids: [B] 用户ID
            intent_centers: [B, D] 意图中心向量
            t: [B] 当前时间步张量
            t_index: 标量时间步
            prev_t: 上一个时间步
        Returns:
            x: [B, D] 下一步状态
        """
        # CFG预测
        cond_pred = self.forward_enhanced(x, input_seq, user_ids, intent_centers, t, training=False)
        uncond_pred = self.forward_uncon_enhanced(x, t)
        
        # CFG引导
        x_start = (1 + self.w) * cond_pred - self.w * uncond_pred
        
        # DDIM更新公式（确定性）
        alpha_t = extract(self.diff.alphas_cumprod, t, x.shape)
        alpha_prev = extract(self.diff.alphas_cumprod, torch.full_like(t, prev_t), x.shape)
        
        # 预测噪声
        pred_noise = (x - torch.sqrt(alpha_t) * x_start) / torch.sqrt(1 - alpha_t)
        
        # DDIM更新
        x_prev = (
            torch.sqrt(alpha_prev) * x_start + 
            torch.sqrt(1 - alpha_prev) * pred_noise
        )
        
        return x_prev


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster # 聚类簇数 k 32
        self.max_points_per_centroid = 4096 # 训练时每个簇最多抽样的点数（FAISS的加速/稳健参数）
        self.min_points_per_centroid = 0  # 每簇最少样本（过低会导致空簇更容易出现）
        self.gpu_id = 0
        self.device = device
        self.first_batch = True # 标志位（这里暂未使用）
        self.hidden_size = hidden_size  # 特征维度 d（embedding 向量长度）64
        # self.clus: FAISS 的 Clustering（KMeans 训练器）。它只负责训练出簇心，簇心保存在 self.clus.centroids
        # self.index: 一个 GpuIndexFlatL2(D)（暴力 L2 距离索引）。后续你需要把簇心加进去，才能用它对样本做“最近簇”的检索
        self.clus, self.index = self.__init_cluster(self.hidden_size) # 初始化 FAISS 的 Clustering 实例和一个 GPU 索引
        self.centroids = [] # 存放最终的簇心（torch.Tensor [k, d]）
        
        self.seq2cluster = None  # 序列ID → 簇ID 的映射（list 或 tensor）
        self.cluster2sequences = defaultdict(list)  # 反向映射：簇ID → 该簇里的序列ID列表


    # 定义了一个私有方法 __init_cluster，用于初始化 KMeans 聚类器 和 GPU 索引
    # hidden_size：向量维度（即每个样本的 embedding 维度）
    # verbose：是否打印迭代收敛信息
    # niter：KMeans 迭代次数
    # nredo：重启次数（KMeans 会多次随机初始化中心，取最好的一次）
    # max_points_per_centroid / min_points_per_centroid：控制每个簇中参与训练的样本数量上下限（防止簇过大或过小）
    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        # print(" cluster train iterations:", niter)
        # 创建一个 FAISS 的 KMeans 聚类器对象，参数 (d, k)：d=hidden_size 表示输入向量维度，k=self.num_cluster 表示聚类簇数，注意：这是一个聚类训练器，真正的样本向量是通过 train(x, index) 方法传入的
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose # 是否输出训练过程的日志（损失、收敛情况）
        clus.niter = niter # 聚类时的最大迭代次数 20
        clus.nredo = nredo # 重新初始化的次数 5
        clus.seed = self.seed
        # 设置每个簇参与训练的样本上限/下限
        clus.max_points_per_centroid = max_points_per_centroid # 4096
        clus.min_points_per_centroid = min_points_per_centroid # 5

        # 创建一个 GPU 资源管理对象，用来分配显存/临时缓冲区。在后续构建 GPU 索引时必须传入
        res = faiss.StandardGpuResources()
        # 禁止使用临时显存缓冲区。启用这个选项后，FAISS 不会在 GPU 上开辟额外的临时内存池，更节省显存，但可能稍微降低速度
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig() # 配置 GPU 索引
        cfg.useFloat16 = False # 使用 FP32 浮点数（计算更准，但显存占用更高）
        cfg.device = self.gpu_id # 指定 GPU id，比如 0 表示第一张显卡
        # 创建一个 L2 距离的 GPU 索引，用于最近邻搜索
        # GpuIndexFlatL2 是“暴力搜索”的 GPU 版，没有训练过程，直接存储所有向量，查询时计算 L2 距离
        # 参数：
        # res：GPU 资源句柄
        # hidden_size：向量维度 d
        # cfg：GPU 配置（精度/设备号）
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x, sequence_ids):
        # 先清空上一轮的反向映射，避免跨 epoch 累积
        self.cluster2sequences = defaultdict(list)
        
        # --- 统一成"球面 KMeans"：训练也要按行 L2 归一化 ---
        x_train = x.astype('float32')
        x_train /= (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-12)

        if x_train.shape[0] > self.num_cluster:
            self.clus.train(x_train, self.index)

        # 取心、归一化、重建索引
        centroids = faiss.vector_to_array(self.clus.centroids)\
            .reshape(self.num_cluster, self.hidden_size).astype('float32')
        centroids = torch.from_numpy(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

        self.index.reset()
        self.index.add(self.centroids.detach().cpu().numpy().astype('float32'))

        # --- 与度量对齐：分配时也用归一化的查询向量 ---
        x_search = x_train  # 已经是单位向量
        D, I = self.index.search(x_search, 1)
        seq2cluster = [int(n[0]) for n in I] # (13128)
        self.seq2cluster = seq2cluster  

        # 遍历每个样本的序列 ID 和它对应的簇 ID

        # 举个小例子
        # 假设：num_cluster=2, hidden_size=2
        # 输入 4 个样本：
        # x = [[1,1], [1.2,1.1], [5,5], [5.1,5.2]]
        # sequence_ids = [101, 102, 201, 202]
        # 1.训练：clus.train(x, index) 找到 2 个簇心，大概在 [1.1,1.05] 和 [5.05,5.1]
        # 2.取簇心：变成 torch.Tensor，并 L2 归一化
        # 3.分配样本：
        # 样本 [1,1]、[1.2,1.1] → 最近簇心是簇 0
        # 样本 [5,5]、[5.1,5.2] → 最近簇心是簇 1
        # 所以 seq2cluster = [0,0,1,1]
        # 4.建立簇→样本ID映射：
        # cluster2sequences = {
        #     0: [101, 102],
        #     1: [201, 202]
        # }
        # cluster2sequences 是一个字典：簇ID → 全局子序列索引（seq_id）列表 的映射
        # 键：cluster_id（0..K-1）
        # 值：属于该簇的 seq_id 列表（这些 seq_id 可用于从 self.sequences_list[seq_id] 取回那条子序列的 token 序列）

        # 总结：这段 train 的作用就是 训练 FAISS KMeans → 得到簇心 → 把每个输入样本分到对应簇 → 建立簇→样本ID映射


        # cluster_id（簇 ID）
        # KMeans/FAISS 把每个样本分到的簇的编号。范围是 0 .. (K-1)，K=簇数（如 32）。
        # 来自 I = index.search(x, 1) 的结果里那一列索引（I[i,0]）

        # seq_id（序列 ID）
        # 你在做聚类前为每条子序列分配的全局行号，用于能从大矩阵 self.sequences_list 里把那条子序列（原始 token 序列）精确取出来
        # 代码里通过：
        # batch_sequence_ids = np.arange(sequence_counter, sequence_counter + batch_size)
        # sequence_counter += batch_size
        # sequence_ids.extend(batch_sequence_ids)
        # 所以 seq_id 实际就是你把各个 batch 拼起来后的行索引：0..N-1
        # 重点：seq_id 不是 user_id。一个 user 可能对应多个 seq_id（不同子序列/切片），而每个 seq_id 恰好对应 self.sequences_list 的一行
        for seq_id, cluster_id in zip(sequence_ids, seq2cluster):
            # self.cluster2sequences：字典，key=簇 ID，value=该簇下的序列 ID 列表
            # 实现了簇 → 样本 ID 的映射
            self.cluster2sequences[cluster_id].append(seq_id)

    # 小型数值示例
    # 假设 K=3, d=2，簇心：
    # self.centroids =
    # tensor([[1.0, 1.0],    # 簇0
    #         [5.0, 5.0],    # 簇1
    #         [-2.0, 0.0]])  # 簇2

    # 输入批 x（N=4）：
    # x =
    # [[1.2, 0.8],     # 离簇0最近
    #  [4.9, 5.1],     # 离簇1最近
    #  [-1.8, 0.3],    # 离簇2最近
    #  [4.7, 4.8]]     # 离簇1最近

    # 如果 self.index 中已经 add 了上面的 3 个簇心，那么
    # I = [[0],[1],[2],[1]]（每行一个最近簇心的 ID）
    # seq2cluster = tensor([0,1,2,1], dtype=torch.long)
    # self.centroids[seq2cluster] 得到形状 [4,2] 的簇心行向量，顺序与样本一一对应
    # 没有用到 query()
    def query(self, x):
        x = x.astype('float32')
        # 与上面 train() 的度量保持一致（这里按余弦来）
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

        D, I = self.index.search(x, 1)
        seq2cluster = torch.as_tensor(I[:, 0], dtype=torch.long, device=self.device)
        # self.centroids 的形状是 [K, d]：K 个簇中心，每个是 d 维向量
        # seq2cluster 的形状是 [B]：B 条样本各自被分到的簇编号（0 到 K-1），dtype 必须是 long，并且在同一 device 上
        # 假设：
        # K = 3, d = 4
        # self.centroids =
        # [
        #   [ 1.0,  1.1,  1.2,  1.3],   # 簇0的中心
        #   [ 2.0,  2.1,  2.2,  2.3],   # 簇1的中心
        #   [ 3.0,  3.1,  3.2,  3.3],   # 簇2的中心
        # ]

        # seq2cluster = [1, 0, 1, 2]    # B = 4
        # 那么：
        # self.centroids[seq2cluster] = self.centroids[[1, 0, 1, 2]]
        # = [
        #     [2.0, 2.1, 2.2, 2.3],  # 第0个样本 → 取簇1
        #     [1.0, 1.1, 1.2, 1.3],  # 第1个样本 → 取簇0
        #     [2.0, 2.1, 2.2, 2.3],  # 第2个样本 → 取簇1
        #     [3.0, 3.1, 3.2, 3.3],  # 第3个样本 → 取簇2
        #   ]
        # 形状是 [B, d] = [4, 4]

        # 返回两个量：
        # seq2cluster: [B] 的簇编号；
        # self.centroids[seq2cluster]: 对应的质心向量，形状 [B, d]（你的 train() 里把质心做了 L2 归一化）
        return seq2cluster, self.centroids[seq2cluster]
