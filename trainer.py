import numpy as np
from tqdm import tqdm
import random
import gc
# import seaborn as sns  # 未使用
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import KMeans
from utils import recall_at_k, ndcg_k, generate_padded_sequences_tensor


class Trainer:
    def __init__(self, model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model # InDiRec
        self.seq_model = self.model.seq_model # SASRecModel

        self.batch_size = self.args.batch_size # 256
        self.sim = self.args.sim # dot

        cluster = KMeans(
            num_cluster=args.intent_num, # 簇的数量（意图数） 32
            seed=1,
            hidden_size=64, # 聚类向量维度（必须与你拿去聚类的表示维度 D 一致）
            device=torch.device("cuda"),
        ) # 实例化 KMeans
        self.clusters = [cluster]
        # 把聚类器放进列表，再套一层列表
        # 这种“双层结构”通常是为了支持“多路/多头/多层”聚类器接口（比如第 0 路只有 1 个聚类器，将来也可以扩成多路多聚类器）
        # 现在的结构是 [[KMeans()]]
        self.clusters_t=[self.clusters]

        self.train_dataloader = train_dataloader # 真正反传更新时用
        self.cluster_dataloader=cluster_dataloader # 只在每个 epoch 开头先遍历一遍，抽取序列最后一步表征进行聚类（无梯度）

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # 统一的设备转移：先转移整个模型，再转移seq_model
        if self.device.type == 'cuda':
            self.model.cuda()
            # seq_model已经是model的一部分，不需要单独转移
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader,self.cluster_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        # 形状：[item_num hidden_size]
        # 物品嵌入表的权重矩阵，形状是 [V, D]：第 v 行是物品 v 的 D 维向量
        test_item_emb = self.model.item_embeddings.weight
        # seq_output 形状是 [B, D]（通常是把 forward(...)[ :, -1, :] 取最后时刻得到的“用户状态向量”）
        # 右乘 test_item_emb.T（[D, V]）得到 [B, V]：其中每一行代表一个用户对V个物品的打分，比如：第一行就是 第一个样本（batch 里第 1 个用户） 对所有物品的打分列表
        # 第 b 行第 v 列：logits[b, v] = dot(seq_output[b], test_item_emb[v])
        # 就是用户状态与物品向量的点积相似度, 点积越大，代表越“匹配”
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        # 返回对全体物品的原始打分（未 softmax），后续会做历史屏蔽、Top-K 选取、计算 HR/NDCG
        return rating_pred
    
    def calculate_cl_loss(self, x0_hat_rep, x0_rep, target):
        # x0_hat_rep：逆扩散生成的 intent-aligned 视图（^𝑥0）
        # x0_rep：SASRec 最后一步的用户表示
        # 当前步的真实目标 item id（做负样本屏蔽）
        batch_size = x0_rep.shape[0] # 取 batch 大小
        # x0_hat_out：逆扩散生成的 intent-aligned 视图（^𝑥0），本身就是位于 item embedding 空间 的一个 [B,D] 向量（与你的 item_embeddings 同维）。它不是一条序列，再送进 SASRec（序列编码器）不合适、也没必要
        # 注意：当前 sample_from_reverse_process 是 @torch.no_grad()：cl_loss 不会更新扩散分支，只会训练 SASRec；如果希望 CL 也影响去噪器，可改用单步预测 predicted_x_0（有梯度）来做对比，或移除 no_grad（注意显存与稳定性）

        # 把 x0_hat_out（逆扩散生成的 ^𝑥0 ，物品嵌入空间）与 SASRec(input_seq)[:,-1,:]（用户状态，序列表示空间）做 InfoNCE：
        sem_nce_logits, sem_nce_labels = self.info_nce(x0_hat_rep,x0_rep,self.args.temperature, batch_size, self.sim, target)
        cl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cl_loss
    
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot',intent_id=None):

        # 定义本函数内部使用的 N 是两份视图合并后的样本数。后面所有矩阵都是 N × N
        N = 2 * batch_size
        # 按 batch 维拼接两份视图：
        # z_i, z_j 是 [B, D]，那 z 就是 [2B,D]
        # 索引 0..B-1 来自 z_i，B..2B-1 来自 z_j
        z = torch.cat((z_i, z_j), dim=0) # (512,64)
        if sim == 'cos':
            # 余弦相似度：通过 z[ :, None, : ] 与 z[ None, :, : ] 广播得到每对样本的相似度，再除以温度
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            # 点积相似度：直接矩阵乘 z · zᵀ，再除以温度
            sim = torch.mm(z, z.t()) / temp

        # 取偏移对角线来拿正样本对的相似度：
        # torch.diag(A, k) 取矩阵 A 的第 k 条对角线（k>0 在主对角线上方，k<0 在下方）
        # 这里 batch_size 对角线恰好是 (i, i+B)（view1 → view2），-batch_size 对角线是 (i+B, i)（view2 → view1）
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # mask = mask_correlated_samples(intent_id) 返回一个 2B×2B 的布尔矩阵：
        # True 表示"标签不同"（可作为负样本）；
        # False 表示"标签相同"（包括自己、正对、以及同 item 的其它样本）
        mask = self.mask_correlated_samples(intent_id)
        negative_samples = sim
        # 负样本 是 除去自己 r（主对角线）；
        # 配对 r±B（已经作为正样本单独拿出来，不应出现在负样里）；
        # batch 中其它与锚"同标签"的样本（同一个 target）

        # 这行代码正是在排除上面的三类：
        # 自己 r（主对角线）
        # 配对的正样本 r±B（已经作为正样本单独放在第 0 列，不能再出现在负样里）
        # batch 中与锚同标签（同 target）的其它样本（两份视图里的都算）

        # 因为 mask_correlated_samples 返回的是一个 [2B×2B] 布尔矩阵，其中
        # True = 标签不同（可作为负样）；
        # False = 标签相同（包括自己、正对、以及其它同标签样本）
        # 所以这行代码就把同标签的位置全部屏蔽（置为 −∞），只留下"不同标签"作为负样参与 softmax，从而避免假负
        negative_samples[mask==0]=float("-inf")

        # 全 0，正样本所在列的索引,表示正样本在第 0 列
        labels = torch.zeros(N).to(positive_samples.device).long()
        # 每一行就是一个 anchor 的分类问题：1 个正类（第 0 列） + (N-2) 个负类
        # 第 0 列是正样本 logit，后面几列就是这些负样本的 logit，顺序按布尔索引填充
        # 最终按交叉熵训练，目标就是让每行的第 0 列（正样本）比分列更大
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels
    
    def mask_correlated_samples(self, label):
        label=label.view(1,-1) # [1, B]
        label=label.expand((2,label.shape[-1])).reshape(1,-1) # [2, B] → [1, 2B]
        label = label.contiguous().view(-1, 1) # [2B, 1]
        # torch.eq(label, label.t()) 会两两比较行标签和列标签是否相等，结果是一个 [2B, 2B] 的布尔矩阵
        # 因此为 True 的位置包含：
        # True = 相同标签（包括：自己 (r,r)、跨视图正对 (r, r±B)、以及同标签的其它样本）
        mask = torch.eq(label, label.t()) # [2B, 2B]
        # 取反：不同标签的地方变成 True，同标签（包括自身 & 正对 & 其它同标签样本）变成 False
        return mask==0
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HR@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HR@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)


    # 新的函数：获取输入序列对应的簇中心向量（参考ICSRec实现）
    # 👉 给定一批输入序列，返回它们各自对应的簇中心向量作为意图信号
    @torch.no_grad()
    def get_intent_centers_for_sequences(self, input_sequence):
        input_sequence = input_sequence.to(self.device)
        # 仅对 seq_model 做临时 eval，避免其中的 Dropout 抖动
        was_training = self.seq_model.training
        self.seq_model.eval()
        try:
            seq_repr = self.seq_model(input_sequence)[:, -1, :].detach().cpu().numpy().astype('float32')
        finally:
            if was_training:
                self.seq_model.train()

        cluster = self.clusters_t[0][0]
        _, intent_centers = cluster.query(seq_repr)   # [B, D]
        return intent_centers
class InDiRecTrainer(Trainer):
    def __init__(self, model, train_dataloader,cluster_dataloader,eval_dataloader, test_dataloader, device, args):
        super(InDiRecTrainer, self).__init__(model, train_dataloader, cluster_dataloader,eval_dataloader, test_dataloader, device, args) # 调用父类 Trainer.__init__，把父类里需要的属性都初始化好。对象仍然是一个 InDiRecTrainer 实例，只是它也拥有 Trainer 初始化过的那些字段

    def iteration(self, epoch, dataloader,cluster_dataloader=None, train=True):
        if train:
            print("Preparing Clustering:")
            # 先抽取稳定的序列表征做聚类，不是在反传更新参数。原因有三：
            # 1.去掉随机性
            # eval() 会关闭 dropout（以及 BatchNorm 的动量更新）
            # 聚类需要可重复、稳定的特征；若在 train() 下有 dropout，特征会抖动，簇中心会漂，后面“同簇取样”就不稳定
            # 2.不需要梯度
            # 这里只是前向提取特征，后面你也 detach().cpu().numpy() 了，完全不走梯度。
            # 用 eval()（最好再包一层 with torch.no_grad():）能省显存和时间。
            # 3.训练与聚类分阶段
            # 聚类完马上就 self.model.train()，再进入真正的多任务训练（有反传）
            # 也就是：提特征（eval, no grad）→ 聚类 → 训练（train, 有 grad）
            # 如果你故意想让聚类“带点噪声”当数据增强，确实可以用 train() 开着 dropout，但代价是簇不稳定、波动大，通常不如 eval() 的效果稳
            self.model.eval()
            kmeans_training_data = [] # 每条子序列的最后一步表征（作为聚类特征）
            sequence_ids = [] # 为每条子序列分配一个全局 ID，便于聚类器内部索引对齐
            sequence_counter = 0  # 累积计数器，给每个 batch 里的样本生成不重复的 ID

            # 遍历“聚类数据”的 dataloader   total=len(cluster_dataloader) 批次数
            rec_t_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
            for i, (rec_batch) in rec_t_data_iter:

                # 把 batch 里每个张量统一搬到 self.device（GPU 或 CPU），避免“张量不在同一设备”的报错
                # .to(self.device) 只改设备，不改 dtype（除非你额外传 dtype=）
                # 这里把 batch 重建成一个 tuple，保证后面按固定顺序解包
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                # subsequence 是 Beauty_s.txt 中的序列输入，形状是 [B, S]
                user_id, subsequence, _, _, _ = rec_batch 
                batch_size = subsequence.size(0) # ()

                # 用序列模型提取"聚类特征"（序列最后一步表示）
                sequence_output_a = self.seq_model(subsequence) # (B,S,D)
                sequence_output_b = sequence_output_a[:,-1,:]  # 取最后一个时间步的表示，得到 [B, D]  (256,64)
                # detach()：与计算图断开，不追踪梯度（聚类不需要梯度，省显存、防止误用）
                # cpu()：从 GPU 拿回到 CPU，才可以 numpy()
                # .numpy()：把张量转成 numpy 数组以便后续用 FAISS/KMeans
                kmeans_training_data.append(sequence_output_b.detach().cpu().numpy())
                # Beauty_s.txt 中的数据，序列输入 subsequence 加入列表

                batch_sequence_ids = np.arange(sequence_counter, sequence_counter + batch_size)
                sequence_counter += batch_size
                sequence_ids.extend(batch_sequence_ids)

            # 把 ID 列表转成 [N] 的 numpy 数组
            sequence_ids = np.array(sequence_ids) # (131413)

            # 把特征列表堆成 [N, D]
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) # (131413,64)
            # 再外面套一层列表（[...]）：这是为了兼容多路/多头聚类器的通用接口（当前只有 1 路，所以放第 0 个槽）
            kmeans_training_data_t = [kmeans_training_data]

            # ID 套一层列表，作为"第 i 路"的数据
            sequence_ids_t = [sequence_ids]

            # 作用：外层遍历“第 i 路”的聚类器列表。self.clusters_t 是二维结构，形如 [[cluster_0_0, cluster_0_1, ...], [cluster_1_0, ...], ...]。现在大多是只有 1 路 1 个：[[KMeans()]]
            # 举例：假设你有 2 路聚类器（后续扩展用），self.clusters_t = [[KMeansA, KMeansB], [KMeansC]]，那 i=0 时 clusters=[KMeansA, KMeansB]；i=1 时 clusters=[KMeansC]
            # 注意：保持 self.clusters_t 的层级是为了将来支持多路/多头（比如不同时域、不同特征空间各做一次聚类）
            for i, clusters in enumerate(self.clusters_t): # self.clusters_t 是 [[KMeans()]]
                # 作用：内层遍历“第 i 路”中的第 j 个聚类器
                for j, cluster in enumerate(clusters):
                    # kmeans_training_data_t[i] 是用来聚类的特征矩阵（第 i 路的 [N, D]）
                    # sequence_ids_t[i] 是与这些特征一一对应的“子序列全局行号”列表（形状 [N]，整型）， sequence_ids_t[i] 不是 user_id，而是子序列的全局索引
                    cluster.train(kmeans_training_data_t[i], sequence_ids_t[i])
                    # 把（已训练好的）cluster 写回容器位置，确保 self.clusters_t 里保存的是最新的聚类器
                    self.clusters_t[i][j] = cluster

            # clean memory
            del kmeans_training_data
            del kmeans_training_data_t
            del sequence_ids
            del sequence_ids_t
            gc.collect()

            # 新增：构建用户意图分布（类似GCDR的calc_user_cat_hist）
            # 注意：无论是否启用多意图，都构建用户意图分布，确保训练-评估一致性
            self.build_user_intent_distribution(cluster_dataloader)

            # 切到 训练模式：如 Dropout/BN 会启用训练行为
            self.model.train()
            # 四路损失累加器（完整版本）
            avg_loss = 0.0              # rec_loss累加器
            avg_diff_loss = 0.0         # diff_loss累加器
            avg_diffusion_rec_loss = 0.0  # diffusion_rec_loss累加器
            avg_cl_loss = 0.0           # cl_loss累加器

            # print(f"rec dataset length: {len(dataloader)}")
            # 也是用的 Beauty_s.txt 中的数据
            rec_t_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # minibatch
            for i, (rec_batch) in rec_t_data_iter:

                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_seq, input_seq_len, target_pos, target = rec_batch

                # 保存当前input_seq供BSARec模式的user_loss使用
                self.current_input_seq = input_seq

                # 单意图训练：使用分布加权原型→非线性变换的两步处理
                if hasattr(self, 'user_intent_hist') and self.user_intent_hist is not None:
                    # 使用用户意图分布进行两步处理
                    user_intent_dist = self.user_intent_hist[user_ids]  # [B, intent_num]  (256,32)
                    intent_centers = self.model.process_intent_distribution(user_intent_dist)  # [B, D]
                else:
                    # 回退到聚类阶段的意图中心
                    intent_centers = self.get_intent_centers_for_sequences(input_seq) # [B, D]

                # 扩散分支要"生成的那个干净信号（x₀）"本质上是"真实要预测的下一物品的向量"，所以用 target（真实下一物品 id）去查 item_embeddings 得到 x0_gt 最贴合任务目标；而输入序列更适合作为条件，不是作为"要生成的真值"

                # target_pos[:, -1] 是当前 batch 里每条序列的真实下一个物品的 id（形状 [B]，整型）
                # x0_gt = self.model.calculate_x0(...) 这一步把这些 item id 通过 item_embeddings 查表，映成 D 维向量，得到扩散分支要学习的"干净信号（x₀ 的真值）
                x0_gt = self.model.calculate_x0(target_pos[:, -1]) # B x D (256,64)

                # 为当前 batch 每个样本采样它对应的扩散步数 t
                actual_batch_size = user_ids.shape[0]  # 使用实际batch大小，处理最后一个不完整batch
                t = torch.randint(0, self.args.timesteps, (actual_batch_size, ), device=self.device).long()

                # 扩散模型的训练过程：加噪和训练去噪网络（保持原有的forward_enhanced，内部有条件置空）
                diff_loss, predicted_x_0, x_noisy = self.calculate_diff_loss_enhanced(
                    self.model, x0_gt, input_seq, user_ids, intent_centers, t
                )
                
                # ========== 关键优化：单步预测代替T步逆扩散 ==========
                # 
                # 【原始方案】：
                # x0_hat_out = self.model.sample_from_reverse_process_enhanced(...)
                # - 需要T步逆扩散（T=50时，需要100次前向传播）
                # - 每步都要调用forward_enhanced + forward_uncon_enhanced
                # - 计算成本极高，是训练瓶颈
                #
                # 【当前优化方案】：
                # x0_hat_out = predicted_x_0.detach()
                # - 直接使用calculate_diff_loss_enhanced中已经计算好的单步预测结果
                # - 只需要1次前向传播（已经在上面完成）
                # - 大幅提速：从101次前向变为1次
                #
                # 【为什么这样做是合理的？】
                # 1. diffusion_rec_loss的目标是让物品嵌入空间学习扩散的知识
                # 2. predicted_x_0已经包含了扩散模型在时间步t的去噪能力
                # 3. detach()确保梯度只更新物品嵌入，不影响去噪器
                # 4. 这本质上是teacher-student蒸馏：
                #    - teacher: 扩散去噪器（通过diff_loss训练）
                #    - student: 物品嵌入空间（通过diffusion_rec_loss学习）
                #
                # 【重要修复：强老师必须关掉所有随机性】
                # 问题：predicted_x_0来自training=True，包含：
                # 1. 手写的条件置空（training=True触发）
                # 2. 模块内部的Dropout/BN（模型.training状态触发）
                # 解决：临时切换到eval()模式，获得完全稳定的"强老师"预测
                #
                # 核心思想：
                # - 训练去噪器：用train()模式，学习鲁棒性
                # - 蒸馏知识：用eval()模式，给出最强最稳定的预测
                
                # 老师（稳定/无随机）：eval() + no_grad + training=False
                with torch.no_grad():
                    was_training = self.model.training
                    self.model.eval()  # 关掉模型及其所有子模块的 Dropout/BN

                    # ★ 在 eval() 下重算老师要用的条件（intent_centers_eval）
                    if hasattr(self, 'user_intent_hist') and self.user_intent_hist is not None:
                        user_intent_dist = self.user_intent_hist[user_ids]          # [B, intent_num]
                        intent_centers_eval = self.model.process_intent_distribution(user_intent_dist)  # [B, D]（两步处理，此时 intent_mlp 的 Dropout 已关闭）
                    else:
                        intent_centers_eval = self.get_intent_centers_for_sequences(input_seq)  # 上一步已保证 seq_model 为 eval

                    # 老师前向（不做条件置空）
                    teacher_cond = self.model.forward_enhanced(
                        x_noisy, input_seq, user_ids, intent_centers_eval, t, training=False
                    )

                    # 可选：单步 CFG 老师
                    if getattr(self.args, 'use_single_step_cfg', False):
                        teacher_uncond = self.model.forward_uncon_enhanced(x_noisy, t)
                        w = getattr(self.args, 'single_step_cfg_w', 2.0)
                        x0_hat_out = (1 + w) * teacher_cond - w * teacher_uncond
                    else:
                        x0_hat_out = teacher_cond

                    # 恢复原状态
                    if was_training:
                        self.model.train()

                x0_out = self.seq_model(input_seq) # BxLxD

                # ========== 对比学习损失 ==========
                # 为了与"修改原版"保持一致，对比学习使用T步逆扩散的结果
                # 这样能获得更高质量的扩散生成表示，虽然会增加计算成本
                if hasattr(self.args, 'cl_use_reverse_process') and self.args.cl_use_reverse_process:
                    # 选项A：使用T步逆扩散（与修改原版一致，质量更好但更慢）
                    # 【关键修复】：CL也要用"老师态"的意图条件，避免Dropout随机性
                    with torch.no_grad():
                        was_training2 = self.model.training
                        self.model.eval()
                        # 重算确定性的意图条件（与单步老师保持一致）
                        if hasattr(self, 'user_intent_hist') and self.user_intent_hist is not None:
                            user_intent_dist = self.user_intent_hist[user_ids]              # [B, intent_num]
                            intent_centers_eval_for_cl = self.model.process_intent_distribution(user_intent_dist)  # [B, D]（两步处理）
                        else:
                            intent_centers_eval_for_cl = self.get_intent_centers_for_sequences(input_seq)

                        x0_hat_for_cl = self.model.sample_from_reverse_process_enhanced(
                            input_seq, user_ids, intent_centers_eval_for_cl,
                            use_ddim=getattr(self.args, 'use_ddim', False),
                            ddim_steps=getattr(self.args, 'ddim_steps', 20)
                        )
                        if was_training2:
                            self.model.train()
                else:
                    # 选项B：复用已有的单步预测结果（效率优先）
                    x0_hat_for_cl = x0_hat_out
                
                # 对比学习：对齐扩散空间和序列空间
                # x0_hat_for_cl：扩散生成的物品表示（在物品嵌入空间）
                # x0_out[:,-1,:]：BSARec的用户序列表示（在序列表示空间）
                cl_loss = self.calculate_cl_loss(x0_hat_for_cl, x0_out[:,-1,:], target_pos[:, -1])

                # 取最后一个时间步的用户表示（当前用户状态）→ 形状 [B, D]
                # predict_full 里面做的是最后一个时间步的用户表示与物品嵌入表矩阵乘：
                # test_item_emb = self.model.item_embeddings.weight（形状 [|I|, D]）
                # rating_pred = seq_out @ test_item_emb.T → logits 形状 [B, |I|]
                logits = self.predict_full(x0_out[:, -1, :])  #  Bx|I|
                # target_pos 每个时间步的“下一物品 id”，形状 [B, L]；这里取最后一个 → [B]（每个样本一个类别 id）
                # target_pos[:, -1] 和 target 一样
                # 保持与原始InDiRec一致，使用target_pos[:, -1]作为目标
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])

                # ========== 扩散输出推荐损失 ==========
                # 
                # 【作用】：
                # 让物品嵌入空间（W_I）学习扩散模型的输出分布
                # 这是一种知识蒸馏：扩散先验 -> 物品嵌入
                #
                # 【关键设计：早期步门控（t <= tau）】
                # - 只在噪声较小的时间步（t <= tau）计算这个损失
                # - 原因：早期步的SNR更高，predicted_x_0更接近真实x_0
                # - tau默认为timesteps/4，确保信号质量
                #
                # 【梯度流向】：
                # x0_hat_out.detach() -> logits -> CE loss -> 只更新W_I
                # 不会影响去噪器，保持训练稳定
                
                tau = self.args.tau if self.args.tau >= 0 else self.args.timesteps // 4
                t_for_diffusion_rec = t  # 单意图：直接使用

                diffusion_rec_loss = self.calculate_diffusion_rec_loss(
                    x0_hat_out, target_pos[:, -1], t_for_diffusion_rec, tau
                )

                # ========== 四路损失组合 ==========
                #
                # 1. rec_loss: BSARec的直接监督信号
                #    - 输入：BSARec最后一步输出 x0_out[:, -1, :]
                #    - 更新：BSARec参数
                #    - 作用：保证序列建模能力
                #
                # 2. diff_loss: 扩散去噪损失
                #    - 输入：加噪的x_t -> 预测x_0
                #    - 更新：去噪器参数
                #    - 作用：学习去噪映射
                #
                # 3. diffusion_rec_loss: 扩散蒸馏损失
                #    - 输入：predicted_x_0.detach()（单步预测）
                #    - 更新：物品嵌入W_I
                #    - 作用：将扩散知识蒸馏到嵌入空间
                #
                # 4. cl_loss: 对比学习损失
                #    - 输入：扩散生成表示 vs 序列表示
                #    - 更新：对齐两个表示空间
                #    - 作用：增强表示学习
                #
                # 权重建议：rec_weight=1, diff_weight=1, diffusion_rec_weight=0.05, cl_weight=0.1
                
                multi_task_loss = (self.args.rec_weight * rec_loss +
                                 self.args.diff_weight * diff_loss +
                                 self.args.diffusion_rec_weight * diffusion_rec_loss +
                                 self.args.cl_weight * cl_loss)
                

                self.optimizer.zero_grad()
                multi_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # 累加各项损失用于监控
                avg_loss += rec_loss.item()
                avg_diff_loss += diff_loss.item()
                avg_diffusion_rec_loss += diffusion_rec_loss.item()
                avg_cl_loss += cl_loss.item()

            # 四路损失监控日志
            num_batches = len(rec_t_data_iter)
            log_record = {
                "epoch": epoch,
                "rec_loss": "{:.4f}".format(avg_loss / num_batches),
                "diff_loss": "{:.4f}".format(avg_diff_loss / num_batches),
                "diffusion_rec_loss": "{:.4f}".format(avg_diffusion_rec_loss / num_batches),
                "cl_loss": "{:.4f}".format(avg_cl_loss / num_batches),
                "total_loss": "{:.4f}".format((avg_loss + avg_diff_loss + avg_diffusion_rec_loss + avg_cl_loss) / num_batches)
            }

            if (epoch+1) % self.args.print_log_freq == 0:
                print(str(log_record))

            with open(self.args.log_file, "a") as f:
                f.write(str(log_record) + "\n")

        else: # Val or Test
            # 开始用 Beauty.txt 的数据
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader)) 
            self.model.eval()
            # pred_list：累计本轮所有 batch 的预测 Top-K  answer_list：累计本轮所有 batch 的真实答案
            pred_list, answer_list = None, None


            for i, batch in rec_data_iter:
                # batch 的结构
                # 在验证/测试 DataLoader 里，DatasetForInDiRec.__getitem__ 返回的是一个 5 元组：
                # (
                #   user_id,        # [B]  当前 batch 的样本 ID
                #   input_seq,      # [B, L]  输入序列（补齐后的）
                #   input_seq_len,  # [B]  每条序列真实长度
                #   target_pos,     # [B, L]  每个时间步的目标物品 ID
                #   answer          # [B]  真实下一个物品
                # )
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids,input_seq_len, target_pos, answers = batch
                rec_output = self.seq_model(input_ids)
                rec_output = rec_output[:,-1,:]


                rating_pred = self.predict_full(rec_output)
                rating_pred = rating_pred.detach().cpu().numpy().copy()
                batch_user_index = user_ids.detach().cpu().numpy()

                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(rating_pred, -20)[:, -20:] 
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind] 
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1] 
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.detach().cpu().numpy()

                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.detach().cpu().numpy(), axis=0) 

            # metrices_list 是列表(metrices_list = [
            #     HR@5,     # recall[0]
            #     NDCG@5,   # ndcg[0]
            #     HR@10,    # recall[1]
            #     NDCG@10,  # ndcg[1]
            #     HR@20,    # recall[3]
            #     NDCG@20   # ndcg[3]
            # ])

            # log_info 是 str(post_fix)
            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)


            return metrices_list, log_info
                    

    # 新增：扩散输出的推荐损失（参考GCDR的diffusion recommendation loss）
    def calculate_diffusion_rec_loss(self, x0_hat, target, t, tau=None):
        """
        计算扩散模型生成的x0用于推荐的损失（纯单意图版本）
        Args:
            x0_hat: [B, D] 扩散模型生成的物品表示
            target: [B] 目标物品ID
            t: [B] 时间步
            tau: 时间步阈值，只在早期步骤(t <= tau)计算损失
        Returns:
            loss: 标量损失值
        """
        # 计算推荐分数
        logits = torch.matmul(x0_hat, self.model.item_embeddings.weight.T)  # [B, V]
        
        # 时间步门控（仅在早期步骤计算损失）
        if tau is not None:
            # ========== 门控后的归一化（避免尺度漂移） ==========
            #
            # 【问题】：原写法 torch.mean(tau_coeff * loss_vec) 会导致尺度漂移
            # - 当有效样本比例 p = (# {i | t_i<=tau}) / B 变化时
            # - loss 会按 p 成比例缩放，导致相对权重不稳定
            #
            # 【解决方案】：对有效样本数归一化，而不是对整个batch归一化
            # - 这样得到的始终是"有效样本上的平均CE"
            # - 不会因为有效样本比例波动导致整体尺度变化
            
            loss_vec = F.cross_entropy(logits, target, reduction='none')  # [B]
            mask = (t <= tau).float()                                      # [B]
            denom = mask.sum().clamp_min(1.0)                             # 标量，避免除零
            loss = (mask * loss_vec).sum() / denom                        # 有效样本上的平均
        else:
            loss = F.cross_entropy(logits, target)
        
        return loss
    

    # 新增：增强版扩散损失计算，使用长短期兴趣和意图信号
    def calculate_diff_loss_enhanced(self, model, x_start, input_seq, user_ids, intent_centers, t, noise=None, loss_type="l2"):
        """
        计算增强版扩散损失（纯单意图版本）
        Args:
            model: InDiRec模型
            x_start: [B, D] 目标物品嵌入（x0_gt）
            input_seq: [B, S] 输入序列
            user_ids: [B] 用户ID
            intent_centers: [B, D] 意图中心向量
            t: [B] 时间步
            noise: [B, D] 可选的噪声（默认随机采样）
            loss_type: 损失类型
        Returns:
            loss: 扩散损失
            predicted_x: [B, D] 预测的x0
        """
        if noise is None:
            noise = torch.randn_like(x_start) # (256,64)

        # 前向加噪
        x_noisy = model.forward_process(x_start=x_start, t=t, noise=noise)
        
        # 增强版去噪网络预测（训练时只做条件置空，不做CFG组合）
        predicted_x = model.forward_enhanced(x_noisy, input_seq, user_ids, intent_centers, t, training=True)
        
        # ========== 可选：单步CFG增强（不需要T步逆扩散） ==========
        #
        # 【背景】：
        # 标准CFG需要在T步逆扩散中每步都做条件/无条件组合
        # 这里提供一个“单步CFG”的选项，只在当前时间步做一次组合
        #
        # 【优势】：
        # 1. 保留CFG的“拉近/推远”效果
        # 2. 只需额外一次无条件前向，成本很低
        # 3. 可以在不做T步逆扩散的情况下获得CFG增强
        #
        # 【公式】：
        # predicted_x_cfg = (1 + w) * predicted_x_cond - w * predicted_x_uncond
        # 其中w是CFG引导强度，通常取值1-3
        #
        # 【注意】：
        # - 需要在main.py中添加--use_single_step_cfg参数启用
        # - predicted_x_uncond使用no_grad，不参与梯度计算
        # - diff_loss仍然用predicted_x（有条件）计算，不用CFG结果
        
        if hasattr(self.args, 'use_single_step_cfg') and self.args.use_single_step_cfg:
            # 无条件预测（不需要梯度）
            with torch.no_grad():
                predicted_x_uncond = model.forward_uncon_enhanced(x_noisy, t)
                # 单步CFG组合
                w = getattr(self.args, 'single_step_cfg_w', 2.0)
                predicted_x_cfg = (1 + w) * predicted_x - w * predicted_x_uncond
        else:
            predicted_x_cfg = predicted_x  # 不使用CFG，直接返回有条件预测
        
        # ========== 计算扩散损失 ==========
        #
        # 【重要】：diff_loss始终使用predicted_x（有条件预测）计算
        # 而不是predicted_x_cfg，因为：
        # 1. 去噪器应该学习的是真实的去噪映射
        # 2. CFG是一种推理时的增强技巧，不应该影响去噪器训练
        # 3. 这样保持了训练目标的清晰性
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            # loss = F.mse_loss(x_start, predicted_x)
            # 方式 B：只对 predicted_x 求梯度
            loss = F.mse_loss(x_start.detach(), predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        # 返回：
        # - loss: 用于更新去噪器
        # - predicted_x_cfg: 用于diffusion_rec_loss（如果启用CFG，否则就是predicted_x）
        # - x_noisy: 加噪后的x_t，可以复用于单步CFG
        return loss, predicted_x_cfg, x_noisy
    
    def build_user_intent_distribution(self, cluster_dataloader):
        """
        构建用户意图分布（类似GCDR的calc_user_cat_hist）
        """
        print("构建用户意图分布...")
        self.model.eval()
        
        # 初始化用户意图计数矩阵
        user_intent_counts = torch.zeros(self.args.num_users, self.args.intent_num, device=self.device) # (22363,32)
        
        with torch.no_grad():
            for i, (rec_batch) in enumerate(tqdm(cluster_dataloader, desc="统计用户意图分布")):
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_seq, input_seq_len, target_pos, target = rec_batch
                
                # 获取序列表示
                x0_out = self.seq_model(input_seq)  # [B, S, D]
                sequence_representation = x0_out[:, -1, :].detach().cpu().numpy()  # [B, D] -> numpy
                
                # 预测意图簇
                intent_ids, _ = self.clusters[0].query(sequence_representation)  # [B] -> tensor
                # 注意：KMeans.query已经返回torch.LongTensor，无需再次转换
                
                # 累计每个用户的意图计数
                user_ids_np = user_ids.detach().cpu().numpy()
                intent_ids_np = intent_ids.detach().cpu().numpy()
                
                for batch_idx, uid in enumerate(user_ids_np):
                    intent_id = intent_ids_np[batch_idx]
                    user_intent_counts[uid, intent_id] += 1
        
        # 归一化为概率分布
        user_intent_counts = user_intent_counts + 1e-8
        self.user_intent_hist = user_intent_counts / user_intent_counts.sum(dim=-1, keepdim=True) # (22362,32)
        
        # 保存意图中心向量到trainer和模型中
        self.intent_centers = self.clusters[0].centroids.clone().to(self.device) # (32,64)
        # 设置到模型中，用于分布加权原型融合
        self.model.intent_centers = self.intent_centers # (32,64)
        
        print(f"用户意图分布构建完成: {self.user_intent_hist.shape}") # torch.Size([22363, 32])
        print(f"意图中心向量: {self.intent_centers.shape}") # torch.Size([32, 64])

        # 添加意图分布统计信息
        print(f"意图中心最大相似度: {torch.max(torch.mm(self.intent_centers, self.intent_centers.t()) - torch.eye(self.intent_centers.size(0), device=self.device)).item():.4f}")
        print(f"意图中心平均相似度 (除对角线): {((torch.mm(self.intent_centers, self.intent_centers.t()) - torch.eye(self.intent_centers.size(0), device=self.device)).sum() / (self.intent_centers.size(0) * (self.intent_centers.size(0) - 1))).item():.4f}")

        # 将分布信息传递给模型
        self.model.user_intent_hist = self.user_intent_hist # (22362,32)
        self.model.intent_centers = self.intent_centers # (32,64)

    # ============================================================
    # RQ3 实验：按用户意图复杂度分层评估
    # ============================================================
    def stratify_test(self, cluster_dataloader, test_dataloader, n_groups=4):
        """
        分层评估流程：
        1. 用 cluster_dataloader（训练集）跑聚类 + 构建 user_intent_hist
        2. 对每个测试用户计算意图熵 H(p_u)
        3. 按 H 值四分位分组
        4. 各组分别计算 HR@20 / NDCG@20，并报告平均序列长度

        Args:
            cluster_dataloader: 训练集 DataLoader（用于聚类和构建意图分布）
            test_dataloader:    测试集 DataLoader
            n_groups:           分组数，默认 4（四分位）
        """
        # ----------------------------------------------------------
        # Step 1: 跑聚类阶段（eval + no_grad，不反传）
        # ----------------------------------------------------------
        print("===== [stratify_test] Step1: 聚类阶段 =====")
        self.model.eval()
        kmeans_training_data = []
        sequence_ids = []
        sequence_counter = 0

        with torch.no_grad():
            for rec_batch in tqdm(cluster_dataloader, desc="提取聚类特征"):
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_id, subsequence, _, _, _ = rec_batch
                batch_size = subsequence.size(0)

                seq_repr = self.seq_model(subsequence)[:, -1, :]  # [B, D]
                kmeans_training_data.append(seq_repr.detach().cpu().numpy())

                batch_ids = np.arange(sequence_counter, sequence_counter + batch_size)
                sequence_counter += batch_size
                sequence_ids.extend(batch_ids)

        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)  # [N, D]
        sequence_ids = np.array(sequence_ids)

        # 训练 K-Means
        self.clusters_t[0][0].train(kmeans_training_data, sequence_ids)

        # ----------------------------------------------------------
        # Step 2: 构建用户意图分布 → user_intent_hist [num_users, K]
        # ----------------------------------------------------------
        print("===== [stratify_test] Step2: 构建用户意图分布 =====")
        self.build_user_intent_distribution(cluster_dataloader)
        # 此时 self.user_intent_hist: [num_users, K]，在 device 上

        # ----------------------------------------------------------
        # Step 3: 计算每个用户的意图熵 H(p_u)
        # ----------------------------------------------------------
        print("===== [stratify_test] Step3: 计算意图熵 =====")
        dist = self.user_intent_hist.float()                      # [num_users, K]
        log_dist = torch.log(dist + 1e-10)                        # 防止 log(0)
        entropy = -(dist * log_dist).sum(dim=-1)                  # [num_users]
        entropy_np = entropy.detach().cpu().numpy()               # numpy [num_users]

        # ----------------------------------------------------------
        # Step 4: 按四分位数划定分组边界
        # ----------------------------------------------------------
        percentiles = np.percentile(entropy_np, np.linspace(0, 100, n_groups + 1))
        group_labels = np.digitize(entropy_np, percentiles[1:-1])  # 0 ~ n_groups-1
        group_names = {i: f"Q{i+1}" for i in range(n_groups)}
        print(f"熵分布 — min:{entropy_np.min():.3f} max:{entropy_np.max():.3f} "
              f"边界:{[f'{v:.3f}' for v in percentiles[1:-1]]}")

        # ----------------------------------------------------------
        # Step 5: 推理 + 收集每个 batch 的预测结果和用户组标签
        # ----------------------------------------------------------
        print("===== [stratify_test] Step4: 测试推理 =====")
        self.model.eval()

        # 用列表按组收集 (pred_top20, answer, seq_len)
        group_preds    = {i: [] for i in range(n_groups)}
        group_answers  = {i: [] for i in range(n_groups)}
        group_seq_lens = {i: [] for i in range(n_groups)}

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="测试推理"):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, input_seq_len, target_pos, answers = batch

                # 序列推荐前向
                rec_output = self.seq_model(input_ids)[:, -1, :]   # [B, D]
                rating_pred = self.predict_full(rec_output)         # [B, V]
                rating_pred = rating_pred.detach().cpu().numpy().copy()

                # 屏蔽历史交互
                batch_user_index = user_ids.detach().cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # Top-20
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_top20 = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                answers_np   = answers.detach().cpu().numpy()
                seq_lens_np  = input_seq_len.detach().cpu().numpy()

                # 按用户意图熵分组归集
                for b_idx, uid in enumerate(batch_user_index):
                    g = int(group_labels[uid])
                    group_preds[g].append(batch_pred_top20[b_idx])
                    group_answers[g].append(int(answers_np[b_idx]))
                    group_seq_lens[g].append(int(seq_lens_np[b_idx]))

        # ----------------------------------------------------------
        # Step 6: 按组计算指标并打印
        # ----------------------------------------------------------
        print("\n===== [stratify_test] 分层评估结果 =====")
        header = f"{'Group':<8} {'N':>6} {'AvgSeqLen':>10} {'HR@20':>8} {'NDCG@20':>10}"
        print(header)
        print("-" * len(header))

        results = {}
        for i in range(n_groups):
            preds   = np.array(group_preds[i])     # [N_g, 20]
            answers = [[a] for a in group_answers[i]]
            avg_len = float(np.mean(group_seq_lens[i])) if group_seq_lens[i] else 0.0
            n_users = len(preds)

            if n_users == 0:
                hr20 = ndcg20 = 0.0
            else:
                hr20   = self._recall_at_k(answers, preds, 20)
                ndcg20 = self._ndcg_k(answers, preds, 20)

            results[group_names[i]] = {
                "N": n_users,
                "AvgSeqLen": round(avg_len, 2),
                "HR@20": round(hr20, 4),
                "NDCG@20": round(ndcg20, 4),
            }
            print(f"{group_names[i]:<8} {n_users:>6} {avg_len:>10.2f} {hr20:>8.4f} {ndcg20:>10.4f}")

        print()
        # 写入 log 文件
        with open(self.args.log_file, "a") as f:
            f.write("\n===== Stratified Evaluation (by Intent Entropy) =====\n")
            f.write(header + "\n")
            for i in range(n_groups):
                r = results[group_names[i]]
                f.write(f"{group_names[i]:<8} {r['N']:>6} {r['AvgSeqLen']:>10} "
                        f"{r['HR@20']:>8} {r['NDCG@20']:>10}\n")

        return results

    # ------ 内部用的指标函数（避免依赖 self.args 之外的状态） ------
    @staticmethod
    def _recall_at_k(actual, predicted, topk):
        sum_recall = 0.0
        true_users = 0
        for i in range(len(predicted)):
            act_set  = set(actual[i])
            pred_set = set(predicted[i][:topk])
            if act_set:
                sum_recall += len(act_set & pred_set) / float(len(act_set))
                true_users += 1
        return sum_recall / true_users if true_users > 0 else 0.0

    @staticmethod
    def _ndcg_k(actual, predicted, topk):
        import math
        res = 0.0
        for i in range(len(actual)):
            idcg = sum(1.0 / math.log(j + 2, 2) for j in range(min(topk, len(actual[i]))))
            if idcg == 0:
                idcg = 1.0
            dcg = sum(
                int(predicted[i][j] in set(actual[i])) / math.log(j + 2, 2)
                for j in range(topk)
            )
            res += dcg / idcg
        return res / float(len(actual)) if actual else 0.0
