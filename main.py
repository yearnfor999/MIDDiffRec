import os
import numpy as np
import torch
import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import datetime

from models import InDiRec
from utils import EarlyStopping, set_seed, check_path, show_args_info
from datasets import D,D_random, get_seqs_and_matrixes, DatasetForInDiRec
from trainer import InDiRecTrainer



def get_args():
    # 创建一个 命令行参数解析器对象
    # 它的作用就是帮你写的 Python 脚本自动识别命令行参数
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--model_name", default="InDiRec", type=str)
    parser.add_argument("--data_dir", default="./datasets/", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--stratify_eval", action="store_true",
                        help="load best checkpoint, re-run clustering, then evaluate per intent-entropy quartile")
    parser.add_argument("--model_idx", default=0, type=int, 
                        help="model idenfier 10, 20, 30...")
    parser.add_argument('--cuda', type=int, default=0,
                            help='cuda device.')

    # robustness experiments
    parser.add_argument("--noise_ratio", type=float, default=0.0,
                        help="percentage of negative interactions in a sequence",)

    ## intent discovery args (保留意图相关参数，但移除对比学习特定参数)
    # parser.add_argument("--temperature", default=1.0, type=float,
    #                     help="softmax temperature (default:  1.0).")
    parser.add_argument("--intent_num",default=256,type=int,
                        help="the multi intent nums.")
    # parser.add_argument("--sim",default='dot',type=str,
    #                     help="the calculate ways of the similarity.")

    # model args
    parser.add_argument("--hidden_size", type=int, default=64, 
                        help="Number of hidden factors, i.e., embedding size.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, 
                        help="number of layers")
    parser.add_argument("--num_attention_heads", type=int, default=2,
                        help="number of attention heads")
    parser.add_argument("--hidden_act", type=str, default="gelu", 
                        help="active function type.") 

    # dropout
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, 
                        help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, 
                        help="hidden dropout p")
    parser.add_argument('--p', type=float, default=0.1,
                        help='uncondition dropout')

    

    # train args
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--print_log_freq", type=int, default=1, 
                        help="per epoch print res")
    parser.add_argument("--max_seq_length", type=int, default=50, 
                        help="max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight_decay of adam")
    parser.add_argument("--seed", default=2022, type=int)

    parser.add_argument("--initializer_range", type=float, default=0.02)


    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, 
                        help="weight of rating prediction")
    parser.add_argument("--diff_weight", type=float, default=1, 
                        help="weight of intent-aware diffusion task")
    # 移除对比学习损失权重
    # parser.add_argument("--cl_weight", type=float, default=0.2, 
    #                     help="weight of contrastive learning task")
    # 新增：用户长期兴趣损失权重
    parser.add_argument("--user_weight", type=float, default=0.05,
                        help="weight of user long-term interest loss")
    # 新增：扩散输出推荐损失权重
    parser.add_argument("--diffusion_rec_weight", type=float, default=0.05,
                        help="weight of diffusion output recommendation loss")
    parser.add_argument("--short_term_weight", type=float, default=0.01,
                        help="weight of short-term interest loss (BSARec frequency branch)")
    parser.add_argument("--tau", type=int, default=-1,
                        help="time step threshold for gated diffusion rec loss (default: -1 means timesteps//4)")
    
    # 对比学习参数
    parser.add_argument("--cl_weight", type=float, default=0.2,
                        help="weight of contrastive learning loss")
    parser.add_argument("--temperature", type=float, default=1,
                        help="temperature for InfoNCE loss")
    parser.add_argument("--sim", type=str, default="dot",
                        help="similarity function for contrastive learning (dot or cos)")
    parser.add_argument("--cl_use_reverse_process", action="store_true",
                        help="use T-step reverse process for contrastive learning (like 修改原版)")
    
    # BSARec需要的参数
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="BSARec frequency layer alpha parameter")
    parser.add_argument("--c", type=int, default=10,
                        help="BSARec frequency layer cutoff parameter for low/high frequency separation")

    # ablation study
    parser.add_argument('--without_segment', action="store_true",
                        help='dropout ')

    # control guidance signal
    parser.add_argument('--w', type=float, default=2.0,
                        help='control the strength of intent-guided signal s')
    
    # 单步CFG参数
    parser.add_argument('--use_single_step_cfg', action='store_true',
                        help='Use single-step CFG instead of T-step reverse diffusion')
    parser.add_argument('--single_step_cfg_w', type=float, default=2.0,
                        help='CFG weight for single-step CFG (recommended: 1.5-3.0)')
    
    # DDIM采样参数
    parser.add_argument('--use_ddim', action='store_true',
                        help='Use DDIM sampling instead of DDPM sampling')
    parser.add_argument('--ddim_steps', type=int, default=20,
                        help='Number of DDIM sampling steps (default: 20). Only used when --use_ddim is enabled')

    # diffusion
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='beta schedule')
    

    # parser.parse_args() 返回的就是一个 argparse.Namespace 对象，里面存着所有你定义过的参数和值
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    check_path(args.output_dir) # args.output_dir: output
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    print("Using Cuda:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.data_file = args.data_dir + args.data_name + ".txt" # 原始数据文件路径 ./datasets/Beauty.txt

    if args.without_segment:
        args.segmented_file = args.data_dir + args.data_name + "_random.txt"
        D_random(args.data_file,args.segmented_file,args.max_seq_length)

    else:
        args.segmented_file = args.data_dir + args.data_name + "_s.txt" # ./datasets/Beauty_s.txt
    
    if not os.path.exists(args.segmented_file):
        D(args.data_file,args.segmented_file,args.max_seq_length)

    # 训练数据
    # 参数 type="training"，文件是 Beauty_s.txt ，这个文件是之前用 D()切分生成的，它里面是已经切成前缀片段的训练样本
    # train_seq = user_seq，不是"每个用户的一条完整序列"，而是"所有用户被切出来的前缀子序列的集合
    train_user_ids, train_seq = get_seqs_and_matrixes("training", args.segmented_file)
    # 验证 + 测试数据
    # 原始数据文件 Beauty.txt
    # user_seq：每个用户的交互序列
    # max_item：全局最大物品 ID（比如 12101）
    # valid_rating_matrix：验证阶段用的"历史交互屏蔽矩阵
    # test_rating_matrix：测试阶段用的"历史交互屏蔽矩阵
    # 为什么要"屏蔽历史交互"？
    # 用户在训练/历史中已经点过的物品 不能再算作推荐目标
    # 如果不屏蔽，模型可能"推荐"用户已经交互过的物品，评测指标会被虚高
    eval_user_ids, user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_seqs_and_matrixes("rating", args.data_file)
    
    # 添加用户数量参数和用户ID映射（用于创建用户嵌入表）
    # 建立用户ID到连续索引的映射
    all_user_ids = set(train_user_ids + eval_user_ids) # (22636)
    sorted_user_ids = sorted(all_user_ids) # (22636)    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
    user_id_map = {uid: idx for idx, uid in enumerate(sorted_user_ids)} # {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,...}
    args.num_users = len(sorted_user_ids) # 22363
    args.user_id_map = user_id_map
    print(f"数据集用户数量: {args.num_users}") # 22363
    print(f"用户ID范围: {min(all_user_ids)} - {max(all_user_ids)}") # 1 - 22363
    
    # 将原始用户ID映射为连续索引
    train_user_indices = [user_id_map[uid] for uid in train_user_ids] # [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,...]
    eval_user_indices = [user_id_map[uid] for uid in eval_user_ids]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ...]

    # 设定"物品嵌入表"的大小。之所以 +2，是给两个特殊 token 预留位置：
    # 0：PAD（padding，用于补齐序列）
    # max_item+1：MASK（掩码/占位，用于mask预测或扩散任务）
    args.item_size = max_item + 2 # 12103
    # 显式记录 MASK 的ID
    args.mask_id = max_item + 1 # 12102

    # args.model_name → 模型名字，"InDiRec"
    # args.data_name → 数据集名字，比如 "Beauty"
    # args.model_idx → 模型编号，比如 1
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}" # InDiRec-Beauty-1
    args.log_file = os.path.join(args.output_dir, args_str + ".txt") # output/InDiRec-Beauty-1.txt
    show_args_info(args) # 把当前所有配置（超参、路径等）打印到控制台，便于你在日志里快速核对这次跑的设置
    # open(..., "a")：以 追加模式 打开文件
    # 如果文件不存在 → 自动创建；
    # 如果文件存在 → 不会覆盖原内容，而是在文件末尾继续写
    with open(args.log_file, "a") as f:
        # str(args)：把 argparse.Namespace 对象转成字符串
        # + "\n"：加换行，保证写入的内容和下一次写入分开
        f.write(str(args) + "\n")

    checkpoint = args_str + ".pt" # 生成模型权重文件名  InDiRec-Beauty-1.pt
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint) # output/InDiRec-Beauty-1.pt

    if args.eval_only:
        # 只评测时，不需要"聚类/训练/验证"这三种数据流，直接用空列表占位
        cluster_dataloader, training_dataloader, eval_dataloader = [],[],[]
        testing_dataset = DatasetForInDiRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)
    elif args.stratify_eval:
        # 分层评估：需要 cluster_dataloader（用于重建聚类和意图分布）和 testing_dataloader
        args.train_matrix = test_rating_matrix

        cluster_dataset = DatasetForInDiRec(args, train_seq, user_ids=train_user_indices, data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=False)

        testing_dataset = DatasetForInDiRec(args, user_seq, user_ids=eval_user_indices, data_type="test")
        testing_sampler = SequentialSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=False)

        training_dataloader, eval_dataloader = [], []
    else:
        # 训练不需要"屏蔽矩阵"，验证和测试必须要。原因是只有验证和测试需要NDCG这些指标
        args.train_matrix = valid_rating_matrix

        # 修改：传入映射后的连续用户索引用于长期兴趣建模
        cluster_dataset = DatasetForInDiRec(args, train_seq, user_ids=train_user_indices, data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset) # 注意这边是 SequentialSampler
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=False)

        training_dataset = DatasetForInDiRec(args, train_seq, user_ids=train_user_indices, data_type="train")
        # train → RandomSampler（随机顺序）
        # 这样每个 epoch 都能看到不同的样本顺序，打乱时序依赖，提升泛化
        # shuffle=True 其实内部就是调用了一个 RandomSampler, RandomSampler 和 shuffle=True 本质上是等价的，只是写法不同
        training_sampler = RandomSampler(training_dataset)
        # drop_last=True
        training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=args.batch_size, drop_last=False)

        eval_dataset = DatasetForInDiRec(args, user_seq, user_ids=eval_user_indices, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset) # 验证使用顺序采样，确保结果可重现
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=False)

        testing_dataset = DatasetForInDiRec(args, user_seq, user_ids=eval_user_indices, data_type="test")
        testing_sampler = SequentialSampler(testing_dataset) # 测试使用顺序采样，确保结果可重现
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=False)


    model = InDiRec(device, args=args) # 实例化 InDiRec

    trainer = InDiRecTrainer(model, training_dataloader, cluster_dataloader, eval_dataloader, testing_dataloader, device, args) # 实例化 InDiRecTrainer 调用 init()

    model.to(device)
    

    if args.eval_only:
        # 将评测用屏蔽矩阵换成测试集
        trainer.args.train_matrix = test_rating_matrix
        # 载入已有权重
        trainer.load(args.checkpoint_path)

        print(f"==================== Loading trained model from {args.checkpoint_path} for test...")
        # 直接测试并打印结果
        scores, log_info = trainer.test(0) # set epoch equal 0 to print results directly.

    elif args.stratify_eval:
        # 载入最优权重（不重新训练）
        trainer.load(args.checkpoint_path)
        print(f"==================== Loaded best checkpoint from {args.checkpoint_path}")
        print(f"==================== Running stratified evaluation ...")
        trainer.stratify_test(cluster_dataloader, testing_dataloader, n_groups=4)

    else:
        print(f"==================== Training ...")
        start_time = datetime.now()
        with open(args.log_file, "a") as f:
            f.write(f"====================  Training InDiRec at: {start_time} ====================" + "\n")
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True) # 实例化

        for epoch in range(args.epochs):
            # main
            #  └─ InDiRecTrainer(...)       # 子类构造 → super().__init__(...) 初始化父类字段
            #  └─ trainer.train(epoch)      # 调父类 Trainer.train
            #      └─ self.iteration(... )  # 动态绑定 → 子类 InDiRecTrainer.iteration
            #          ├─ 聚类阶段（eval + no_grad）
            #          └─ 训练阶段（train + backward）

            # 要点就这两个：(1) self 是子类实例，所以 iteration 走子类实现；(2) train/valid/test 只是把不同 dataloader + train 标志传给同一个入口
            trainer.train(epoch) # ① 训练一个 epoch

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                scores, _ = trainer.valid(epoch) # ② epoch = 0, 5, 10, 15, ... 和最后一个epoch做验证
                # 只用NDCG@20做早停（避免HR@20和NDCG@20互相冲突的问题）
                early_stopping(np.array(scores[-1:]), trainer.model) # ③ 用NDCG@20指标做早停监控，并在提升时保存最佳权重
                # ④ 如果连续 patience 轮没提升，就停
                if early_stopping.early_stop:
                    print("==================== Early stopping triggered!")
                    break

        # 将评测用屏蔽矩阵换成测试集
        trainer.args.train_matrix = test_rating_matrix
        print("====================  Testing Phase  ====================")
        # ⑤ 载回最优验证分数对应的权重
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        # ⑥ 在测试集上做最终一次评测
        scores, result_info = trainer.test(0)
        
        end_time = datetime.now()
        cost_time = str(end_time-start_time)
        print(args_str)
        print(result_info)
        with open(args.log_file, "a") as f:
            f.write(f"Finished at: {end_time} ==============================================================================================================" + "\n")
            f.write(f" diff_weight:{args.diff_weight} || intent_num: {args.intent_num} || dropout: {args.hidden_dropout_prob} || w: {args.w} || p: {args.p} || timesteps: {args.timesteps} || time_cost: {cost_time}" + "\n")
            f.write(result_info + "\n")
            f.write("======================================================================================================================================================" + "\n")