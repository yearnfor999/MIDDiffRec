#!/bin/bash

# 测试多意图机制的不同配置

echo "=== 测试多意图机制 ==="

# 1. 基线：不使用多意图
echo "1. 基线测试（无多意图）..."
python main.py --data_name Beauty --model_idx 1 --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 32 --w 2 --epochs 50

# 2. 多意图 with nmc=1 (Top-1)
echo "2. 多意图测试 nmc=1..."
python main.py --data_name Beauty --model_idx 2 --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 32 --w 2 --use_multi_intent --nmc 1 --epochs 50

# 3. 多意图 with nmc=3 (Top-3) + max聚合
echo "3. 多意图测试 nmc=3 max聚合..."
python main.py --data_name Beauty --model_idx 3 --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 32 --w 2 --use_multi_intent --nmc 3 --mc_func max --epochs 50

# 4. 多意图 with nmc=3 (Top-3) + mean聚合
echo "4. 多意图测试 nmc=3 mean聚合..."
python main.py --data_name Beauty --model_idx 4 --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 32 --w 2 --use_multi_intent --nmc 3 --mc_func mean --epochs 50

# 5. 多意图 with nmc=5 (Top-5) + max聚合
echo "5. 多意图测试 nmc=5 max聚合..."
python main.py --data_name Beauty --model_idx 5 --diff_weight 1 --user_weight 0.05 --diffusion_rec_weight 0.05 --timesteps 50 --intent_num 32 --w 2 --use_multi_intent --nmc 5 --mc_func max --epochs 50

echo "=== 测试完成 ==="
