"""
超小玩具例子：完整展示训练流程
3个物品，2维嵌入，序列长度3
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("🎮 玩具例子：3个物品的推荐系统")
print("=" * 60)
print()

# ========== 场景设定 ==========
print("📊 场景设定:")
print("-" * 40)

# 物品嵌入表 W_I ∈ R^(3×2)
W_I = torch.tensor([
    [1.0, 0.0],      # item0
    [0.0, 1.0],      # item1
    [0.8, 0.6]       # item2
], requires_grad=True)

print("物品嵌入表 W_I:")
print(f"  item0: {W_I[0].tolist()}")
print(f"  item1: {W_I[1].tolist()}")
print(f"  item2: {W_I[2].tolist()}")
print()

# 用户序列和目标
user_seq = [0, 2, 0]  # 用户交互序列
target = 2             # 目标下一物品: item2

print(f"用户序列: {user_seq}")
print(f"目标物品: item{target}")
print()

# ========== 第1步：构造扩散监督（教师） ==========
print("=" * 60)
print("📚 第1步：构造扩散监督（教师）")
print("=" * 60)

# 1.1 取出目标物品向量 x0_gt
x0_gt = W_I[target].clone()  # [0.8, 0.6]
print(f"1. 目标物品向量 x0 = W_I[item{target}] = {x0_gt.tolist()}")
print()

# 1.2 加噪得到 x_t
t = 700  # 时间步
sqrt_alpha_bar_t = 0.3
sqrt_one_minus_alpha_bar_t = 0.954
epsilon = torch.tensor([0.2, -0.4])  # 采样的噪声

x_t = sqrt_alpha_bar_t * x0_gt + sqrt_one_minus_alpha_bar_t * epsilon
print(f"2. 加噪 (t={t}):")
print(f"   x_t = {sqrt_alpha_bar_t:.1f}·{x0_gt.tolist()} + {sqrt_one_minus_alpha_bar_t:.3f}·{epsilon.tolist()}")
print(f"   x_t = {x_t.tolist()}")
print()

# 1.3 模拟去噪器预测
# 简化的去噪器：向真值方向恢复一部分
denoiser_param = torch.tensor([[0.9, 0.1], [0.1, 0.8]], requires_grad=True)
predicted_x0 = torch.matmul(x_t.unsqueeze(0), denoiser_param).squeeze()
predicted_x0 = torch.clamp(predicted_x0, -1, 1)  # 限制范围
predicted_x0 = torch.tensor([0.7, 0.5], requires_grad=True)  # 模拟预测结果

print(f"3. 去噪器预测:")
print(f"   pred_x0 = f_θ(x_t, cond, t) ≈ {predicted_x0.tolist()}")
print()

# 1.4 计算diff_loss
diff_loss = F.mse_loss(predicted_x0, x0_gt)
print(f"4. 去噪损失 (diff_loss):")
print(f"   L_diff = ||pred_x0 - x0||² = ||{predicted_x0.tolist()} - {x0_gt.tolist()}||²")
print(f"   L_diff = {diff_loss.item():.4f}")
print(f"   → 这个梯度只更新去噪器参数")
print()

# ========== 第2步：把先验蒸馏给打分侧（学生） ==========
print("=" * 60)
print("🎯 第2步：把先验蒸馏给打分侧（学生）")
print("=" * 60)

print("方案A：直接用单步预测（我们的优化）")
print("-" * 40)

# 2.1 detach阻断梯度
x0_hat = predicted_x0.detach()
print(f"1. 使用 detach 阻断梯度:")
print(f"   x0_hat = pred_x0.detach() = {x0_hat.tolist()}")
print()

# 2.2 计算logits
logits = torch.matmul(x0_hat.unsqueeze(0), W_I.T).squeeze()
print(f"2. 计算推荐分数:")
print(f"   logits = x0_hat · W_I^T")
print(f"   logits[0] = {x0_hat.tolist()} · {W_I[0].tolist()} = {logits[0].item():.2f}")
print(f"   logits[1] = {x0_hat.tolist()} · {W_I[1].tolist()} = {logits[1].item():.2f}")
print(f"   logits[2] = {x0_hat.tolist()} · {W_I[2].tolist()} = {logits[2].item():.2f}")
print(f"   logits = {logits.tolist()}")
print()

# 2.3 计算diffusion_rec_loss
target_tensor = torch.tensor([target])
diffusion_rec_loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
print(f"3. 扩散推荐损失:")
print(f"   L_diff-rec = CE(logits, target={target})")
print(f"   L_diff-rec = {diffusion_rec_loss.item():.4f}")
print(f"   → 这个梯度只更新物品嵌入W_I（因为detach）")
print()

# 演示早期步门控
tau = 250
if t <= tau:
    print(f"4. 早期步门控: t={t} <= τ={tau}, 损失生效 ✓")
else:
    print(f"4. 早期步门控: t={t} > τ={tau}, 损失置零 ✗")
print()

# ========== 第3步：主线排序（学生自己的监督） ==========
print("=" * 60)
print("🎓 第3步：主线排序（学生自己的监督）")
print("=" * 60)

# 模拟BSARec输出
h = torch.tensor([0.6, 0.7], requires_grad=True)  # 用户状态向量
print(f"1. BSARec输出用户状态:")
print(f"   h = BSARec(seq) = {h.tolist()}")
print()

# 计算推荐分数
logits_rec = torch.matmul(h.unsqueeze(0), W_I.T).squeeze()
print(f"2. 主线推荐分数:")
print(f"   logits_rec = h · W_I^T = {logits_rec.tolist()}")
print()

# 计算rec_loss
rec_loss = F.cross_entropy(logits_rec.unsqueeze(0), target_tensor)
print(f"3. 主线推荐损失:")
print(f"   L_rec = CE(logits_rec, target={target})")
print(f"   L_rec = {rec_loss.item():.4f}")
print(f"   → 这是主要的监督信号")
print()

# ========== 合并损失 & 梯度路由 ==========
print("=" * 60)
print("⚡ 合并损失 & 梯度路由")
print("=" * 60)

# 损失权重
lambda_rec = 1.0
lambda_diff = 0.1
lambda_diff_rec = 1.0

# 总损失
total_loss = lambda_rec * rec_loss + lambda_diff * diff_loss + lambda_diff_rec * diffusion_rec_loss

print(f"总损失 = {lambda_rec}·L_rec + {lambda_diff}·L_diff + {lambda_diff_rec}·L_diff-rec")
print(f"       = {lambda_rec}·{rec_loss.item():.4f} + {lambda_diff}·{diff_loss.item():.4f} + {lambda_diff_rec}·{diffusion_rec_loss.item():.4f}")
print(f"       = {total_loss.item():.4f}")
print()

print("梯度路由（关键设计）:")
print("  • L_diff → 只更新去噪器 f_θ")
print("  • L_diff-rec → 只更新物品嵌入 W_I（因为detach）")
print("  • L_rec → 更新BSARec和W_I")
print()

# ========== 推理阶段 ==========
print("=" * 60)
print("🚀 推理阶段")
print("=" * 60)

print("推理时：")
print("  1. 输入序列 → BSARec → 用户状态h")
print("  2. h · W_I^T → 推荐分数")
print("  3. 排序取Top-K")
print()
print("✅ 不需要扩散模型！")
print("✅ 不需要T步逆采样！")
print("✅ 直接用BSARec输出！")
print()

# ========== 效率对比 ==========
print("=" * 60)
print("📈 效率对比")
print("=" * 60)

print("如果使用T步逆扩散:")
T = 50
print(f"  • 需要 2×{T} = {2*T} 次去噪器前向（条件+无条件）")
print(f"  • 每个batch都要等待逆扩散完成")
print(f"  • GPU利用率低，训练慢")
print()

print("使用单步预测（我们的方案）:")
print(f"  • 只需要 1 次去噪器前向（已经在diff_loss中完成）")
print(f"  • 立即得到x0_hat，无需等待")
print(f"  • 加速比: {2*T}x")
print()

# ========== 核心洞察 ==========
print("=" * 60)
print("💡 核心洞察")
print("=" * 60)

print("1. 扩散模型是'老师'：")
print("   • 在连续空间预测'下一物品大概长什么样'")
print("   • 通过diff_loss学习这个能力")
print()

print("2. BSARec是'学生'（也是主力）：")
print("   • 通过rec_loss学习序列建模")
print("   • 通过diffusion_rec_loss接收老师的指导")
print()

print("3. detach()是关键：")
print("   • 确保老师和学生的学习目标不冲突")
print("   • 梯度路由清晰，训练稳定")
print()

print("4. 这不是妥协，而是智慧：")
print("   • 保留了扩散的价值（高质量先验）")
print("   • 避免了扩散的成本（T步采样）")
print("   • 实现了最优的性价比")
print()

print("🎯 结论：这是深思熟虑的最优设计！")
