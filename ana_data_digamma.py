import numpy as np
import torch
from torch.testing import assert_close

# 加载输入数据，允许pickle
x_data = np.load("all_tensors.npz")['numpy_tensor']


# 加载数据
grads_data = np.load("grads_comparison.npz")
paddle_grad_np = grads_data['paddle_grad']
torch_grad_np = grads_data['torch_grad']

# 转换为 PyTorch Tensor
paddle_grad_tensor = torch.from_numpy(paddle_grad_np)
torch_grad_tensor = torch.from_numpy(torch_grad_np)

# 打印形状确认
print("Paddle 梯度 shape:", paddle_grad_tensor.shape)
print("Torch 梯度 shape:", torch_grad_tensor.shape)



nan_torch = torch.isnan(torch_grad_tensor).sum().item()
pos_inf_torch = (torch_grad_tensor == float('inf')).sum().item()
neg_inf_torch = (torch_grad_tensor == float('-inf')).sum().item()
total_torch = nan_torch + pos_inf_torch + neg_inf_torch

# Paddle
nan_paddle = torch.isnan(paddle_grad_tensor).sum().item()
pos_inf_paddle = (paddle_grad_tensor == float('inf')).sum().item()
neg_inf_paddle = (paddle_grad_tensor == float('-inf')).sum().item()
total_paddle = nan_paddle + pos_inf_paddle + neg_inf_paddle

# 打印结果
print(f"Torch - NaN: {nan_torch}, +Inf: {pos_inf_torch}, -Inf: {neg_inf_torch}, Total: {total_torch}")
print(f"Paddle - NaN: {nan_paddle}, +Inf: {pos_inf_paddle}, -Inf: {neg_inf_paddle}, Total: {total_paddle}")
print(f"Difference (Torch - Paddle): {total_torch - total_paddle}")

# ---------- 找出 Torch 中为 Inf 的位置 ----------
inf_mask_torch = ~torch.isfinite(torch_grad_tensor)  # 包含 NaN, +Inf, -Inf
inf_indices = inf_mask_torch.nonzero(as_tuple=False)  # [N, D] tensor

print(f"Torch 中非 finite 的元素数量: {inf_indices.shape[0]}")

# 将 x_data 转为 torch.Tensor（假设它和梯度同 shape）
x_tensor = torch.from_numpy(x_data)

# 提取对应位置的值
torch_inf_vals = torch_grad_tensor[inf_mask_torch]
paddle_vals_at_inf = paddle_grad_tensor[inf_mask_torch]
x_vals_at_inf = x_tensor[inf_mask_torch]

# 打印一些统计信息
print("== Torch 梯度为 inf 的位置对应信息 ==")
print(f"对应 Paddle 梯度范围: min={paddle_vals_at_inf.min().item():.6e}, max={paddle_vals_at_inf.max().item():.6e}")
print(f"对应输入 x 的范围: min={x_vals_at_inf.min().item():.6e}, max={x_vals_at_inf.max().item():.6e}")

# 可选：打印前几个位置的值看样例
print("\n前几个为 inf 的 Torch 梯度对应的数据：")
for i in range(min(10, torch_inf_vals.numel())):
    print(f"[{i}] Torch: {torch_inf_vals[i].item():.3e}, Paddle: {paddle_vals_at_inf[i].item():.3e}, x: {x_vals_at_inf[i].item():.3e}")


paddle_grad_finite = abs(paddle_grad_tensor[inf_mask_torch])
print(f"[Paddle | Torch为finite时] min: {paddle_grad_finite.min().item()}, max: {paddle_grad_finite.max().item()}")





# 设置容差并断言
# atol = 0.01
# rtol = 0.01
# assert_close(paddle_grad_tensor, torch_grad_tensor, atol=atol, rtol=rtol)
# # 梯度差异
# diff = paddle_grad - torch_grad
# print("最大绝对差:", np.abs(diff).max())
# print("最小绝对差:", np.abs(diff).min())
# print("均值绝对差:", np.abs(diff).mean())
# print("差异为0的元素比例:", np.sum(diff == 0) / diff.size)

# threshold = 1e-3
# large_diff_indices = np.where(np.abs(diff) > threshold)
# print(f"差异大于 {threshold} 的元素数量:", large_diff_indices[0].size)
