# import paddle
# import paddle.nn.functional as F

# # 构造形状为 (2, 3) 的张量
# x = paddle.to_tensor([[1.0, 2.0, 3.0],
#                       [4.0, 5.0, 6.0]])

# # 按行(axis=1)归一化（对每一行的向量做L2归一化）
# out_axis1 = F.normalize(x, axis=1, p=2)

# # 按列(axis=0)归一化（对每一列的向量做L2归一化）
# out_axis0 = F.normalize(x, axis=0, p=2)

# print("按行归一化 (axis=1) 结果:\n", out_axis1.numpy())
# print("按列归一化 (axis=0) 结果:\n", out_axis0.numpy())
import numpy as np

# 加载数据
grads_data = np.load("npzFile/pnorm_grads_compare.npz")
x_data = np.load("npzFile/pnorm_x.npz")

paddle_arr = grads_data['paddle']
torch_arr = grads_data['torch']
x_arr = x_data['x']

# 打印基本信息
print("paddle shape:", paddle_arr.shape)
print("torch shape:", torch_arr.shape)
print("x shape:", x_arr.shape)

# NaN 和 Inf 统计
print("paddle NaNs:", np.isnan(paddle_arr).sum())
print("paddle Infs:", np.isinf(paddle_arr).sum())
print("torch NaNs:", np.isnan(torch_arr).sum())
print("torch Infs:", np.isinf(torch_arr).sum())

# 掩码：找出 paddle 出现 nan/inf，而 torch 正常的地方
paddle_nan_inf_mask = np.isnan(paddle_arr) | np.isinf(paddle_arr)
torch_valid_mask = ~(np.isnan(torch_arr) | np.isinf(torch_arr))
problem_mask = paddle_nan_inf_mask & torch_valid_mask

print("Number of problem elements (paddle nan/inf, torch valid):", np.sum(problem_mask))

# 找到对应的索引（前10个）
problem_indices = np.argwhere(problem_mask)
print(f"找到异常位置数量: {len(problem_indices)}")
print("前10个符合条件的位置索引:")

# 打印前10个位置的信息
for i, idx in enumerate(problem_indices[:10]):
    idx_tuple = tuple(idx)
    paddle_val = paddle_arr[idx_tuple]
    torch_val = torch_arr[idx_tuple]
    x_val = x_arr[idx_tuple]
    print(f"\n[{i+1}] 索引: {idx_tuple}")
    print(f"    paddle: {paddle_val}")
    print(f"    torch : {torch_val}")
    print(f"    x     : {x_val}")


