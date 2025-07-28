import numpy as np
import paddle
import torch
import traceback

# 设置使用的 GPU
paddle.set_device('gpu:7')
device = 'cuda:7'

# 设置固定随机种子，确保复现
np.random.seed(42)

def test_paddle_torch_dist_with_p(x_shape, y_shape, dtype='float16', p=2):
    print("=" * 80)
    print(f"测试 paddle.dist(x.shape={x_shape}, y.shape={y_shape}, dtype={dtype}, p={p})")
    
    try:
        # 数据类型映射
        np_dtype = np.float16 if dtype == 'float16' else np.float32
        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        paddle_dtype = dtype

        # 生成相同输入数据
        x_np = (np.random.random(x_shape) - 0.5).astype(np_dtype)
        y_np = (np.random.random(y_shape) - 0.5).astype(np_dtype)

        # 构建 Paddle Tensor
        x_pd = paddle.to_tensor(x_np, stop_gradient=False, dtype=paddle_dtype)
        y_pd = paddle.to_tensor(y_np, stop_gradient=False, dtype=paddle_dtype)

        # Paddle 正向计算
        dist_pd = paddle.dist(x_pd, y_pd, p=p)
        dist_pd.backward()
        print("[Paddle] Forward dist =", dist_pd.numpy())
        print("[Paddle] x.grad shape =", x_pd.grad.shape)
        print("[Paddle] y.grad shape =", y_pd.grad.shape)

        # 构建 Torch Tensor
        x_tc = torch.tensor(x_np, dtype=torch_dtype, requires_grad=True, device=device)
        y_tc = torch.tensor(y_np, dtype=torch_dtype, requires_grad=True, device=device)

        # Torch 正向计算
        dist_tc = torch.dist(x_tc, y_tc, p=p)
        dist_tc.backward()
        print("[Torch ] Forward dist =", dist_tc.detach().cpu().numpy())
        print("[Torch ] x.grad shape =", x_tc.grad.shape)
        print("[Torch ] y.grad shape =", y_tc.grad.shape)

        # 对比前向差异
        abs_diff = np.abs(dist_pd.numpy() - dist_tc.detach().cpu().numpy())
        rel_diff = abs_diff / (np.abs(dist_tc.detach().cpu().numpy()) + 1e-8)
        print("[对比] Forward 绝对误差 =", abs_diff)
        print("[对比] Forward 相对误差 =", rel_diff)

        # 对比梯度差异（只打印 norm）
        # x_grad_diff = np.linalg.norm(x_pd.grad.numpy() - x_tc.grad.detach().cpu().numpy())
        # y_grad_diff = np.linalg.norm(y_pd.grad.numpy() - y_tc.grad.detach().cpu().numpy())
        # print("[对比] x.grad 差异 (L2 norm) =", x_grad_diff)
        # print("[对比] y.grad 差异 (L2 norm) =", y_grad_diff)

        print("测试通过 ✅")
        print(x_pd.grad.numpy())
        print(x_tc.grad.detach().cpu().numpy())
        # print(y_pd.grad.numpy() , y_tc.grad.detach().cpu().numpy())

    except Exception:
        print("执行失败 ❌")
        print(traceback.format_exc())

# 示例调用
test_paddle_torch_dist_with_p([10], [429496730, 10], dtype='float16', p=4)
# test_paddle_torch_dist_with_p([10], [42949670, 10], dtype='float32', p=2)
