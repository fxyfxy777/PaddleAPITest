# import numpy as np
# import paddle
# import torch

# 20452226, 5, 6, 7
# # 设置统一 shape 和 dtype
# shape = (20452226, 5, 6, 7)  # 可修改，例如 (2, 6)、(91, 1)、(2, 1000000)
# dtype = 'float16'
# axis = 1  # 归一化的轴（与 torch 的 dim 对应）
# p = 4
# GPU_DEVICE_ID = 2  
# # 设置GPU设备

# # 设置随机种子保证可复现
# np.random.seed(42)
# paddle.seed(42)
# torch.manual_seed(42)

# # ==================== 构造输入数据 ====================
# data_np = np.random.randn(*shape).astype(dtype)

# # ==================== Torch ====================
# # 使用示例
# device = torch.device(f"cuda:{GPU_DEVICE_ID}" if torch.cuda.is_available() else "cpu")
# torch_input = torch.tensor(data_np, requires_grad=True, device=device)
# torch_norm = torch.nn.functional.normalize(torch_input, p=p)
# torch_loss = torch_norm.mean()
# torch_loss.backward()
# torch_output = torch_norm.detach().cpu().numpy()
# torch_grad = torch_input.grad.detach().cpu().numpy()

# # ==================== Paddle ====================
# paddle.set_device("gpu:{}".format(GPU_DEVICE_ID), place="gpu")
# paddle_input = paddle.to_tensor(data_np, stop_gradient=False)
# paddle_norm = paddle.nn.functional.normalize(paddle_input, p=p)
# paddle_loss = paddle_norm.mean()
# paddle_loss.backward()
# paddle_output = paddle_norm.numpy()
# paddle_grad = paddle_input.grad.numpy()

# # ==================== 误差比较 ====================
# output_diff = np.mean(np.abs(paddle_output - torch_output))
# grad_diff = np.mean(np.abs(paddle_grad - torch_grad))

# print("====== Normalize 测试 ======")
# print(f"输入 shape: {shape}，归一化 axis: {axis}")
# print(f"输出平均误差: {output_diff:.6e}")
# print(f"梯度平均误差: {grad_diff:.6e}")
import paddle
import paddle.nn.functional as F

# 配置日志和环境
paddle.set_device('gpu')  # 可改为 'gpu' 如果你想在 GPU 上跑

# 定义测试样本（输入 shape, dtype, kwargs, 是否pass）
samples = [
    (([2281701379, 1], 'float32'), {'axis': 1}, False),
    (([1073741825, 4], 'float16'), {}, False),
    (([2, 2147483649], 'float16'), {}, True),
    (([20452226, 5, 6, 7], 'float16'), {'p': 4}, False),
    (([20452226, 5, 6, 7], 'float16'), {'p': 4, 'axis': 3}, False),
    (([2147483649, 2], 'float16'), {'p': 1.2}, False),
    (([4, 25565282, 6, 7], 'float16'), {'p': 4}, True),
    (([4, 25565282, 6, 7], 'float16'), {'p': 4, 'axis': 3}, False),
    (([4, 5, 214748365], 'float16'), {}, False),
    (([4, 5, 30678338, 7], 'float16'), {}, False),
    (([4, 5, 30678338, 7], 'float16'), {'p': 4}, False),
    (([4, 5, 30678338, 7], 'float16'), {'p': 4, 'axis': 3}, False),
    (([4, 5, 6, 35791395], 'float16'), {}, True),
    (([4, 5, 6, 35791395], 'float16'), {'p': 4}, False),
    (([4, 5, 6, 35791395], 'float16'), {'p': 4, 'axis': 3}, True),
    (([4294967297], 'float16'), {'axis': 0}, True),
    (([2, 2147483649], 'float16'), {'p': 2, 'axis': -1}, True),
]

for i, ((shape, dtype), kwargs, expected_pass) in enumerate(samples):
    print("=" * 30 + f" Case {i + 1} " + "=" * 30)
    print(f"Input shape: {shape}, dtype: {dtype}, kwargs: {kwargs}")
    try:
        # 注意：这里构造的 shape 不能超过实际机器内存限制，所以我们替代为合理的测试
        test_shape = [s for s in shape]  # 避免 OOM
        x = paddle.randn(test_shape, dtype=dtype)
        x.stop_gradient = False

        y = F.normalize(x, **kwargs)
        loss = y.mean()
        loss.backward()

        print("Output shape:", y.shape)
        print("Backward grad shape:", x.grad.shape)
        print("Status: ✅ Pass" if expected_pass else "Status: ❌ Should Fail but Passed")
    except Exception as e:
        print("Exception:", str(e))
        # print("Status: ✅ Fail as Expected" if not expected_pass else "Status: ❌ Should Pass but Failed")
