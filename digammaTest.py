import re
import numpy as np
import paddle
import torch
id = 1
def parse_paddle_tensor_call(call_str):
    """
    解析类似 'paddle.digamma(x=Tensor([6, 6, 19884108, 6],"float16"))'
    提取 shape 和 dtype
    """
    shape_match = re.search(r'Tensor\(\[([^\]]+)\]', call_str)
    dtype_match = re.search(r'"(float\d+)"', call_str)

    if not shape_match or not dtype_match:
        raise ValueError("Invalid format")

    shape = [int(x.strip()) for x in shape_match.group(1).split(',')]
    dtype = dtype_match.group(1)

    return shape, dtype

def test_digamma_from_call_string_cuda(call_str, atol=0.01):
    shape, dtype = parse_paddle_tensor_call(call_str)
    print(f"Parsed shape={shape}, dtype={dtype} from input")
    np.random.seed(42)

    # numpy dtype
    np_dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[dtype]
    # np_input = (np.random.uniform(1.0, 1.2, size=shape)/1000).astype(np_dtype)
    np_input =(np.random.random(shape) - 0.5).astype(np_dtype)

    # PyTorch (on CUDA:1)
    torch_device = torch.device(f"cuda:{id}")
    torch_input = torch.tensor(np_input, requires_grad=True, device=torch_device)
    torch_output = torch.special.digamma(torch_input)
    torch_output.sum().backward()
    torch_out_np = torch_output.detach().cpu().numpy()
    torch_grad_np = torch_input.grad.cpu().numpy()

    # Paddle (on CUDA:1)
    paddle.set_device(f"gpu:{id}")
    paddle_input = paddle.to_tensor(np_input, stop_gradient=False, place=paddle.CUDAPlace(1))
    paddle_output = paddle.digamma(paddle_input)
    paddle_output.sum().backward()
    paddle_out_np = paddle_output.cpu().numpy()
    paddle_grad_np = paddle_input.grad.cpu().numpy()
    # print (f"[Paddle Output] {paddle_out_np}")
    # print (f"[Paddle Grad] {paddle_grad_np}")
    # print (f"[Torch Output] {torch_out_np}")
    # print (f"[Torch Grad] {torch_grad_np}")
    # Compare results
    np.testing.assert_allclose(torch_out_np, paddle_out_np, atol=atol, rtol=0.01)
    np.testing.assert_allclose(torch_grad_np, paddle_grad_np, atol=atol, rtol=0.01)
    
    # 找出 NaN 和 Inf 的位置
    nan_indices = np.argwhere(np.isnan(paddle_grad_np))
    inf_indices = np.argwhere(np.isinf(paddle_grad_np))

    for idx in nan_indices[0:min(10, len(nan_indices))]:
        print("NaN indices and values:")
        
        idx_tuple = tuple(idx)
        print(f"Index: {idx_tuple}")
        print(f"paddle_grad_np value: {paddle_grad_np[idx_tuple]}")
        print(f"torch_grad_np value: {torch_grad_np[idx_tuple]}")
        print(f"对应输入值: {np_input[idx_tuple]}")  # np_input 是对应的输入数组
        print("-----------------------------------")

    print("Inf indices and values:")
    for idx in inf_indices[0:min(10, len(inf_indices))]:
        idx_tuple = tuple(idx)
        print("Inf indices and values:")
        print(f"Index: {idx_tuple}")
        print(f"paddle_grad_np value: {paddle_grad_np[idx_tuple]}")
        print(f"torch_grad_np value: {torch_grad_np[idx_tuple]}")
        print(f"对应输入值: {np_input[idx_tuple]}")
        print("-----------------------------------")
    



if __name__ == "__main__":
    # test_digamma_from_call_string("paddle.digamma(x=Tensor([6, 6, 19884108, 6],\"float16\"), )")
    # test_digamma_from_call_string_cuda('paddle.digamma(x=Tensor([6, 6, 19884108, 6],"float16"))')
    test_digamma_from_call_string_cuda('paddle.digamma(x=Tensor([10, 21474836, 10],"float16"))')
    # paddle.digamma(x=Tensor([6, 6, 19884108, 6],"float16"))
