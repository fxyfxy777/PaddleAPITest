import paddle

def main():
    # 1. 检测可用 GPU 设备数量
    gpu_device_count = paddle.device.cuda.device_count()
    print(f"可用 GPU 设备数量: {gpu_device_count}")

    if gpu_device_count < 2:
        print("警告：没有足够的 GPU 设备（需要至少 2 个），将使用 CPU 或第一个 GPU。")
        device = paddle.CUDAPlace(0) if gpu_device_count > 0 else paddle.CPUPlace()
    else:
        # 2. 选择第二个 GPU（索引为 1）
        device = paddle.CUDAPlace(1)
        print(f"使用 GPU 设备: cuda:1")

    # 3. 在指定设备上生成随机数据并计算均值
    with paddle.fluid.dygraph.guard(device):
        # 生成 1000 个随机数（形状为 [1000]）
        data = paddle.randn([1000])
        mean = paddle.mean(data)
        print(f"生成的数据均值: {mean.numpy()[0]}")

if __name__ == "__main__":
    main()