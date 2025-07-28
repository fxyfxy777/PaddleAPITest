# occupy_gpu.py
import torch
import time
import argparse

def occupy_memory(gpu_id: int, mem_gb: float):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # 占用指定显存，float32 每个元素 4 字节
    num_elems = int(mem_gb * 1e9 / 4)
    _ = torch.empty(num_elems, dtype=torch.float32, device=device)

    print(f"GPU {gpu_id}: Occupied ~{mem_gb} GB memory. Press Ctrl+C to stop.")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID to occupy")
    parser.add_argument("--mem", type=float, default=60.0, help="Memory to occupy in GB")
    args = parser.parse_args()

    occupy_memory(args.gpu, args.mem)
