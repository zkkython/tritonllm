import torch
import triton
import triton.language as tl
import time

# 1，内核定义
@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                        # 获取当前块的 ID
    block_start = pid * BLOCK_SIZE                # 计算当前块的起始索引
    offsets = tl.arange(0, BLOCK_SIZE)            # 生成当前块的线程偏移量
    idx = block_start + offsets                   # 计算每个线程负责的索引
    mask = idx < N                                # 创建掩码，防止越界

    x = tl.load(X_ptr + idx, mask=mask)           # 加载 X 的值
    y = tl.load(Y_ptr + idx, mask=mask)           # 加载 Y 的值
    z = x + y                                     # 执行加法

    tl.store(Z_ptr + idx, z, mask=mask)           # 存储结果

# 2，内核调用
def vector_add_triton(X, Y):
    assert X.shape == Y.shape, "输入张量形状必须相同"
    N = X.numel()
    Z = torch.empty_like(X)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    vector_add_kernel[grid](X, Y, Z, N, BLOCK_SIZE=1024)

    return Z

# 3，主函数
if __name__ == "__main__":
    N = 10_000_000
    X = torch.randn(N, device='cuda', dtype=torch.float32)
    Y = torch.randn(N, device='cuda', dtype=torch.float32)

    # GPU 预热
    for _ in range(10):
        Z_triton = vector_add_triton(X, Y)

    # Triton 向量加法时间
    start_time = time.time()
    Z_triton = vector_add_triton(X, Y)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    # PyTorch 向量加法时间
    start_time = time.time()
    Z_pytorch = X + Y
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # 验证结果
    if torch.allclose(Z_triton, Z_pytorch):
        print("Triton 向量加法成功！")
    else:
        print("Triton 向量加法失败！")

    # 输出时间
    print(f"Triton 向量加法时间: {triton_time * 1000:.2f} ms")
    print(f"PyTorch 向量加法时间: {pytorch_time * 1000:.2f} ms")