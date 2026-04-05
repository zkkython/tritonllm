# 古法编程Triton学习：从 0 到 1 读懂向量加法、Softmax 和矩阵乘法

## 1 本篇内容

1. Triton 三个核心案例的“新手可理解版”：Vector Add、Fused Softmax、Matmul。
2. 保留原始代码（不改动），并在代码旁边解释“它到底在做什么”。

## 2 先把最重要的一句话记住

在 GPU 上，很多时候**不是算得慢，而是搬得慢**。
也就是说，瓶颈往往在显存读写（memory IO），而不是浮点运算本身。

所以 Triton 的核心价值是：

- 让你更直接控制“如何分块处理数据”
- 让数据尽量在更快的层级停留（减少来回 DRAM），减少数据搬运
- 把多个步骤融合在一个 kernel 内完成

> 重点提示
> 对初学者来说，先别追求“语法全懂”，先追求“数据流全懂”。你能画出数据从哪里来、在哪里算、最后写到哪里，就入门了。

---

## 3 基础概念

### 3.1. 什么是 Triton 里的 `program`？

可以粗略理解为：

- 你写的 kernel 会被启动很多个并行实例
- 每个实例处理一小块数据
- 每个实例都有一个编号 `pid = tl.program_id(axis=...)`

它有点像 CUDA 里的 block 概念（不是完全等价，但初学阶段可这样理解）。

### 3.2. 什么是 `BLOCK_SIZE`？

就是“每个 program 一次处理多少元素”。

- 太小：并行度可能高，但调度开销/访存模式不一定好
- 太大：寄存器/共享资源压力可能变大

所以后面你会看到 `autotune`：让 Triton 自动帮你试参数。

### 3.3. 什么是 `mask`？

边界保护。
例如长度 1000，`BLOCK_SIZE=256`，最后一块会超出边界。`mask` 可以确保越界位置不读不写。

### 3.4. 什么是 `stride`？

stride 是“沿某个维度走一步，地址要跳多少”。

- 对连续二维张量 `A(M, K)`，通常 `A.stride(0)=K`，`A.stride(1)=1，  A.stride(0) 代表在第0维的指针间隔就是列数K，针对A.stride(1) 代表就是在第1维那个轴的指针间隔，就是一个个挨着的，所以是1`
- 你写指针算术时，stride 是把二维索引映射到线性地址的关键

### 3.5. 为什么经常提 `contiguous()`？

因为很多高性能 kernel 默认希望内存布局规整，便于向量化/合并访问。
如果数据布局乱（比如复杂转置后未整理），性能会明显受影响，甚至逻辑要改。

---

## 4 看懂性能背景：GPU 内存层级

![](image/triton-tutorial-part1-wechat/img_3_b580c2ea039c.jpeg)

![](image/triton-tutorial-part1-wechat/img_4_833fbd53f698.jpeg)

新手可以先记“结论版”：

- DRAM：容量大，慢
- L2/片上更近存储：容量小，快

你在 Triton 里做的很多优化，本质就是：

- 把一块数据 load 进来
- 在块内尽量多做事
- 最后一次性写回

> 重点提示
> 一个常见误区是“公式一样，速度就该差不多”。在 GPU 世界里，公式一样但数据流不同，性能可能差数倍。

---

## 5 第一关：Vector Addition（建立最小心智模型）

先看原始 kernel（保持不变）：

```python
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 5.1 逐段讲解（新手版）

1. `@triton.jit`

- 表示这是要编译到 GPU 上执行的 Triton kernel。

2. 参数里的 `x_ptr/y_ptr/output_ptr`

- 在 kernel 视角，它们是指针，不是 Python 对象。

3. `pid = tl.program_id(axis=0)`

- 获取当前 program 编号。每个 program 负责不同区间。 整个数据会分块（Block)， 针对每个Block 都有一个program编号索引代表第几个块，所以处理数据的时候先要知道处理的是哪个块，然后并行对这个块进行数据处理

4. `block_start + tl.arange(...)`

- 构造当前 program 要处理的索引范围。 拿到这部分块的所有数据

5. `mask = offsets < n_elements`

- 防止最后一块越界。

6. `tl.load` / `tl.store`

- 从显存读、向显存写；`mask` 会作用在读写上。

下面是 Python 包装函数（保持不变）：

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

### 5.2 这里最关键的两行

- `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)`
  - 表示总共要启动多少个 program。
- `add_kernel[grid](...)`
  - 这是 Triton kernel 的启动方式。

> 重点提示
> 看到 `grid` 就问自己：我总共要处理多少数据？每个 program 处理多少？那要启动几个？

---

## 6 第二关：Fused Softmax（第一次真正感受 IO 优化）

先看 PyTorch naive 版本（保持不变）：

```python
import torch

import triton
import triton.language as tl

@torch.jit.script
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

这段代码逻辑很清晰，但会经历多轮读写。
在大矩阵上，这些读写成本很高。

再看 Triton 版本（保持不变）：

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

### 6.1 为什么这个版本更快（初学者要点）

1. 以“行”为单位并行：`row_idx = tl.program_id(0)`。
2. 一次把一行（或补齐到块）加载到更快存储里。
3. 在块内完成 `max -> exp -> sum -> div`。
4. 最后一次写回结果。

### 6.2 三个容易困惑的问题

1. 为什么 `BLOCK_SIZE = next_power_of_2(n_cols)`？

- 很多底层执行模式对 2 的幂更友好，通常更容易拿到好性能。

2. 为什么 `other=-float('inf')`？

- 被 mask 的位置 `exp(-inf)=0`，不会影响 softmax 分母。

3. `num_warps` 是什么？

- 你可以理解成“并行执行资源配置”的一个旋钮。不同 shape/GPU 下最优值不同。

> 重点提示
> softmax 这个例子最值得学的不是“公式”，而是“把多个步骤融合在一次数据加载里做完”。

---

## 7 第三关：Matrix Multiplication（从会用到会调）

先看分块乘法伪代码（保持不变）：

```python
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```

![](image/triton-tutorial-part1-wechat/img_5_1e4077dbd001.png)

这个伪代码表达的是：

- 把大矩阵切成小块
- 每次算一个输出块 `C_block`
- 在 K 维度上累计 dot 结果

指针定位伪代码（保持不变）：

```cpp
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
```

对应 Triton 指针构造（保持不变）：

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
```

### 7.1 这段最需要理解什么？

- `offs_am / offs_bn / offs_k`：三个方向的局部坐标。
- `a_ptrs / b_ptrs`：把局部坐标映射到真实地址。
- 之后循环里不断移动 `a_ptrs/b_ptrs`，沿 K 维推进。

L2 分组调度代码（保持不变）：

```python
# Program ID
pid = tl.program_id(axis=0)
# Number of program ids along the M axis
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# Number of programs ids along the N axis
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# Number of programs in group
num_pid_in_group = GROUP_SIZE_M * num_pid_n
# Id of the group this program is in
group_id = pid // num_pid_in_group
# Row-id of the first program in the group
first_pid_m = group_id * GROUP_SIZE_M
# If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# *Within groups*, programs are ordered in a column-major order
# Row-id of the program in the *launch grid*
pid_m = first_pid_m + (pid % group_size_m)
# Col-id of the program in the *launch grid*
pid_n = (pid % num_pid_in_group) // group_size_m
```

![](image/triton-tutorial-part1-wechat/img_6_35b1cdd59686.png)

这段分组调度的直觉是：

- 按某种顺序安排 program 计算块
- 让相邻 program 更可能复用缓存中的数据
- 减少重复从慢速内存拿数据

完整 matmul 代码（保持不变）：

```python
import torch

import triton
import triton.language as tl

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c
```

### 7.2 Matmul 初学者关注清单

1. `assert a.is_contiguous()` / `assert b.is_contiguous()`

- 内存布局不规整时，很多优化假设会失效。

2. `accumulator` 用 `fp32`，最后转 `fp16`

- 常见技巧：中间累加用更高精度，减少误差。

3. `@triton.autotune(...)`

- 实战里你很难一次选对参数，自动调参很重要。

4. `mask` 在写回阶段同样关键

- 处理边界块时避免越界。

> 重点提示
> 新手先把“正确性跑通”，再做参数扫描。别一上来就追求极限性能。

---

## 8 从“看懂”到“会用”的练习路径

### 练习 1：Vector Add 改参数

- 固定向量长度，测试 `BLOCK_SIZE = 256/512/1024/2048`。
- 记录耗时，观察变化。

### 练习 2：Softmax 改 shape

- 固定行数，改变列数（如 512, 1024, 2048, 4096）。
- 观察 `num_warps` 对性能影响。

### 练习 3：Matmul 改分块

- 分别改变 `BLOCK_SIZE_M/N/K`。
- 对照 `autotune` 结果，建立参数直觉。

### 练习 4：正确性先行

- 每次改 kernel 后都和 PyTorch 输出做 `allclose` 对比。
- 先确保对，再谈快。

---

## 9 常见坑（新手高频）

1. 只看平均耗时，不做 warmup。
2. 没有 `torch.cuda.synchronize()` 就计时，结果不准。
3. 忘记 mask，边界 silently 出错。
4. 输入不是 contiguous，性能和预期差很多。
5. 一次改太多参数，找不到性能变化原因。

---

## 10. 结尾互动

接下来最想继续深入哪一块？

1. Triton 的 `autotune` 参数如何系统化搜索？
2. Softmax/Attention 如何做更深入的 kernel fusion？
3. Matmul 在不同 GPU（消费卡/数据中心卡）参数差异有多大？
4. 下一篇要不要写「Triton + PyTorch 自定义算子完整接入」？

欢迎留言你的 GPU 型号、驱动版本和你跑出来的 benchmark。
