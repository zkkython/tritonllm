# 课时2 - vLLM 核心组件：Engine 模块和流式执行

这章不再按 ZeroMQ 模式逐项罗列。更好的讲法是：**拿一个异步请求，从进入 `AsyncLLM` 开始，一路追到它怎样变成流式输出。**

这样一来，通信模式、对象流转、模块关系都会自然落在链路里，不会碎。

## 1. 这章要回答的 4 个问题

1. `AsyncLLM` 为什么不自己做调度和模型执行？
2. `EngineCoreClient` 到底是“客户端”，还是“桥接层”？
3. `EngineCore` 为什么必须单独存在？
4. 流式输出为什么不是“模型一出 token 就直接写 HTTP”？

## 2. 先看总关系图：前端门面、桥接层、内核循环

![vLLM Engine 总关系图（Style 1 Flat Icon）](./assets/lesson-02-engine-overview-flat-icon.svg)

这张图里，最容易被忽略的关键点有两个：

- `AsyncLLM` 是门面，不是执行内核
- `OutputProcessor` 不是装饰层，而是把内核输出翻译成用户输出的必要桥梁

## 3. 先纠偏：REQ/REP 是对照概念，不是当前主链

你原来的提纲里有 `REQ/REP`，这个概念本身没有问题，但如果直接拿它当源码主链，就会把当前 v1 实现讲歪。

### 3.1 `REQ/REP` 能帮我们建立什么直觉

它能帮助我们先建立最简单的 RPC 心智模型：

- 客户端发请求
- 服务端回响应
- 一来一回，严格配对

### 3.2 为什么 vLLM 没把它当主链

因为 vLLM 的引擎通信不只是“问一句，答一句”：

- 请求可能持续很多 step
- 输出可能是增量流
- 一个前端可能管理多个 engine / DP rank
- 有控制面消息，也有输出面消息

所以当前 v1 的主链实际是：

- 输入控制面：`ROUTER / DEALER`
- 输出回流面：`PULL / PUSH`
- 辅助信号与聚合：`PAIR`、`XPUB / XSUB`

这就是更接近真实系统形态的实现。

## 4. 主调用链：一个异步请求如何穿过 Engine

### 4.1 第一步：`AsyncLLM.generate()` 接住上层请求

`vllm/vllm/v1/engine/async_llm.py` 里的 `generate()` 是整个异步链路的起点。它做的事情可以压缩成四步：

1. 确保输出处理任务在跑
2. 把用户输入转成标准请求
3. 把请求交给 `EngineCore`
4. 从请求专属队列里不断取回 `RequestOutput` 并 `yield`

### 4.2 关键片段 #1：`AsyncLLM.generate()` 真正的角色是“前台协调者”

```python
q = await self.add_request(
    request_id,
    prompt,
    sampling_params,
    ...
)

finished = False
while not finished:
    out = q.get_nowait() or await q.get()
    finished = out.finished
    yield out
```

这段代码很值得细看，因为它把 `AsyncLLM` 的定位暴露得非常清楚：

- 它不自己生成 token
- 它不自己做调度
- 它的工作是把请求送入内核，再把结果流回调用者

如果把 `AsyncLLM` 比作餐厅前台，它不是后厨，也不是配送员；它负责接单、关联订单状态、把后厨回来的菜按正确顺序端给顾客。

## 5. 第二步：输入为什么必须先过 `InputProcessor`

### 5.1 不同输入形态不能直接扔给内核

用户传进来的可能是：

- 纯字符串
- token ids
- chat message
- 多模态输入
- 带 LoRA、trace headers、priority 的请求

如果把这些差异直接扔给 `EngineCore`，后面的 scheduler、executor、worker 都得跟着处理入口分歧，整个系统会迅速变脏。

### 5.2 关键片段 #2：`add_request()` 的本质是“把用户输入标准化”

在 `AsyncLLM.add_request()` 里，最关键的转换是：

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    arrival_time,
    lora_request,
    tokenization_kwargs,
    trace_headers,
    priority,
    data_parallel_rank,
)
```

它的意义不是“做一次简单预处理”，而是创建出整个内核能理解的标准请求对象。也就是说，**这里发生的不是字符串处理，而是边界收口。**

### 5.3 对象流转表

| 层                 | 输入对象                  | 输出对象               | 为什么要变                     |
| ------------------ | ------------------------- | ---------------------- | ------------------------------ |
| API/上层           | prompt / chat / token ids | 原始输入               | 面向用户，形态很多             |
| `InputProcessor` | 原始输入                  | `EngineCoreRequest`  | 面向内核，必须统一             |
| `EngineCore`     | `EngineCoreRequest`     | `Request` / 调度对象 | 面向 scheduler，需要可调度状态 |

这张表才是真正的“代码之间关系图”。因为文件之间的关系，本质上是对象在不同层被谁创建、谁消费、谁转换。

## 6. 第三步：`EngineCoreClient` 为什么必须存在

### 6.1 它不是普通 SDK client，而是前后端桥接层

`v1/engine/core_client.py` 这个文件最容易被低估。名字叫 client，很多人会以为它只是“发消息的壳”。实际上它承担了两个系统边界职责：

- 把前端引擎和后台内核隔开
- 根据同步/异步、单进程/多进程、DP 场景选择不同 client 形态

### 6.2 关键片段 #3：`make_async_mp_client()` 在决定“系统怎么连”

```python
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        return DPAsyncMPClient(*client_args)
    return DPLBAsyncMPClient(*client_args)
return AsyncMPClient(*client_args)
```

这里真正发生的不是“选一个类名”，而是在决定：

- 是否多进程
- 是否数据并行
- 负载均衡在哪一层做
- 前端到底管理一个 engine，还是一组 engine

也就是说，`EngineCoreClient` 不是通信细节，它是**前端如何连接后端拓扑**的落点。

## 7. 第四步：为什么主链实际是 ROUTER/DEALER + PULL/PUSH

### 7.1 输入控制面

在 `core_client.py` 里，前端会创建 `ROUTER` 输入 socket；在 `core.py` 里，后台 `EngineCore` 进程用 `DEALER` 连回来。

这个组合比 `REQ/REP` 更适合当前场景，因为：

- 前端需要识别不同引擎身份
- 后端不应该被一问一答的同步语义卡死
- 多 engine / DP 场景下需要更灵活的路由

### 7.2 输出回流面

输出侧则是 `PUSH -> PULL`。

这很自然，因为 EngineCore 产生的是持续输出流，不是一次性答复。尤其在流式生成中，一个请求的输出会分很多批回来，用生产者/消费者模型更贴切。

### 7.3 把通信模式放回链路里看

```text
AsyncLLM
  -> EngineCoreClient
  -> ROUTER / DEALER   # 控制面：送请求、送控制消息
  -> EngineCore
  -> PUSH / PULL       # 输出面：回传阶段性结果
  -> OutputProcessor
```

这样讲，通信模式就不再是抽象名词，而是清楚地挂在请求生命周期里的两个位置。

## 8. 第五步：`EngineCore` 为什么是独立内核

### 8.1 它负责的不是“某个函数”，而是整个引擎循环

在 `v1/engine/core.py` 里，`EngineCore.step()` 是当前链路真正的内核循环。

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

这一小段代码几乎就是 vLLM 内核的浓缩版：

1. 调度器决定这一步算什么
2. 执行器把任务交给 worker
3. 执行结果回来
4. 调度器根据结果更新请求状态
5. 引擎构造输出，发回前端

### 8.2 关键片段 #4：`EngineCore.step()` 把调度和执行焊在一起

这段代码重要的不是“先 schedule 再 execute”这么简单，而是它定义了内核边界：

- 前面输入层不需要知道 worker 细节
- 后面 worker 层不需要知道 API 细节
- `EngineCore` 是调度与执行交汇的位置

没有这一层，前端门面就会直接和 worker、scheduler、output 互相纠缠，系统很快失控。

## 9. 第六步：流式输出为什么一定要过 `OutputProcessor`

### 9.1 内核输出不是用户输出

EngineCore 回来的不是“已经可直接发给用户的文本”，而是一组更偏内部语义的输出：

- token 级别结果
- 状态变化
- finished 标记
- logprobs
- 统计信息

如果 API 层直接理解这些内部对象，它就会和内核结构强耦合。

### 9.2 关键片段 #5：后台输出处理任务在做“翻译”

`AsyncLLM._run_output_handler()` 的核心逻辑是：

```python
outputs = await engine_core.get_output_async()
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

这说明流式输出真正的形态是：

1. 后台 core 持续吐引擎输出
2. `OutputProcessor` 把它翻译成请求级语义
3. 每个请求对应的 collector/queue 收到增量结果
4. `generate()` 再把它 `yield` 给上层

所以流式输出不是“模型一出 token 就直接往 HTTP 写”，而是经过了一层**输出语义重组**。

## 10. 本章最重要的对象流转

| 阶段     | 对象                  | 谁创建                                | 谁消费                                |
| -------- | --------------------- | ------------------------------------- | ------------------------------------- |
| 用户入口 | 原始 prompt           | API/调用方                            | `InputProcessor`                    |
| 标准请求 | `EngineCoreRequest` | `InputProcessor`                    | `EngineCoreClient` / `EngineCore` |
| 调度输入 | `Request`           | `EngineCore.preprocess_add_request` | `Scheduler`                         |
| 调度结果 | `SchedulerOutput`   | `Scheduler`                         | `Executor`                          |
| 执行结果 | `ModelRunnerOutput` | `Worker / ModelRunner`              | `Scheduler.update_from_output`      |
| 引擎输出 | `EngineCoreOutputs` | `EngineCore`                        | `OutputProcessor`                   |
| 用户输出 | `RequestOutput`     | `OutputProcessor`                   | `AsyncLLM.generate()` / API 层      |

如果把这张表记住，课时 2 的大部分代码关系就不再乱。

## 11. 回到 example：如何用示例验证这条链

### 11.1 推荐示例

- `examples/offline_inference/async_llm_streaming.py`
- 在线服务 + `examples/online_serving/openai_completion_client.py --stream`

### 11.2 验证方式

你可以不用一开始就单步调试到底，而是先观察这三件事：

1. 请求是先进入队列，再逐步拿到输出，不是一次性返回
2. 输出是增量返回，但 finished 标记最终收束一次请求
3. 外层看起来像 OpenAI streaming，内层其实是 `AsyncLLM -> EngineCoreClient -> EngineCore -> OutputProcessor`

## 12. 本章应该带走的心智模型

### 12.1 不是“谁的职责是什么”，而是“请求怎么穿过去”

这一章最核心的理解不是背类名，而是：

- `AsyncLLM` 负责接单与回流
- `InputProcessor` 负责边界收口
- `EngineCoreClient` 负责桥接前后端
- `EngineCore` 负责调度与执行主循环
- `OutputProcessor` 负责把内核输出变成用户输出

### 12.2 一句话总结

**Engine 模块不是一堆零散类，而是一条把请求从入口送到内核、再把结果送回用户的输送带。**

## 13. 小结与下章衔接

这章把前端门面、桥接层、内核循环和输出回流串起来了。

下一章继续顺着这条链往后走，不过焦点不再是“请求如何到达内核”，而是 `SchedulerOutput` 一旦生成之后，`Executor`、`Worker`、`ModelRunner` 究竟怎样把它变成一次真实的 GPU 执行。
