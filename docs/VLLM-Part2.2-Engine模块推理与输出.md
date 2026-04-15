# 1 Engine模块推理与输出流程
**<font style="color:#DF2A3F;">在读取这个内容之前，还是先建议先读一下VLLM-Part2.1-VLLM使用的ZMQ通信部分了解一下ZMQ。</font>**

在VLLM-Part2.1-VLLM使用的ZMQ通信基础知识部分我们已经学习了ZMQ的通信方式，它有ROUTER/DEALER PULL/PUSH模式，在这里我们会学习到它具体是怎么发挥作用的。在VLLM中正好使用了这两个PAIR对。这里我们以一个异步请求从进入到AsyncLLM开始，一路追踪它最终是怎么变成流式输出的。完整的流程图见下图所示：

<!-- 这是一张图片，ocr 内容为：内核执行层 前端协调层 桥接与通信层 API SERVER ENGINECORE 上层调用入口 ENGINECORECLIENT 内核循环:SCHEDULE->EXECUTE->UPDATE 独立存在,隔离API与WORKER细节 前后端桥接层 选择单进程/多进程/DP CLIENT 分解STEP 调度后 ASYNCLLM 桥接请求 执行 接单,建队列,YIELD流式结果 发送请求 门面,不直接调度模型 SCHEDULER EXECUTOR/ ZEROMQ面 WORKER 决定本STEP 标准化 执行什么 执行模型前向 PULL ROUTER/DEALER ROUTER 输出回流 控制面 结果产出 状态更新 输入请求与控制消息走ROUT R/DEALER OUTPUT INPUT 阶段性结果走PUSH/PULL PROCESSOR TOCESSOR MODELRUNNER 译内核输出 统一请求形态 内部输出翻译 OUTPUT 阶段性执行结果 PUSH/PULL 输出回流 组装用户 可见结果 按请求队列回流 REQUESTOUTPUT 图例 最终被ASYNCLLM 流式 YIELD 给调用方 蓝线:主调用/控制 绿线:执行与模型输出 紫线:流式结果回流 -->
![](https://cdn.nlark.com/yuque/0/2026/png/23012885/1776262408324-d56c2755-6fa6-4960-bb6f-af7ac69e10b7.png)

可以看到大致有如下的链路：

```python
UserInput->AsyncLLM
  -> EngineCoreClient
  -> ROUTER / DEALER   # 控制面：送请求Request、送控制消息
  -> EngineCore        # 通过step来完成真正的推理
  -> PUSH / PULL       # 输出面：回传模型token结果
  -> OutputProcessor   # 处理EngineCore的结果变成传递给用户的结果
```

# 2 流转的数据结构
如果我们希望能够很清楚的理解这个过程，我觉得理清楚下面的几个数据结构就能把整条链路理的很清晰。

| 阶段 | 对象 | 谁创建 | 谁消费 |
| --- | --- | --- | --- |
| 用户入口 | 原始 prompt | API/调用方 | `InputProcessor` |
| 标准请求 | `EngineCoreRequest` | `InputProcessor` | `EngineCoreClient` / `EngineCore` |
| 调度输入 | `Request` | `EngineCore.preprocess_add_request` | `Scheduler` |
| 调度结果 | `SchedulerOutput` | `Scheduler` | `Executor` |
| 模型执行结果 | `ModelRunnerOutput` | `Worker / ModelRunner` | `Scheduler.update_from_output` |
| 引擎输出 | `EngineCoreOutputs` | `EngineCore` | `OutputProcessor` |
| 用户输出 | `RequestOutput` | `OutputProcessor` | `AsyncLLM.generate()` / API 层 |


针对这些数据结构的流转过程，我们尝试用少量的vLLM代码讲清楚.

# 3 Engine模块主调用链路
接下来将会逐一展开这个链路过程：

```python
UserInput->AsyncLLM
  -> EngineCoreClient
  -> ROUTER / DEALER   # 控制面：送请求Request、送控制消息
  -> EngineCore        # 通过step来完成真正的推理
  -> PUSH / PULL       # 输出面：回传模型token结果
  -> OutputProcessor   # 处理EngineCore的结果变成传递给用户的结果
```

## 3.1 AsyncLLM.generate()
`vllm/vllm/v1/engine/async_llm.py` 里的 `generate()` 是整个异步链路的起点。它做的事情可以压缩成下面的内容：

1. 启动一个异步线程output_handler 专门用于处理EngineCore产生的输出结果EngineCoreOutputs
2. 把用户输入转成标准请求，并通过add_request把请求交给 `EngineCore`
3. 从请求专属队列里不断取回 `RequestOutput` 并 `yield`

```python
self._run_output_handler()
...
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

这段代码可以看出AsyncLLM的职责：

1. 它不自己生成token
2. 它不自己做调度
3. 它的工作就是把请求送给EngineCore, 然后再从EngineCore拿回结果

## 3.2 AsyncLLM.add_request() -> InputProcessor 处理
之所以用户的请求需要先经过InputProcessor 而不是直接传递给EngineCore主要是因为用户传进来的可能是：

+ 纯字符串
+ token ids
+ chat message
+ 多模态输入
+ 带 LoRA、trace headers、priority 的请求

如果把这些差异直接扔给 `EngineCore`，后面的 scheduler、executor、worker 都得跟着处理入口分歧，整个系统会多处进行处理容易造成混乱。

在 `AsyncLLM.add_request()` 里，最关键的代码是：

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

它的意义不是“做一次简单预处理”，而是创建出EngineCore认可的标准请求对象。属于请求的**边界收口。**

| **层** | **输入对象** | **输出对象** | **为什么要变** |
| --- | --- | --- | --- |
| **API/上层** | **prompt / chat / token ids** | **原始输入** | **面向用户，形态很多** |
| `**InputProcessor**` | **原始输入** | `**EngineCoreRequest**` | **面向EngineCore，必须统一** |
| `**EngineCore**` | `**EngineCoreRequest**` | `**Request**`** / 调度对象** | **面向 scheduler，需要可调度状态** |


## 3.3 EngineCoreClient的作用
我们通过流程可以知道请求并没有直接到EngineCore，而是过了EngineCoreClient， 但是它的责任是不是就是替AsyncLLM"发消息的壳“呢？ 从代码层面看，EngineCoreClient起了如下的作用：

+ 把前端引擎（非页面那个前端，用户态的前端）和 EngineCore隔离开
+ 根据同步/异步、单进程/多进程、DP场景来选择不同的client形态。

核心代码块在：make_async_mp_client() 它决定了到底使用哪个客户端去连接EngineCore

```python
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        return DPAsyncMPClient(*client_args)
    return DPLBAsyncMPClient(*client_args)
return AsyncMPClient(*client_args)
```

可以看到它在根据系统的不同信息去创建不同的Client连接EngineCore

+ 是否多进程
+ 是否数据并行
+ 负载均衡在哪一层做
+ 前端到底管理一个 engine，还是一组 engine

## 3.4 书接上回ROUTER/DEALER + PULL/PUSH在VLLM中的作用
### 3.4.1 输入层面
在 `core_client.py` 里，AsyncMPClient(EngineCoreClient)会创建 `ROUTER` input socket；在 `core.py` 里，后台 `EngineCore` 进程用 `DEALER` 连回来。这样构成了ROUTER/DEALER, EngineCoreRequests 推到EngineCore就是用的input_socket这个方式。

这个组合比 `REQ/REP（一问一答的方式）` 更适合当前场景，因为：

+ 前端需要识别不同引擎身份（假设有DP并行的话，就会有多个EngineCore）
+ 后端不应该被一问一答的同步语义卡死

### 3.4.2 输出层面
输出层面使用的是PUSH->PULL， 因为 EngineCore 产生的是持续输出流（自回归的Decode，一个词一个词的蹦），不是一次性答复。尤其在流式生成中，一个请求的输出会分很多批回来，用生产者/消费者模型更加贴切。

EngineCoreOutputs回到EngineCoreClient就是使用的output_socket

### 3.4.3 ROUTER/DEALER + PULL/PUSH是如何连接的
先重点看下MPClient代码：

```python
self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True
            )
self.resources.output_socket = make_zmq_socket(
    self.ctx, output_address, zmq.PULL
)
```

这个input_socket 是用来构建ROUTER/DEALER的， output_socket是用来构建PULL/PUSH的.

#### 针对input_socket: ROUTER/DEALER
+ **<font style="color:#DF2A3F;">AsyncMPClient(EngineCoreClient)是ROUTER  收数据</font>**

```python
self.input_socket = self.resources.input_socket = make_zmq_socket(
    self.ctx, input_address, zmq.ROUTER, bind=True
)
```

+ **<font style="color:#DF2A3F;">core.py EngineCore 是DEALER  发数据</font>**

```python
input_thread = threading.Thread(
    target=self.process_input_sockets,
    args=(
        addresses.inputs,
        addresses.coordinator_input,
        identity,
        ready_event,
    ),
    daemon=True,
)
...
def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""
       ...
        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            ...
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
```

**<font style="color:#DF2A3F;"></font>**

+ **<font style="color:#DF2A3F;">core_client.py中的AsyncMPClient使用input_socket发送请求 </font>**

```python
def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future
```

+ **<font style="color:#DF2A3F;">core.py EngineCore 使用input_socket接收请求</font>**

```python
type_frame, *data_frames = input_socket.recv_multipart(copy=False)
request_type = EngineCoreRequestType(bytes(type_frame.buffer))

```

#### 针对output_socket：PULL/PUSH
+ **<font style="color:#DF2A3F;">AsyncMPClient(EngineCoreClient)是PULL, 作用是拉回来EngineCore模型推理的结果</font>**

```python
self.resources.output_socket = make_zmq_socket(
    self.ctx, output_address, zmq.PULL
)
```

+ **<font style="color:#DF2A3F;">core.py EngineCore 是PUSH，发模型结果</font>**

这个链接咋建立的呢，就稍微有点神奇了，它是先借助input_socket通过握手的方式，从EngineCoreClient那边拿到的output_socket地址，然后再建立的PUSH，代码如下：

1. 握手：发HELLO

```python
def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
) -> EngineZmqAddresses:
    # Send registration message.
    handshake_socket.send(
        msgspec.msgpack.encode(
            {
                "status": "HELLO",
                "local": local_client,
                "headless": headless,
            }
        )
    )
```

2. 收到HELLO，回地址addresses, 里面就包含需要的output_socket

```python
if status == "HELLO" and engine.state == CoreEngineState.NEW:
    # Send init message with DP config info and config hash.
    # The config hash ensures all DP workers have compatible configs.
    init_message = msgspec.msgpack.encode(
        EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={
                k: getattr(parallel_config, k)
                for k in (
                    "data_parallel_master_ip",
                    "data_parallel_master_port",
                    "_data_parallel_master_port_list",
                    "data_parallel_size",
                )
            },
            parallel_config_hash=parallel_config.compute_hash()
            if parallel_config.data_parallel_size > 1
            else None,
        )
    )
    handshake_socket.send_multipart((eng_identity, init_message), copy=False)
    conn_pending[0 if local else 1] -= 1
    start_pending[0 if local else 1] += 1
    engine.state = CoreEngineState.CONNECTED
```

3. 拿到了output_socket之后，core.py EngineCore 在结果输出的时候，就建立PUSH, 通过while循环不断输出

```python
self.output_thread = threading.Thread(
    target=self.process_output_sockets,
    args=(
        addresses.outputs,
        addresses.coordinator_output,
        self.engine_index,
    ),
    daemon=True,
)

...

with ExitStack() as stack, zmq.Context() as ctx:
        sockets = [
            stack.enter_context(
                make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
            )
            for output_path in output_paths
        ]
        coord_socket = (
            stack.enter_context(
                make_zmq_socket(
                    ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                )
            )
            if coord_output_path is not None
            else None
        )
    ...
        while True:
                output = self.output_queue.get()
                ...
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
```

4. core_client.py AsyncEngineClient通过PULL拿到结果开始处理

```python
async def process_outputs_socket():
        try:
            while True:
                frames = await output_socket.recv_multipart(copy=False)
                ...
                outputs: EngineCoreOutputs = decoder.decode(frames)
                ...

                if output_handler is not None:
                   ...
                    await output_handler(_self, outputs)

                if outputs.outputs or outputs.scheduler_stats:
                    outputs_queue.put_nowait(outputs)
```

## 3.5 EngineCore: 最核心的引擎干活中心
在 `v1/engine/core.py` 里，`EngineCore.step()` 是当前链路真正的核心

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

这一小段代码几乎就是 vLLM 的浓缩版：

1. 调度器决定这一步算什么
2. 执行器把任务交给 worker
3. 模型执行结果取回来
4. 调度器根据结果更新请求状态
5. 引擎构造结果输出，发回前端



## 3.6 流式结果输出还要过一道OutputProcessor
EngineCore 回来的并不是“可直接发给用户的文本”，而是一组更偏内部语义的输出：

+ token 级别结果
+ 状态变化
+ finished 标记
+ logprobs
+ 统计信息

如果 API 层直接理解这些内部对象，它就会和EngineCore的结构强耦合。所以我觉得需要这个OutputProcessor做一下”翻译“

`AsyncLLM._run_output_handler()` 的核心逻辑是：

```python
outputs = await engine_core.get_output_async()
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

1. EngineCore 持续吐引擎输出
2. `OutputProcessor` 把它翻译成请求级语义的结果RequestOutput
3.  RequestOutputCollector处理每个请求对应的增量结果
4. `generate()` 再把它 `yield` 给上层



# 4 本文总结
以上就是vLLM Engine推理与输出的大流程，我觉得主要就是记住第二部分的几个数据结构之间的流转关系，然后理解清楚几个核心大类的功能，基本上就能把整个过程串下来。

+ `AsyncLLM` 负责接收用户请求与返回用户结果
+ `InputProcessor` 负责处理用户请求统一成EngineCoreRequest喂给EngineCore
+ `EngineCoreClient` 负责承上启下
+ `EngineCore` 负责调度与模型执行的主循环
+ `OutputProcessor` 负责把EngineCore的输出EngineCoreOutput变成用户输出RequestOutput
+ ZMQ 整个vLLM请求与结果这两个数据层面的通信核心方法



通过Part2.1和Part2.2应该可以了解ZMQ的原理以及ZMQ在vLLM推理中的真实作用（包括它的建链、收发数据等），同时也可以梳理清楚核心的主流程，后面看看是否在Part2.3的部分再着手展开一下更详细的代码。本部分主要还是让大家串清楚整个过程，记住最核心的几个类，记住最核心的几个数据结构，明白这些流转过程。Bye。

