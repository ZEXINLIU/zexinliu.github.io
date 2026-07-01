---
layout: post
title: "Inside nano-vLLM: Scheduler, Paged KV Cache, Prefill/Decode, and Sampling"
date: 2026-04-20 00:00:00
description: 沿请求生命周期拆解 scheduler、paged KV cache、prefill/decode、prefix cache 和采样。
tags: vllm inference large-language-models ai-infra
categories: ai-infra
_styles: |
  #markdown-content pre code,
  #markdown-content figure.highlight pre code,
  #markdown-content div.highlighter-rouge pre code {
    white-space: pre;
    word-wrap: normal;
    overflow-wrap: normal;
  }
---

<a id="top"></a>

nano-vllm 的价值在于把 vLLM 风格推理引擎压缩到一个足够小、但仍能跑通关键闭环的代码面。沿着一个 batch 的 prompt 追下去，可以看到请求调度、KV cache 分配、模型 forward、采样和后处理并不是五个孤立模块，而是围绕“下一轮还能不能继续写入和读取 KV”这件事互相约束。

这条主线里最值得抓住的是：

- **Continuous batching 的本质是每步重新做资源选择**：`waiting` 和 `running` 不是静态 batch，而是 scheduler 每轮根据 token budget、KV block 是否足够、是否还有 prefill 请求来重新组批。
- **Paged KV cache 把“逻辑上下文”和“物理显存块”拆开**：`Sequence.block_table` 是理解整套系统的关键，它让释放、复用、prefix cache 和 attention 访问都围绕 block id 运转。
- **chunked prefill、prefix cache、preemption 是同一个资源问题的三种处理方式**：长 prompt 占用 token budget，KV block 是稀缺显存资源，调度器必须在继续算、复用旧 KV、释放请求之间选择。
- **warmup 把冷启动成本前置成容量估算的一部分**：nano-vllm 先用模拟 prefill 暴露峰值显存，再用这个峰值决定可分配多少 KV blocks，因此 warmup 直接影响后续 scheduler 的资源上限。
- **prefill/decode 的差别不只是 batch shape 不同**：prefill 负责把 prompt KV 写入 cache 并采样第一个 completion token；decode 每轮只输入上一个采样 token，却要通过 `block_tables` 读完整历史 KV。
- **Tensor Parallel 是模型内部的数据布局约束**：QKV、MLP、embedding、LM head 的切分方式决定了哪些地方保留局部分片，哪些地方必须 all-reduce 或 gather。

本文以 `nano-vllm/` 中的官方源码为正确性基准；历史手写分析文件中的有效注释和小例子已经压缩进主教程与 `examples/`，后续不再把根目录同名草稿作为材料入口。

## 目录

- [1. 先用一句话建立全局图景](#mental-model)
- [2. 代码地图：每个文件在 pipeline 中的位置](#code-map)
- [3. 初始化：引擎启动时准备了什么](#init)
- [4. 请求进入：prompt 如何变成 Sequence](#request)
- [5. 一次 step：调度、执行、后处理的最小闭环](#step)
- [6. Prefill 调度：prompt 如何拿到 KV blocks](#prefill-schedule)
- [7. BlockManager：paged KV cache 的元数据核心](#block-manager)
- [8. Prefill 执行：模型如何写入 prompt KV](#prefill-run)
- [9. Postprocess：prefill 后如何进入 decode](#postprocess-prefill)
- [10. Decode 调度：每轮为什么只处理一个 token](#decode-schedule)
- [11. Decode 执行：如何读取历史 KV 并写入新 KV](#decode-run)
- [12. Prefix Cache：哪些 KV 可以复用](#prefix-cache)
- [13. Preemption：KV block 不够时怎么办](#preemption)
- [14. 完成输出：什么时候释放 KV cache](#finish)
- [15. 一条请求的端到端时间线](#timeline)
- [16. Tensor Parallel：模型内部如何分片执行](#tensor-parallel)
- [17. 采样策略：temperature 与 exponential race](#sampling)
- [18. nano-vllm 与生产 vLLM 的关系](#nanovllm-vs-vllm)
- [19. 理解检查与表达线索](#check)

<a id="mental-model"></a>

## 1. 先用一句话建立全局图景

nano-vllm 的推理循环可以概括成一句话：

```text
Scheduler 每轮从 waiting/running 里挑一批 seq，
ModelRunner 按 prefill 或 decode 的形态执行模型，
Attention 根据 slot_mapping/block_tables 写入或读取 paged KV cache，
Sampler 采样下一个 token，
Scheduler.postprocess 再把 token 和 KV 元数据更新回 seq。
```

如果你刚接触推理引擎，建议先抓住三个对象：

- `Sequence`：一个请求的状态。它记录 token、当前是否完成、已经缓存多少 token、用了哪些 KV blocks。
- `Scheduler`：决定本轮谁能跑。它维护 `waiting` 和 `running` 两个队列。
- `ModelRunner`：真正跑模型。它准备输入张量，设置 attention context，执行 Qwen3 forward 和采样。

这三个对象之间的关系：

```text
LLMEngine.generate
  -> add_request 创建 Sequence
  -> Scheduler.schedule 选择 Sequence
  -> ModelRunner.run 执行模型
  -> Scheduler.postprocess 更新 Sequence
  -> 重复直到所有 Sequence finished
```

[回到目录](#top)

<a id="code-map"></a>

## 2. 代码地图：每个文件在 pipeline 中的位置

按请求生命周期看，官方源码中的核心文件可以这样理解：

| 文件                                         | 角色                       | 在 pipeline 中的位置                                  |
| -------------------------------------------- | -------------------------- | ----------------------------------------------------- |
| `nano-vllm/nanovllm/engine/llm_engine.py`    | 总入口                     | 接收 prompts，循环调用 `step()`，收集输出             |
| `nano-vllm/nanovllm/engine/sequence.py`      | 请求状态                   | 记录 token、KV block table、cached/scheduled token 数 |
| `nano-vllm/nanovllm/engine/scheduler.py`     | 调度器                     | 在 waiting/running 中选择 prefill 或 decode batch     |
| `nano-vllm/nanovllm/engine/block_manager.py` | KV block 元数据管理        | 分配、释放、复用物理 KV blocks                        |
| `nano-vllm/nanovllm/engine/model_runner.py`  | 模型执行器                 | 准备 prefill/decode 输入，调用模型 forward，采样      |
| `nano-vllm/nanovllm/utils/context.py`        | attention 上下文           | 把调度信息传给 Attention                              |
| `nano-vllm/nanovllm/layers/attention.py`     | KV 写入与 attention kernel | 通过 `slot_mapping` 写 KV，通过 `block_tables` 读 KV  |
| `nano-vllm/nanovllm/models/qwen3.py`         | 模型结构                   | embedding、decoder layers、attention、MLP、lm head    |
| `nano-vllm/nanovllm/layers/sampler.py`       | 采样                       | 根据 logits 和 temperature 采样下一个 token           |

Tensor Parallel 相关内容主要在 `nano-vllm/nanovllm/layers/linear.py` 和 `nano-vllm/nanovllm/layers/embed_head.py`。本文把它放在 [第 16 节](#tensor-parallel)，CPU 仿真代码对应 `examples/tensor_cpu_sim.py`。

[回到目录](#top)

<a id="init"></a>

## 3. 初始化：引擎启动时准备了什么

入口是：

```python
llm = LLMEngine(model, **kwargs)
```

### 3.1 Config：把运行参数和模型配置合并

`LLMEngine.__init__()` 先筛出属于 `Config` 的参数：

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
```

`Config.__post_init__()` 会：

1. 检查 `model` 是本地目录。
2. 检查 `kvcache_block_size % 256 == 0`。
3. 检查 `1 <= tensor_parallel_size <= 8`。
4. 用 `AutoConfig.from_pretrained(self.model)` 读取 HuggingFace config。
5. 把 `max_model_len` 限制在模型最大 position embeddings 内。

随后：

```python
Sequence.block_size = config.kvcache_block_size
```

这一步很关键。`Sequence.num_blocks`、`Sequence.block(i)`、`BlockManager` 里的 block 计算，都依赖同一个 `block_size`。

### 3.2 ModelRunner：模型、warmup 和 KV cache 的拥有者

`ModelRunner(config, rank, event)` 做几件大事：

1. 初始化 `torch.distributed`。
2. 设置当前 CUDA device。
3. 构造 `Qwen3ForCausalLM(hf_config)`。
4. `load_model(self.model, config.model)` 加载 safetensors 权重。
5. 构造 `Sampler()`。
6. `warmup_model()` 跑一次模拟 prefill。
7. `allocate_kv_cache()` 分配 KV cache。
8. 如果允许 CUDA graph，则 `capture_cudagraph()` 捕获 decode 图。

这里的顺序很有信息量：nano-vllm 不是先用静态规则估算 KV cache，再假设后续 forward 一定能放得下；它先用一次模拟 prefill 测出模型执行路径的峰值显存，再把剩余显存换算成 `num_kvcache_blocks`。这就是本项目里 cold start / warmup 和调度容量之间的直接关系。

大模型推理里的冷启动通常指一个服务进程刚启动、还没有稳定处理请求时，需要一次性付出的初始化成本。常见成本包括：

- 权重从磁盘加载到 GPU。
- CUDA context、allocator、Triton/torch compile、attention kernel 等路径第一次触发。
- 某些 decode shape 的 CUDA Graph 捕获。
- 根据真实执行峰值预留足够显存，避免第一个大请求触发 OOM。

nano-vllm 把这些动作放在 `ModelRunner.__init__()` 里，让真实请求进入 `Scheduler` 之前，模型路径和显存预算已经准备好。

`warmup_model()` 的作用不是生成真实文本，而是用假 sequence 触发一次接近上限的 prefill forward：

```python
seq_len = min(max_num_batched_tokens, max_model_len)
num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)
seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
for seq in seqs:
    seq.num_scheduled_tokens = seq_len
self.run(seqs, True)
```

这段代码前后还有两句容易忽略：

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
...
torch.cuda.empty_cache()
```

`reset_peak_memory_stats()` 把峰值统计清零，随后 fake prefill 触发模型 forward。这样下一步 `allocate_kv_cache()` 能看到这次 forward 过程中真实出现过的 activation/workspace 峰值。

warmup 时 `Sequence.block_table` 是空的，KV cache 也还没有分配。`prepare_prefill()` 里有一个特别分支：

```python
if not seq.block_table:    # warmup
    continue
```

因此 warmup 不会构造真实 `slot_mapping`，Attention 里也不会把 K/V 写入正式 KV cache。它主要是在回答一个问题：在没有 KV cache 占满显存之前，模型跑最大 prefill 形态时还需要多少瞬时显存？

之后 `allocate_kv_cache()` 根据显存情况计算可用 KV blocks：

```python
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

可以把后半段理解成：

```text
可用于 KV cache 的预算
  = total * gpu_memory_utilization
  - 当前已经占用的显存 used
  - warmup 期间出现过的额外峰值 (peak - current)
```

源码写成 `- used - peak + current`，等价于上面这个式子。`used` 覆盖权重等常驻占用，`peak - current` 覆盖 forward 过程中短暂出现、但结束后释放的激活和 workspace。这样分配出来的 KV cache 不会把所有空闲显存吃干净，而是给后续 prefill forward 留出同等级别的瞬时空间。

为什么 `block_bytes` 要乘 `num_hidden_layers`？因为一个物理 block id 不是只属于某一层，而是每一层都有同一个 block id 对应的 K/V 存储位置。最终 KV cache 的形状是：

```python
(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
```

其中 `2` 表示 K 和 V。

随后代码遍历模型模块：

```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

也就是说，整个模型共享一块大的 KV cache tensor，每层 Attention 拿自己的 layer slice。

此时 `config.num_kvcache_blocks` 才有确定值。后面的 `Scheduler(config)` 会用这个值创建：

```python
BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
```

所以 warmup 不是对真实请求做预调度，也不是提前填充 prefix cache；它影响的是调度器之后能看到的物理 KV block 总量。block 总量越少，`can_allocate()` 越容易返回 `-1`，decode 中也越容易触发 preemption。

`capture_cudagraph()` 里还有第二类 warmup：

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
with torch.cuda.graph(graph, self.graph_pool):
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
```

这一段针对 decode batch size bucket 做图捕获。它和前面的 `warmup_model()` 目标不同：前者为 KV cache 容量估算服务，后者为后续 decode replay 减少调度和 kernel launch 开销服务。两者都属于冷启动成本前置，只是一个服务于显存预算，一个服务于执行路径稳定化。

### 3.3 Scheduler：请求队列和 KV block 元数据

`Scheduler(config)` 初始化：

```python
self.waiting = deque()
self.running = deque()
self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
```

两个队列的语义：

- `waiting`：还没有完成 prefill 的 sequence，或者被抢占后需要重新 prefill 的 sequence。
- `running`：prefill 已完成，后续每轮 decode 一个 token 的 active sequence。

注意：`running` 不是指“此刻正在 GPU 上运行 kernel”。它更像 vLLM 语境里的 active set：已经有可用 KV cache，可以进入 decode。

[回到目录](#top)

<a id="request"></a>

## 4. 请求进入：prompt 如何变成 Sequence

用户调用：

```python
outputs = llm.generate(prompts, sampling_params)
```

`generate()` 先把每个 prompt 加入 scheduler：

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

`add_request()` 做两件事：

```python
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

### 4.1 Sequence 保存了什么

`Sequence.__init__()` 中最重要的字段：

```python
self.seq_id = next(Sequence.counter)
self.status = SequenceStatus.WAITING
self.token_ids = copy(token_ids)
self.last_token = token_ids[-1]
self.num_tokens = len(self.token_ids)
self.num_prompt_tokens = len(token_ids)
self.num_cached_tokens = 0
self.num_scheduled_tokens = 0
self.is_prefill = True
self.block_table = []
```

初始状态可以理解为：

```text
token_ids = prompt tokens
num_cached_tokens = 0
num_scheduled_tokens = 0
block_table = []
status = WAITING
```

`block_table` 为空，表示这个请求还没有拿到任何物理 KV block。

### 4.2 num_tokens、num_cached_tokens、num_scheduled_tokens 的区别

这三个字段非常容易混：

- `num_tokens`：当前 sequence 总长度，包括 prompt 和已经采样出来的 completion token。
- `num_cached_tokens`：已经写入 KV cache 的 token 数。
- `num_scheduled_tokens`：本轮 scheduler 决定要计算的 token 数。

例如 prompt 长度是 6，刚完成 prefill 并采样了第一个 completion token 后：

```text
token_ids = [6 个 prompt token, 1 个 sampled token]
num_tokens = 7
num_cached_tokens = 6
last_token = 第 1 个 sampled token
```

为什么 `num_cached_tokens` 还是 6？因为刚采样出来的 token 还没有作为输入跑过模型，它的 KV 会在下一轮 decode 写入。

[回到目录](#top)

<a id="step"></a>

## 5. 一次 step：调度、执行、后处理的最小闭环

`generate()` 的核心循环：

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

`LLMEngine.step()` 是整个引擎的最小闭环：

```python
seqs, is_prefill = self.scheduler.schedule()
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
return outputs, num_tokens
```

这四步分别对应：

1. **调度**：`schedule()` 选择一批 sequence，并标记本轮是 prefill 还是 decode。
2. **执行**：`model_runner.run()` 把 sequence 变成模型输入张量，执行 forward 和采样。
3. **后处理**：`postprocess()` 更新 sequence 状态、KV block 元数据和完成队列。
4. **输出收集**：只收集本轮刚 finished 的 sequence。

`num_tokens` 的正负也来自这里：

- prefill 阶段：统计本轮实际处理多少 prompt tokens，所以是正数。
- decode 阶段：每个 sequence 处理一个 token，所以用 `-len(seqs)` 表示 decode batch size。

[回到目录](#top)

<a id="prefill-schedule"></a>

## 6. Prefill 调度：prompt 如何拿到 KV blocks

只要 `waiting` 非空，`Scheduler.schedule()` 会优先尝试 prefill。

初始变量：

```python
scheduled_seqs = []
num_batched_tokens = 0
```

进入 prefill loop：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
```

这里的 `remaining` 是本轮还能放多少 token。它对应 vLLM 里经常讨论的 token budget：prefill 可能很长，如果一个长 prompt 占满整个 step，就会影响其他请求的延迟；如果过度切碎，又会影响吞吐。这就是 chunked prefill 背后的吞吐与延迟权衡。

### 6.1 第一次进入 prefill：block_table 为空

如果：

```python
if not seq.block_table:
```

说明该 sequence 还没有物理 KV blocks。可能是全新请求，也可能是被 preempt 后释放过 block 的请求。

此时 scheduler 调用：

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
```

返回值含义：

- `-1`：当前 free blocks 不够，不能调度这个 sequence。
- `>= 0`：可以调度，并且这个数字表示命中了多少个完整 prefix cache blocks。

如果不能分配：

```python
if num_cached_blocks == -1:
    break
```

prefill loop 退出，后面如果没有安排任何 prefill，才会进入 decode。

### 6.2 计算本轮要处理多少 token

如果是首次分配或抢占后重新分配：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

命中的 cached blocks 对应的 token 已经有 KV，不需要重新算。

如果 `seq.block_table` 不为空，说明这是 chunked prefill 的后续 step，它已经有 block table，只是 prompt 还没算完：

```python
num_tokens = seq.num_tokens - seq.num_cached_tokens
```

### 6.3 chunked prefill 只允许本轮第一个 seq 被切

关键判断：

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

这句的意思是：

- 如果当前 sequence 放不下，并且本轮已经安排过别的 sequence，就不要再切这个 sequence。
- 如果当前 sequence 是本轮第一个，那么即使放不下完整 prompt，也可以调度 `remaining` 个 token。

所以 nano-vllm 的 chunked prefill 是一个简化版本：只允许本轮第一个 sequence 被切块 prefill。

### 6.4 分配 block 并放入 scheduled_seqs

如果 block table 为空：

```python
self.block_manager.allocate(seq, num_cached_blocks)
```

然后：

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
num_batched_tokens += seq.num_scheduled_tokens
```

如果本轮后 prompt 已经全部 prefill 完：

```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
```

这一步非常关键：sequence 在模型 forward 前就先被放入 `running`。为什么可以这样？因为 scheduler 已经决定本轮会把剩余 prompt tokens 算完；forward 和 postprocess 之后，它就真的具备 decode 条件了。

如果 prompt 没算完，sequence 仍留在 `waiting`，等待下一次 prefill。

[回到目录](#top)

<a id="block-manager"></a>

## 7. BlockManager：paged KV cache 的元数据核心

Paged KV cache 的核心思想是：不要给每条 sequence 分配一整段连续 KV 内存，而是切成固定大小 blocks。

在 `Sequence` 里：

```python
block_table = [physical_block_id_0, physical_block_id_1, ...]
```

它表示：

```text
sequence 的逻辑第 0 个 block -> 物理 block 7
sequence 的逻辑第 1 个 block -> 物理 block 3
sequence 的逻辑第 2 个 block -> 物理 block 12
```

Attention kernel 后面通过 `block_tables` 就能找到历史 KV。

### 7.1 Block 元数据

每个 `Block` 有：

```python
self.block_id = block_id
self.ref_count = 0
self.hash = -1
self.token_ids = []
```

含义：

- `ref_count`：有多少 sequence 正在引用这个物理 block。
- `hash`：完整 block 的 prefix hash，用于 prefix cache。
- `token_ids`：这个完整 block 对应的 token ids，用于二次校验 hash collision。

### 7.2 can_allocate：同时检查 cache 命中和可用空间

`can_allocate(seq)` 初始化：

```python
h = -1
num_cached_blocks = 0
num_new_blocks = seq.num_blocks
```

然后遍历完整 blocks：

```python
for i in range(seq.num_blocks - 1):
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        break
    num_cached_blocks += 1
    if block_id in self.used_block_ids:
        num_new_blocks -= 1
```

几个细节：

1. `range(seq.num_blocks - 1)` 不处理最后一个可能未满的 block。
2. prefix cache 必须从头连续命中，中间断了就停。
3. 即使命中 hash，也要比较 `token_ids`，避免 hash collision。
4. 如果命中的 block 正在 used 中，就不需要新 free block，只增加引用计数即可。

最后：

```python
if len(self.free_block_ids) < num_new_blocks:
    return -1
return num_cached_blocks
```

### 7.3 allocate：先复用，再新分配

`allocate(seq, num_cached_blocks)` 分两段。

先处理缓存命中的 blocks：

```python
for i in range(num_cached_blocks):
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block_id = self.hash_to_block_id[h]
    block = self.blocks[block_id]
    if block_id in self.used_block_ids:
        block.ref_count += 1
    else:
        block.ref_count = 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
    seq.block_table.append(block_id)
```

再为剩余 blocks 分配新空间：

```python
for i in range(num_cached_blocks, seq.num_blocks):
    seq.block_table.append(self._allocate_block())
```

最后：

```python
seq.num_cached_tokens = num_cached_blocks * self.block_size
```

这表示命中的 prefix blocks 已经有 KV 了。

[回到目录](#top)

<a id="prefill-run"></a>

## 8. Prefill 执行：模型如何写入 prompt KV

调度结束后，`LLMEngine.step()` 调用：

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

如果是 TP 多进程，rank0 会先通过 shared memory 通知其他 rank 也执行同样的方法：

```python
if self.world_size > 1 and self.rank == 0:
    self.write_shm(method_name, *args)
method = getattr(self, method_name, None)
return method(*args)
```

### 8.1 prepare_prefill：把 Sequence 变成张量

prefill 输入不是简单 padding 成 `[batch, max_len]`，而是拼成 varlen 格式。

对每个 seq：

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
seqlen_k = end
input_ids.extend(seq[start:end])
positions.extend(range(start, end))
```

含义：

- `input_ids` 只包含本轮真正要算的新 tokens。
- `positions` 是这些 tokens 在原 sequence 里的绝对位置。
- `seqlen_q` 是本轮 query 长度。
- `seqlen_k` 是这个 query 能看到的上下文长度。

如果没有 prefix cache：

```text
prompt = [A, B, C, D, E, F]
start = 0
seqlen_q = 6
seqlen_k = 6
input_ids = [A, B, C, D, E, F]
positions = [0, 1, 2, 3, 4, 5]
```

如果前 4 个 token 命中 prefix cache：

```text
prompt = [A, B, C, D, E, F]
start = 4
seqlen_q = 2
seqlen_k = 6
input_ids = [E, F]
positions = [4, 5]
```

此时 query 只有 `[E, F]`，但它们要看见 `[A, B, C, D, E, F]` 的上下文，因此 `seqlen_k > seqlen_q`。

### 8.2 slot_mapping：新 KV 写到哪里

`prepare_prefill()` 根据 `seq.block_table` 计算 `slot_mapping`：

```python
slot_start = seq.block_table[i] * self.block_size
slot_mapping.extend(range(slot_start, slot_end))
```

`slot_mapping` 的作用是告诉 `store_kvcache()`：本轮第几个 token 的 K/V 要写到物理 KV cache 的哪个 slot。

例如：

```text
block_size = 4
seq.block_table = [7, 3]
```

则逻辑 token 0-3 写入物理 block 7，对应 slots 28-31；逻辑 token 4-7 写入物理 block 3，对应 slots 12-15。

### 8.3 block_tables：历史 KV 从哪里读

如果存在 prefix cache：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

`block_tables` 是一个二维表：

```text
[
  [seq0_block0, seq0_block1, ...],
  [seq1_block0, seq1_block1, ...],
]
```

长度不够的地方补 `-1`。

prefill 没有 prefix cache 时，attention 可以直接用本轮连续的 `k/v`。prefill 命中 prefix cache 时，历史 KV 已经在 paged cache 里，所以要传 `block_tables` 给 flash-attn。

### 8.4 set_context：把调度信息交给 Attention

`prepare_prefill()` 最后：

```python
set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
```

模型 forward 没有显式传入这些参数。Attention 内部会：

```python
context = get_context()
```

这是一种简化实现：通过全局 context 把 scheduler/model_runner 准备好的元数据传给 attention layer。

### 8.5 Qwen3 forward：KV 在 Attention 中写入

`run_model()` 中：

```python
return self.model.compute_logits(self.model(input_ids, positions))
```

模型结构主链路：

```text
Qwen3ForCausalLM
  -> Qwen3Model
    -> VocabParallelEmbedding
    -> Qwen3DecoderLayer x N
      -> RMSNorm
      -> Qwen3Attention
      -> RMSNorm
      -> Qwen3MLP
    -> RMSNorm
  -> ParallelLMHead
```

在 `Qwen3Attention.forward()`：

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)
output = self.o_proj(o.flatten(1, -1))
```

真正写 KV 的地方在 `layers/attention.py`：

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

所以要特别记住：`scheduler.postprocess()` 不写 KV tensor。它只更新调度元数据。KV 张量内容是在 Attention forward 期间写入的。

### 8.6 Prefill logits 为什么只取最后位置

`ParallelLMHead.forward()` 中：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

prefill 可能一次算很多 prompt tokens，但采样下一个 token 只需要每条 sequence 最后一个 query 位置的 hidden。因此 prefill 阶段不会对所有 prompt token 都计算完整 vocab logits，只取最后位置。

[回到目录](#top)

<a id="postprocess-prefill"></a>

## 9. Postprocess：prefill 后如何进入 decode

模型返回 `token_ids` 后，进入：

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

对每个 `(seq, token_id)`：

```python
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
```

这表示：本轮被模型实际计算过的 token，它们的 KV 已经在 Attention 中写入了。

如果是 chunked prefill，并且 prompt 还没算完：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

这时不 append 采样 token，因为 prompt 还没有完整 prefill，不能开始生成 completion。

如果 prompt 已经 prefill 完：

```python
seq.append_token(token_id)
```

这个 `token_id` 是基于完整 prompt 采样出的第一个 completion token。append 后：

```text
token_ids = prompt tokens + [first completion token]
num_tokens = prompt_len + 1
num_cached_tokens = prompt_len
```

下一轮 decode 会把这个 first completion token 作为输入，并写入它自己的 KV。

[回到目录](#top)

<a id="decode-schedule"></a>

## 10. Decode 调度：每轮为什么只处理一个 token

当 `waiting` 中没有可调度 prefill 时，`Scheduler.schedule()` 进入 decode。

decode loop：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
```

对每个 running seq，decode 只调度一个 token：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
scheduled_seqs.append(seq)
```

为什么只调度一个？因为自回归生成中，第 t+1 个 token 依赖第 t 个 token 的采样结果。decode 阶段每条 sequence 每轮只能向前推进一个 token。

### 10.1 can_append：当前 last_token 是否需要新 block

在安排 decode 前：

```python
while not self.block_manager.can_append(seq):
```

`can_append()`：

```python
return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`len(seq)` 此时已经包含上一步刚采样出来、即将作为 decode 输入的 `last_token`。

如果：

```text
len(seq) % block_size == 1
```

说明这个 `last_token` 是新 block 的第一个 token，需要额外分配一个 free block。

如果不是 1，说明最后一个 block 还有空间，不需要新 block。

### 10.2 may_append：必要时扩展 block_table

如果需要新 block：

```python
if len(seq) % self.block_size == 1:
    seq.block_table.append(self._allocate_block())
```

这样 decode forward 中的 `slot_mapping` 才能找到 `last_token` 的写入位置。

### 10.3 为什么 scheduled_seqs 要放回 running

decode 调度完成后：

```python
self.running.extendleft(reversed(scheduled_seqs))
```

作用是把本轮 active sequence 放回 `running`。这样后面的 `postprocess()` 如果发现某个 seq 结束，可以执行：

```python
self.running.remove(seq)
```

这里不是为了 round-robin 公平调度，而是维护 active set。

[回到目录](#top)

<a id="decode-run"></a>

## 11. Decode 执行：如何读取历史 KV 并写入新 KV

decode 阶段也调用：

```python
model_runner.run(seqs, is_prefill=False)
```

但输入准备不同。

### 11.1 prepare_decode

对每个 seq：

```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

含义：

- `input_ids`：只包含每条 sequence 的最后一个 token。
- `positions`：这个 token 在完整 sequence 中的位置。
- `context_lens`：当前 sequence 总长度，attention 需要知道能看多长上下文。
- `slot_mapping`：这个 token 的 K/V 要写入的物理 slot。

随后：

```python
block_tables = self.prepare_block_tables(seqs)
set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
```

decode 一定需要 `block_tables`，因为它只输入一个 query token，但要读取完整历史 KV。

### 11.2 Attention decode 分支

`Attention.forward()` 中：

```python
if context.is_prefill:
    ...
else:
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True,
    )
```

decode 的计算形态：

```text
query: 当前 last_token 的 q
key/value: 历史所有 tokens 的 KV cache + 当前 token 刚写入的 KV
```

它通过 `block_tables` 找到该 sequence 的所有物理 KV blocks，通过 `context_lens` 知道有效长度。

### 11.3 decode 后处理的节奏

decode 后 `postprocess()`：

1. `hash_blocks(seq)`：如果刚好形成完整 block，就注册 prefix cache。
2. `seq.num_cached_tokens += 1`。
3. `seq.append_token(token_id)`，把新采样 token 放入 sequence。
4. 检查 EOS 或 `max_tokens`。

节奏可以写成：

```text
第 t 轮 decode:
  输入 token_t
  写入 token_t 的 KV
  采样 token_{t+1}
  append token_{t+1}

第 t+1 轮 decode:
  输入 token_{t+1}
  写入 token_{t+1} 的 KV
  采样 token_{t+2}
```

[回到目录](#top)

<a id="prefix-cache"></a>

## 12. Prefix Cache：哪些 KV 可以复用

prefix cache 的价值在于：如果一个新请求的前缀 token 已经在 KV cache 里出现过，就不需要重新计算这些前缀 token 的 K/V。nano-vllm 的实现很小，但边界非常清楚：它只复用从序列开头开始连续命中的完整 blocks。

相关状态都在 `BlockManager`：

```python
block.hash
block.token_ids
self.hash_to_block_id
self.free_block_ids
self.used_block_ids
```

### 12.1 注册：只有完整 block 会进入 cache

prefix cache 的注册发生在 `Scheduler.postprocess()`：

```python
self.block_manager.hash_blocks(seq)
```

`hash_blocks()` 只处理本轮新完成的完整 blocks：

```python
start = seq.num_cached_tokens // self.block_size
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
if start == end:
    return
```

如果本轮只让某个 block 从 6 个 token 推进到 7 个 token，而 `block_size = 4`，这个 block 仍然没有新形成完整边界，`start == end`，不会注册 hash。只有跨过完整 block 边界时，才会执行：

```python
block.update(h, token_ids)
self.hash_to_block_id[h] = block.block_id
```

这解释了一个细节：`append_token()` 把刚采样出的 token 放进 `Sequence.token_ids`，但这个 token 的 KV 要到下一轮 decode 才会写入。因此不能因为逻辑 token 已经在 sequence 里，就立刻认为对应 block 可以进入 prefix cache。必须等该 token 真的作为 input 跑过 Attention，并让 `num_cached_tokens` 前进之后，才能注册完整 block。

### 12.2 命中：链式 hash 保证是同一个完整前缀

如果只 hash 当前 block tokens，那么下面两个上下文的第二个 block 可能相同：

```text
Seq A: [A, B, C, D] [X, Y, Z, W]
Seq B: [E, F, G, H] [X, Y, Z, W]
```

第二个 block token 都是 `[X, Y, Z, W]`，但前文不同，KV 不能复用。

所以 `compute_hash()` 支持传入前一个 block 的 hash：

```python
if prefix != -1:
    h.update(prefix.to_bytes(8, "little"))
h.update(np.array(token_ids).tobytes())
```

`hash_blocks()` 和 `can_allocate()` 都按同样顺序滚动：

```python
h = self.compute_hash(token_ids, h)
```

这等价于：

```text
h0 = hash(block0_tokens)
h1 = hash(h0, block1_tokens)
h2 = hash(h1, block2_tokens)
```

因此同一个 block token 只有在它前面的完整上下文也一致时才会命中。

### 12.3 二次校验：源码只校验当前 block token_ids

`can_allocate()` 命中 hash 后还会比较当前 block 的 token ids：

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

这一步能挡住常见的“hash 指向了一个当前 block tokens 不同的 block”的错误复用。更严格地说，它没有把历史所有 token 再完整比较一遍；完整前缀一致主要由链式 hash 保证，`token_ids` 比较是当前 block 层面的防线。xxh64 碰撞概率很低，但这里不要把它理解成完整 prefix 的逐 token 校验。

### 12.4 为什么候选请求的最后一个 block 不复用

`can_allocate()` 遍历的是：

```python
for i in range(seq.num_blocks - 1):
```

这意味着它始终不复用候选 sequence 的最后一个逻辑 block。这个最后 block 可能未满，也可能刚好是满的。原理上，prefix cache 复用的是历史 KV；但是 prefill 还需要至少计算一段 query hidden，才能在 `ParallelLMHead` 中取最后 query 的 logits 并采样第一个 completion token。

例如 `block_size = 4`，已有请求注册了两个完整 blocks：

```text
cached: [A, B, C, D] [E, F, G, H]
```

如果新请求正好也是：

```text
[A, B, C, D] [E, F, G, H]
```

它有 2 个逻辑 blocks，`range(seq.num_blocks - 1)` 只会尝试复用第 0 个 block，最后一个 `[E, F, G, H]` 会重新 prefill。这样 `input_ids` 不会变成空，模型仍能产生最后位置的 logits。

如果新请求更长：

```text
[A, B, C, D] [E, F, G, H] [I]
```

它有 3 个逻辑 blocks，前两个 blocks 都不再是候选的最后 block，因此都可以参与 prefix cache 命中。

### 12.5 free block 为什么还能用于 prefix cache

`deallocate(seq)` 会把 `ref_count == 0` 的 block 从 used 移到 free：

```python
self.used_block_ids.remove(block_id)
self.free_block_ids.append(block_id)
```

但它不会清空 `block.hash` 和 `block.token_ids`。这些元数据会保留，直到该 block 被 `_allocate_block()` 重新分配覆盖。

因此：一个已经结束的请求释放了 block，不代表它的 prefix cache 信息立刻消失。只要物理 block 内容还没被复写，后续请求仍可能复用。

这里还有一个资源计数细节。`can_allocate()` 初始化：

```python
num_new_blocks = seq.num_blocks
```

命中 cached block 后，只有当这个 block 正在 `used_block_ids` 中，才会：

```python
num_new_blocks -= 1
```

如果命中的 block 在 free 队列里，它虽然不需要重新计算 KV，但重新启用它会把这个 block 从 `free_block_ids` 移到 `used_block_ids`，仍然消耗一个 free slot。因此源码不会减少 `num_new_blocks`。这不是 bug，而是在用 free block 数量同时表达“可新分配”和“可重新启用”的物理容量。

### 12.6 同 step 内的新 prefix 不会立刻去重

`hash_blocks()` 发生在模型 forward 之后的 `postprocess()`。所以如果两个新请求在同一个 scheduler 决策里刚好有相同前缀，第二个请求不能复用第一个请求本轮才会写出来的 KV，因为第一个请求的 block hash 此时还没有注册。

它可以复用的是：

- 更早 step 已经完成并注册的完整 blocks。
- 当前仍被其他 running sequence 引用的完整 blocks。
- 已经释放到 free 队列、但还没有被 `_allocate_block()` 覆盖的完整 blocks。

CPU 小实验对应 `examples/prefix_cache_cpu_sim.py`。它模拟了三件事：

```text
1. S1 刚 allocate 但还没 postprocess 时，S2 相同 prefix 不能命中。
2. S1 hash_blocks 后，S2 可以复用 S1 的第一个完整 block，并增加 ref_count。
3. 候选请求的最后一个 block 不复用：同长请求只复用前 n-1 个 block，更长请求可以复用更多完整前缀。
```

[回到目录](#top)

<a id="preemption"></a>

## 13. Preemption：KV block 不够时怎么办

decode 中如果当前 seq 需要新 block，但没有 free block：

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
```

如果还有其他 running sequence，就抢占队尾 sequence。

`preempt(seq)` 做：

```python
seq.status = SequenceStatus.WAITING
seq.is_prefill = True
self.block_manager.deallocate(seq)
self.waiting.appendleft(seq)
```

被抢占后：

- 它不再保留当前 `block_table`。
- 它回到 waiting 队首。
- 后续重新走 prefill。
- 如果之前的完整 blocks 已经注册 prefix cache，重新 prefill 时可能复用这些 blocks。

这就是一种简单的“释放 KV cache 换取其他请求继续运行”的策略。生产 vLLM 的抢占和调度策略会复杂得多，但核心问题一样：KV cache 是稀缺资源，调度器必须在吞吐、延迟和显存之间取舍。

[回到目录](#top)

<a id="finish"></a>

## 14. 完成输出：什么时候释放 KV cache

`postprocess()` 中检查结束条件：

```python
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

结束条件有两个：

1. 采样 token 是 EOS，并且没有设置 `ignore_eos`。
2. completion token 数达到 `max_tokens`。

完成后：

- KV blocks 被释放。
- sequence 从 `running` 移除。
- `LLMEngine.step()` 收集：

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

`generate()` 最后按 `seq_id` 排序，保证输出顺序和请求顺序一致：

```python
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

[回到目录](#top)

<a id="timeline"></a>

## 15. 一条请求的端到端时间线

下面用一条请求把前面的模块串起来。

### 15.1 请求刚进入

```text
prompt -> tokenizer.encode -> token_ids
token_ids -> Sequence
Sequence.status = WAITING
Sequence.block_table = []
Scheduler.waiting.append(seq)
```

### 15.2 第一次 step，进入 prefill

```text
Scheduler.schedule
  -> 发现 waiting 非空
  -> BlockManager.can_allocate(seq)
  -> BlockManager.allocate(seq)
  -> seq.num_scheduled_tokens = prompt_len 或 chunk size
  -> 如果 prompt 本轮能算完，seq 进入 running
```

### 15.3 prefill forward

```text
ModelRunner.run(prefill)
  -> prepare_prefill
    -> input_ids = prompt 中本轮未缓存 tokens
    -> positions = token 绝对位置
    -> slot_mapping = 新 KV 写入位置
    -> block_tables = prefix cache 命中时的历史 KV block 表
  -> set_context(is_prefill=True)
  -> Qwen3 forward
    -> Attention.store_kvcache 写 prompt KV
    -> flash_attn_varlen_func 做 prefill attention
  -> ParallelLMHead 取最后 query logits
  -> Sampler 采样第一个 completion token
```

### 15.4 prefill postprocess

```text
Scheduler.postprocess(prefill)
  -> hash_blocks 注册完整 prompt blocks
  -> num_cached_tokens += num_scheduled_tokens
  -> 如果 prompt 还没完，继续留在 waiting
  -> 如果 prompt 已完成，append 第一个 completion token
```

### 15.5 后续 decode

```text
Scheduler.schedule
  -> waiting 没有可调度 prefill
  -> 从 running 取 seq
  -> can_append 检查 last_token 是否需要新 block
  -> may_append 必要时分配新 block

ModelRunner.run(decode)
  -> prepare_decode，只输入 last_token
  -> Attention.store_kvcache 写 last_token KV
  -> flash_attn_with_kvcache 读取完整历史 KV
  -> Sampler 采样下一个 token

Scheduler.postprocess(decode)
  -> num_cached_tokens += 1
  -> append 新 token
  -> 检查 EOS / max_tokens
```

### 15.6 完成

```text
EOS or max_tokens
  -> status = FINISHED
  -> BlockManager.deallocate
  -> running.remove(seq)
  -> generate 收集 completion_token_ids
  -> tokenizer.decode
```

[回到目录](#top)

<a id="tensor-parallel"></a>

## 16. Tensor Parallel：模型内部如何分片执行

Tensor Parallel 不改变 scheduler 的请求生命周期，但它决定 `ModelRunner.run_model()` 内部每个 rank 到底保存哪部分权重、计算哪部分激活、在哪里通信。nano-vllm 的 TP 可以概括成一句话：

```text
qkv/gate_up 用列并行产生局部分片，
attention/activation 在本 rank 本地消费这些分片，
o_proj/down_proj 用行并行 all-reduce 回完整 hidden，
embedding/head 按 vocab 维度切分。
```

相关源码集中在：

- `nano-vllm/nanovllm/layers/linear.py`
- `nano-vllm/nanovllm/layers/embed_head.py`
- `nano-vllm/nanovllm/models/qwen3.py`

CPU 数值实验对应 `examples/tensor_cpu_sim.py`，用 Python list 模拟多个 rank，验证局部分片经过 concat 或 sum 后与 full linear 等价。

### 16.1 ColumnParallelLinear：切输出维度

普通线性层：

```text
x: [batch, input_size]
W: [output_size, input_size]
y = x @ W.T
```

Column parallel 切 `W` 的第 0 维，也就是输出维度：

```text
rank0: W0 = W[0 : output_size/TP, :]
rank1: W1 = W[output_size/TP : 2*output_size/TP, :]
y0 = x @ W0.T
y1 = x @ W1.T
```

源码里的 `ColumnParallelLinear.forward()` 只返回本 rank 的局部输出：

```python
return F.linear(x, self.weight, self.bias)
```

它不做 gather，因为后续 attention heads 或 MLP intermediate 可以继续以分片形式计算。CPU 仿真里额外 `torch.cat(local_outputs, dim=-1)`，只是为了验证拼起来等于完整 `F.linear`。

### 16.2 RowParallelLinear：切输入维度并 all-reduce

Row parallel 切 `W` 的第 1 维，也就是输入维度：

```text
W0: [output_size, input_size/TP]
W1: [output_size, input_size/TP]
x0: [batch, input_size/TP]
x1: [batch, input_size/TP]
p0 = x0 @ W0.T
p1 = x1 @ W1.T
y = p0 + p1
```

源码中：

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)
return y
```

bias 只在 rank0 加一次，否则 all-reduce 后 bias 会被加 `TP` 次。这里通信是求和，所以对应 CPU 仿真里的 `sum(partial_outputs)`。

### 16.3 QKVParallelLinear：Q heads 和 KV heads 分开切

`QKVParallelLinear` 是特殊的 column parallel。Q heads 和 KV heads 可能不同，例如 GQA：

```text
total_num_heads = 32
total_num_kv_heads = 8
TP = 4
```

每个 rank 得到：

```text
Q heads: 8
K heads: 2
V heads: 2
```

因此 loader 不能只把一个大 qkv 权重平均切开，而要根据 `loaded_shard_id in ["q", "k", "v"]` 分别把 checkpoint 的 `q_proj/k_proj/v_proj` 放进合并后的本地 `qkv_proj` 区域。`Qwen3Attention.forward()` 随后按本 rank 的尺寸切开：

```python
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

这个设计解释了为什么 KV cache 里每个 rank 只保存 `num_key_value_heads // world_size` 个 KV heads：attention 的计算和 KV cache 的布局都已经按 TP rank 分片。

### 16.4 MLP：gate/up 列并行，down 行并行

Qwen3 MLP 中：

```python
self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2)
self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
```

`gate_proj` 和 `up_proj` 输入相同，可以合并成一次大投影：

```text
gate_up = x @ concat(W_gate, W_up).T
gate, up = split(gate_up)
out = silu(gate) * up
```

在 TP 下，每个 rank 只拿 gate/up 的 intermediate 分片，`SiluAndMul` 可以本地完成；最后 `down_proj` 把各 rank 的 partial hidden 通过 all-reduce 合并回完整 hidden。

### 16.5 VocabParallelEmbedding 与 ParallelLMHead

Embedding 和 LM head 都按 vocab 维度切。

Embedding 中，每个 rank 只负责一段 token id：

```python
mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
y = F.embedding(local_ids, self.weight)
y = mask.unsqueeze(1) * y
dist.all_reduce(y)
```

不属于当前 rank 的 token 输出为 0，all-reduce 后每个 token 的 embedding 来自唯一负责它的 rank。

LM head 则不能 all-reduce，因为每个 rank 算的是不同 vocab 范围的 logits：

```python
logits = F.linear(x, self.weight)
dist.gather(logits, all_logits, 0)
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

所以 RowParallelLinear 用 all-reduce，是因为各 rank 贡献同一个输出向量的部分和；ParallelLMHead 用 gather，是因为各 rank 贡献不同 vocab 区间的 logits。

### 16.6 CPU 仿真如何对应源码

运行：

```bash
python examples/tensor_cpu_sim.py
```

脚本检查四组等价关系：

```text
ColumnParallelLinear: cat(local outputs) == full linear
MergedColumnParallelLinear + RowParallelLinear: reduced MLP == full MLP
QKVParallelLinear: local q/k/v shards == full q/k/v shards
VocabParallelEmbedding / ParallelLMHead: reduced or gathered result == full op
```

这个实验不模拟 NCCL，也不需要 GPU；它只验证切分维度、局部计算和通信语义是否与源码一致。

[回到目录](#top)

<a id="sampling"></a>

## 17. 采样策略：temperature 与 exponential race

采样发生在 `ModelRunner.run()` 的最后：

```python
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
logits = self.run_model(input_ids, positions, is_prefill)
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

nano-vllm 只实现了 temperature sampling，没有实现 top-k、top-p、repetition penalty、beam search 等策略。`SamplingParams` 中也只有：

```python
temperature: float = 1.0
max_tokens: int = 64
ignore_eos: bool = False
```

并且：

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

也就是说，这份实现刻意不走 greedy path；每次采样都通过概率分布产生 token。

### 17.1 temperature 如何改变分布

源码中先做：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
```

公式是：

```text
p_i = exp(logit_i / T) / sum_j exp(logit_j / T)
```

`T` 越小，最大的 logit 会被进一步放大，分布更尖；`T` 越大，不同 token 之间的差距被压平，分布更平。举一个固定 logits：

```text
logits = [0, 1, 2, 4]
```

大致会有：

```text
T = 0.5: 最大 token 概率非常高
T = 1.0: 保留原始 logit 差距
T = 2.0: 低分 token 获得更多概率
```

更准确地说，temperature 改变的是 logits 差距进入 softmax 前的尺度；“温度越高越随机”只是这个尺度变化的外在表现。

### 17.2 exponential race 为什么等价于按 probs 抽样

`Sampler.forward()` 没有直接调用 `torch.multinomial`，而是：

```python
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

也就是对每个 token 采样一个独立随机变量：

```text
E_i ~ Exp(1)
score_i = p_i / E_i
token = argmax_i score_i
```

这个过程可以理解为 exponential race。等价形式是：

```text
argmax_i p_i / E_i
= argmax_i log(p_i) - log(E_i)
```

而 `-log(E_i)` 服从 Gumbel 分布，所以它也等价于 Gumbel-max trick：

```text
token = argmax_i (log p_i + G_i)
G_i ~ Gumbel(0, 1)
```

这个技巧的结果是：第 `i` 个 token 被选中的概率正好是 `p_i`。所以源码虽然没有写 `multinomial`，但语义仍然是从 softmax 分布中采样。

### 17.3 为什么只在 rank0 采样

TP 场景下，前面的模型计算可能分布在多个 rank 上，但采样只在 rank0 执行：

```python
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
...
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

这是因为 `ParallelLMHead` 会把 vocab-parallel logits gather 到 rank0。采样需要看到完整 vocab 分布；如果每个 rank 只拿自己的一段 vocab logits 独立采样，就无法得到全局 softmax 的正确结果。

### 17.4 数值小实验

CPU 小实验对应 `examples/sampling_cpu_sim.py`。它做两件事：

```text
1. 对同一组 logits 分别用 T=0.5、1.0、2.0 采样，观察经验分布如何靠近 softmax 目标分布。
2. 比较 exponential race 和 torch.multinomial 的经验分布，验证两者都在采样同一个 categorical 分布。
```

运行：

```bash
python examples/sampling_cpu_sim.py
```

如果后续要研究 top-k/top-p，应把它们作为生产推理系统的扩展采样策略来讲；在 nano-vllm 当前源码里，它们不是 `Sampler` 的行为。

[回到目录](#top)

<a id="nanovllm-vs-vllm"></a>

## 18. nano-vllm 与生产 vLLM 的关系

nano-vllm 适合用来学习 vLLM 的核心概念，但不能等同于生产 vLLM。

相同主线：

- 都区分 prefill 和 decode。
- 都围绕 continuous batching 做请求级动态调度。
- 都用 paged KV cache / block table 思想管理 KV。
- 都需要 attention kernel 根据 block table 访问历史 KV。
- 都需要在有限 KV cache 下处理抢占、释放和复用。

nano-vllm 的简化：

- scheduler 策略更简单，prefill 优先，公平性和复杂优先级较少。
- chunked prefill 只实现了简化策略。
- KV cache manager 没有生产 vLLM 那么多策略和后端。
- 模型、采样参数、量化、LoRA、speculative decoding、服务接口等能力都更少。
- 错误处理、可观测性、性能优化、分布式部署都不是重点。

可以这样定位 nano-vllm 与生产 vLLM 的关系：

```text
nano-vllm 是帮助理解 vLLM 核心机制的最小化实现。
它可以帮助讲清楚请求生命周期、continuous batching、prefill/decode、
paged KV cache、prefix cache、抢占和模型执行链路。
生产 vLLM 在这些核心思想上扩展了更复杂的 scheduler、cache manager、
kernel backend、模型生态和服务化能力。
```

[回到目录](#top)

<a id="check"></a>

## 19. 理解检查与表达线索

如果要完整讲清楚 vLLM 风格推理流程，可以按这个顺序：

1. 请求进入后会被包装成 `Sequence`，放入 scheduler 的 `waiting` 队列。
2. scheduler 每轮先尝试 prefill，根据 token budget 和 KV block 资源决定调度哪些请求。
3. prefill 前通过 BlockManager 分配 paged KV blocks，建立 sequence 的 `block_table`。
4. ModelRunner 把 sequence 转成 varlen attention 输入，设置 context。
5. Attention 根据 `slot_mapping` 把新 token 的 K/V 写入物理 KV cache。
6. prefill 用最后一个 query 的 logits 采样第一个 completion token。
7. postprocess 更新 `num_cached_tokens`、注册 prefix cache，并把采样 token append 到 sequence。
8. 后续 decode 每轮每个 sequence 只输入上一个采样 token，通过 `block_tables` 读取完整历史 KV。
9. 如果 KV block 不够，scheduler 会 preempt 一些 running 请求，把它们释放后放回 waiting。
10. 请求遇到 EOS 或达到 max_tokens 后释放 KV blocks，输出 completion。

如果被追问几个重点：

- **Continuous Batching**：每个 step 都重新从 waiting/running 选择请求，而不是固定 batch 跑到底。
- **Chunked Prefill**：长 prompt 可以切成多轮 prefill，避免一次占满 token budget，但切分策略会影响吞吐和延迟。
- **Paged KV Cache**：sequence 的逻辑 block 通过 `block_table` 映射到物理 KV block，避免为每条请求申请连续大块显存。
- **Prefix Cache**：完整 block 通过链式 hash 注册，后续相同前缀可以复用已有 KV。
- **Sampling**：nano-vllm 用 temperature softmax 得到概率分布，再用 exponential race 等价地从该分布中抽样。
- **Decode**：每轮只推进一个 token，但通过 KV cache 看完整历史上下文。

[回到目录](#top)
