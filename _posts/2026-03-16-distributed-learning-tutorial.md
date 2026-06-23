---
layout: post
title: 分布式训练教程：从通信原语到大模型并行训练
date: 2026-03-16 00:00:00
description: 从 collective 通信、DDP/FSDP/ZeRO 到 TP/PP/SP/EP 的大模型训练并行笔记。
tags: distributed-training large-language-models ai-infra deep-learning-systems
categories: ai-infra
---

## 开篇总结

大模型训练的核心张力是：显存希望每个 rank 少存一点，吞吐希望每个 rank 多算一点，数学等价性又要求该同步的数据必须同步。分布式训练的难点正是在这三者之间做切分和通信的交换。

- 通信原语不是背景知识，而是并行策略的“接口”：DDP 的梯度同步落在 All-Reduce，FSDP/ZeRO-3 的参数重建落在 All-Gather，梯度分片落在 Reduce-Scatter，MoE token 路由落在 All-to-All。
- 并行策略的区别不在名字，而在切分对象：DP 切 batch，TP 切单层矩阵，PP 切层，SP/CP 切序列或上下文，EP 切 expert，ZeRO/FSDP 切模型状态。
- 显存估算必须逐项拆开：parameter、gradient、optimizer state、activation、communication buffer、临时 full parameter 的峰值不能混在一句“省 N 倍”里。
- 3D/4D/5D 并行不是固定标准术语；更可追问的表达是说明每个 process group 负责哪种切分，以及 forward/backward/optimizer step 中触发什么通信。
- 没有多 GPU 环境时，CPU-only 仿真仍然有价值：它不能替代真实性能 profiling，但能把“每个 rank 通信前后持有什么”固定成可打印的 tensor。

主线可以压缩成一条链：

```text
单卡训练为什么不够
-> 哪些 tensor 或状态可以被切分
-> 切分后需要哪些 collective 通信
-> 通信量和显存如何变化
-> 多种并行策略如何组合成 3D/4D/5D 训练
-> PyTorch FSDP、DeepSpeed ZeRO、Megatron-LM 等框架分别落在哪些抽象层
```

相比普通通识介绍，本文更关注三个抓手：

- 数据流：一次 step 中 batch、activation、parameter、gradient、optimizer state 到底在哪里产生、在哪里通信、在哪里释放。
- 通信成本：All-Gather、Reduce-Scatter、All-Reduce 不只看输入输出，还要看每个 rank 发送多少字节、需要多少通信轮次。
- 组合视角：DP/DDP、TP、PP、SP/CP、EP、ZeRO/FSDP 不是互斥选项，而是分别切不同维度，最终组成大模型训练中的多维并行。

配套代码：

- `examples/fsdp_zero3_sim.py`：CPU-only FSDP / ZeRO-3 数值实验，用两个虚拟 rank 展示参数分片、all-gather、reduce-scatter 和 shard optimizer update。
- `examples/collectives_cost_sim.py`：CPU 上模拟常见 collective 的输入输出语义，并用简单 `alpha-beta` 模型估算通信开销。
- `examples/collective_cost_model.py`：共享的 `alpha-beta` 与 ring collective 成本模型，供 collective 语义脚本和策略估算脚本复用。
- `examples/memory_comm_estimator.py`：把 DDP、ZeRO-1/2/3、FSDP 的模型状态显存和通信量放在同一张估算表里。
- `examples/tp_linear_sim.py`：手写 column-parallel linear 和 row-parallel linear，并对照完整 Linear 验证 forward/backward 等价。

## 目录

- [1. 学习边界和心智模型](#1-学习边界和心智模型)
- [2. 从单卡训练 step 出发](#2-从单卡训练-step-出发)
- [3. 通信原语：语义、出现位置和通信量](#3-通信原语语义出现位置和通信量)
- [4. 数据并行：DP 与 DDP](#4-数据并行dp-与-ddp)
- [5. 状态分片：ZeRO 与 FSDP](#5-状态分片zero-与-fsdp)
- [6. 张量并行：把单层矩阵乘法切开](#6-张量并行把单层矩阵乘法切开)
- [7. 流水线并行：把层按 stage 切开](#7-流水线并行把层按-stage-切开)
- [8. 序列并行和上下文并行](#8-序列并行和上下文并行)
- [9. 专家并行和 MoE 通信](#9-专家并行和-moe-通信)
- [10. 从 3D 并行到 5D 并行](#10-从-3d-并行到-5d-并行)
- [11. 数值精度如何配合分布式训练](#11-数值精度如何配合分布式训练)
- [12. 主流训练框架的关系](#12-主流训练框架的关系)
- [13. 关键时间线和出处](#13-关键时间线和出处)
- [14. 如何使用本项目的代码实验](#14-如何使用本项目的代码实验)
- [15. 面试表达线索](#15-面试表达线索)

## 1. 学习边界和心智模型

本项目面向 AI infra 大模型算法岗位，不以 CUDA 算子开发或 NCCL kernel 实现为主线。更重要的是能解释清楚：

- 大模型训练的瓶颈来自哪里：显存、算力、通信、训练稳定性。
- 哪些对象可以切：数据、参数、层、张量维度、序列维度、专家、优化器状态。
- 切完以后如何恢复数学等价性：通过 All-Reduce、All-Gather、Reduce-Scatter、All-to-All 等 collective。
- 每种并行策略牺牲什么换来什么：显存、吞吐、通信量、bubble、负载均衡和实现复杂度。

可以用四层心智模型组织所有内容：

```text
资源层：多 GPU / 多机提供显存、算力和网络带宽
通信层：collective 定义跨 rank tensor 如何交换
并行层：DP、TP、PP、SP/CP、EP、ZeRO/FSDP 决定切分方式
训练层：一次 step 中 parameter、activation、gradient、optimizer state 的生命周期
```

不要把“模型并行”理解成一个单一方法。更准确的说法是：不同并行策略切的是不同对象。

| 策略      | 主要切分对象                       | 主要解决问题                       | 典型通信                                  |
| --------- | ---------------------------------- | ---------------------------------- | ----------------------------------------- |
| DP/DDP    | batch/data                         | 提升吞吐                           | gradient All-Reduce                       |
| ZeRO/FSDP | parameter/gradient/optimizer state | 降低模型状态显存                   | All-Gather、Reduce-Scatter                |
| TP        | 单层矩阵维度                       | 单层太大或计算太重                 | All-Reduce、All-Gather、Reduce-Scatter    |
| PP        | layer/stage                        | 整模型层数太多                     | activation send/recv                      |
| SP/CP     | sequence/context                   | 长上下文 activation/attention 显存 | All-Gather、Reduce-Scatter、ring exchange |
| EP        | expert/token routing               | MoE 大参数规模                     | All-to-All / All-to-AllV                  |

## 2. 从单卡训练 step 出发

单卡训练的抽象非常简单：

```python
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

分布式训练的所有复杂性，本质上都来自这几行代码里的对象被拆开了：

- `x` 被数据并行切成不同 rank 的 batch shard。
- `model` 的参数可能被 DDP 复制，也可能被 FSDP/ZeRO-3 分片。
- `activation` 可能跨 PP stage 传递，也可能按 sequence 维度分片。
- `loss.backward()` 产生的梯度可能需要 All-Reduce，也可能只保留 Reduce-Scatter 后的 shard。
- `optimizer.step()` 可能在每个 rank 更新完整参数，也可能只更新本 rank 的参数分片。

### 纯 DDP 的 step

```text
global batch
-> DistributedSampler 按 DP rank 切分 batch
-> rank i 拿到 local batch shard
-> 每个 rank 用完整模型副本 forward
-> 每个 rank 计算 local loss
-> backward 产生本地梯度
-> DDP bucket 触发梯度 All-Reduce
-> 每个 rank 用同步后的梯度执行 optimizer step
-> 所有 rank 继续保持相同参数
```

关键点：

- DDP 梯度同步通常发生在每个 training step 的 backward 过程中，而不是每个 epoch 结束。
- PyTorch DDP 会按 bucket 组织梯度，某个 bucket 中的梯度就绪后就可以启动通信，从而和剩余 backward 计算重叠。
- DDP 的数学目标是让每个 rank 都拿到等价的全局平均梯度。

### FSDP 的 step

```text
persistent:
rank i 常驻 parameter_shard_i 和 optimizer_state_shard_i

forward:
for each wrapped module:
    all-gather 当前 module 的 parameter shards
    用完整参数执行本地 batch shard 的 forward
    reshard/free 完整参数

backward:
for each wrapped module in reverse:
    all-gather 当前 module 的 parameter shards
    计算本地完整梯度
    reduce-scatter 梯度，得到 rank i 的 gradient shard
    free 完整参数和完整梯度

optimizer:
rank i 只用自己的 gradient shard 和 optimizer state shard 更新 parameter shard
```

一句话记忆：

```text
FSDP 参数需要时临时 all-gather，梯度产生后 reduce-scatter 回 shard，optimizer 始终只更新 shard。
```

## 3. 通信原语：语义、出现位置和通信量

通信原语要同时记三件事：

- 语义：输入输出 tensor 如何变化。
- 位置：在训练 step 的哪里出现。
- 成本：每个 rank 大概发送多少数据，需要多少通信轮次。

下面假设有 `N` 个 rank，完整 tensor 大小为 `D` bytes。为了做 chunk-based collective，完整 tensor 通常被切成 `N` 块，每块大小约 `D/N`。

实际通信时间可以粗略写成：

```text
time ~= alpha * num_steps + beta * bytes_per_rank
```

其中 `alpha` 表示每轮通信延迟，`beta` 表示带宽倒数。大 tensor 更受带宽影响，小 tensor 更受延迟影响。

### All-Gather

语义：每个 rank 贡献一个 shard，所有 rank 最终都拿到完整拼接结果。

```text
before:
rank0 = [a0]
rank1 = [a1]
rank2 = [a2]
rank3 = [a3]

after:
rank0 = [a0, a1, a2, a3]
rank1 = [a0, a1, a2, a3]
rank2 = [a0, a1, a2, a3]
rank3 = [a0, a1, a2, a3]
```

特点：

- 不做 sum/max 等规约，只收集并拼接。
- 常用于 FSDP/ZeRO-3 中从参数 shard 临时重建完整参数。
- Ring all-gather 中，每个 rank 发送约 `(N-1)/N * D` bytes，通信轮次约 `N-1`。
- 不应简单说“通信量与节点数无关”。更准确是：当 `N` 增大时，单 rank 发送量趋近于 `D`，但通信轮次仍随 `N` 增长。

### Reduce-Scatter

语义：先对所有 rank 的完整 tensor 做规约，再把规约结果按 chunk 分散给不同 rank。

```text
before:
rank0 = [x_00, x_01, x_02, x_03]
rank1 = [x_10, x_11, x_12, x_13]
rank2 = [x_20, x_21, x_22, x_23]
rank3 = [x_30, x_31, x_32, x_33]

after with SUM:
rank0 = [x_00 + x_10 + x_20 + x_30]
rank1 = [x_01 + x_11 + x_21 + x_31]
rank2 = [x_02 + x_12 + x_22 + x_32]
rank3 = [x_03 + x_13 + x_23 + x_33]
```

特点：

- 每个 rank 只得到规约结果的 `1/N`。
- 常用于 ZeRO-2 和 FSDP/ZeRO-3 的梯度同步。
- Ring reduce-scatter 中，每个 rank 发送约 `(N-1)/N * D` bytes，通信轮次约 `N-1`。
- 它比 All-Reduce 更适合“后续只需要梯度分片”的场景，因为没有必要把完整规约梯度再复制给每个 rank。

### All-Reduce

语义：所有 rank 的 tensor 做规约，然后每个 rank 都得到完整规约结果。

```text
All-Reduce = Reduce-Scatter + All-Gather
```

DDP 中典型用法是对梯度 bucket 做 All-Reduce，使所有 rank 拿到相同的全局平均梯度。实现上经常先 sum，再除以 world size。

Ring all-reduce 成本：

```text
bytes_per_rank ~= 2 * (N - 1) / N * D
steps ~= 2 * (N - 1)
```

它的带宽利用率通常较好，但小 tensor 或 rank 数很多时，延迟项会变明显。

### Broadcast

语义：root rank 把一份完整数据复制给其他 rank。

常见位置：

- DDP 初始化时同步参数。
- 某些全局元数据或 checkpoint state 分发。

### All-to-All

语义：每个 rank 都把不同 chunk 发给不同目标 rank，每个 rank 也从所有 rank 接收属于自己的 chunk。

常见位置：

- MoE expert parallelism 中 token dispatch 和 combine。
- 某些 sequence/context parallel 的 token 或 block 重排。

All-to-AllV 表示不同 rank 到不同目标 rank 的数据量可以不同。MoE 的 token routing 往往动态且不均匀，所以工程上经常需要处理 All-to-AllV 式的不规则通信。

## 4. 数据并行：DP 与 DDP

数据并行切的是 batch，不切模型。每个 rank 都有完整模型副本，处理不同数据子集。

### DP/DDP 的核心逻辑

```text
rank0: model(theta), batch_0 -> grad_0
rank1: model(theta), batch_1 -> grad_1
rank2: model(theta), batch_2 -> grad_2
rank3: model(theta), batch_3 -> grad_3

All-Reduce:
grad = average(grad_0, grad_1, grad_2, grad_3)

每个 rank:
theta <- optimizer(theta, grad)
```

优点：

- 最容易理解和落地。
- 对模型代码侵入小。
- 适合通过增大 global batch 提升吞吐。

限制：

- 每个 rank 都复制完整 parameters、gradients、optimizer states。
- 模型状态显存随模型规模线性增长，不会因为 DP rank 增加而下降。
- 跨 rank 同步完整梯度，通信量随模型参数量增长。

### PyTorch DDP 的实现抓手

面试中可以这样描述 DDP：

```text
DDP 初始化时同步参数。
它给参数注册 autograd hook。
backward 时梯度逐步产生，DDP Reducer 把梯度放入 bucket。
bucket ready 后启动 All-Reduce。
All-Reduce 完成后，每个 rank 的梯度一致。
optimizer 在每个 rank 本地执行相同更新。
```

关键词：

- `DistributedSampler`
- gradient bucket
- autograd hook
- overlap communication with backward
- `no_sync()` gradient accumulation

## 5. 状态分片：ZeRO 与 FSDP

普通 DDP 最大的问题是模型状态冗余。以 Adam mixed precision 训练为例，单个参数可能对应：

- BF16/FP16 parameter：2 bytes
- BF16/FP16 gradient：2 bytes
- FP32 master parameter：4 bytes
- FP32 Adam first moment `m`：4 bytes
- FP32 Adam second moment `v`：4 bytes

粗略就是 `16 bytes * 参数量`，还没算 activation、通信 buffer 和临时 workspace。

ZeRO 的核心思想是：数据并行 rank 之间不必重复保存所有模型状态。按分片对象不同，分成三个阶段。

### ZeRO-1：切 optimizer states

常驻状态：

- parameters：完整复制。
- gradients：完整复制。
- optimizer states：按 DP rank 分片。

流程：

1. 每个 rank 用完整参数 forward/backward。
2. 梯度通过 All-Reduce 得到全局平均梯度。
3. 每个 rank 只更新自己负责的 parameter shard，因为只有这部分 optimizer states。
4. 更新后的 parameter shard 再 All-Gather 成完整参数。

### ZeRO-2：继续切 gradients

常驻状态：

- parameters：完整复制。
- gradients：分片。
- optimizer states：分片。

流程：

1. forward/backward 仍使用完整参数。
2. 梯度产生后通过 Reduce-Scatter 做规约并分片。
3. 每个 rank 用自己的 gradient shard 和 optimizer state shard 更新 parameter shard。
4. parameter shard 再 All-Gather 成完整参数。

### ZeRO-3 / FSDP：继续切 parameters

常驻状态：

- parameters：分片。
- gradients：分片。
- optimizer states：分片。

流程：

1. 进入某个 module 前 All-Gather 当前 module 的参数 shard。
2. 使用临时完整参数计算 forward。
3. forward 后释放完整参数，只保留 shard。
4. backward 到该 module 时再次 All-Gather 参数。
5. 计算本地梯度后 Reduce-Scatter，得到 gradient shard。
6. optimizer 只更新本 rank 的 parameter shard。

### FSDP 和 ZeRO-3 的关系

可以说 FSDP 思想上接近 ZeRO-3，但不要把二者说成完全同一个东西。

- ZeRO 是 DeepSpeed 提出的减少数据并行冗余状态的一组方法。
- FSDP 是 PyTorch 中围绕 module wrapping、FlatParameter、reshard、prefetch、mixed precision 等机制实现的 fully sharded data parallel。
- FSDP 的工程核心是“按 wrapped module 管理参数 all-gather 和 reshard”，而不是一次性 all-gather 整个模型。

### 显存估算要逐项算

不要脱离假设说“ZeRO-3 一定降低 64 倍”。更专业的说法是：

```text
如果 world size = N，被分片的状态项理论上常驻显存近似降低到 1/N。
但实际峰值还要加上当前 FSDP unit 的完整参数、prefetch buffer、通信 buffer、activation、padding 和 allocator 碎片。
```

因此分析 FSDP 显存时，应至少拆成：

- persistent parameter shards
- persistent optimizer state shards
- gradient shards
- gathered full parameter for current module
- activation
- communication buffer

以 mixed precision Adam 为例，如果每个参数对应 BF16/FP16 parameter、BF16/FP16 gradient、FP32 master parameter、Adam FP32 `m/v`，模型状态可以粗略估成：

| 策略          | 每 rank 常驻模型状态                  | 额外峰值                                                |
| ------------- | ------------------------------------- | ------------------------------------------------------- |
| DDP           | `param + grad + master + m + v`       | 无参数 all-gather 峰值                                  |
| ZeRO-1        | `param + grad + (master + m + v) / N` | 更新参数 shard 后可能需要 all-gather                    |
| ZeRO-2        | `param + (grad + master + m + v) / N` | 更新参数 shard 后可能需要 all-gather                    |
| ZeRO-3 / FSDP | `(param + grad + master + m + v) / N` | 当前 FSDP unit 的完整参数、通信 buffer、prefetch buffer |

通信量也要和 collective 对上：

```text
DDP gradient sync      ~= All-Reduce(full_grad)
ZeRO-1 optimizer step  ~= All-Reduce(full_grad) + All-Gather(updated_param)
ZeRO-2 gradient sync   ~= Reduce-Scatter(full_grad) + All-Gather(updated_param)
FSDP/ZeRO-3 one step   ~= All-Gather(param) in forward
                         + All-Gather(param) in backward
                         + Reduce-Scatter(full_grad)
```

`examples/collective_cost_model.py` 保存这套共享 ring 公式；`examples/memory_comm_estimator.py` 用它估算策略级通信项，`examples/collectives_cost_sim.py` 用它标注单个 collective 的语义输出。两个可运行脚本配合看，可以把“公式里的 D bytes”和“rank 上实际拿到什么 tensor”对齐起来。

## 6. 张量并行：把单层矩阵乘法切开

张量并行切的是单层内部的矩阵计算。以 PyTorch 线性层为例：

```text
Y = X @ W.T + b
W shape = [out_features, in_features]
```

### Column Parallel Linear

按 output dimension 切 `W`：

```text
W = concat([W_0, W_1, ..., W_k], dim=0)
Y_i = X @ W_i.T
Y = concat([Y_0, Y_1, ..., Y_k], dim=-1)
```

特点：

- 输入 `X` 通常复制到 TP group 内所有 rank。
- 每个 rank 计算一部分 output features。
- 如果下一层可以直接消费分片输出，就可以延迟 All-Gather。

反向：

- 每个 rank 计算自己的 `dW_i`。
- 因为每个 rank 的输出都依赖同一个 `X`，`dX` 需要把各 rank 的局部贡献相加，常见通信是 All-Reduce。

### Row Parallel Linear

按 input dimension 切 `W`，同时切 `X`：

```text
X = concat([X_0, X_1, ..., X_k], dim=-1)
W = concat([W_0, W_1, ..., W_k], dim=1)
Y_i = X_i @ W_i.T
Y = sum_i(Y_i)
```

特点：

- 每个 rank 只持有一段 input features 和对应权重。
- forward 的 partial output 需要求和，常见通信是 All-Reduce 或 Reduce-Scatter 变体。
- `dX_i` 可以留在本 rank，完整 `dX` 是各分片拼接。

### Transformer 中的典型组合

Megatron-style TP 常把 MLP 写成：

```text
X
-> ColumnParallelLinear: 得到分片 hidden
-> GeLU/SwiGLU: 本地计算
-> RowParallelLinear: 得到 partial output
-> All-Reduce partial output
-> residual add
```

这个设计的直觉是：尽量让中间大 hidden tensor 保持分片，只在必要位置通信。

`examples/tp_linear_sim.py` 把这里的两个 Linear 拆法都写成了 CPU 数值实验：

- Column parallel：每个 rank 持有 output-feature shard，forward 输出需要 concat/all-gather，backward 的 `dX` 需要 sum/all-reduce。
- Row parallel：每个 rank 持有 input-feature shard，forward partial output 需要 sum/all-reduce，backward 的 `dX` 是各 input shard 的 concat。

脚本会对照完整 Linear 层打印 `Y`、`dX`、`dW`、`db` 的最大误差。误差为 0 或浮点舍入级别，说明 TP 只是重排计算和通信位置，不改变线性层的数学结果。

## 7. 流水线并行：把层按 stage 切开

流水线并行切的是层。它解决的是“整模型太深，或者完整模型无法放进单个 rank/group”的问题。

核心概念：

- stage：一组连续层。
- micro-batch：把一个 global batch 切成多个小批次，用于填满流水线。
- bubble：stage 等待输入或等待反向梯度时的空闲。

### GPipe

GPipe 的基本调度是：

```text
先跑完所有 micro-batch forward
再跑所有 micro-batch backward
```

优点是调度简单，bubble 相对容易分析。缺点是 activation 存活时间较长，因为较早 micro-batch 的 forward activation 要等到 backward 才能释放。

### 1F1B

1F1B 的基本思想是 warmup 后交替执行：

```text
one forward, one backward
```

它可以缩短 activation 存活时间，显存更友好，是实际大模型流水线训练中更常见的调度思想。

### Interleaved 1F1B

一个 rank 持有多个 virtual stage，减少 pipeline bubble，但代价是调度和通信更复杂。

PP 的通信主要是相邻 stage 间发送 activation 和 activation gradient，不是 All-Reduce。把 PP 和 TP/DP 混合时，同一 step 中会同时出现 stage send/recv、TP collective 和 DP/FSDP 梯度同步。

## 8. 序列并行和上下文并行

序列相关并行容易混淆，建议拆成两个概念。

### Megatron-style Sequence Parallelism

它通常和 TP 配套，把 LayerNorm、Dropout 等 activation 沿 sequence 维度切分。核心收益是减少 activation 显存，并把部分 TP 中的 All-Reduce 拆成：

```text
Reduce-Scatter + All-Gather
```

这样通信量可以接近等价，但 activation 不再完整复制在每个 TP rank 上。

### Context Parallelism

CP 面向长上下文 attention。它把 sequence/context 维度切给多个 rank，使每个 rank 不必保存完整长序列的 attention 中间状态。

难点是 attention 的 softmax 需要看到足够的 K/V 信息。常见实现会通过：

- all-gather K/V
- ring attention
- block-wise exchange
- reduce-scatter output

来避免单 rank 上构造完整 `seq_len x seq_len` 的注意力矩阵。

## 9. 专家并行和 MoE 通信

专家并行用于 MoE。它切的是 expert，而不是普通 dense layer 的矩阵维度。

一次 MoE layer 的数据流：

```text
hidden states
-> router 为每个 token 选择 top-k expert
-> 按目标 expert/rank 打包 token
-> All-to-All dispatch
-> local experts compute
-> All-to-All combine
-> 按原 token 顺序恢复输出
```

EP 的主要难点：

- 负载不均衡：某些 expert 收到过多 token。
- capacity factor：限制每个 expert 的最大 token 数。
- token drop 或 padding：处理超过容量和通信 shape 对齐。
- All-to-AllV：不同 expert 的 token 数不均匀，通信量动态变化。

MoE 的独特性在于：参数量可以很大，但每个 token 只激活少量 expert。因此它常常提升“总参数规模”，而不是等比例提升“每 token 计算量”。

## 10. 从 3D 并行到 5D 并行

所谓 3D/4D/5D 并行不是固定标准术语，而是一种训练配置的组织方式。关键是说清楚每一维切什么。

### 3D 并行

常见指：

```text
DP + TP + PP
```

- DP 切 batch。
- TP 切单层矩阵。
- PP 切层/stage。

3D 并行适合 dense Transformer 大模型，是理解 Megatron-LM 类训练栈的基础。

### 4D 并行

在 3D 基础上再加入一种维度，常见有两种语境：

```text
DP + TP + PP + ZeRO/FSDP
```

或：

```text
DP + TP + PP + SP/CP
```

前者强调模型状态分片，后者强调长序列 activation/attention 分片。回答时不要只说“4D”，要明确第四维到底是什么。

### 5D 并行

大模型 MoE 或长上下文训练中，常见可以组织成：

```text
DP/FSDP + TP + PP + SP/CP + EP
```

每一维的职责：

- DP/FSDP：不同数据 shard 和模型状态分片。
- TP：单层 dense 计算切分。
- PP：层间 stage 切分。
- SP/CP：序列或上下文维度切分。
- EP：MoE expert 和 token routing 切分。

面试中更专业的表达是：

```text
我不会把 3D/5D 当成死记术语，而会先说明每个 process group 对应哪种切分维度，以及这些 group 在 forward/backward/optimizer step 中分别产生哪些通信。
```

## 11. 数值精度如何配合分布式训练

数值精度不是分布式训练的附属概念。它直接影响显存、通信量、吞吐和训练稳定性。

### 常见格式

| 格式 | 大致特点                               | 常见用途                                        |
| ---- | -------------------------------------- | ----------------------------------------------- |
| FP32 | 精度和动态范围较好，成本高             | optimizer state、master weights、部分 reduction |
| TF32 | NVIDIA Ampere 后常见的 matmul 加速格式 | FP32 输入的矩阵乘加加速                         |
| FP16 | 显存低、吞吐高，但动态范围小           | mixed precision training                        |
| BF16 | 与 FP32 类似的 exponent，动态范围更好  | 大模型训练常用                                  |
| FP8  | 更低精度，常见 E4M3/E5M2               | 需要配套 scaling 和框架支持                     |

如果看到“BF8”这个说法，需要谨慎。更常见、标准的术语是 BF16 和 FP8。FP8 又常细分为 E4M3、E5M2 等格式。

### 为什么 BF16 常用于大模型

FP16 的 mantissa 和 exponent 都较小，容易 overflow/underflow，因此经常需要 loss scaling。BF16 的 mantissa 更短，但 exponent 和 FP32 接近，动态范围更大，对大模型训练更稳。

### 精度和分布式通信的关系

精度会影响：

- 参数和 activation 显存。
- 通信 tensor 的字节数，例如 BF16 梯度通信约为 FP32 的一半。
- reduction 的数值稳定性。有些框架会用低精度通信，但在 FP32 中累积或维护 optimizer state。
- FSDP/ZeRO 的状态估算。optimizer state 常用 FP32，即使 forward/backward 用 BF16。

一个更专业的表述是：

```text
mixed precision 不是简单把所有 tensor 改成 FP16/BF16。
通常 forward/backward 用低精度提升吞吐和降低显存，optimizer state 或 master weight 保留高精度保证更新稳定。
分布式场景下，通信 tensor 的 dtype 还会直接影响通信量。
```

## 12. 主流训练框架的关系

这些框架可以按抽象层理解，而不是互相替代地背名字。

### PyTorch DDP / FSDP

- DDP：数据并行，完整模型副本，gradient All-Reduce。
- FSDP：PyTorch 原生 fully sharded data parallel，参数、梯度、optimizer state 分片。

PyTorch 的 collective API 通过 c10d `ProcessGroup` 调到底层 backend。GPU 训练通常使用 NCCL，CPU 或测试场景可能使用 Gloo。

### DeepSpeed ZeRO

DeepSpeed 的 ZeRO 系列专注减少数据并行冗余状态：

- ZeRO-1：optimizer state sharding。
- ZeRO-2：optimizer state + gradient sharding。
- ZeRO-3：optimizer state + gradient + parameter sharding。

DeepSpeed 还常和 offload、activation checkpointing、pipeline parallelism 等组合使用。

### Megatron-LM

Megatron-LM 的代表性价值在于大模型并行策略组合，尤其是：

- tensor parallelism
- pipeline parallelism
- sequence parallelism
- context parallelism
- 和 data parallel / distributed optimizer 的组合

理解 Megatron-LM 时，重点不是“它有几个并行开关”，而是它如何构造不同 process group，并让每个 group 承担不同维度的通信。

### Accelerate、NeMo、Colossal-AI 等

这些工具或框架往往提供更高层的训练配置、launcher、策略封装和生态集成。面试中可以把它们放在“编排层”理解：它们可能调用 PyTorch DDP/FSDP、DeepSpeed 或 Megatron 风格并行策略，而不是每个都从零实现所有 collective。

## 13. 关键时间线和出处

这些条目用于给主教程中的判断建立出处锚点，不是完整论文综述。

| 时间                           | 工作或文档            | 和本文的关系                                                                                             |
| ------------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------- |
| 2018                           | GPipe                 | 把 micro-batch pipeline 作为训练大模型的一种系统化调度方式。                                             |
| 2019                           | Megatron-LM           | 系统展示 Transformer tensor model parallelism，并奠定 TP + PP + DP 组合讨论的基础。                      |
| 2019/2020                      | ZeRO                  | 提出按 optimizer state、gradient、parameter 逐步消除数据并行冗余，是理解 ZeRO-1/2/3 和 FSDP 的核心来源。 |
| PyTorch 官方文档               | DDP notes / FSDP docs | DDP bucket、autograd hook、FSDP wrapping、sharding、reshard 等工程语义以官方文档为准。                   |
| NVIDIA Transformer Engine 文档 | FP8 primer            | FP8 E4M3/E5M2、scaling 与低精度训练相关概念的工程参考。                                                  |

参考来源：

- PyTorch DistributedDataParallel notes: <https://docs.pytorch.org/docs/stable/notes/ddp.html>
- PyTorch FullyShardedDataParallel docs: <https://docs.pytorch.org/docs/stable/fsdp.html>
- DeepSpeed ZeRO tutorial: <https://www.deepspeed.ai/tutorials/zero/>
- ZeRO paper: <https://arxiv.org/abs/1910.02054>
- Megatron-LM paper: <https://arxiv.org/abs/1909.08053>
- GPipe paper: <https://arxiv.org/abs/1811.06965>
- NVIDIA Transformer Engine FP8 primer: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html>

## 14. 如何使用本项目的代码实验

这些实验不是为了替代真实多 GPU profiling，而是在没有多 GPU 环境时，把分布式训练中最容易抽象化的对象固定成可打印、可检查的 CPU tensor。阅读时应关注每个虚拟 rank 在通信前后持有什么，而不是把脚本当作高性能实现。

### FSDP 数值实验

运行：

```bash
python examples/fsdp_zero3_sim.py
```

这个脚本用两个虚拟 rank 模拟：

- 参数 flatten 后分片。
- forward 前 all-gather 当前 layer 的完整参数。
- forward 后释放完整参数。
- backward 再次 all-gather 参数。
- 梯度 reduce-scatter 回 shard。
- Adam 的 `m/v` optimizer state 始终保持分片。

验证点是：FSDP shard update 后重新 gather 成完整模型，应该与单进程完整模型训练一步的结果一致，差异只来自浮点误差。

### Collective 和通信开销实验

运行：

```bash
python examples/collectives_cost_sim.py
```

这个脚本不是模拟真实网络，而是做两件事：

- 用 tensor 展示 All-Gather、Reduce-Scatter、All-Reduce、All-to-All 的输入输出。
- 按 ring 或 pairwise 直觉统计每个 rank 发送 bytes 和通信 step，并用 `alpha-beta` 模型估算通信时间。

它和本文第 3 节对应：先理解 collective 语义，再把通信量和训练中的使用位置对上。

通信成本公式来自 `examples/collective_cost_model.py`。该文件只保存共享的 `alpha-beta` 和 ring collective 记账逻辑，不单独作为实验入口。

### 显存与通信量估算实验

运行：

```bash
python examples/memory_comm_estimator.py
```

这个脚本和 collective 实验共享 `examples/collective_cost_model.py` 中的 ring 成本公式，但视角从“单个 collective 怎么变换 tensor”切到“一种并行策略在一个 training step 中触发哪些 collective”。它会打印：

- DDP、ZeRO-1、ZeRO-2、ZeRO-3/FSDP 的 per-rank persistent model-state memory。
- FSDP 当前 unit all-gather 带来的额外峰值。
- DDP/ZeRO/FSDP 每 step 的主要模型状态通信量。

### Tensor Parallel Linear 数值实验

运行：

```bash
python examples/tp_linear_sim.py
```

这个脚本和本文第 6 节对应：完整 Linear 作为 reference，column-parallel 和 row-parallel 分别手写 forward/backward，再比较 `Y`、`dX`、`dW`、`db`。它适合用来检查自己是否真的理解“列并行为什么要 all-reduce dX，行并行为什么要 all-reduce output”。

## 15. 面试表达线索

回答分布式训练问题时，可以固定按这条链路组织：

1. 训练瓶颈是什么：显存、计算、通信、长上下文、MoE 负载。
2. 切分对象是什么：data、parameter、gradient、optimizer state、layer、tensor、sequence、expert。
3. 引入什么通信：All-Reduce、All-Gather、Reduce-Scatter、All-to-All、send/recv。
4. 显存和通信怎么变：哪些状态从完整复制变成 `1/N`，哪些地方增加临时 buffer 或额外通信。
5. 数据流是否仍数学等价：梯度如何聚合，参数如何保持一致或按 shard 更新。
6. 工程代价是什么：bucket、prefetch、reshard、bubble、load balance、activation checkpointing、mixed precision。

可以用一句完整的话收束：

```text
分布式训练的核心不是某个并行名词，而是为每类 tensor 选择合适的切分维度，并用相应 collective 在需要的位置恢复数学等价性，同时控制显存峰值和通信开销。
```
