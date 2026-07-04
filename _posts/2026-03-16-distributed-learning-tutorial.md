---
layout: post
title: "Distributed Training for Large Models: Collectives, FSDP/ZeRO, DeviceMesh, and Multi-Dimensional Parallelism"
date: 2026-03-16 00:00:00
description: 从一次 training step 中 parameter、activation、gradient、optimizer state 的生命周期出发，梳理 Broadcast/All-Gather/Reduce-Scatter/All-Reduce/All-to-All、DDP、ZeRO/FSDP1/FSDP2、DeviceMesh、TP/PP/SP/CP/EP、process group topology 与显存-通信权衡。
tags: distributed-training large-language-models ai-infra deep-learning-systems
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

## 开篇总结

本文重点如下：

- 分布式训练首先是对象生命周期问题。一次 step 里，batch、parameter、activation、gradient、optimizer state 分别在哪里产生、是否复制、何时通信、何时释放，决定了显存峰值、通信量和数学等价性。
- 单卡、单机多卡、多机多卡对应不同瓶颈。DDP 主要沿 batch 维度扩展吞吐，也能在固定 global batch 时降低每卡 activation；FSDP/ZeRO 切模型状态；TP/PP/SP/CP/EP 分别切单层矩阵、层、序列/上下文和 expert。多机多卡时，process group 的拓扑比“用了几 D 并行”更重要。
- 通信操作决定切分后的 tensor 如何重新对齐。All-Gather 临时重建 FSDP 参数，Reduce-Scatter 把梯度规约回 shard，All-Reduce 让 DDP 的完整模型副本得到同一份全局梯度，All-to-All 承担 MoE token 路由。
- 显存估算不能只说“省 N 倍”。参数、梯度、optimizer state、activation、communication buffer、prefetch buffer、当前 FSDP unit 的 full parameter 必须分开计算；通信 tensor 的 dtype 也会直接改变 bytes/rank。
- 组合并行要落到 process group。先说明每个 group 切什么对象，再说明 forward、backward、optimizer step 中触发哪些 collective，以及这些 collective 落在单机高速互联还是跨机网络上。

## 目录

- [1. 从一次训练 step 看可切分对象](#1-从一次训练-step-看可切分对象)
- [2. 通信操作：rank 之间如何交换 tensor](#2-通信操作rank-之间如何交换-tensor)
- [3. 数据并行：DP 与 DDP](#3-数据并行dp-与-ddp)
- [4. 状态分片：ZeRO 与 FSDP](#4-状态分片zero-与-fsdp)
  - [FSDP1：FlatParameter 版本的完整生命周期](#fsdp1flatparameter-版本的完整生命周期)
  - [FSDP2：DTensor / DeviceMesh 版本的完整生命周期](#fsdp2dtensor--devicemesh-版本的完整生命周期)
- [5. 张量并行：把单层矩阵乘法切开](#5-张量并行把单层矩阵乘法切开)
- [6. 流水线并行：把层按 stage 切开](#6-流水线并行把层按-stage-切开)
- [7. 序列并行和上下文并行](#7-序列并行和上下文并行)
- [8. 专家并行和 MoE 通信](#8-专家并行和-moe-通信)
- [9. 从单机多卡到多机多卡：process group 与拓扑](#9-从单机多卡到多机多卡process-group-与拓扑)
- [10. 从 3D 并行到 5D 并行](#10-从-3d-并行到-5d-并行)
- [11. 数值精度如何配合分布式训练](#11-数值精度如何配合分布式训练)
- [12. 主流训练框架的关系](#12-主流训练框架的关系)
- [13. 关键时间线和出处](#13-关键时间线和出处)
- [14. 如何使用本项目的代码实验](#14-如何使用本项目的代码实验)

## 1. 从一次训练 step 看可切分对象

单卡训练的一步可以写成：

```python
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

分布式训练就是把这一步里的对象放到不同 rank 上，并在需要恢复数学等价时通信。先把对象分清楚，后面的 DP、FSDP、TP、PP、SP/CP、EP 才不会混在一起：

| 对象                   | 可以怎么切                                     | 典型策略                            | 需要的通信                             |
| ---------------------- | ---------------------------------------------- | ----------------------------------- | -------------------------------------- |
| batch / data           | 按样本维度分给不同 rank                        | DP/DDP                              | gradient All-Reduce                    |
| parameter              | 复制，或按 rank 分片                           | ZeRO-3 / FSDP                       | parameter All-Gather                   |
| gradient               | 完整同步，或规约后只保留 shard                 | DDP、ZeRO-2/3、FSDP                 | All-Reduce、Reduce-Scatter             |
| optimizer state        | 完整复制，或跟 parameter shard 对齐            | ZeRO-1/2/3、FSDP                    | 通常随 optimizer step 本地更新         |
| activation             | 按 layer、sequence/context 或 micro-batch 拆开 | PP、SP/CP、activation checkpointing | send/recv、All-Gather、Reduce-Scatter  |
| tensor dimension       | 按矩阵行/列、attention head 等拆开             | TP                                  | All-Reduce、All-Gather、Reduce-Scatter |
| layer / stage          | 连续层分给不同 rank group                      | PP                                  | activation send/recv                   |
| expert / token routing | expert 分布到不同 rank，token 动态路由         | EP / MoE                            | All-to-All / All-to-AllV               |

单卡阶段只有一个执行者，所有参数、梯度、optimizer state 和 activation 都在同一张卡上。瓶颈主要来自显存容量、单卡算力和显存带宽；这里没有跨 rank 通信。

当模型或 batch 不再适合单卡时，常见扩展路径有两类：

- 增加吞吐：复制模型，让不同 GPU 处理不同 batch shard，这就是 DP/DDP。
- 降低单卡显存：切 parameter、gradient、optimizer state、activation 或 layer，这会引入 FSDP/ZeRO、TP、PP、SP/CP 等策略。

DDP 对显存的作用取决于固定什么量：

- 固定 global batch 时，DDP 把 batch 拆成更小的 local batch，每卡 input 和 activation 会下降。
- 固定 local batch 时，DDP 扩大 global batch，主要收益是吞吐提升，每卡 activation 不会因为 DDP 自动下降。

无论哪种设置，DDP 都不会切 parameters、gradients、optimizer states；如果模型状态本身放不下，需要 ZeRO/FSDP 这类状态分片策略。

后文按切分对象展开：先看哪一类 tensor 或状态放不下、算不快、通信太贵，再决定切 batch、切状态、切矩阵、切层、切序列，还是切 expert。

## 2. 通信操作：rank 之间如何交换 tensor

系统论文和框架文档常把 Broadcast、All-Gather、Reduce-Scatter 这类接口称为 collective communication primitives，中文常译为集合通信原语。为了行文自然，下面统一称为通信操作或 collective。

按 `Broadcast -> Scatter -> Gather -> Reduce -> All-Gather -> Reduce-Scatter -> All-Reduce -> All-to-All` 的顺序梳理，原因很直接：前四个是复制、分发、收集、规约；后四个是大模型训练中更常见的全员 collective 或重排 collective。

这个顺序也能避免一个误区：collective 的语义不等于某个固定实现。比如 All-Gather 可以用 naive gather+broadcast 理解，也可以用 ring 实现；All-Reduce 经常用 ring Reduce-Scatter + ring All-Gather 实现。真正要记的是“输入输出语义、训练中出现的位置、通信量级和瓶颈”。

下面假设有 `N` 个 rank，完整 tensor 大小为 `D` bytes，chunk 大小约为 `D/N`。本文的通信量默认统计发送侧 `bytes_per_rank`：

```text
time ~= alpha * logical_steps + beta * bytes_per_rank
```

`alpha` 是每轮通信延迟，`beta` 是带宽倒数。大 tensor 更受带宽影响，小 tensor 或 rank 很多时更受延迟影响。

| 通信操作       | 语义                                                     | 分布式训练中的典型位置                         |
| -------------- | -------------------------------------------------------- | ---------------------------------------------- |
| Broadcast      | root 的完整 tensor 复制给其他 rank                       | DDP 初始化参数、元数据分发                     |
| Scatter        | root 的 tensor 切成多份分给不同 rank                     | 数据/任务分发的基础操作                        |
| Gather         | 多个 rank 的 shard 收集到 root                           | checkpoint、指标或 naive All-Gather 的第一阶段 |
| Reduce         | 多个 rank 的 tensor 规约到 root                          | 指标聚合、参数服务器式规约                     |
| All-Gather     | 每个 rank 的 shard 拼成完整 tensor，并让所有 rank 都拿到 | FSDP/ZeRO-3 参数临时重建，TP 输出拼接          |
| Reduce-Scatter | 先规约，再让每个 rank 只保留一个 reduced shard           | ZeRO-1 owner update，ZeRO-2/3/FSDP 梯度分片    |
| All-Reduce     | 规约后每个 rank 都拿到完整结果                           | DDP 梯度同步                                   |
| All-to-All     | 每个 rank 给每个目标 rank 发送不同 chunk                 | MoE token dispatch/combine，某些重排型并行     |

### Broadcast

Broadcast 是一对多复制：root rank 有完整 tensor，其他 rank 接收同一份副本。

```text
before:
rank0 = W
rank1 = -
rank2 = -
rank3 = -

after:
rank0 = W
rank1 = W
rank2 = W
rank3 = W
```

DDP 初始化时常用 broadcast-like 同步，让所有模型副本从相同参数开始。它的瓶颈取决于实现：star broadcast 会让 root 成为发送热点，tree broadcast 能降低 root 压力，但仍然是从一个源扩散同一份数据。

### Scatter

Scatter 是一对多分发：root rank 把完整 tensor 切成多个 shard，每个 rank 得到不同 shard。

```text
before:
rank0 = [a0, a1, a2, a3]

after:
rank0 = [a0]
rank1 = [a1]
rank2 = [a2]
rank3 = [a3]
```

它不是大模型训练中最常被直接点名的 collective，但它是理解数据分发、parameter shard 初始分配和 Reduce-Scatter 中 “scatter” 部分的基础。

### Gather

Gather 是多对一收集：每个 rank 贡献一个 shard，root rank 得到完整拼接结果。

```text
before:
rank0 = [a0]
rank1 = [a1]
rank2 = [a2]
rank3 = [a3]

after on root:
rank0 = [a0, a1, a2, a3]
```

Gather 容易形成 root 接收热点。All-Gather 可以被 naive 地理解成 Gather + Broadcast，但这只是实现视角，不是 All-Gather 的语义定义。

### Reduce

Reduce 是多对一规约：每个 rank 有同形状 tensor，root rank 得到 sum/max/min 等规约结果。

```text
before:
rank0 = g0
rank1 = g1
rank2 = g2
rank3 = g3

after on root with SUM:
rank0 = g0 + g1 + g2 + g3
```

DDP 不使用普通 Reduce 做最终梯度同步，因为只有 root 拿到结果会导致其他 rank 无法更新自己的完整模型副本。DDP 需要的是 All-Reduce：每个 rank 都拿到规约后的完整梯度。

### All-Gather

All-Gather 是“所有 rank 都拿到完整拼接结果”：

```text
before:
rank0 = [p0]
rank1 = [p1]
rank2 = [p2]
rank3 = [p3]

after:
rank0 = [p0, p1, p2, p3]
rank1 = [p0, p1, p2, p3]
rank2 = [p0, p1, p2, p3]
rank3 = [p0, p1, p2, p3]
```

FSDP/ZeRO-3 中最典型的 All-Gather 对象是当前 FSDP unit 的 parameter shard：

```text
p_i = rank i 常驻的当前 module parameter shard
P   = concat(p_0, p_1, ..., p_(N-1))
```

forward 或 backward 进入该 module 前，每个 rank 临时 all-gather 出完整 `P` 来计算；Adam `m/v`、FP32 master weight 等 optimizer state 不参与前向 All-Gather，仍然保持 sharded。

Ring All-Gather 中，每轮每个 rank 发送一个 `D/N` chunk 给右邻居，同时从左邻居接收一个 chunk。经过 `N-1` 轮后，每个 shard 都传播到所有 rank：

```text
bytes_per_rank ~= (N - 1) / N * D
logical_steps  ~= N - 1
```

它的优势是没有 root 热点，适合大 tensor 和带宽主导通信；瓶颈是 `N-1` 个 logical steps 的延迟，以及 All-Gather 结束后每个 rank 都会临时持有完整 `D`，因此 FSDP 仍要关注 full parameter、communication buffer 和 prefetch 带来的峰值显存。

### Reduce-Scatter

Reduce-Scatter 是“先规约，再分片保留”。在 FSDP/ZeRO-3 的梯度同步里，先把对象定义清楚：

```text
B_i = rank i 上的 local batch shard
P_j = 常驻在 rank j 上的 parameter shard
G_i = rank i 用 B_i 算出来的 local full gradient
x_{j,i} = G_i 中对应 P_j 位置的 gradient chunk
```

`x` 不是数据，也不是模型权重，而是梯度的一段。这里把 parameter shard / output chunk 放在第一维，把 source rank / local batch 放在第二维。这个写法和 NCCL 文档中 ReduceScatter 的 output chunk 视角更接近：每个 output rank 拿到一个 chunk 在所有 input rank 上的 reduction 结果。

每个 rank 在内存里仍然先有自己的 local full gradient；如果把这些 local full gradient 按列摆进表里，会得到：

```text
                 source rank / local batch
                 rank0 B_0  rank1 B_1  rank2 B_2  rank3 B_3
P_0 row:         x_00      x_01      x_02      x_03
P_1 row:         x_10      x_11      x_12      x_13
P_2 row:         x_20      x_21      x_22      x_23
P_3 row:         x_30      x_31      x_32      x_33

rank0 local full gradient G_0 = [x_00, x_10, x_20, x_30]
rank1 local full gradient G_1 = [x_01, x_11, x_21, x_31]
rank2 local full gradient G_2 = [x_02, x_12, x_22, x_32]
rank3 local full gradient G_3 = [x_03, x_13, x_23, x_33]
```

一行表示不同 local batch 对同一个 parameter shard 的梯度贡献；一列表示同一个 rank 对不同 parameter shard 的梯度贡献。Reduce-Scatter 要做的是按行规约：

```text
y_0 = x_00 + x_01 + x_02 + x_03 -> rank0 更新 P_0
y_1 = x_10 + x_11 + x_12 + x_13 -> rank1 更新 P_1
y_2 = x_20 + x_21 + x_22 + x_23 -> rank2 更新 P_2
y_3 = x_30 + x_31 + x_32 + x_33 -> rank3 更新 P_3
```

这里的 `y_j` 就是 `P_j` 对应的 reduced gradient shard。以 `rank0` 为例，`y_0` 不是 rank0 自己 local batch 的梯度，也不是完整模型梯度；它是所有 rank 的 local batch 对 `P_0` 这段参数产生的梯度之和。因为 FSDP/ZeRO-3 中 `P_0` 常驻在 rank0，rank0 的 optimizer 只需要 `P_0`、`P_0` 的 optimizer state，以及 `P_0` 对应的 `y_0`，所以 Reduce-Scatter 到这里就够了。

Ring Reduce-Scatter 不设置 root。每个目标 chunk 沿 ring 访问所有 rank，把同一行的局部贡献累加起来，最后落到拥有对应 parameter shard 的 rank。消息不是“发送给某个 parameter shard”，而是“携带某个 parameter shard 对应的 gradient partial sum”，接收方再把自己对同一 shard 的局部梯度加进去。

以 `N = 4`、发送方向 `rank0 -> rank1 -> rank2 -> rank3 -> rank0` 为例，如果希望最终 `y_j` 落到拥有 `P_j` 的 rank j，那么第一轮通常不会让 rank0 发送 `x_00`。原因是 `x_00` 已经在最终 owner rank0 本地；如果它先离开 rank0，走完整个 ring 回来需要 4 跳，而 Reduce-Scatter 只有 `N - 1 = 3` 轮。

一种更合适的排程是让每个 chunk 从 owner 的下一个 rank 开始，最后一跳回到 owner：

```text
目标 P_0 / y_0:
rank1(x_01) -> rank2(+x_02) -> rank3(+x_03) -> rank0(+x_00) = y_0

目标 P_1 / y_1:
rank2(x_12) -> rank3(+x_13) -> rank0(+x_10) -> rank1(+x_11) = y_1

目标 P_2 / y_2:
rank3(x_23) -> rank0(+x_20) -> rank1(+x_21) -> rank2(+x_22) = y_2

目标 P_3 / y_3:
rank0(x_30) -> rank1(+x_31) -> rank2(+x_32) -> rank3(+x_33) = y_3
```

按轮次展开就是：

```text
step 1:
rank0 sends x_30                 -> rank1,  target P_3
rank1 sends x_01                 -> rank2,  target P_0
rank2 sends x_12                 -> rank3,  target P_1
rank3 sends x_23                 -> rank0,  target P_2

after receive/add:
rank1 has x_30 + x_31            target P_3
rank2 has x_01 + x_02            target P_0
rank3 has x_12 + x_13            target P_1
rank0 has x_23 + x_20            target P_2

step 2:
rank0 sends x_23 + x_20          -> rank1,  target P_2
rank1 sends x_30 + x_31          -> rank2,  target P_3
rank2 sends x_01 + x_02          -> rank3,  target P_0
rank3 sends x_12 + x_13          -> rank0,  target P_1

after receive/add:
rank1 has x_23 + x_20 + x_21     target P_2
rank2 has x_30 + x_31 + x_32     target P_3
rank3 has x_01 + x_02 + x_03     target P_0
rank0 has x_12 + x_13 + x_10     target P_1

step 3:
rank0 sends x_12 + x_13 + x_10   -> rank1,  target P_1
rank1 sends x_23 + x_20 + x_21   -> rank2,  target P_2
rank2 sends x_30 + x_31 + x_32   -> rank3,  target P_3
rank3 sends x_01 + x_02 + x_03   -> rank0,  target P_0

final after receive/add:
rank0 gets y_0 = x_01 + x_02 + x_03 + x_00
rank1 gets y_1 = x_12 + x_13 + x_10 + x_11
rank2 gets y_2 = x_23 + x_20 + x_21 + x_22
rank3 gets y_3 = x_30 + x_31 + x_32 + x_33
```

实际实现会把所有 chunk 流水化，每轮每个 rank 发送一个 `D/N` 大小的 partial sum：

```text
bytes_per_rank ~= (N - 1) / N * D
logical_steps  ~= N - 1
```

它和 Ring All-Gather 的发送侧通信量相同，但语义相反：All-Gather 是从 shard 扩散成 full tensor；Reduce-Scatter 是从 local full gradient 规约并收缩成 reduced gradient shard。FSDP/ZeRO-3 喜欢 Reduce-Scatter，是因为 optimizer 只需要本 rank parameter shard 对应的 gradient shard，没有必要把完整梯度再复制给所有 rank。

### All-Reduce

All-Reduce 是“规约后每个 rank 都拿到完整结果”。DDP 梯度同步是最典型场景：

```text
B_i = rank i 的 local batch shard
G_i = rank i 用完整模型副本和 B_i 算出的 local full gradient
G   = sum_i G_i / N
```

DDP 中每个 rank 都持有完整参数副本，所以每个 rank 都需要完整的同步梯度 `G` 来执行相同 optimizer step：

```text
before:
rank0 = G_0
rank1 = G_1
rank2 = G_2
rank3 = G_3

after All-Reduce with SUM:
rank0 = G_0 + G_1 + G_2 + G_3
rank1 = G_0 + G_1 + G_2 + G_3
rank2 = G_0 + G_1 + G_2 + G_3
rank3 = G_0 + G_1 + G_2 + G_3
```

Ring All-Reduce 通常可以理解成两段：

```text
All-Reduce = Reduce-Scatter + All-Gather
```

第一段 Reduce-Scatter：把每个 local full gradient `G_i` 切成 chunks `x_{j,i}`，按行规约成 `y_j`，并让 rank j 暂时持有 `y_j`。

```text
G_0 = [x_00, x_10, x_20, x_30]
G_1 = [x_01, x_11, x_21, x_31]
G_2 = [x_02, x_12, x_22, x_32]
G_3 = [x_03, x_13, x_23, x_33]

y_0 = x_00 + x_01 + x_02 + x_03
y_1 = x_10 + x_11 + x_12 + x_13
y_2 = x_20 + x_21 + x_22 + x_23
y_3 = x_30 + x_31 + x_32 + x_33

after Reduce-Scatter:
y_0 -> rank0
y_1 -> rank1
y_2 -> rank2
y_3 -> rank3
```

第二段 All-Gather：把这些 reduced gradient shard 再 all-gather 给所有 rank。逻辑上，完整规约梯度由所有 parameter shard 的 reduced gradient 组成：

```text
G_reduced = [y_0, y_1, y_2, y_3]
```

如果前面是按一个 flat buffer 切 shard，那么 All-Gather 后通常就是按 shard 顺序拼接回一个完整 gradient buffer；如果是按原始参数逐个切 shard，框架可以把这些 shard 放回对应参数的 gradient view。无论存储形式是 flat buffer 还是 parameter views，语义都是“每个 rank 都拥有所有 parameter shard 的规约后梯度”。

以 `N = 4` 为例，这一段可以理解成 `y_j` 沿 ring 继续扩散：

```text
初始:
rank0 = [y_0]
rank1 = [y_1]
rank2 = [y_2]
rank3 = [y_3]

step 1:
rank0 -> rank1: y_0
rank1 -> rank2: y_1
rank2 -> rank3: y_2
rank3 -> rank0: y_3

rank0 = [y_0, y_3]
rank1 = [y_1, y_0]
rank2 = [y_2, y_1]
rank3 = [y_3, y_2]

step 2:
rank0 -> rank1: y_3
rank1 -> rank2: y_0
rank2 -> rank3: y_1
rank3 -> rank0: y_2

rank0 = [y_0, y_3, y_2]
rank1 = [y_1, y_0, y_3]
rank2 = [y_2, y_1, y_0]
rank3 = [y_3, y_2, y_1]

step 3:
rank0 -> rank1: y_2
rank1 -> rank2: y_3
rank2 -> rank3: y_0
rank3 -> rank0: y_1
```

最后每个 rank 都拿到完整规约梯度，只需要按 shard index 重排：

```text
rank0 = [y_0, y_1, y_2, y_3]
rank1 = [y_0, y_1, y_2, y_3]
rank2 = [y_0, y_1, y_2, y_3]
rank3 = [y_0, y_1, y_2, y_3]
```

所以 Ring All-Reduce 的成本约为一轮 Ring Reduce-Scatter 加一轮 Ring All-Gather：

```text
bytes_per_rank ~= 2 * (N - 1) / N * D
logical_steps  ~= 2 * (N - 1)
```

它的好处是每个 rank 都得到完整规约结果，适合 DDP 这种完整模型副本场景；代价是通信量约为 Reduce-Scatter 的两倍。如果后续只需要 gradient shard，例如 FSDP/ZeRO-3 optimizer step，就没有必要做完整 All-Reduce。

### All-to-All

All-to-All 是全员重排：每个 rank 都把不同 chunk 发给不同目标 rank，每个 rank 也从所有 rank 接收属于自己的 chunk。

```text
rank0 sends: to0=a00, to1=a01, to2=a02, to3=a03
rank1 sends: to0=a10, to1=a11, to2=a12, to3=a13

rank0 receives: a00, a10, ...
rank1 receives: a01, a11, ...
```

MoE expert parallelism 中，token 会按 router 分配给不同 expert owner rank，这就是典型 All-to-All / All-to-AllV 场景。All-to-All 的难点不是规约，而是数据量不规则、目标 rank 不均匀和跨节点拓扑敏感；expert 负载不均衡时，慢的 rank 会拖住整步训练。

## 3. 数据并行：DP 与 DDP

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

### PyTorch DDP 的实现要点

DDP 的关键机制可以概括为：

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

## 4. 状态分片：ZeRO 与 FSDP

普通 DDP 最大的问题是模型状态冗余。以 Adam mixed precision 训练为例，单个参数可能对应：

- BF16/FP16 parameter：2 bytes
- BF16/FP16 gradient：2 bytes
- FP32 master parameter：4 bytes
- FP32 Adam first moment `m`：4 bytes
- FP32 Adam second moment `v`：4 bytes

粗略就是 `16 bytes * 参数量`，还没算 activation、通信 buffer 和临时 workspace。

ZeRO 的核心思想是：数据并行 rank 之间不必重复保存所有模型状态。按分片对象不同，分成三个阶段。

先把三个阶段的边界写清楚：

| 阶段          | 常驻 parameter | backward 中的 local gradient                          | reduce 后的 gradient                                        | optimizer state | 参数同步方式                                               |
| ------------- | -------------- | ----------------------------------------------------- | ----------------------------------------------------------- | --------------- | ---------------------------------------------------------- |
| ZeRO-1        | 完整复制       | 每个 rank 仍会产生完整 local gradient                 | 只需要 owner rank 拿到对应 reduced gradient shard，用完即可 | 分片            | owner 更新自己的参数分区后，再 All-Gather 更新后的参数分区 |
| ZeRO-2        | 完整复制       | local gradient bucket 产生后尽早 Reduce-Scatter       | reduced gradient shard 作为分片状态保留到 optimizer step    | 分片            | owner 更新自己的参数分区后，再 All-Gather 更新后的参数分区 |
| ZeRO-3 / FSDP | 常驻分片       | 当前 module 临时 All-Gather 参数后计算 local gradient | Reduce-Scatter 成 gradient shard                            | 分片            | 下一次计算需要时按 module All-Gather 参数，不常驻完整参数  |

这里的“参数分区”在 ZeRO-1/2 中指 optimizer owner 负责更新的那一段参数；它不是常驻 parameter shard。ZeRO-1/2 在 forward/backward 前后仍然让每个 rank 持有完整参数副本，更新后才需要把各 owner 更新过的参数分区同步回所有 rank。ZeRO-3/FSDP 才把参数本身也变成常驻分片。

### ZeRO-1：切 optimizer states

常驻状态：

- parameters：完整复制。
- gradients：完整复制；这里指 backward 期间仍会产生完整 local gradient，不代表 optimizer step 一定要 all-gather 出完整规约梯度。
- optimizer states：按 DP rank 分片。

流程：

1. 每个 rank 用完整参数 forward/backward。
2. 梯度可以通过 Reduce-Scatter 得到本 rank 负责参数分区对应的 reduced gradient shard。
3. 每个 rank 用自己的 optimizer state shard 更新自己负责的参数分区。
4. 更新后的参数分区再 All-Gather 成完整参数副本。

这里容易和 DDP 式 All-Reduce 混淆。All-Reduce 可以拆成 `Reduce-Scatter(gradient) + All-Gather(reduced gradient shards)`；但 ZeRO-1 不需要第二段 gradient All-Gather，因为 optimizer states 已经分片，rank j 只会更新 `P_j` 这段参数分区，只需要 `P_j` 对应的 reduced gradient shard `y_j`。ZeRO-1 后面的 All-Gather 是为了把“更新后的参数分区”拼回每个 rank 的完整参数副本，用于下一轮 forward，而不是为了恢复完整梯度。

ZeRO-1/2 的差异在 gradient 的生命周期：ZeRO-1 的 gradient shard 主要是 owner update 的通信结果，用完即可；ZeRO-2 把 reduced gradient shard 作为分片状态保留到 optimizer step，gradient storage 才真正进入显存优化范围。

### ZeRO-2：继续切 gradients

常驻状态：

- parameters：完整复制。
- gradients：分片。
- optimizer states：分片。

流程：

1. forward/backward 仍使用完整参数。
2. gradient bucket 产生后通过 Reduce-Scatter 做规约并分片，只保留本 rank 负责的 reduced gradient shard。
3. 每个 rank 用自己的 gradient shard 和 optimizer state shard 更新自己的参数分区。
4. 更新后的参数分区再 All-Gather 成完整参数副本。

ZeRO-2 的参数 All-Gather 和 ZeRO-1 一样，都是为了恢复完整参数副本；新增收益来自 gradient partition。DeepSpeed 官方对 Stage 2 的描述也是：用于更新模型权重的 reduced gradients 会被 partition，每个 process 只保留与自己 optimizer states 对应的那部分 gradients。

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

ZeRO-3 和 ZeRO-1/2 的关键差别在参数生命周期：ZeRO-1/2 的参数在计算前后都是完整复制，只在 optimizer update 的归属上切成分区；ZeRO-3 的参数常驻状态就是分片，All-Gather 只在 forward/backward 需要当前 module 参数时临时发生，计算后再次 partition / reshard。DeepSpeed 官方也把 Stage 3 定义为 16-bit model parameters partitioned，并在 forward/backward 中自动 collect 和 partition。

### FSDP 和 ZeRO-3 的关系

FSDP 思想上接近 ZeRO-3，但二者不应混为同一个实现。

- ZeRO 是 DeepSpeed 提出的减少数据并行冗余状态的一组方法。
- FSDP1 是 PyTorch 中围绕 module wrapping、FlatParameter、reshard、prefetch、mixed precision 等机制实现的 fully sharded data parallel。
- FSDP2 保留 FSDP/ZeRO-3 的 All-Gather / Reduce-Scatter 语义，但用 DTensor 表示逐参数 shard，并用 DeviceMesh 描述 rank 拓扑。
- FSDP 的工程核心是“按通信单元管理参数 all-gather 和 reshard”，而不是一次性 all-gather 整个模型。

FSDP1 和 FSDP2 的详细例子都比较长，先用一张骨干表固定生命周期：

| 阶段               | FSDP1：FlatParameter                                                             | FSDP2：DTensor / DeviceMesh                                                            |
| ------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 常驻状态           | `flat_shard_i`、`optimizer_state_shard_i`                                        | 每个原始参数的 DTensor local shard、对应 optimizer state shard                         |
| forward pre-hook   | All-Gather `flat_shard_i -> flat_full`，再从 1D buffer 创建原始参数 views        | All-Gather 当前 unit 中各参数的 DTensor shards，临时得到完整原始参数                   |
| forward compute    | rank i 用 local batch `B_i` 和 full parameter views 计算                         | rank i 用 local batch `B_i` 和 full original parameters 计算                           |
| forward post-hook  | 释放 `flat_full` 和 views，保留 `flat_shard_i`                                   | 释放 full original parameters，保留 DTensor local shards                               |
| backward pre-hook  | 再次 All-Gather `flat_shard_i -> flat_full`，重建 views                          | 再次 All-Gather 当前 unit 参数 shards                                                  |
| backward compute   | 产生与 `flat_full` 对齐的 `grad_full_i`，再按 flat shard 位置切成 `x_{j,i}`      | 产生每个原始参数的 local full gradient，再按该参数的 DTensor shard 位置切成 `x_{j,i}`  |
| backward post-hook | Reduce-Scatter 后只保留 `grad_shard_i`，释放 full parameter                      | Reduce-Scatter 后只保留每个参数自己的 gradient local shard，释放 full parameter        |
| optimizer step     | 用 `flat_shard_i`、`grad_shard_i`、`optimizer_state_shard_i` 更新本地 flat shard | 用 DTensor local shard、gradient local shard、optimizer state shard 更新本地参数 shard |

这张表只保留主干：FSDP1 的通信围绕 1D `FlatParameter`，FSDP2 的通信围绕保留原始参数语义的 DTensor。下面两个小节分别展开同一个 `unit1` 例子。

### FSDP1：FlatParameter 版本的完整生命周期

FSDP1 指传统 PyTorch FSDP 中以 wrapped module 为单位管理参数的方式：每个 FSDP unit 内部把原始参数 flatten 成一个 1D `FlatParameter`，然后按 data-parallel rank 分片。它适合作为理解 FSDP/ZeRO-3 生命周期的基线，因为 All-Gather、reshard、Reduce-Scatter 都围绕这个 1D buffer 发生。

先固定几个对象：

```text
FSDP unit = 一组被同一个 FSDP wrapper 管理的 module
flat_full = 这个 unit 内所有参数拼成的 1D 完整参数
flat_shard_i = rank i 常驻的 flat_full 分片
full views = all-gather 后从 flat_full 切回原始 weight/bias shape 的 view
grad_full_i = rank i 用本地 batch 算出的这个 unit 的完整梯度
grad_shard_i = reduce-scatter 后 rank i 保留的梯度分片
```

一个模型可以被切成多个 FSDP unit。例如：

```text
unit0 = [layer0, layer3]
unit1 = [layer1, layer2]
unit2 = [layer4, layer5]
```

这个划分通常由手动 wrapping 或 auto wrap policy 决定。FSDP 的核心不是一次 all-gather 整个模型，而是在执行到某个 unit 时，只临时 all-gather 这个 unit 的参数；执行完以后再 reshard/free，让其他 unit 继续保持 sharded 状态。

下面只看 `unit1 = [layer1, layer2]`，假设 `world_size = 2`。

#### 1D FlatParameter 是怎么创建的

假设 `unit1` 里有两个 Linear：

```text
layer1.weight: shape [2, 3], numel = 6
layer1.bias:   shape [2],    numel = 2
layer2.weight: shape [2, 2], numel = 4
layer2.bias:   shape [2],    numel = 2

unit1 total numel = 14
```

FSDP1 会记录每个原始参数在 1D buffer 中的区间，然后把它们拼成一个 `flat_full`：

```text
flat_full =
[
  layer1.weight[0:6],
  layer1.bias[6:8],
  layer2.weight[8:12],
  layer2.bias[12:14]
]

flat_full shape = [14]
```

这些区间元数据很重要。forward/backward 真正计算时，Linear 仍然需要 `[2, 3]`、`[2]`、`[2, 2]` 这样的原始 shape；FSDP 只是把存储变成 1D，计算前再从 1D buffer 创建 view：

```text
layer1.weight view = flat_full[0:6].reshape(2, 3)
layer1.bias   view = flat_full[6:8]
layer2.weight view = flat_full[8:12].reshape(2, 2)
layer2.bias   view = flat_full[12:14]
```

然后按 rank 分片：

```text
rank0 flat_shard_0 = flat_full[0:7]
rank1 flat_shard_1 = flat_full[7:14]
```

注意，shard 边界不一定和原始参数边界对齐。这个例子中 `flat_full[6:8]` 是 `layer1.bias`，但分片点在 index 7：

```text
rank0 持有 layer1.bias 的一部分
rank1 持有 layer1.bias 的另一部分
```

这说明 FSDP 的 shard 是沿 `flat_full` 这个 1D storage 切的，不是逐个 weight/bias 保持完整边界切分。原始参数 shape 依赖元数据恢复。

如果 `flat_full.numel` 不能被 `world_size` 整除，FSDP 会做 padding，让每个 rank 的 shard 长度一致。padding 只服务于通信和分片对齐，不属于真实模型参数。

#### 初始化阶段发生什么

初始化可以理解成按 unit 处理：

```text
1. materialize / initialize unit1 的原始参数
2. 按确定顺序 flatten 成 flat_full
3. 建立 flat_full slice -> 原始参数 shape 的元数据
4. 按 world_size 切出 flat_shard_i
5. 每个 rank 只保留自己的 flat_shard_i
6. 释放 full flat parameter
```

如果使用 meta tensor / deferred init，目的也是避免所有完整参数同时真实分配到 GPU 上。更准确的理解是：先用低成本方式构造 module，再在 FSDP 管理的初始化流程里 materialize 当前 unit，初始化后立刻 flatten/shard。核心目标仍然是不要长时间保存完整模型。

初始化完成后，`unit1` 在两个 rank 上的常驻状态是：

```text
rank0: flat_shard_0, optimizer_state_shard_0
rank1: flat_shard_1, optimizer_state_shard_1
```

此时没有完整的 `layer1.weight`、`layer1.bias`、`layer2.weight`、`layer2.bias` 常驻在任意单个 rank 上；只有当前 rank 的 1D shard 常驻。

#### Forward 进入 unit1 前：All-Gather 参数

forward 执行到 `unit1` 之前，每个 rank 只有自己的 shard：

```text
rank0: flat_shard_0 = flat_full[0:7]
rank1: flat_shard_1 = flat_full[7:14]
```

但 `layer1` 和 `layer2` 的矩阵乘法需要完整参数。因此 FSDP 在 unit1 的 pre-forward hook 中触发 All-Gather：

```text
All-Gather(flat_shard_0, flat_shard_1)

rank0: flat_full = concat(flat_shard_0, flat_shard_1)
rank1: flat_full = concat(flat_shard_0, flat_shard_1)
```

然后每个 rank 用同一份 `flat_full` 创建原始参数 view：

```text
rank0:
  layer1.weight view -> flat_full[0:6].reshape(2, 3)
  layer1.bias   view -> flat_full[6:8]
  layer2.weight view -> flat_full[8:12].reshape(2, 2)
  layer2.bias   view -> flat_full[12:14]

rank1:
  创建同样的 views
```

注意：两个 rank 的参数相同，但输入数据不同：

```text
rank0 用 local batch B_0 计算 unit1 forward
rank1 用 local batch B_1 计算 unit1 forward
```

这一步通信只重建 `unit1` 的参数，不重建 optimizer state。

#### Forward 离开 unit1 后：Reshard / free full params

`layer1` 和 `layer2` forward 完成后，如果当前策略是 full shard 并且 `reshard_after_forward=True`，FSDP 会释放刚 all-gather 出来的 full parameter，只保留本地 shard：

```text
forward compute finished
-> free flat_full
-> keep flat_shard_i

rank0: flat_shard_0
rank1: flat_shard_1
```

这一步常被叫做 reshard。它的意义是把峰值显存限制在“当前 unit 的完整参数 + 其他 unit 的分片参数”，而不是让所有 unit 的完整参数都同时留在 GPU 上。

如果后面马上进入 `unit2`，FSDP 会对 `unit2` 做同样流程：

```text
unit1 reshard
-> unit2 all-gather
-> unit2 forward
-> unit2 reshard
```

所以 forward 过程中通常只有当前正在计算的 FSDP unit 是 unsharded/full parameter 状态。

#### Backward 进入 unit1 前：再次 All-Gather 参数

forward 后 full parameter 已经释放。backward 回到 `unit1` 时，需要重新拿到完整参数来计算梯度，尤其是计算上游梯度和 weight gradient 都需要原始参数 shape。

因为 backward 顺序和 forward 相反，假设先处理完 `unit2`，再回到 `unit1`：

```text
before unit1 backward:
rank0: flat_shard_0
rank1: flat_shard_1

pre-backward hook:
All-Gather(flat_shard_0, flat_shard_1)

after all-gather:
rank0: flat_full + full views
rank1: flat_full + full views
```

然后每个 rank 用自己的 local activation / grad output 计算本地完整梯度：

```text
rank0: grad_full_0 = dLoss(B_0) / d flat_full
rank1: grad_full_1 = dLoss(B_1) / d flat_full
```

`grad_full_i` 的 shape 也是 1D `[14]`，和 `flat_full` 对齐：

```text
grad_full_0 = [x_00, x_10]
grad_full_1 = [x_01, x_11]
```

这里用 `x_{j,i}` 表示梯度 chunk：第一下标 `j` 是目标 parameter shard，第二下标 `i` 是产生这段梯度的 source rank / local batch：

```text
x_00 = rank0 local batch 对 flat_full[0:7] 这段参数的梯度
x_10 = rank0 local batch 对 flat_full[7:14] 这段参数的梯度
x_01 = rank1 local batch 对 flat_full[0:7] 这段参数的梯度
x_11 = rank1 local batch 对 flat_full[7:14] 这段参数的梯度
```

#### Backward 离开 unit1 后：Reduce-Scatter 梯度

local gradient 还不是全局梯度。FSDP 需要把不同 rank 的 local batch 对同一段参数的梯度加起来，并且只把对应 shard 留给 owner rank。

Reduce-Scatter 做的事是：

```text
rank0 输入 grad_full_0 = [x_00, x_10]
rank1 输入 grad_full_1 = [x_01, x_11]

Reduce over same shard position:
y_0 = x_00 + x_01
y_1 = x_10 + x_11

Scatter reduced gradient shard:
rank0 gets grad_shard_0 = y_0
rank1 gets grad_shard_1 = y_1
```

如果训练代码对梯度取平均，那么还会除以 `world_size`，但这个缩放通常由框架或 optimizer 约定处理。关键是：Reduce-Scatter 后每个 rank 只保留自己参数 shard 对应的梯度 shard。

随后 FSDP 释放 backward 时重新 all-gather 的 full parameter：

```text
free flat_full and full views
keep flat_shard_i
keep grad_shard_i
```

此时 `unit1` 的状态是：

```text
rank0:
  param: flat_shard_0 = flat_full[0:7]
  grad:  grad_shard_0 = y_0

rank1:
  param: flat_shard_1 = flat_full[7:14]
  grad:  grad_shard_1 = y_1
```

#### Optimizer step：只更新本地 shard

optimizer step 不需要完整参数。rank i 拥有：

```text
flat_shard_i
grad_shard_i
optimizer_state_shard_i
```

所以每个 rank 只更新自己的 shard：

```text
rank0:
  flat_shard_0 <- optimizer(flat_shard_0, grad_shard_0, optimizer_state_shard_0)

rank1:
  flat_shard_1 <- optimizer(flat_shard_1, grad_shard_1, optimizer_state_shard_1)
```

更新完成后，完整参数仍然没有常驻在任何 rank 上。下一次 forward 进入 unit1 时，再通过 All-Gather 临时重建。

#### unit1 的完整生命周期

把上面的流程压成一个时间线：

```text
init:
  original params
  -> flatten to flat_full
  -> shard to flat_shard_i
  -> free flat_full

forward pre-hook:
  All-Gather flat_shard_i -> flat_full on every rank
  create full parameter views

forward compute:
  rank i uses local batch B_i and full views

forward post-hook:
  reshard/free flat_full
  keep flat_shard_i

backward pre-hook:
  All-Gather flat_shard_i -> flat_full again
  recreate full parameter views

backward compute:
  rank i computes grad_full_i for this unit

backward post-hook:
  Reduce-Scatter grad_full_i -> grad_shard_i
  free flat_full
  keep flat_shard_i and grad_shard_i

optimizer step:
  update flat_shard_i with grad_shard_i and optimizer_state_shard_i
```

FSDP1 的关键点是：1D `FlatParameter` 是每个 FSDP unit 内部的真实存储抽象；All-Gather 和 Reduce-Scatter 都围绕这个 1D buffer 发生。原始参数的多维 shape 通过 view 暂时恢复出来服务计算，计算结束后再回到 sharded 1D storage。

### FSDP2：DTensor / DeviceMesh 版本的完整生命周期

FSDP2 不是把 FSDP 的通信语义换掉。它仍然是 ZeRO-3/FSDP 的流程：参数分片常驻，计算前 All-Gather 成完整参数，梯度算完后 Reduce-Scatter 回梯度分片，optimizer 只更新本 rank 的参数分片。

FSDP2 真正变化的是参数表示和调度接口：FSDP1 用 1D `FlatParameter` 表示一个 FSDP unit 的参数；FSDP2 用 `DTensor` 表示每个原始参数自己的分片，并用 `DeviceMesh` 描述这些分片放在哪些 rank 上。官方入口是 `torch.distributed.fsdp.fully_shard`：

- PyTorch FSDP2 API: <https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html>
- PyTorch FSDP2 tutorial: <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>

#### 先固定三个新对象

`DeviceMesh` 是 rank 的逻辑拓扑。最简单的 FSDP2 可以只有一个 1D mesh：

```text
DeviceMesh = [rank0, rank1]
```

这表示 rank0 和 rank1 组成一个 sharding group。参数、梯度和 optimizer state 都沿这个 group 分片。

更复杂的混合并行会用 2D 或更高维 mesh。例如 HSDP 可以把一维用来 shard，另一维用来 replicate；TP/FSDP 混合时，也可以给 TP group 和 FSDP group 各自一维。这里先只看 1D FSDP mesh。

`DTensor` 是带分布式元数据的 tensor。一个 DTensor 至少包含三层信息：

```text
global shape: 这个参数完整时的 shape
local tensor: 当前 rank 实际持有的 shard
placement:   这个 tensor 如何分布在 DeviceMesh 上
```

FSDP2 的参数通常是：

```text
DTensor(global_shape=原始参数 shape, placement=Shard(0), local_tensor=本 rank 的 dim-0 shard)
```

`Shard(0)` 表示沿参数第 0 维切分。对 Linear weight 来说，第 0 维通常是 output feature；对 bias 来说，第 0 维就是 bias entry。

如果第 0 维不能被 `world_size` 整除，`Shard(0)` 不是要求模型结构必须改成可整除，而是按类似 `torch.chunk` 的语义切成不等长 shard。DTensor 会记录每个 rank 的 local shard size 和 offset：

```text
weight shape = [3, 3], world_size = 2

rank0 local shard: rows [0:2], shape [2, 3]
rank1 local shard: rows [2:3], shape [1, 3]
```

如果第 0 维比 rank 数还小，后面的 rank 甚至可能拿到空 shard：

```text
weight shape = [2, 3], world_size = 4

rank0 local shard: row [0], shape [1, 3]
rank1 local shard: row [1], shape [1, 3]
rank2 local shard: empty, shape [0, 3]
rank3 local shard: empty, shape [0, 3]
```

语义上这仍然是合法的 sharding：All-Gather 时根据 offset 拼回完整 `[2, 3]`，Reduce-Scatter 时也只把每个 rank 负责的梯度 shard 留下来。但不均匀 sharding 会带来负载不均、通信实现中的 padding/metadata 处理和空 shard 边界问题。PyTorch DTensor 文档也把 uneven sharding 标为 experimental，所以工程上通常会让 FSDP shard 维度远大于 FSDP rank 数；大模型里的 Linear output feature 一般足够大，这个问题更多出现在教学小例子、小模块或过细 wrapping 中。

`fully_shard(module)` 是 FSDP2 的包装入口。它会把 module 里的参数转换成 DTensor，并给 module 加上 forward/backward hook。这个 module 就成为一个 FSDP 通信单元：执行到它时 unshard，执行完再 reshard。

#### 用 FSDP1 同一个 unit1 对比

沿用 FSDP1 里的例子：

```text
unit1 = [layer1, layer2]
world_size = 2

layer1.weight: shape [2, 3], numel = 6
layer1.bias:   shape [2],    numel = 2
layer2.weight: shape [2, 2], numel = 4
layer2.bias:   shape [2],    numel = 2
```

FSDP1 的常驻参数是一个 1D flat shard：

```text
flat_full shape = [14]

rank0 flat_shard_0 = flat_full[0:7]
rank1 flat_shard_1 = flat_full[7:14]
```

这个切法可能切开原始参数边界。前面例子里 `layer1.bias = flat_full[6:8]`，但分片点在 index 7，所以 rank0 和 rank1 各持有 `layer1.bias` 的一部分。

FSDP2 不创建这个 1D `FlatParameter`。它直接让每个原始参数成为自己的 DTensor：

```text
layer1.weight [2, 3]
  rank0 local shard: row [0], shape [1, 3]
  rank1 local shard: row [1], shape [1, 3]

layer1.bias [2]
  rank0 local shard: entry [0], shape [1]
  rank1 local shard: entry [1], shape [1]

layer2.weight [2, 2]
  rank0 local shard: row [0], shape [1, 2]
  rank1 local shard: row [1], shape [1, 2]

layer2.bias [2]
  rank0 local shard: entry [0], shape [1]
  rank1 local shard: entry [1], shape [1]
```

所以 FSDP2 的常驻状态可以写成：

```text
rank0:
  layer1.weight = DTensor(global=[2, 3], placement=Shard(0), local=row 0)
  layer1.bias   = DTensor(global=[2],    placement=Shard(0), local=entry 0)
  layer2.weight = DTensor(global=[2, 2], placement=Shard(0), local=row 0)
  layer2.bias   = DTensor(global=[2],    placement=Shard(0), local=entry 0)

rank1:
  layer1.weight = DTensor(global=[2, 3], placement=Shard(0), local=row 1)
  layer1.bias   = DTensor(global=[2],    placement=Shard(0), local=entry 1)
  layer2.weight = DTensor(global=[2, 2], placement=Shard(0), local=row 1)
  layer2.bias   = DTensor(global=[2],    placement=Shard(0), local=entry 1)
```

这里的核心差异是：FSDP1 的 shard 是 `flat_full` 上的 index 区间；FSDP2 的 shard 是每个原始参数在 dim-0 上的一段。通信实现仍然可以把多个参数合并调度，但参数语义不再丢失。

从这个例子看，FSDP1 先把 unit 内参数混成一个 1D storage，再按 rank 切 storage；FSDP2 保留每个参数自己的身份、完整 shape 和 local offset，再让每个参数分别按 mesh 切 shard。后面 checkpoint、debug 和混合并行的收益，都来自这个参数语义没有丢失。

#### FSDP2 的完整流程

初始化时先构造模型，再 bottom-up 调用 `fully_shard`。optimizer 要在 `fully_shard` 之后创建，因为 optimizer state 也要跟随 DTensor 参数分片：

```python
from torch.distributed.fsdp import fully_shard

model = Transformer()

for block in model.layers:
    fully_shard(block)

fully_shard(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

对 `unit1` 来说，初始化后的常驻状态不是 full parameter，而是每个参数的 DTensor local shard：

```text
rank0: layer1.weight row0, layer1.bias entry0, layer2.weight row0, layer2.bias entry0
rank1: layer1.weight row1, layer1.bias entry1, layer2.weight row1, layer2.bias entry1
```

forward 进入 `unit1` 前，Linear 计算需要完整 weight/bias。FSDP2 的 pre-forward hook 对这个 unit 的 DTensor 参数做 All-Gather：

```text
All-Gather layer1.weight:
  rank0 row0 + rank1 row1 -> full layer1.weight [2, 3]

All-Gather layer1.bias:
  rank0 entry0 + rank1 entry1 -> full layer1.bias [2]

All-Gather layer2.weight:
  rank0 row0 + rank1 row1 -> full layer2.weight [2, 2]

All-Gather layer2.bias:
  rank0 entry0 + rank1 entry1 -> full layer2.bias [2]
```

All-Gather 之后，两个 rank 都临时看到普通完整参数：

```text
rank0: full layer1/layer2 parameters + local batch B_0
rank1: full layer1/layer2 parameters + local batch B_1
```

forward 离开 `unit1` 后，如果 `reshard_after_forward=True`，FSDP2 释放这些 full parameter，把参数恢复成 DTensor shard：

```text
free full layer1.weight / layer1.bias / layer2.weight / layer2.bias
keep local DTensor shards
```

backward 回到 `unit1` 前，如果 forward 后已经 reshard，就需要再次 All-Gather 参数。原因和 FSDP1 一样：计算 input gradient 和 parameter gradient 时需要完整参数。

backward 中每个 rank 先得到一份 local full gradient。以 `layer1.weight [2, 3]` 为例：

```text
rank0: dW1_0 = rank0 local batch B_0 对完整 layer1.weight 的梯度
rank1: dW1_1 = rank1 local batch B_1 对完整 layer1.weight 的梯度

dW1_0 shape = [2, 3]
dW1_1 shape = [2, 3]
```

然后按 FSDP2 的参数 shard 方式做 Reduce-Scatter。因为 `layer1.weight` 是沿 dim-0 切，所以梯度也沿 dim-0 切：

```text
dW1_0 = [x_00, x_10]
dW1_1 = [x_01, x_11]
```

这里的 `x_{j,i}` 含义是：

```text
j = 对 layer1.weight 哪个 row shard 的梯度贡献
i = local batch 来自哪个 rank

x_00 = rank0 local batch 对 layer1.weight row0 的梯度
x_10 = rank0 local batch 对 layer1.weight row1 的梯度
x_01 = rank1 local batch 对 layer1.weight row0 的梯度
x_11 = rank1 local batch 对 layer1.weight row1 的梯度
```

Reduce-Scatter 后：

```text
rank0 gets grad shard for row0 = x_00 + x_01
rank1 gets grad shard for row1 = x_10 + x_11
```

`layer1.bias`、`layer2.weight`、`layer2.bias` 同理：每个参数的 gradient shard 和这个参数自己的 DTensor shard 对齐。若训练约定使用平均梯度，reduce 后还会除以 `world_size`。

optimizer step 不需要完整参数。rank i 只需要：

```text
parameter DTensor local shard
gradient local shard
optimizer state local shard
```

所以更新发生在本地 shard 上：

```text
rank0 更新 layer1.weight row0、layer1.bias entry0、layer2.weight row0、layer2.bias entry0
rank1 更新 layer1.weight row1、layer1.bias entry1、layer2.weight row1、layer2.bias entry1
```

下一轮 forward 再 All-Gather 最新的 DTensor shards，临时恢复完整参数。

#### FSDP2 实战代码骨架

下面的代码骨架展示 FSDP2 应用时最关键的几个顺序：先初始化分布式环境和 `DeviceMesh`，再 bottom-up 调用 `fully_shard`，最后创建 optimizer。训练 loop 仍然是普通 PyTorch 写法，参数 unshard、reshard、gradient reduce-scatter 由 FSDP2 hook 在 forward/backward 前后触发。

```python
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def init_distributed():
    # torchrun 会注入 RANK、LOCAL_RANK、WORLD_SIZE 等环境变量。
    # FSDP2 的 DeviceMesh 构建依赖已经初始化好的 process group。
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")


def build_fsdp2_model(model):
    # 1D mesh 表示这里只做 FSDP 这一维切分。
    # 如果后续组合 TP/FSDP 或 HSDP，可以把 mesh 扩展成 2D。
    mesh = init_device_mesh(
        "cuda",
        (dist.get_world_size(),),
        mesh_dim_names=("dp",),
    )

    # FSDP2 推荐 bottom-up wrapping：先 shard 子模块，再 shard root。
    # 每个 block 成为一个通信单元，执行到该 block 时临时 All-Gather 参数。
    for block in model.layers:
        fully_shard(block, mesh=mesh)

    # root wrapper 处理 embedding、lm_head 或其他未被子模块接管的参数。
    fully_shard(model, mesh=mesh)
    return model


def train_one_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)

    # forward 前：当前 FSDP unit 的 DTensor parameter shards 被 All-Gather。
    # forward 后：若 reshard_after_forward=True，full parameters 被释放。
    loss = model(**batch).loss

    # backward 前：需要时再次 All-Gather 参数。
    # backward 后：local full gradients 被 Reduce-Scatter 成 gradient shards。
    loss.backward()

    # optimizer 只看到并更新本 rank 的 DTensor local shard 及其 optimizer state shard。
    optimizer.step()
    return loss.detach()


def save_sharded_checkpoint(model, optimizer, step):
    # FSDP2 参数是 DTensor，model.state_dict() 可以保留每个参数自己的 shard 语义。
    # Distributed Checkpoint 负责把不同 rank 的 shards 写成一个分布式 checkpoint。
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    dcp.save(state, checkpoint_id=f"ckpt_step_{step}")


def main():
    init_distributed()

    model = build_model().cuda()
    model = build_fsdp2_model(model)

    # optimizer 必须在 fully_shard 之后创建。
    # 这样 AdamW 的 m/v state 才会和 DTensor parameter shard 对齐。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, batch in enumerate(dataloader()):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        loss = train_one_step(model, optimizer, batch)

        if step % 1000 == 0:
            save_sharded_checkpoint(model, optimizer, step)

    dist.destroy_process_group()
```

这段骨架里有四个容易写错的点：

1. `DeviceMesh` 不是参数本身，而是 rank 拓扑；`DTensor` 才是“带 global shape、local shard、placement 的参数”。
2. `fully_shard(block)` 的粒度决定通信单元。包得太细，collective 次数更多；包得太粗，单次 All-Gather 更大，峰值 full parameter 显存更高。
3. optimizer 要在 `fully_shard` 之后创建，否则 optimizer state 可能绑定到 sharding 前的普通参数对象，而不是最终训练的 DTensor 参数。
4. checkpoint 时优先按 sharded state dict / Distributed Checkpoint 理解：保存的是带参数名和 shard 语义的分布式状态；需要单进程 full checkpoint 时，再把 shards 汇总成完整 tensor。

#### FSDP2 相比 FSDP1 的改进点

FSDP2 的核心通信量不必然更少。相同参数量、相同 world size、相同 reshard 策略下，它仍然是：

```text
forward pre-hook:  All-Gather parameters
backward pre-hook: All-Gather parameters
backward post-hook: Reduce-Scatter gradients
```

它主要改进的是工程表示和调度能力：

| 改进点                                 | 具体含义                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| 不再依赖 1D `FlatParameter`            | 参数名、shape、shard 关系更直接，不需要从 flat index 反推原始参数        |
| 使用 `DTensor` 表示分片                | 参数自带 global shape、local shard 和 placement，checkpoint 和调试更自然 |
| 使用 `DeviceMesh` 描述 rank 拓扑       | 1D FSDP、HSDP、TP+FSDP 等组合可以用统一 mesh 表达                        |
| `fully_shard` 是 composable API        | 可以 bottom-up 包装 block/root，通信边界和模型结构更一致                 |
| 显式 `unshard/reshard` 控制            | 可以更清楚地控制 full parameter 何时出现、何时释放                       |
| 更适合和 DCP / DTensor state dict 配合 | 分布式 checkpoint 可以保留每个参数自己的 shard 语义                      |

这些改进服务于同一个目标：保留 FSDP/ZeRO-3 的通信流程，同时让 sharded state 继续携带原始参数语义。

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
| ZeRO-1        | `param + grad + (master + m + v) / N` | 更新参数分区后需要 all-gather 恢复完整参数副本          |
| ZeRO-2        | `param + (grad + master + m + v) / N` | 更新参数分区后需要 all-gather 恢复完整参数副本          |
| ZeRO-3 / FSDP | `(param + grad + master + m + v) / N` | 当前 FSDP unit 的完整参数、通信 buffer、prefetch buffer |

通信量也要和 collective 对上：

```text
DDP gradient sync      ~= All-Reduce(full_grad)
ZeRO-1 optimizer step  ~= Reduce-Scatter(full_grad) + All-Gather(updated_param_partitions)
ZeRO-2 optimizer step  ~= Reduce-Scatter(full_grad) + All-Gather(updated_param_partitions)
FSDP/ZeRO-3 one step   ~= All-Gather(param) in forward
                         + All-Gather(param) in backward
                         + Reduce-Scatter(full_grad)
```

ZeRO-1 和 ZeRO-2 在这个粗略通信式里看起来相同，区别主要在显存生命周期：ZeRO-1 切 optimizer state，ZeRO-2 继续切 gradient storage。如果把 ZeRO-1 朴素实现成 `All-Reduce(full_grad) + All-Gather(updated_param_partitions)`，通信会比 ZeRO 的标准分解更重；这个写法只适合作为“DDP 式同步梯度再做 owner update”的概念对照，不适合作为 ZeRO-1 的通信估算。

`examples/collective_cost_model.py` 保存这套共享 ring 公式；`examples/memory_comm_estimator.py` 用它估算策略级通信项，`examples/collectives_cost_sim.py` 用它标注单个 collective 的语义输出。两个可运行脚本配合看，可以把“公式里的 D bytes”和“rank 上实际拿到什么 tensor”对齐起来。

## 5. 张量并行：把单层矩阵乘法切开

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
- 如果后续计算需要完整 `Y`，就要 All-Gather / concat 所有 `Y_i`。
- 如果下一层本身按 input features 切分，就可以直接把 `Y_i` 当成下一层的 `X_i`，不必立刻 All-Gather。这就是“消费分片输出”：下一层只需要自己那段 feature shard，而不是完整 activation。

反向：

- 每个 rank 计算自己的 `dW_i`。
- 每个 rank 还会计算一个 `dX_i_partial = dY_i @ W_i`。它不是 `X` 某一段的梯度，而是完整 `X` 梯度的一部分贡献。
- 完整 `dX = sum_i dX_i_partial`，所以需要在 TP group 内做 All-Reduce / sum。`dW_i` 和 `db_i` 属于本 rank 持有的参数 shard，不需要跨 rank reduce。

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

这个设计的直觉是：尽量让中间大 hidden tensor 保持分片，只在必要位置通信。第一层 column-parallel 产生的 `Y_i` 正好是第二层 row-parallel 需要的 input-feature shard；GeLU/SwiGLU 是逐元素操作，也可以在每个 shard 上本地计算。因此中间 hidden 不需要先 All-Gather，通信被推迟到 row-parallel output 的求和位置。

用两个 rank 看这个组合：

```text
Column-parallel:
rank0: Y_0 = X @ W_col0.T
rank1: Y_1 = X @ W_col1.T

GeLU/SwiGLU:
rank0: H_0 = activation(Y_0)
rank1: H_1 = activation(Y_1)

Row-parallel:
rank0: O_0 = H_0 @ W_row0.T
rank1: O_1 = H_1 @ W_row1.T
All-Reduce/Sum: O = O_0 + O_1
```

如果 column-parallel 后面接的是一个需要完整 hidden 的普通非切分层，就不能这样延迟通信，必须先 All-Gather 得到 `Y = concat([Y_0, Y_1], dim=-1)`。

`examples/tp_linear_sim.py` 把这里的两个 Linear 拆法都写成了 CPU 数值实验：

- Column parallel：每个 rank 持有 output-feature shard，forward 输出需要 concat/all-gather，backward 的 `dX` 需要 sum/all-reduce。
- Row parallel：每个 rank 持有 input-feature shard，forward partial output 需要 sum/all-reduce，backward 的 `dX` 是各 input shard 的 concat。

脚本会对照完整 Linear 层打印 `Y`、`dX`、`dW`、`db` 的最大误差。误差为 0 或浮点舍入级别，说明 TP 只是重排计算和通信位置，不改变线性层的数学结果。

## 6. 流水线并行：把层按 stage 切开

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

## 7. 序列并行和上下文并行

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

## 8. 专家并行和 MoE 通信

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

## 9. 从单机多卡到多机多卡：process group 与拓扑

前面各章分别讲了 DDP、FSDP、TP、PP、SP/CP 和 EP。真正训练时，这些策略落在不同 process group 上，并在同一次 forward/backward 中交替触发通信。单机多卡和多机多卡的差别，主要体现在这些 group 放在哪些 GPU 上、是否跨节点、通信频率和通信量是否能被网络承受。

### 单机多卡：先把高频通信放在高速互联内

单机多卡通常先从 DDP 开始，因为它最接近单卡训练：每个 GPU 持有完整模型副本，处理不同 local batch，最后用 gradient All-Reduce 同步完整梯度。

如果完整模型状态放不下，再引入 FSDP/ZeRO-3：参数、梯度、optimizer state 常驻分片；计算当前 unit 时临时 All-Gather 参数，backward 后 Reduce-Scatter 梯度，optimizer 只更新本地 parameter shard。

如果单层矩阵乘法太大或希望提升单层计算并行度，就引入 TP；如果模型层数太多，就引入 PP。单机内通信通常走 NVLink、NVSwitch 或 PCIe，带宽和延迟比跨机网络更友好，所以 TP 这类高频通信策略更倾向放在单机高速互联内。

### 多机多卡：每个 group 都要看跨不跨节点

多机多卡不能只看总 GPU 数。跨机网络延迟更高、带宽更稀缺，策略选择通常要区分 process group：

- DP/FSDP group 常常跨节点扩展总 GPU 数，用于提升 batch 吞吐或分摊模型状态。
- TP group 更依赖高带宽低延迟，实践中经常优先限制在同节点或同高速互联域内。
- PP group 可以跨节点切 layer，但会引入 activation send/recv 和 pipeline bubble。
- CP group 的 KV/attention 交换可能很重，长上下文训练时要关注它是否跨节点。
- EP/MoE 的 All-to-All 对网络拓扑很敏感，跨节点 token dispatch 可能成为主要瓶颈。

一个训练任务可能同时有：

```text
DP/FSDP group: 负责 batch 维度和参数状态分片
TP group:      负责单层矩阵切分
PP group:      负责 layer/stage 切分
SP/CP group:   负责 sequence/context 切分
EP group:      负责 expert/token routing
```

分析拓扑时，先写出每类 group 的 rank 列表，再判断这些 rank 是否在同一台机器、同一 NVLink/NVSwitch 域、还是跨机网络上。多 D 并行的配置优劣，取决于高频、大流量的 group 有没有放在更快的互联上。

## 10. 从 3D 并行到 5D 并行

3D/4D/5D 并行通常表示一个训练配置里同时使用了几个相互独立的切分维度。第 9 节已经讨论拓扑位置，这一节重点看 rank coordinate 和 process group 如何组织。判断一个多 D 方案是否讲清楚，至少要回答三件事：

- 每一维切什么对象：batch、parameter state、tensor dimension、layer、sequence/context、expert。
- 每一维对应哪些 process group：同一个 rank 会同时属于 TP group、FSDP group、PP group 等不同 group。
- 一次 forward/backward 中，每个 group 在什么时候通信，通信对象是什么。

进入多维并行通常是因为单一策略只解决一个瓶颈：DDP 扩 batch，FSDP/ZeRO 切模型状态，TP 切单层矩阵，PP 切层，SP/CP 切长序列 activation 或 attention，EP 切 expert 和 token 路由。组合时重点不是维度数量本身，而是这些维度对应的 group 是否放在合适的互联拓扑上。

### 3D 并行

3D 并行常见指：

```text
DP + TP + PP
```

- DP 切 batch。
- TP 切单层矩阵。
- PP 切层/stage。

3D 并行适合 dense Transformer 大模型，是理解 Megatron-LM 类训练栈的基础。

在大模型训练中，DP 维度经常不是完整参数复制的 DDP，而是 FSDP/ZeRO-3 这样的状态分片。此时可以把 3D 写成：

```text
FSDP + TP + PP
```

这三个维度分别解决不同问题：

| 维度 | 切分对象                               | 典型通信                                      | 解决的问题                       |
| ---- | -------------------------------------- | --------------------------------------------- | -------------------------------- |
| FSDP | parameter / gradient / optimizer state | parameter All-Gather、gradient Reduce-Scatter | 降低模型状态显存                 |
| TP   | 单层矩阵或 attention head              | All-Reduce、All-Gather、Reduce-Scatter        | 单层太大或单层计算太重           |
| PP   | 连续层组成的 stage                     | activation send/recv                          | 整模型层数太多，单卡放不下所有层 |

### 一个 3D 并行例子：16 GPUs = PP 2 _ FSDP 4 _ TP 2

假设有 16 张 GPU，组织成：

```text
PP = 2
FSDP = 4
TP = 2

rank coordinate = (pp_idx, fsdp_idx, tp_idx)
rank = pp_idx * 8 + fsdp_idx * 2 + tp_idx
```

rank 坐标表：

| rank | pp_idx | fsdp_idx | tp_idx |
| ---- | ------ | -------- | ------ |
| 0    | 0      | 0        | 0      |
| 1    | 0      | 0        | 1      |
| 2    | 0      | 1        | 0      |
| 3    | 0      | 1        | 1      |
| 4    | 0      | 2        | 0      |
| 5    | 0      | 2        | 1      |
| 6    | 0      | 3        | 0      |
| 7    | 0      | 3        | 1      |
| 8    | 1      | 0        | 0      |
| 9    | 1      | 0        | 1      |
| 10   | 1      | 1        | 0      |
| 11   | 1      | 1        | 1      |
| 12   | 1      | 2        | 0      |
| 13   | 1      | 2        | 1      |
| 14   | 1      | 3        | 0      |
| 15   | 1      | 3        | 1      |

三类 process group 来自同一张坐标表的不同切片。

TP group 固定 `pp_idx` 和 `fsdp_idx`，只变化 `tp_idx`：

```text
pp0, fsdp0: ranks [0, 1]
pp0, fsdp1: ranks [2, 3]
pp0, fsdp2: ranks [4, 5]
pp0, fsdp3: ranks [6, 7]
pp1, fsdp0: ranks [8, 9]
pp1, fsdp1: ranks [10, 11]
pp1, fsdp2: ranks [12, 13]
pp1, fsdp3: ranks [14, 15]
```

这些 group 负责单层内部的 tensor parallel 计算。例如 column-parallel Linear 中，不同 `tp_idx` 持有不同 output feature shard；row-parallel Linear 中，不同 `tp_idx` 持有不同 input feature shard。

FSDP group 固定 `pp_idx` 和 `tp_idx`，只变化 `fsdp_idx`：

```text
pp0, tp0: ranks [0, 2, 4, 6]
pp0, tp1: ranks [1, 3, 5, 7]
pp1, tp0: ranks [8, 10, 12, 14]
pp1, tp1: ranks [9, 11, 13, 15]
```

这些 group 负责当前 stage、当前 TP shard 上参数状态的分片。注意这里 FSDP All-Gather 重建的是“某个 TP shard 对应的参数”，不是整个 dense layer 的全量参数；完整 dense layer 还需要 TP group 中不同 `tp_idx` 的参数 shard 共同组成。

PP group 固定 `fsdp_idx` 和 `tp_idx`，只变化 `pp_idx`：

```text
fsdp0, tp0: ranks [0, 8]
fsdp0, tp1: ranks [1, 9]
fsdp1, tp0: ranks [2, 10]
fsdp1, tp1: ranks [3, 11]
fsdp2, tp0: ranks [4, 12]
fsdp2, tp1: ranks [5, 13]
fsdp3, tp0: ranks [6, 14]
fsdp3, tp1: ranks [7, 15]
```

这些 group 负责 stage 之间传 activation 和 activation gradient。`pp_idx=0` 可能持有前半部分 Transformer blocks，`pp_idx=1` 持有后半部分 blocks。同一个 `fsdp_idx` 和 `tp_idx` 的 rank 之间形成一条 pipeline lane。

### 一次 micro-batch 中三维如何协同

以 rank5 为例，它的坐标是：

```text
rank5 = (pp_idx=0, fsdp_idx=2, tp_idx=1)
```

它同时属于：

```text
TP group:   [4, 5]
FSDP group: [1, 3, 5, 7]
PP group:   [5, 13]
```

一次 micro-batch 的 forward 可以按对象生命周期理解：

```text
1. FSDP group [1,3,5,7]
   All-Gather 当前 layer、当前 TP shard 的 parameter shards。

2. TP group [4,5]
   用各自的 tensor shard 计算当前层。
   如果该层需要跨 TP shard 汇总，就触发 All-Reduce / All-Gather / Reduce-Scatter。

3. PP group [5,13]
   rank5 把当前 stage 的 activation shard 发送给 rank13。

4. FSDP group [1,3,5,7]
   当前 layer forward 结束后 reshard/free full parameter。
```

backward 反向走同一组关系：

```text
1. PP group [5,13]
   rank13 把 activation gradient 发回 rank5。

2. FSDP group [1,3,5,7]
   如果 forward 后已经 reshard，backward 前再次 All-Gather 当前 layer、当前 TP shard 的 parameter shards。

3. TP group [4,5]
   完成 tensor-parallel backward 中的必要 collective。

4. FSDP group [1,3,5,7]
   对 local full gradient 做 Reduce-Scatter，只保留 rank5 负责的 gradient shard。

5. optimizer step
   rank5 只更新自己持有的 parameter shard 和 optimizer state shard。
```

这个例子说明：多 D 并行不是把几种策略串成独立阶段，而是同一个 rank 在同一次 forward/backward 中不断切换通信语境。计算单层内部矩阵时看 TP group；需要完整当前参数 shard 时看 FSDP group；跨 layer stage 传 activation 时看 PP group。

### 4D 并行

在 3D 基础上再加入一种维度，常见有两种语境：

```text
DP + TP + PP + ZeRO/FSDP
```

或：

```text
DP + TP + PP + SP/CP
```

前者强调模型状态分片，后者强调长序列 activation/attention 分片。只说“4D”没有足够信息，必须明确第四维到底是什么。

如果沿用上面的坐标表，加入 context parallel 可以写成：

```text
32 GPUs = PP 2 * FSDP 4 * TP 2 * CP 2
rank coordinate = (pp_idx, fsdp_idx, tp_idx, cp_idx)
```

新增 CP group 固定 `pp_idx`、`fsdp_idx`、`tp_idx`，只变化 `cp_idx`。它负责把 sequence/context 维度切开，并在 attention 中交换 KV 或 partial attention 结果。此时同一个 rank 可能同时属于四类 group：

```text
TP group:   单层矩阵切分
FSDP group: 参数/梯度/optimizer state 分片
PP group:   stage 间 activation 传递
CP group:   长上下文 attention 通信
```

Sequence Parallelism 在 Megatron-style TP 中经常和 TP group 绑定，不一定总是新增一条独立 mesh 维度；Context Parallelism 更常被当成额外维度来理解。写“4D”时应说明第四维到底是 SP、CP、FSDP shard 维度，还是 HSDP 的 replicate 维度。

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

从 4D 继续加入 EP，可以写成：

```text
64 GPUs = PP 2 * FSDP 4 * TP 2 * CP 2 * EP 2
rank coordinate = (pp_idx, fsdp_idx, tp_idx, cp_idx, ep_idx)
```

EP group 负责 expert 参数归属和 token routing。MoE layer 中，dense attention/MLP 以外的 expert 计算会多出一段：

```text
router 选 expert
-> EP group 内按目标 expert 打包 token
-> All-to-All / All-to-AllV dispatch
-> local expert compute
-> All-to-All / All-to-AllV combine
-> 恢复原 token 顺序
```

5D 并行里最容易混淆的是“模型参数被切了几次”。更准确的说法是：

- TP 切 dense layer 内部的 tensor 维度。
- FSDP 切每个 TP shard 上的参数状态。
- PP 切 layer/stage。
- CP/SP 切 activation 或 attention 的 sequence/context 维度。
- EP 切 expert 归属，并让 token 按路由结果跨 rank 移动。

这些切分作用在不同对象上。分析一个真实训练配置时，应先写出 rank coordinate 和每一维大小，再列出每类 process group、通信操作、通信对象和拓扑位置。TP、CP、EP 往往更依赖高速互联；FSDP 的 All-Gather/Reduce-Scatter 和 PP 的 activation send/recv 也可能跨机，具体放置要结合模型形状、batch/micro-batch、网络带宽和节点内 GPU 拓扑判断。

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
- FSDP：PyTorch 原生 fully sharded data parallel。FSDP1 以 FlatParameter 管理分片；FSDP2 以 DTensor / DeviceMesh 管理逐参数分片。

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

这些工具或框架往往提供更高层的训练配置、launcher、策略封装和生态集成。它们可以放在“编排层”理解：可能调用 PyTorch DDP/FSDP、DeepSpeed 或 Megatron 风格并行策略，而不是每个都从零实现所有 collective。

## 13. 关键时间线和出处

这些条目用于给主教程中的判断建立出处锚点，不是完整论文综述。

| 时间                           | 工作或文档                             | 和本文的关系                                                                                                            |
| ------------------------------ | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 2018                           | GPipe                                  | 把 micro-batch pipeline 作为训练大模型的一种系统化调度方式。                                                            |
| 2019                           | Megatron-LM                            | 系统展示 Transformer tensor model parallelism，并奠定 TP + PP + DP 组合讨论的基础。                                     |
| 2019/2020                      | ZeRO                                   | 提出按 optimizer state、gradient、parameter 逐步消除数据并行冗余，是理解 ZeRO-1/2/3 和 FSDP 的核心来源。                |
| PyTorch 官方文档               | DDP notes / FSDP docs / FSDP2 tutorial | DDP bucket、autograd hook、FSDP1 wrapping、FSDP2 `fully_shard`、DTensor、DeviceMesh、reshard 等工程语义以官方文档为准。 |
| NVIDIA NCCL 文档               | Collectives                            | AllGather、ReduceScatter、AllReduce 等 collective 的输入输出 buffer 布局和 output chunk 视角。                          |
| NVIDIA Transformer Engine 文档 | FP8 primer                             | FP8 E4M3/E5M2、scaling 与低精度训练相关概念的工程参考。                                                                 |

参考来源：

- PyTorch DistributedDataParallel notes: <https://docs.pytorch.org/docs/stable/notes/ddp.html>
- PyTorch FullyShardedDataParallel docs: <https://docs.pytorch.org/docs/stable/fsdp.html>
- PyTorch FSDP2 `fully_shard` docs: <https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html>
- PyTorch FSDP2 tutorial: <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
- NVIDIA NCCL collectives: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html>
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

它和本文第 2 节对应：先理解 collective 语义，再把通信量和训练中的使用位置对上。

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

这个脚本和本文第 5 节对应：完整 Linear 作为 reference，column-parallel 和 row-parallel 分别手写 forward/backward，再比较 `Y`、`dX`、`dW`、`db`。它适合用来检查自己是否真的理解“列并行为什么要 all-reduce dX，行并行为什么要 all-reduce output”。
