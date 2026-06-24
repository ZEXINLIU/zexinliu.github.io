---
layout: post
title: "Modern Attention for LLMs: MHA/MQA/GQA, RoPE, MLA, FlashAttention, and MoE"
date: 2026-02-22 00:00:00
description: 从 MHA/MQA/GQA、RoPE、MLA 到 FlashAttention、MoE 的注意力机制笔记。
tags: attention transformer large-language-models inference
categories: ai-infra
---

## 开篇总结

本文重点包括：

1. **MHA、MQA、GQA 的核心差异不是 attention 公式变了，而是 K/V head 的组织方式变了。** MHA 给每个 query head 配独立 K/V；MQA 让所有 query head 共享一组 K/V；GQA 介于两者之间。真正的收益主要出现在自回归解码：KV cache 更小，显存带宽压力更低。本文会把 `num_q_heads`、`num_kv_heads`、`repeat_kv`、cache shape 和复杂度放在同一个接口里解释。
2. **Transformer encoder-decoder 与 GPT decoder-only 的差异，最后会落到 attention 调用接口上。** Encoder self-attention、decoder causal self-attention、cross-attention 都可以复用同一个注意力核心，但 Q/K/V 来源、mask、KV cache 生命周期完全不同。本文会解释为什么 cross-attention 是 `Q=decoder, K/V=encoder`，而 decoder-only decode 阶段只输入一个新 token 却能看见全部历史。
3. **位置编码的难点不只是公式，而是“位置如何进入 QK 点积”。** 原始正弦位置编码把位置加到输入 embedding；RoPE 把 Q/K 当作二维旋转对；ALiBi 直接修改 attention logits。尤其 RoPE，本文会保留原始代码中有价值的多实现视角：偶奇维公式、向量化 rotate、复数乘法、split-half/nanovllm 风格布局。它们解释的是同一个数学对象，但暴露了不同的工程布局问题。
4. **MLA 的关键不是一句“压缩 KV cache”，而是两次矩阵吸收如何成立，以及为什么 RoPE 路径不能一起吸收。** 内容路径可以利用结合律把 $$(c^QW^{UQ})(c^{KV}W^{UK})^\top$$ 改写成 $$c^Q\left(W^{UQ}(W^{UK})^\top\right)(c^{KV})^\top$$；输出路径也可以把 $W^{UV}$ 和 $W^O$ 合并。但 RoPE key 是位置相关路径，旋转矩阵随位置变化，不能简单并进同一个 latent KV。
5. **FlashAttention 不是把 $O(T^2)$ attention 变成线性 attention，而是在 exact attention 下减少中间矩阵和 HBM/SRAM 往返。** 它的灵魂是 online softmax：只维护每行的最大值 $m$、归一化分母 $l$ 和输出累积量 $O$，就能 block by block 合并完整 softmax。本文会解释 v1/v2 在循环顺序和输出写回上的差异，也会说明 CPU-only 环境能复现什么、不能复现什么。
6. **长上下文、Sparse Attention、Linear Attention 不是同一类优化。** 长上下文 RoPE 扩展主要改位置到角度的映射；Sparse Attention 改 token 可见图；Linear Attention 改 softmax kernel 或计算结合方式。它们都服务长序列，但牺牲点完全不同。
7. **MoE 与 Attention 的关系经常被混在一起，但它通常不是 attention 机制本身。** MoE 多数时候替换 FFN 层，用路由器让不同 token 走不同专家；MLA/FlashAttention 处理 attention/cache/IO，MoE 处理参数容量和条件计算。现代模型可以同时使用这些机制。
8. **离散选择是 MoE、VQ-VAE 等模型训练中的共同难点。** Top-1/Top-k 路由、categorical sampling、codebook argmin 都会在前向中产生硬选择，但 argmax/argmin 本身不可导。本文会补充 Gumbel-Softmax、soft relaxation、Straight-Through estimator 等常见处理方式，并用 VQ-VAE 的 codebook lookup 作为对照例子。

读这些机制时，可以始终追问一个问题：它到底是在改变**表达能力**、改变**接口组织**、改变**cache/内存布局**，还是改变**硬件执行路径**。这个问题比记住更多缩写更重要，因为很多看似相似的 attention 变体，真正改变的层次并不一样。

## 阅读地图

- [技术发展路径](#技术发展路径)：先把各机制放到时间线上，避免孤立理解。
- [MHA、MQA、GQA](#2-mhamqagqa)：理解 query head 与 KV head 的接口差异。
- [Transformer Encoder-Decoder 与 Decoder-Only](#3-transformer-encoder-decoder-与-decoder-only)：理解 attention 核心如何在不同结构中被调用。
- [位置编码](#4-位置编码)：从绝对位置、RoPE 到 ALiBi，重点看位置如何进入 score。
- [MLA](#5-mla从-kv-cache-压缩到矩阵吸收)：理解 latent KV、矩阵吸收和 RoPE 路径边界。
- [FlashAttention](#6-flashattentionexact-attention-的-io-优化)：理解 online softmax 和 CPU-only 实验边界。
- [Sparse / Linear Attention](#7-sparse--linear-attention改变可见图或核函数)：区分稀疏可见图和线性核技巧。
- [MoE 与 Attention](#8-moe-与-attention-的关系)：理解注意力之外的条件计算、容量扩展和离散路由训练。
- [面试复写线索](#9-面试复写线索)：把主线浓缩为可快速复写的实现不变量。

## 技术发展路径

| 时间       | 技术                       | 主要出处                                                                                                                                                     | 本项目关注点                                                        |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 2013-08-15 | Straight-Through estimator | [Bengio et al.](https://arxiv.org/abs/1308.3432)                                                                                                             | 为离散/随机神经元估计或传播梯度，包含 straight-through 思路         |
| 2016-11-03 | Gumbel-Softmax             | [Categorical Reparameterization](https://arxiv.org/abs/1611.01144)                                                                                           | 用可微的 Gumbel-Softmax 分布近似 categorical sample                 |
| 2016-11-02 | Concrete distribution      | [Concrete Distribution](https://arxiv.org/abs/1611.00712)                                                                                                    | categorical 离散变量的连续松弛                                      |
| 2017-01-23 | Sparsely-Gated MoE         | [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)                                                                                       | 用条件计算扩大参数容量，每个样本只激活部分专家                      |
| 2017-06-12 | Transformer / MHA          | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                | Scaled dot-product attention、encoder-decoder、并行 self-attention  |
| 2017-11-02 | VQ-VAE                     | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)                                                                                  | 用向量量化离散 latent，并用 straight-through 让 encoder 可训练      |
| 2018-06-11 | GPT-style decoder-only     | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 只保留 causal decoder，用 next-token prediction 训练                |
| 2019-11-06 | MQA                        | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)                                                                | 所有 query head 共享一组 K/V，减少增量解码带宽                      |
| 2020-04-10 | Longformer                 | [Longformer](https://arxiv.org/abs/2004.05150)                                                                                                               | 用局部窗口和全局 token 做长文档 sparse attention                    |
| 2020-06-29 | Linear Transformer         | [Transformers are RNNs](https://arxiv.org/abs/2006.16236)                                                                                                    | 用核函数重写 attention，支持线性复杂度递推                          |
| 2020-07-28 | BigBird                    | [BigBird](https://arxiv.org/abs/2007.14062)                                                                                                                  | 用局部、随机、全局边组合 sparse attention，并分析表达能力           |
| 2020-09-30 | Performer                  | [Performer](https://arxiv.org/abs/2009.14794)                                                                                                                | 用 FAVOR+ 随机特征近似 softmax attention                            |
| 2021-01-11 | Switch Transformer         | [Switch Transformer](https://arxiv.org/abs/2101.03961)                                                                                                       | 简化 MoE 路由，每个 token 选择一个专家                              |
| 2021-04-20 | RoPE                       | [RoFormer](https://arxiv.org/abs/2104.09864)                                                                                                                 | 用旋转把绝对位置编码进 Q/K，并在点积里体现相对位置                  |
| 2021-08-27 | ALiBi                      | [Train Short, Test Long](https://arxiv.org/abs/2108.12409)                                                                                                   | 不加位置 embedding，而是给 score 加线性距离惩罚                     |
| 2022-05-27 | FlashAttention             | [FlashAttention](https://arxiv.org/abs/2205.14135)                                                                                                           | IO-aware exact attention，避免物化完整 attention matrix             |
| 2023-05-22 | GQA                        | [GQA](https://arxiv.org/abs/2305.13245)                                                                                                                      | 在 MHA 和 MQA 之间折中 KV head 数量                                 |
| 2023-06-27 | Position Interpolation     | [PI](https://arxiv.org/abs/2306.15595)                                                                                                                       | 将超长位置线性压回训练上下文范围，缓解 RoPE 外推                    |
| 2023-07-17 | FlashAttention-2           | [FlashAttention-2](https://arxiv.org/abs/2307.08691)                                                                                                         | 改善 work partitioning，减少 non-matmul FLOPs 和 shared memory 通信 |
| 2023-09-01 | YaRN                       | [YaRN](https://arxiv.org/abs/2309.00071)                                                                                                                     | 对 RoPE context extension 做更高效的频率缩放和微调策略              |
| 2024-01-11 | DeepSeekMoE                | [DeepSeekMoE](https://arxiv.org/abs/2401.06066)                                                                                                              | 通过更细粒度专家与共享专家增强 MoE 专家分工                         |
| 2024-02-21 | LongRoPE                   | [LongRoPE](https://arxiv.org/abs/2402.13753)                                                                                                                 | 面向百万级上下文的 RoPE 扩展与搜索策略                              |
| 2024-05-07 | MLA                        | [DeepSeek-V2](https://arxiv.org/abs/2405.04434)                                                                                                              | 用 latent KV 压缩 cache，并配合矩阵吸收提升推理效率                 |
| 2024-07-11 | FlashAttention-3           | [FlashAttention-3](https://arxiv.org/abs/2407.08608)                                                                                                         | 面向 Hopper GPU 的异步流水、warp specialization、FP8                |
| 2024-12-27 | DeepSeek-V3                | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)                                                                                             | 继续使用 MLA，并在训练系统上扩展 FP8/MoE 等工程策略                 |
| 2026-03-05 | FlashAttention-4           | [FlashAttention-4](https://arxiv.org/abs/2603.05451)                                                                                                         | 面向 Blackwell GPU 的算法与 kernel pipeline co-design               |

## 1. Scaled Dot-Product Attention

设输入 hidden states 为 $X\in\mathbb{R}^{B\times T\times d_{model}}$。标准 self-attention 中：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

单头 attention 为：

$$
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V
$$

其中 $M$ 是 mask，常见有三类：

- causal mask：decoder 不能看未来 token。
- padding mask：batch 中 padding token 不应被 attention 到。
- task-specific attention mask：用于屏蔽任意指定位置。

为什么除以 $\sqrt{d_h}$：如果 $Q,K$ 每个维度方差近似为 1，点积的方差会随 $d_h$ 增大。softmax 对尺度敏感，未缩放时 logits 容易过大，使概率分布过早接近 one-hot，梯度也更不稳定。

实现上最关键的 shape 是：

```text
Q:      (batch, num_q_heads, target_len, head_dim)
K/V:    (batch, num_kv_heads, source_len, head_dim)
score:  (batch, num_q_heads, target_len, source_len)
output: (batch, target_len, d_model)
```

这个 shape 约定会贯穿 MQA/GQA、cross-attention、KV cache、MLA。

## 2. MHA、MQA、GQA

### 2.1 先看接口，而不是先背名字

MHA 的多头不是简单复制 attention，而是把 $d_{model}$ 分成多个 head 子空间：

$$
\mathrm{head}_i=\mathrm{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})
$$

然后拼接后过输出投影：

$$
O=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O
$$

MQA/GQA 的 attention 公式没有变，变的是 K/V 投影输出的 head 数：

| 类型 | Query heads |       KV heads | cache 每 token 元素数 | 直觉                                   |
| ---- | ----------: | -------------: | --------------------: | -------------------------------------- |
| MHA  |       $h_q$ |          $h_q$ |             $2h_qd_h$ | 每个 query head 有独立 K/V，表达最完整 |
| GQA  |       $h_q$ | $1<h_{kv}<h_q$ |          $2h_{kv}d_h$ | 一组 K/V 服务一组 query heads          |
| MQA  |       $h_q$ |            $1$ |                $2d_h$ | 所有 query heads 共享一组 K/V          |

注意这里的 cache 每 token 元素数只统计 K/V。训练时仍然要计算 $QK^\top$，所以 MQA/GQA 不会把 attention 的二次复杂度变成线性；它们主要降低自回归解码中的 KV cache 体积和读带宽。

### 2.2 `repeat_kv` 不应该污染 cache

代码中最容易写错的一点是：GQA/MQA 的 K/V 在 cache 里应该保持紧凑，只在计算 attention score 前临时 repeat 到 query head 数。

```python
k_for_attn = repeat_kv(k, num_q_heads // num_kv_heads)
v_for_attn = repeat_kv(v, num_q_heads // num_kv_heads)
```

如果把 repeat 后的 K/V 存进 cache，就把 MQA/GQA 的推理收益抵消了。也就是说，`repeat_kv` 是矩阵乘法前的视图/广播逻辑，不是 cache 逻辑。

### 2.3 为什么 GQA 是折中而不是折磨

MQA 最省 cache，但所有 query heads 只能共享同一套 K/V 表示；MHA 最自由，但 cache 最大。GQA 的价值是：让若干 query heads 共享一个 K/V head，保留一部分多样性，同时显著降低 cache。例如 `num_q_heads=32, num_kv_heads=8` 时，KV cache 是 MHA 的 1/4，但不是像 MQA 那样压到 1/32。

对应实验：`examples/attention_family.py`。

## 3. Transformer Encoder-Decoder 与 Decoder-Only

### 3.1 同一个 attention 核心，三种调用方式

原始 Transformer 是 encoder-decoder 架构：

- Encoder self-attention：输入序列内部全量双向可见，不使用 causal mask。
- Decoder self-attention：目标序列自回归生成，必须使用 causal mask。
- Cross-attention：decoder states 作为 Q，encoder output 作为 K/V。

接口上最容易出错的是 cross-attention：

```text
self-attention:  Q, K, V all come from current hidden states
cross-attention: Q comes from decoder states, K/V come from encoder states
```

这解释了为什么一个通用 attention 模块最好把参数拆成 `query` 和 `key_value`。self-attention 可以让 `key_value=None`，内部默认 `key_value=query`；cross-attention 则显式传入 encoder output。

### 3.2 Decoder-only 为什么和 KV cache 天然绑定

GPT-style decoder-only 去掉 encoder 和 cross-attention，只保留 causal decoder stack。训练时输入整段序列，用 causal mask 保证第 $t$ 个位置只能看见 $\leq t$ 的 token。推理时如果每一步都重新计算整段 K/V，会重复做大量历史 token 投影。

增量解码的状态变化是：

```text
prefill:
  input prompt length = T
  build K_cache, V_cache for all prompt tokens

decode step t:
  input only the newest token
  project q_t, k_t, v_t
  append k_t, v_t to cache
  attention(q_t, K_cache, V_cache)
```

这里 causal mask 的 diagonal 也要考虑 `past_len`。如果当前只输入 1 个 token，source length 是 `past_len + 1`，它可以看见所有历史和自己，不应该被普通上三角 mask 错误屏蔽。

对应实验：`examples/transformer_usage.py`。

## 4. 位置编码

### 4.1 正弦绝对位置编码

原始 Transformer 使用固定正弦位置编码：

$$
PE(pos,2i)=\sin(pos/10000^{2i/d})
$$

$$
PE(pos,2i+1)=\cos(pos/10000^{2i/d})
$$

它被加到 token embedding 上，因此位置信息进入后续所有线性层。它的优点是简单、不增加参数；缺点是位置是“混入输入表示”的，后续层很难显式控制位置如何影响 QK 点积。

### 4.2 RoPE 的核心：位置进入 QK 点积

RoPE 对 Q/K 的二维子空间做旋转。对一对维度 $(x_1,x_2)$：

$$
\begin{bmatrix}
x_1'\\
x_2'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_m & -\sin\theta_m\\
\sin\theta_m & \cos\theta_m
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

其中 $m$ 是位置。关键不是“旋转看起来高级”，而是两个旋转向量点积时：

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k
$$

点积自然依赖相对位移 $n-m$。这就是“RoPE 用绝对位置旋转 Q/K，却在 score 中体现相对位置”的核心。

### 4.3 多实现视角：公式等价和布局等价是两件事

原始代码里保留了多个 RoPE 实现版本，这一点很有价值。整理后可以这样理解：

- 偶奇维直接公式：把 $(x_0,x_1),(x_2,x_3)$ 当作旋转对，最贴近数学定义。
- 向量化 rotate：把公式写成 `x*cos + rotate_pair(x)*sin`，便于对照主流 LLM 代码。
- 复数乘法：把 $(x_0,x_1)$ 看成 $x_0+i x_1$，乘以 $\cos\theta+i\sin\theta$，这是最清楚的证明视角。
- split-half/nanovllm 风格：把前半维和后半维配对，代码更紧凑，但必须意识到内存布局已经变了。

最容易踩的坑是：两个实现输出不同，不一定是数学错了，也可能只是 rotary pair 的布局不同。若 interleaved 输入是：

```text
[x0, x1, x2, x3, ...]
```

split-half 需要先整理成：

```text
[x0, x2, ..., x1, x3, ...]
```

然后再用 `rotate_half`。这正是 `examples/positional_encoding.py`
里保留多实现对照的原因。

### 4.4 ALiBi 的位置观

ALiBi 不把位置向量加到 embedding，也不旋转 Q/K，而是直接给 attention score 加距离惩罚：

$$
score_{h,i,j}=\frac{q_{h,i}k_{h,j}^{\top}}{\sqrt{d_h}}-m_h(i-j)
$$

它的优势是外推直觉简单：训练短上下文时，模型已经学到“越远惩罚越大”的结构；推理长上下文时继续沿用。代价是表达形式更受约束，不像 RoPE 那样在 Q/K 子空间里保留更丰富的相对相位关系。

对应实验：`examples/positional_encoding.py`。

### 4.5 长上下文位置扩展：外推不是免费午餐

RoPE 的优势是相对位置性质强，但长上下文会遇到一个直观问题：如果训练时只见过长度 $L$，推理时直接把位置推到远大于 $L$，旋转角度会进入训练中没见过的相位区域。长上下文位置扩展的核心就是重新设计“位置 $\rightarrow$ 角度”的映射。

最容易混淆的几个方向：

- 直接外推：不改 RoPE，位置继续增长。实现简单，但高频维度相位可能快速转到训练外区域。
- Position Interpolation：把新上下文位置线性压缩回训练上下文范围。例如训练长度 $L$、目标长度 $L'$，位置 $m$ 使用 $m\cdot L/L'$。
- YaRN 类方法：不是所有频率都用同一个缩放，通常会对不同频率维度做分段或平滑缩放，并配合少量微调。
- LongRoPE 类方法：把短上下文保真和长上下文扩展作为联合目标，搜索或设计更细粒度的位置缩放策略。

这几类方法都在改 RoPE 的位置映射，而不是改 attention 的 Q/K/V head 组织，也不是 FlashAttention 那种 IO 优化。它们通常可以和 GQA、MLA、FlashAttention 同时出现。

对应实验：`examples/long_context_position.py`。它展示了原始 RoPE、Position
Interpolation、YaRN-like 频率缩放在相同位置上的角度变化。这个实验不是复现完整论文训练
recipe，而是把“为什么长上下文要改角度映射”这件事跑出来。

## 5. MLA：从 KV cache 压缩到矩阵吸收

### 5.1 MLA 在压缩什么

MLA 的目标不是把 attention matrix 低秩近似掉，而是让 K/V 的生成经过一个低维 latent cache。以 DeepSeek-V2 风格记号表示：

Query 路径：

$$
c_t^Q = \mathrm{RMSNorm}(h_t W^{DQ})
$$

$$
q_t^C=c_t^QW^{UQ},\quad q_t^R=c_t^QW^{QR}
$$

KV 路径：

$$
c_t^{KV}=\mathrm{RMSNorm}(h_tW^{DKV})
$$

$$
k_t^C=c_t^{KV}W^{UK},\quad v_t^C=c_t^{KV}W^{UV}
$$

同时还有一条共享的 RoPE key：

$$
k_t^R=h_tW^{KR}
$$

attention score 拆成内容部分和位置部分：

$$
score = (q^C(k^C)^\top + q^R(k^R)^\top) / \sqrt{d_q}
$$

普通 MHA 每个 token cache 约为：

$$
h(qk\_head\_dim + v\_head\_dim)
$$

DeepSeek-V2 典型配置里：

$$
128(128+128)=32768
$$

MLA cache 只保存：

$$
c^{KV} + k^R = 512+64=576
$$

比例约为 56.9x。这个数字的意义不是“attention 计算少了 56.9x”，而是每个历史 token 要从 cache 里读出的 K/V 表示大幅减少了。

### 5.2 内容路径的矩阵吸收

内容 attention 展开为：

$$
(c^QW^{UQ})(c^{KV}W^{UK})^\top
$$

利用矩阵结合律：

$$
(c^QW^{UQ})(W^{UK})^\top(c^{KV})^\top
=
c^Q\left(W^{UQ}(W^{UK})^\top\right)(c^{KV})^\top
$$

因此可以预先或运行时吸收：

$$
W^{QK}=W^{UQ}(W^{UK})^\top
$$

这样无需显式恢复 $k^C$，而是在 latent 空间里把 query 变成能直接和 $c^{KV}$ 点积的 pseudo query。

输出部分同理：

$$
(\mathrm{attn}\ c^{KV}W^{UV})W^O
=
(\mathrm{attn}\ c^{KV})(W^{UV}W^O)
$$

因此可以吸收：

$$
W^{VO}=W^{UV}W^O
$$

这里有两个实现版本值得同时保留：

- 展开版：显式恢复 $k^C,v$，逻辑最直观，适合训练和理解。
- 吸收版：不显式恢复 $k^C,v$，更接近推理优化思路，适合验证矩阵结合律。

### 5.3 为什么 RoPE 不能直接并进同一个吸收矩阵

RoPE 路径是：

$$
q^R_m = R_m(c^QW^{QR}),\quad k^R_n = R_n(h_nW^{KR})
$$

这里至少有两层障碍：

1. $R_m,R_n$ 随位置变化，不是固定权重矩阵。矩阵吸收依赖固定线性层之间的结合律，而位置旋转会让“同一个权重”在不同 token 位置表现不同。
2. $k^R$ 来自 $hW^{KR}$ 的共享位置 key 路径，不是从 $c^{KV}$ 通过 $W^{UK}$ 恢复出来的内容 key。换句话说，MLA 有意把 content cache 和 positional key 分开保存。

因此 MLA 可以吸收内容部分的 $W^{UQ},W^{UK},W^{UV},W^O$，但仍需要单独处理 RoPE key。这个点如果不拆开，很容易误以为“既然 K/V 都压缩了，RoPE K 也能一起压缩到同一个 latent 里”。

对应实验：`examples/mla.py`。它提供 expanded forward 和 absorbed forward，并做数值等价检查。

## 6. FlashAttention：exact attention 的 IO 优化

### 6.1 FlashAttention 优化的不是公式复杂度

标准 attention 会物化：

$$
S=QK^\top,\quad P=\mathrm{softmax}(S)
$$

它们都是 $T\times T$。计算复杂度仍是 $O(T^2d)$，FlashAttention 并没有把 exact attention 变成线性 attention。它优化的是中间矩阵物化和 HBM/SRAM 数据移动。

在 GPU 语境中，HBM 容量大但带宽相对低，SRAM/shared memory 容量小但带宽高。FlashAttention 的思路是：把 Q/K/V 分块搬进快存储，在块内算局部 score，用 online softmax 合并结果，避免把完整 $S$ 和 $P$ 写回 HBM。

### 6.2 Online Softmax：只维护充分统计量

对一行 score，维护历史最大值 $m$、历史归一化分母 $l$ 和输出 $O$。新 block 的 score 为 $S_j$：

$$
m_{new}=\max(m,\max(S_j))
$$

旧分母要从旧基准 $m$ 换到新基准 $m_{new}$：

$$
l_{new}=e^{m-m_{new}}l + \sum e^{S_j-m_{new}}
$$

输出更新：

$$
O_{new}=
\frac{
e^{m-m_{new}}lO + e^{S_j-m_{new}}V_j
}{l_{new}}
$$

这就是原始代码里大量注释推导的核心：如果新的 block 出现了更大的 row max，历史的 $l$ 和 $O$ 并不是作废，而是乘上 $e^{m-m_{new}}$ 后换到同一个指数基准。

### 6.3 v1 与 v2：循环顺序背后的状态写回

教学上可以先抓住循环顺序：

```text
v1:
  for K/V block:
    for Q block:
      load O_i, l_i, m_i
      update O_i, l_i, m_i
      write O_i, l_i, m_i

v2:
  for Q block:
    keep running O_i, l_i, m_i locally
    for K/V block:
      update local states
    write final O_i once
```

FlashAttention-2 不只是换了 for 循环，它还减少 non-matmul FLOPs、改进 parallelism 和 work partitioning。但在 CPU 教学代码里，最能复现的是两个点：

- v1 更像“每来一个 K/V block，都把某个 Q block 的归一化输出更新并写回”。
- v2 更像“固定一个 Q block，把所有 K/V block 扫完，最后只除一次 $l$ 并写回一次 $O$”。

本项目的 `examples/flash_attention.py` 保留了这种差异，并打印
`output block writes`。在 `seq_len=64, block=16` 的例子里，v1 写 16 次，v2 写 4 次。

### 6.4 CPU-only 环境应该展示什么

CPU 上的 Python block 循环通常会比 PyTorch 一次性大矩阵乘法更慢，所以不应该用这个实验证明 FlashAttention 的性能优势。CPU-only 环境适合展示：

- exactness：block 版本和标准 attention 输出接近。
- online softmax：$m,l,O$ 如何在不物化完整矩阵时合并。
- IO 意识：v1/v2 何时读写输出块。
- 边界意识：v3/v4 的异步流水、TMA、Tensor Core、warp specialization、Blackwell pipeline 等 GPU kernel 优化不能在 CPU 上真实复现。

这比单纯写一个“更慢的 Python 版 FlashAttention”更有意义，因为它明确区分了算法原理和硬件实现。

## 7. Sparse / Linear Attention：改变可见图或核函数

长序列优化里，Sparse Attention、Linear Attention 和 FlashAttention 经常被放在一起讨论，但它们不是同一类东西。

FlashAttention 的目标是 exact softmax attention，数学结果尽量不变，优化中间矩阵和 IO。Sparse Attention 和 Linear Attention 通常会改变 attention 机制本身：

- Sparse Attention 改 token-pair 可见图，不再让每个 token 看见所有历史 token。
- Linear Attention 改 softmax kernel 或计算形式，让 $QK^\top$ 不必完整物化。

### 7.1 Sparse Attention：减少边，而不是换公式

Dense causal attention 中，第 $i$ 个 token 可以看 $0...i$，可见边数量是：

$$
\frac{T(T+1)}{2}
$$

局部窗口 attention 只允许看最近 $w$ 个 token：

$$
j \in [i-w+1, i]
$$

可见边数量近似变成：

$$
O(Tw)
$$

Longformer 使用 sliding window attention，并加入 global attention 来处理任务级全局 token。BigBird 则组合 local、random、global 连接，尝试在稀疏图上保留足够的信息流和理论表达能力。

直觉上，Sparse Attention 的难点是：省掉的边是否真的不重要。如果任务依赖远距离精确匹配，纯局部窗口可能看不到关键 token；所以实际模型常加入全局 token、随机边、分块模式或检索机制。

### 7.2 Linear Attention：把 softmax 核换成可结合形式

标准 softmax attention：

$$
\mathrm{softmax}(QK^\top)V
$$

很难直接改写为只依赖前缀累计的形式。Linear Attention 的思路是用特征映射 $\phi$ 近似或替代 softmax kernel：

$$
\exp(q^\top k)\approx \phi(q)^\top\phi(k)
$$

于是 causal attention 可以写成：

$$
o_t=
\frac{
\phi(q_t)^\top\sum_{i\le t}\phi(k_i)v_i^\top
}{
\phi(q_t)^\top\sum_{i\le t}\phi(k_i)
}
$$

这样只需要维护两个前缀状态：

$$
S_t=\sum_{i\le t}\phi(k_i)v_i^\top,\quad z_t=\sum_{i\le t}\phi(k_i)
$$

Performer 用 FAVOR+ 随机特征近似 softmax attention；Linear Transformer 使用 kernel trick 让 attention 可以像 RNN 一样递推。代价是：这不再是普通 dense softmax attention 的精确结果，质量、稳定性和长距离选择能力都取决于核函数和特征设计。

对应实验：`examples/sparse_linear_attention.py`。它展示 sliding-window sparse attention
的可见边数量变化，以及一个确定性 feature map 的 causal linear attention。实验中
sparse/linear 输出和 dense 输出有差异，这是预期现象，因为机制被改变了。

## 8. MoE 与 Attention 的关系

MoE 经常和 attention 优化一起出现在现代 LLM 架构里，但它通常不是 attention 本身。标准 Transformer block 可以粗略写成：

```text
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

MoE 通常替换的是 FFN：

```text
x = x + Attention(LN(x))
x = x + MoE-FFN(LN(x))
```

路由器根据 token 表示选择专家：

$$
e_t=\mathrm{TopK}(\mathrm{Router}(x_t))
$$

然后只计算被选中的专家：

$$
\mathrm{MoE}(x_t)=\sum_{e\in e_t}g_{t,e}E_e(x_t)
$$

MoE 的收益和 attention 优化的收益不在同一层：

- MQA/GQA/MLA 主要降低 attention 推理 cache 或带宽。
- FlashAttention 主要降低 exact attention 的中间矩阵和 GPU IO。
- MoE 主要增加参数容量，同时让每个 token 只激活部分参数。

这也解释了 DeepSeek-V2/V3 为什么可以同时使用 MLA 和 MoE：MLA 解决 KV cache 和 attention 路径效率，MoE 解决 FFN 参数容量与专家分工。两者不是替代关系。

MoE 的工程难点包括：

- 路由负载不均衡：所有 token 都挤到少数专家会造成质量和吞吐问题。
- 专家容量限制：每个专家最多处理多少 token，会影响 token drop、padding 和通信。
- 分布式通信：专家并行会引入 all-to-all 通信，吞吐瓶颈可能不在矩阵乘法本身。
- 专家分工：共享专家、细粒度专家、路由正则都会影响专家是否真正专业化。

对应实验：`examples/moe_attention.py`。它保留 dense causal attention，然后把 FFN 换成
top-1 MoE，并打印每个专家收到的 token 数量。这个实验的重点是看清楚：attention 负责跨
token 混合，MoE 负责每个 token 后续走哪个 FFN 专家。

### 8.1 Top-k 路由的不可导问题

MoE 路由通常会先得到 router logits：

$$
r_t = \mathrm{Router}(x_t)
$$

然后做 top-k：

$$
S_t = \mathrm{TopK}(r_t)
$$

问题在于 $\mathrm{TopK}$、$\mathrm{argmax}$、$\mathrm{argmin}$ 都是分段常数操作。只要某个 logit 没有跨过排序边界，离散选择结果就不变，所以经典反向传播拿不到有用梯度。直观地说：

```text
router_logits -> argmax -> expert_id -> selected_expert_output -> loss
```

这条链在 `argmax` 处断掉。实际训练不会只靠“硬选择本身”给 router 学习信号，而会使用替代策略。

### 8.2 光滑近似：从 max 到 top-k

苏剑林在[科学空间的相关整理](https://kexue.fm/archives/6620)里给出了一个很有用的统一视角：很多不可导算子可以先找一个带温度参数的光滑近似。这个视角适合放进本教程，但不需要把所有 soft sorting/ranking 技巧都展开；和当前 MoE/VQ-VAE 主线最相关的是下面几类。

**LogSumExp 近似 max。**

$$
\max_i x_i \approx \tau\log\sum_i e^{x_i/\tau}
$$

当 $\tau\to 0$ 时，它趋近于 $\max(x)$；反向传播时，它的梯度是：

$$
\frac{\partial}{\partial x_i}\tau\log\sum_j e^{x_j/\tau}
=
\mathrm{softmax}(x/\tau)_i
$$

这说明“最大值”可以被一个平滑的加权平均式梯度替代：最大元素得到最多梯度，非最大元素也能得到少量信号。

**Softmax 近似 onehot(argmax)。**

$$
\mathrm{onehot}(\arg\max(x)) \approx \mathrm{softmax}(x/\tau)
$$

温度越低，分布越尖锐；温度越高，分布越平滑。MoE router 的 soft relaxation、Gumbel-Softmax、Straight-Through hard sample 都可以看作围绕这个近似做不同取舍。

**SoftArgmax 近似 argmax index。**

如果 index 本身有意义，例如位置、坐标、排序桶，可以用：

$$
\mathrm{softargmax}(x)=\sum_i i\cdot \mathrm{softmax}(x/\tau)_i
$$

但如果类别编号没有序关系，例如“猫=0、狗=1、车=2”，这个期望 index 没有稳定语义，不适合作为分类 loss。

**Soft top-k。**

top-k 可以递归地构造 soft 版本：每轮用 softmax 得到一个软选择，然后抑制已经被选择的位置，再进行下一轮。这个思路适合教学和某些可微软排序/检索场景；大规模 MoE 的工程实现通常还要结合容量限制、负载均衡和硬 dispatch，所以不能只靠一个 soft top-k 公式解决所有问题。

**Soft accuracy / soft F1。**

正确率、F1 这类指标含有 threshold、argmax、计数，因此原始形式不可导。可以用概率替代 hard prediction，构造 soft TP/FP/FN：

$$
TP_{soft}=\sum_i p_i y_i,\quad
FP_{soft}=\sum_i p_i(1-y_i),\quad
FN_{soft}=\sum_i (1-p_i)y_i
$$

$$
F1_{soft}=
\frac{2TP_{soft}}{2TP_{soft}+FP_{soft}+FN_{soft}}
$$

这个 surrogate 是可导的，可以把 $-F1_{soft}$ 作为 loss。但它通常是 batch-level 的有偏估计，分母也依赖 batch 统计，优化轨迹可能不如交叉熵稳定。更稳妥的用法是先用交叉熵训练到合理区域，再用 soft F1/soft accuracy 做小步微调，而不是从头直接优化。

### 8.3 常见做法一：Soft relaxation

最直接的做法是训练时不做硬选择，而使用 softmax 权重：

$$
g_t=\mathrm{softmax}(r_t)
$$

$$
y_t=\sum_e g_{t,e}E_e(x_t)
$$

这样所有专家都有权重，router 可导；缺点是计算不再稀疏，和推理时 top-k 路由不完全一致。因此它常用作教学 baseline、辅助损失或小模型实验，不一定是大规模 MoE 的最终训练路径。

### 8.4 常见做法二：Gumbel-Softmax

Gumbel-Softmax 解决的是 categorical sample 不可导的问题。给 logits 加 Gumbel noise 后做 softmax：

$$
y_i=\frac{\exp((r_i+g_i)/\tau)}{\sum_j\exp((r_j+g_j)/\tau)}
$$

温度 $\tau$ 越低，样本越接近 one-hot；温度越高，分布越平滑。PyTorch 的 `F.gumbel_softmax(logits, hard=True)` 常用 straight-through 形式：前向返回 hard one-hot，反向使用 soft sample 的梯度。

这适合解释“训练时想要离散选择，但又想让 logits 获得梯度”的场景。不过在大规模 MoE 中，路由还会叠加 load balancing loss、capacity constraint、token dropping 或 expert parallel 通信策略。

### 8.5 常见做法三：Straight-Through estimator

Straight-Through 的核心技巧是“前向 hard，反向 pretend soft”。对 hard argmax gate：

$$
y_{hard}=\mathrm{onehot}(\mathrm{argmax}(r))
$$

构造：

$$
y_{st}=y_{hard}-\mathrm{stopgrad}(y_{soft})+y_{soft}
$$

前向时：

$$
y_{st}=y_{hard}
$$

反向时，$y_{hard}$ 和 $\mathrm{stopgrad}(y_{soft})$ 不给梯度，只剩 $y_{soft}$ 的梯度。这是有偏估计，不是“数学上真的让 argmax 可导”，但在很多离散选择模型里非常实用。

### 8.6 VQ-VAE 的 codebook argmin 对照

VQ-VAE 中 encoder 输出 $z_e(x)$，然后从 codebook 中找最近向量：

$$
k=\arg\min_j \lVert z_e(x)-e_j\rVert_2
$$

$$
z_q(x)=e_k
$$

这里的 argmin 也不可导。如果直接用 $z_q$ 接 decoder，decoder loss 的梯度无法回到 encoder。VQ-VAE 使用 straight-through：

$$
z_{q,st}=z_e+\mathrm{stopgrad}(z_q-z_e)
$$

前向值等于 $z_q$，反向梯度流向 $z_e$。同时还需要 codebook loss 和 commitment loss，让 codebook 向 encoder 输出靠近，也让 encoder 不要无限漂移：

$$
\lVert \mathrm{sg}[z_e]-e\rVert_2^2 + \beta \lVert z_e-\mathrm{sg}[e]\rVert_2^2
$$

这和 MoE top-k routing 的共同点是：前向有离散选择，训练需要替代梯度路径。不同点是：VQ-VAE 的离散对象是 latent code，MoE 的离散对象是专家路由。

对应实验：`examples/discrete_gradient_estimators.py`。它展示
`logsumexp≈max`、softargmax、recursive soft top-k、soft F1 surrogate、hard argmax
无梯度，softmax relaxation、Gumbel-Softmax hard sample、straight-through argmax 都能让
router logits 获得梯度，并用 VQ-VAE codebook lookup 展示 encoder/codebook 的梯度路径。

## 9. 面试复写线索

如果要在面试中快速复写，应优先抓住这些不变量：

- MHA/MQA/GQA：写一个通用 attention，参数是 `num_q_heads` 和 `num_kv_heads`，检查 `num_q_heads % num_kv_heads == 0`，cache 存未 repeat 的 K/V。
- Decoder-only cache：prefill 阶段输入整段 prompt；decode 阶段输入一个 token，K/V concat 到 cache，causal mask 要考虑 `past_len`。
- RoPE：先写偶奇维 2D 旋转公式，再写 `x*cos + rotate(x)*sin`；若结果不一致，优先检查 layout。
- 长上下文 RoPE：先说明改变的是位置到角度的映射，再写 Position Interpolation 的 `position * train_len / target_len`。
- MLA：先写 expanded 版，再用结合律写 absorbed 版；明确 content path 能吸收，RoPE path 不能直接一起吸收。
- FlashAttention：不用背 kernel，先写 online softmax 的 $m,l,O$ 更新公式，再解释 v1/v2 的循环顺序和 IO 目的。
- Sparse/Linear Attention：先说明是否改变可见图，还是改变 softmax kernel；不要把它们和 exact FlashAttention 混为一谈。
- MoE：先画出 Attention 后接 FFN/MoE-FFN 的 block 结构，再解释路由、专家和负载均衡。
- 离散选择训练：先指出 argmax/top-k/argmin 不可导，再给出 logsumexp/softmax/softargmax 这类光滑近似，以及 Gumbel-Softmax、Straight-Through 等替代梯度路径。

## 10. 复杂度总表

| 机制             | 训练计算复杂度                   | 推理 cache                  | 主要收益                       | 主要代价                      |
| ---------------- | -------------------------------- | --------------------------- | ------------------------------ | ----------------------------- |
| MHA              | $O(T^2h d_h)$                    | $O(Thd_h)$ for K and V      | 表达完整，标准基线             | cache 和带宽最大              |
| MQA              | 近似同 MHA                       | $O(Td_h)$ for K and V       | 解码带宽最低                   | 可能损失部分质量              |
| GQA              | 近似同 MHA                       | $O(Th_{kv}d_h)$ for K and V | 质量/速度折中                  | 多一个分组超参                |
| RoPE             | 轻量逐元素旋转                   | 不改变 K/V 数量             | 相对位置性质好                 | layout 和长上下文缩放容易混淆 |
| MLA              | attention 仍含 $T^2$             | $O(T(r_{kv}+d_{rope}))$     | 大幅压缩 KV cache              | 实现和权重布局复杂            |
| FlashAttention   | $O(T^2d)$                        | 不直接改变 KV cache         | 减少中间矩阵和 HBM IO          | 依赖 GPU kernel 才能体现速度  |
| Sparse Attention | 常见为 $O(Tw d)$ 或结构化稀疏    | 取决于模式                  | 降低长序列可见边               | 可能丢失远距离信息            |
| Linear Attention | 常见为 $O(Td^2)$ 或 $O(Td)$ 变体 | 可递推状态                  | 不物化 $T\times T$ attention   | 不再是精确 softmax            |
| MoE-FFN          | attention 不变，FFN 条件计算     | 不改变 KV cache             | 增大参数容量                   | 路由、负载均衡和通信复杂      |
| 离散选择梯度估计 | 不改变推理复杂度                 | 不改变 KV cache             | 让 top-k/argmin 等硬选择可训练 | 通常是有偏或近似梯度          |

## 11. 当前项目代码组织

本教程对应的本地实验代码包括：

- `attention_family.py`：MHA/MQA/GQA 统一实现。
- `transformer_usage.py`：encoder-decoder、decoder-only、KV cache 调用方式。
- `positional_encoding.py`：sinusoidal、RoPE 多实现、ALiBi。
- `long_context_position.py`：RoPE 长上下文位置缩放实验。
- `mla.py`：MLA 展开版与矩阵吸收版。
- `flash_attention.py`：FlashAttention v1/v2 CPU 仿真。
- `sparse_linear_attention.py`：Sparse/Linear Attention 机制对照。
- `moe_attention.py`：Attention 后接 MoE-FFN 的路由实验。
- `discrete_gradient_estimators.py`：MoE/VQ-VAE 中离散选择的替代梯度路径。

## 12. 理解检查

1. 为什么 MQA 主要改善推理速度，而不是把训练复杂度从二次降成一次？
2. GQA 的 `num_kv_heads` 越小，cache 越小；为什么它不一定越好？
3. Cross-attention 中 Q/K/V 分别来自哪里？哪些 mask 仍然需要，哪些不需要？
4. RoPE 两个实现输出不一致时，如何区分数学错误和 layout 差异？
5. MLA 的内容部分为什么能做矩阵吸收，RoPE 部分为什么不能完全一起吸收？
6. FlashAttention 为什么是 exact attention？它到底优化的是计算复杂度还是 IO？
7. Position Interpolation 和 FlashAttention 都服务长上下文，它们分别改了系统中的哪一层？
8. Sparse Attention 和 Linear Attention 为什么不能简单说成 FlashAttention 的替代品？
9. MoE 通常替换 Transformer block 中的哪一部分？它为什么可以和 MLA 同时使用？
10. 为什么 top-k/argmax/argmin 会切断梯度？Straight-Through 为什么只是替代估计而不是让离散操作真正可导？
