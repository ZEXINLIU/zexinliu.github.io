---
layout: post
title: "Modern Attention for LLMs: KV Cache, RoPE, MLA, FlashAttention, Sparse/Linear Attention, and MoE"
date: 2026-02-22 00:00:00
description: 从 Q/K/V 调用接口、MHA/MQA/GQA 与 KV cache 出发，梳理 RoPE/ALiBi/RMSNorm、MLA matrix absorption、FlashAttention online softmax、Sparse/Linear Attention、MoE routing 和 discrete gradient estimators。
tags: attention transformer large-language-models inference
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

本文重点包括：

1. **MHA、MQA、GQA 的变化落在 K/V head 和 KV cache。** 标准 attention 公式仍然是 $\mathrm{softmax}(QK^\top/\sqrt{d_h})V$；自回归解码成本主要受 `num_q_heads`、`num_kv_heads`、`repeat_kv` 的位置影响，也取决于 cache 中保存紧凑 K/V 还是 repeat 后的 K/V。
2. **Encoder-decoder、decoder-only、cross-attention 共享一个 attention 核心，却有三套 Q/K/V 来源和 mask 语义。** Cross-attention 是 `Q=decoder, K/V=encoder`；decoder-only 增量解码时只输入一个新 token，但依靠 KV cache 看到全部历史，mask 的 diagonal 也必须随 `past_len` 改变。
3. **位置编码要沿着“位置 $\rightarrow$ 相位 $\rightarrow$ QK score”追踪。** 正弦绝对位置编码把多频率相位加进 hidden state，外推时模型要解释训练外的绝对相位组合；RoPE 只在 Q/K 点积前旋转，长上下文问题集中到相对相位差和角度映射。RoPE 的偶奇维、向量化、复数、split-half 写法也会在这里对齐。
4. **MLA 同时处理 latent 尺度、cache 形态和矩阵吸收边界。** RMSNorm 先稳定 $c^Q,c^{KV}$ 的尺度；content path 可以把 $W^{UQ}(W^{UK})^\top$ 和 $W^{UV}W^O$ 吸收掉；共享 RoPE key $k^R$ 没有 head 维度，但旋转矩阵依赖位置，无法并成同一个固定吸收矩阵。
5. **FlashAttention 保持 exact attention，优化的是 online softmax 状态和 HBM/SRAM 读写。** 一行 attention 可以只维护最大值 $m$、归一化分母 $l$ 和输出累积量 $O$，逐 block 合并完整 softmax；v1/v2 的差异主要体现在循环顺序、状态写回次数和 GPU kernel 的工作划分。
6. **长上下文、Sparse Attention、Linear Attention 分别改三件不同的事。** RoPE 扩展改位置到角度的映射；Sparse Attention 改 token 可见图；Linear Attention 改 softmax kernel 或计算结合方式。它们都服务长序列，但牺牲点和适用边界不同。
7. **MoE 通常接在 attention 之后，负责条件计算和参数容量。** Attention 先完成跨 token 信息混合，MoE-FFN 再让不同 token 进入不同专家；因此 MLA、FlashAttention 和 MoE 可以同时出现，前两者处理 attention/cache/IO，MoE 处理 FFN 容量和专家分工。
8. **离散选择把 MoE 路由、categorical sampling、VQ-VAE codebook lookup 连接到同一个训练问题。** Top-1/Top-k、argmax、argmin 的前向结果是硬选择，反向不能直接传梯度；Gumbel-Softmax、soft relaxation、Straight-Through estimator 和 soft F1 surrogate 都是在不同偏差/稳定性之间取舍。

## 目录

- [技术发展路径](#技术发展路径)
- [1. Scaled Dot-Product Attention](#1-scaled-dot-product-attention)
- [2. MHA、MQA、GQA](#2-mhamqagqa)
- [3. Transformer Encoder-Decoder 与 Decoder-Only](#3-transformer-encoder-decoder-与-decoder-only)
- [4. 位置编码](#4-位置编码)
- [5. MLA：从 KV cache 压缩到矩阵吸收](#5-mla从-kv-cache-压缩到矩阵吸收)
- [6. FlashAttention：exact attention 的 IO 优化](#6-flashattentionexact-attention-的-io-优化)
- [7. Sparse / Linear Attention：改变可见图或核函数](#7-sparse-linear-attention改变可见图或核函数)
- [8. MoE 与 Attention 的关系](#8-moe-与-attention-的关系)
- [9. 快速复写线索](#9-快速复写线索)
- [10. 复杂度总表](#10-复杂度总表)
- [11. 当前项目代码组织](#11-当前项目代码组织)
- [12. 理解检查](#12-理解检查)

## 技术发展路径

| 时间       | 技术                       | 主要出处                                                                                                                                                     | 本项目关注点                                                                     |
| ---------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 2013-08-15 | Straight-Through estimator | [Bengio et al.](https://arxiv.org/abs/1308.3432)                                                                                                             | 为离散/随机神经元估计或传播梯度，包含 straight-through 思路                      |
| 2015-02-11 | BatchNorm                  | [Batch Normalization](https://arxiv.org/abs/1502.03167)                                                                                                      | 按 batch 统计特征均值/方差，说明它和 token 内归一化的切分角度不同                |
| 2016-07-21 | LayerNorm                  | [Layer Normalization](https://arxiv.org/abs/1607.06450)                                                                                                      | 按单个样本/Token 的 hidden 维统计，适合 Transformer 中的 token 表示              |
| 2016-11-02 | Concrete distribution      | [Concrete Distribution](https://arxiv.org/abs/1611.00712)                                                                                                    | categorical 离散变量的连续松弛                                                   |
| 2016-11-03 | Gumbel-Softmax             | [Categorical Reparameterization](https://arxiv.org/abs/1611.01144)                                                                                           | 用可微的 Gumbel-Softmax 分布近似 categorical sample                              |
| 2017-01-23 | Sparsely-Gated MoE         | [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)                                                                                       | 用条件计算扩大参数容量，每个样本只激活部分专家                                   |
| 2017-06-12 | Transformer / MHA          | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                | Scaled dot-product attention、encoder-decoder、并行 self-attention               |
| 2017-11-02 | VQ-VAE                     | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)                                                                                  | 用向量量化离散 latent，并用 straight-through 让 encoder 可训练                   |
| 2018-06-11 | GPT-style decoder-only     | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 只保留 causal decoder，用 next-token prediction 训练                             |
| 2019-10-16 | RMSNorm                    | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)                                                                                     | 和 LayerNorm 一样按 token hidden 维统计，但去掉 centering，只保留 RMS re-scaling |
| 2019-11-06 | MQA                        | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)                                                                | 所有 query head 共享一组 K/V，减少增量解码带宽                                   |
| 2020-04-10 | Longformer                 | [Longformer](https://arxiv.org/abs/2004.05150)                                                                                                               | 用局部窗口和全局 token 做长文档 sparse attention                                 |
| 2020-06-29 | Linear Transformer         | [Transformers are RNNs](https://arxiv.org/abs/2006.16236)                                                                                                    | 用核函数重写 attention，支持线性复杂度递推                                       |
| 2020-07-28 | BigBird                    | [BigBird](https://arxiv.org/abs/2007.14062)                                                                                                                  | 用局部、随机、全局边组合 sparse attention，并分析表达能力                        |
| 2020-09-30 | Performer                  | [Performer](https://arxiv.org/abs/2009.14794)                                                                                                                | 用 FAVOR+ 随机特征近似 softmax attention                                         |
| 2021-01-11 | Switch Transformer         | [Switch Transformer](https://arxiv.org/abs/2101.03961)                                                                                                       | 简化 MoE 路由，每个 token 选择一个专家                                           |
| 2021-04-20 | RoPE                       | [RoFormer](https://arxiv.org/abs/2104.09864)                                                                                                                 | 用旋转把绝对位置编码进 Q/K，并在点积里体现相对位置                               |
| 2021-08-27 | ALiBi                      | [Train Short, Test Long](https://arxiv.org/abs/2108.12409)                                                                                                   | 不加位置 embedding，而是给 score 加线性距离惩罚                                  |
| 2022-05-27 | FlashAttention             | [FlashAttention](https://arxiv.org/abs/2205.14135)                                                                                                           | IO-aware exact attention，避免物化完整 attention matrix                          |
| 2023-05-22 | GQA                        | [GQA](https://arxiv.org/abs/2305.13245)                                                                                                                      | 在 MHA 和 MQA 之间折中 KV head 数量                                              |
| 2023-06-27 | Position Interpolation     | [PI](https://arxiv.org/abs/2306.15595)                                                                                                                       | 将超长位置线性压回训练上下文范围，缓解 RoPE 外推                                 |
| 2023-07-17 | FlashAttention-2           | [FlashAttention-2](https://arxiv.org/abs/2307.08691)                                                                                                         | 改善 work partitioning，减少 non-matmul FLOPs 和 shared memory 通信              |
| 2023-09-01 | YaRN                       | [YaRN](https://arxiv.org/abs/2309.00071)                                                                                                                     | 对 RoPE context extension 做更高效的频率缩放和微调策略                           |
| 2024-01-11 | DeepSeekMoE                | [DeepSeekMoE](https://arxiv.org/abs/2401.06066)                                                                                                              | 通过更细粒度专家与共享专家增强 MoE 专家分工                                      |
| 2024-02-21 | LongRoPE                   | [LongRoPE](https://arxiv.org/abs/2402.13753)                                                                                                                 | 面向百万级上下文的 RoPE 扩展与搜索策略                                           |
| 2024-05-07 | MLA                        | [DeepSeek-V2](https://arxiv.org/abs/2405.04434)                                                                                                              | 用 latent KV 压缩 cache，并配合矩阵吸收提升推理效率                              |
| 2024-07-11 | FlashAttention-3           | [FlashAttention-3](https://arxiv.org/abs/2407.08608)                                                                                                         | 面向 Hopper GPU 的异步流水、warp specialization、FP8                             |
| 2024-12-27 | DeepSeek-V3                | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)                                                                                             | 继续使用 MLA，并在训练系统上扩展 FP8/MoE 等工程策略                              |
| 2026-03-05 | FlashAttention-4           | [FlashAttention-4](https://arxiv.org/abs/2603.05451)                                                                                                         | 面向 Blackwell GPU 的算法与 kernel pipeline co-design                            |

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

除以 $\sqrt{d_h}$ 的理由不能只说“防止 logits 太大”。更底层的问题是：未缩放的点积会让 softmax 很快进入饱和区，而 attention 里的 softmax 是内部路由，不像分类交叉熵那样总能直接得到 $p-y$ 形式的梯度。

先看点积尺度。假设 $q_i,k_i$ 已经经过 LayerNorm 和近似保方差的线性投影，所以每个维度大致满足：

$$
\mathbb{E}[q_i]=\mathbb{E}[k_i]=0,\quad
\mathrm{Var}(q_i)\approx\mathrm{Var}(k_i)\approx 1
$$

这不是说训练中每个 token 永远严格满足独立同分布，而是初始化和归一化设计希望把每个 head 维度放在相近量级上。若进一步把不同维度近似看作独立，则：

$$
s=q^\top k=\sum_{i=1}^{d_h}q_i k_i
$$

$$
\mathbb{E}[q_i k_i]=0,\quad
\mathrm{Var}(q_i k_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]\approx 1
$$

因此：

$$
\mathrm{Var}(s)\approx d_h
$$

也就是说，head_dim 越大，未缩放 score 的典型幅度越大。除以 $\sqrt{d_h}$ 后：

$$
\mathrm{Var}\left(\frac{q^\top k}{\sqrt{d_h}}\right)\approx 1
$$

这一步的目标是让 softmax 输入在训练早期保持可用尺度，而不是让某几个偶然较大的点积直接支配整行 attention。

再看 softmax 梯度。设：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

对任意 $i,j$：

$$
\frac{\partial p_i}{\partial z_j}
=p_i(\delta_{ij}-p_j)
$$

写成矩阵就是：

$$
J_{\mathrm{softmax}}=\mathrm{diag}(p)-pp^\top
$$

主对角线元素是：

$$
J_{ii}=p_i(1-p_i)
$$

非主对角线元素是：

$$
J_{ij}=-p_i p_j,\quad i\ne j
$$

如果 logits 的最大值比其他位置大很多，softmax 会接近 one-hot。设赢家位置为 $c$，则 $p_c\approx 1$，其他 $p_j\approx 0$。这时：

$$
J_{cc}=p_c(1-p_c)\approx 0
$$

$$
J_{jj}=p_j(1-p_j)\approx 0,\quad
J_{ij}=-p_i p_j\approx 0
$$

整块 Jacobian 都接近 0。若 loss 对 attention 概率的梯度记为 $g_i=\partial L/\partial p_i$，则：

$$
\frac{\partial L}{\partial z_j}
=\sum_i g_i\frac{\partial p_i}{\partial z_j}
=p_j\left(g_j-\sum_i p_i g_i\right)
$$

当 $p$ 已经接近 one-hot，非赢家位置被 $p_j\approx 0$ 直接压掉；赢家位置又因为 $\sum_i p_i g_i\approx g_c$，也接近 0。于是模型很难通过微调 logits 改变 attention 分布。缩放项的作用就是把 score 方差从 $O(d_h)$ 拉回 $O(1)$，让 softmax 不至于在初始化或训练早期过早饱和。

核心实现对应下面几行：

```python
scores = torch.matmul(q, k.transpose(-2, -1))
scores = scores / math.sqrt(q.shape[-1])
if attention_mask is not None:
    scores = scores.masked_fill(attention_mask, float("-inf"))
probs = torch.softmax(scores, dim=-1)
out = torch.matmul(probs, v)
```

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

| 类型 | Query heads |             KV heads | cache 每 token 元素数 | 直觉                                   |
| ---- | ----------: | -------------------: | --------------------: | -------------------------------------- |
| MHA  |       $h_q$ |                $h_q$ |             $2h_qd_h$ | 每个 query head 有独立 K/V，表达最完整 |
| GQA  |       $h_q$ | $1\lt h_{kv}\lt h_q$ |          $2h_{kv}d_h$ | 一组 K/V 服务一组 query heads          |
| MQA  |       $h_q$ |                  $1$ |                $2d_h$ | 所有 query heads 共享一组 K/V          |

注意这里的 cache 每 token 元素数只统计 K/V。训练时仍然要计算 $QK^\top$，所以 MQA/GQA 不会把 attention 的二次复杂度变成线性；它们主要降低自回归解码中的 KV cache 体积和读带宽。

### 2.2 `repeat_kv` 不应该污染 cache

代码中最容易写错的一点是：GQA/MQA 的 K/V 在 cache 里应该保持紧凑，只在计算 attention score 前临时 repeat 到 query head 数。下面这个函数就是 `examples/attention_family.py` 里的核心：

```python
def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    # hidden 是应该存进 KV cache 的紧凑形态：
    #   (batch, seq_len, num_kv_heads, head_dim)
    # repeat 后只用于本次 attention matmul：
    #   (batch, seq_len, num_q_heads, head_dim)
    if repeats == 1:
        return hidden

    batch, seq_len, num_kv_heads, head_dim = hidden.shape
    hidden = hidden[:, :, :, None, :].expand(
        batch, seq_len, num_kv_heads, repeats, head_dim
    )
    return hidden.reshape(batch, seq_len, num_kv_heads * repeats, head_dim)


k_for_attn = repeat_kv(k_cache_compact, num_q_heads // num_kv_heads)
v_for_attn = repeat_kv(v_cache_compact, num_q_heads // num_kv_heads)
```

如果把 repeat 后的 K/V 存进 cache，就把 MQA/GQA 的推理收益抵消了。也就是说，`repeat_kv` 是矩阵乘法前的视图/广播逻辑，不是 cache 逻辑。

### 2.3 为什么 GQA 是折中方案

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

训练阶段没有 cache，`target_len == source_len`，普通上三角 mask 就够了：

```python
def causal_mask_for_training(seq_len: int, device: torch.device) -> torch.Tensor:
    # True 表示要屏蔽的位置。diagonal=1 会屏蔽主对角线右上方，
    # 所以第 i 个 token 可以看见 0..i，包括自己。
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
```

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

```python
def causal_mask_with_cache(
    target_len: int,
    source_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    # source_len = past_len + target_len。
    # 如果 decode 时 target_len=1, source_len=past_len+1，
    # diagonal=1 会错误屏蔽历史后面的列；diagonal=1+past_len
    # 才表示“只屏蔽当前 query 之后的未来 key”。
    return torch.triu(
        torch.ones(target_len, source_len, device=device, dtype=torch.bool),
        diagonal=1 + past_len,
    )
```

对应实验：`examples/transformer_usage.py`。

## 4. 位置编码

### 4.1 为什么需要位置：从顺序信息到多频率相位坐标

如果 self-attention 不加入任何位置信息，输入序列被整体重排后，输出也会以同样方式重排。也就是说，模型能看到 token 内容，却没有一个独立信号告诉它“这个 token 在第几个位置”。对双向 encoder 来说，这会直接丢失顺序；对 causal decoder 来说，mask 虽然规定了可见范围，但模型仍然需要知道距离、局部邻近关系和绝对/相对顺序。

位置编码的目标不是让“每个频率维度都能单独区分所有位置”。单个三角频率一定会周期性重复，它只能提供一个相位坐标。真正有用的是一组不同频率共同形成多尺度坐标：

$$
\theta_i(m)=m\omega_i
$$

其中 $m$ 是 token 位置，$\omega_i$ 是第 $i$ 个频率的角速度。对应周期是：

$$
T_i=\frac{2\pi}{\omega_i}
$$

第 $i$ 个频率上的位置坐标是：

$$
(\sin\theta_i(m),\cos\theta_i(m))
$$

固定某个频率 $i$ 时，位置差 $\Delta=m-n$ 对应相位差：

$$
\Delta\theta_i=\Delta\omega_i
$$

如果 $\Delta\theta_i$ 接近 $2\pi$ 的整数倍，这个频率看起来就像“绕了一圈又回到原处”。这里说的“周期性混叠/别名”是借用信号处理里的 aliasing 直觉：单个周期函数无法单独区分相差一个或多个周期的位置；它不是 Transformer 论文里单独定义的专有术语。如果 $\omega_i$ 很小，那么短距离上的 $\Delta\theta_i$ 也很小，这个频率对局部位置变化不敏感。

因此三角位置编码需要高频和低频同时存在：

- 高频维度：$\omega_i$ 大、周期 $T_i$ 短，局部位置变化会产生明显相位差，适合分辨近邻顺序；问题是很快绕圈，单看这个频率时，长距离上容易出现周期性混叠。
- 低频维度：$\omega_i$ 小、周期 $T_i$ 长，长距离内不容易绕回，适合提供更慢变化的全局坐标；问题是短距离变化很小，局部分辨率弱。
- 多频率组合：不要求每个频率都唯一标识位置，而是让不同频率在不同尺度上互补。一个位置最终由所有频率上的相位组合表示。

对应到代码，频率和周期可以直接从 `inv_freq` 看出来：

```python
inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
periods = 2 * math.pi / inv_freq
angles = positions[:, None] * inv_freq[None, :]
```

`inv_freq[0]` 最大，旋转最快；越往后频率越低，周期越长。理解这一点后，再看“上下文扩展”和“外推”，问题就会变成：训练时模型见过哪些相位组合，推理时新的上下文长度会把哪些频率推到训练没覆盖的区域。

`base` 控制这组频率覆盖的尺度范围。原始 Transformer 的正弦位置编码使用：

$$
\omega_i=base^{-2i/d}
$$

当 $i=0$ 时，$\omega_0=1$，周期是 $2\pi$ 个 token；当 $i$ 接近 $d/2$ 时，$\omega_i$ 接近 $1/base$，周期接近 $2\pi\cdot base$。所以 `base=10000` 大致让周期从几个 token 跨到几万 token，形成按对数间隔排列的多尺度相位坐标。

原版 Transformer base 模型使用 $d_{model}=512$，big 模型使用 $d_{model}=1024$；位置编码维度和 $d_{model}$ 相同。以 $d_{model}=512$ 为例，偶奇维组成 256 个 sin/cos 频率对，周期大致从 $2\pi$ token 到 $10000\cdot 2\pi$ token。若改成 $d_{model}=1024$，频率范围基本不变，但频率对变成 512 个，对数频率网格更密。

这和原论文里的任务长度不是同一个量级。Transformer 论文做的是机器翻译，训练样本是句子对；论文提到 batch 里约有 25000 个 source tokens 和 25000 个 target tokens，但这不是单条序列的 context window。也就是说，最慢频率接近 $1/10000$、周期接近 $6.28\times 10^4$ token，远大于当时句子级翻译样本的典型长度。它不是为了让低频在训练长度里覆盖多个周期，而是把最低频做成很慢变化的全局坐标。

Transformer 选择固定正弦/余弦编码的出发点有两个：第一，不引入额外可学习参数；第二，论文作者希望它有机会外推到训练时没见过的序列长度。这里的外推直觉来自三角函数加法公式：固定偏移 $k$ 时，$PE(m+k)$ 可以由 $PE(m)$ 通过同频率下的线性变换表示，因为 $\sin(\theta+k\omega)$ 和 $\cos(\theta+k\omega)$ 都能写成 $\sin\theta,\cos\theta$ 的线性组合。

这里不要把设计目标理解成“每个频率在目标长度内都不能重复”。高频本来就会重复，它负责局部分辨；低频通常不需要在训练长度内覆盖多个周期，反而常常作为慢变化的全局坐标。如果最低频在训练长度内绕了很多圈，它也会失去长距离锚点作用。更合理的直觉是：给定一个预期上下文长度 $L$，希望最快频率能看清局部变化，最慢频率的周期和 $L$ 同量级或更大，中间频率按几何级数填满尺度空隙。

因此 `base` 不是严格从 $L$ 推导出的唯一答案，而是一个控制频率范围的超参数。`base` 太小，所有频率都偏快，长距离上更容易整体绕圈；`base` 太大，很多低频维度在训练长度内几乎不动，局部分辨贡献很弱。改变 $d_{model}$ 则主要改变频率采样密度：维度越大，中间尺度越密，模型更容易组合出不同距离尺度上的位置特征；维度越小，频率网格更稀疏，相邻尺度之间的空档更大。原始 Transformer 的 `10000` 可以理解为一个经验尺度选择，后来的 RoPE scaling、Position Interpolation、YaRN、LongRoPE 本质上都在重新分配“位置长度”和“频率尺度”之间的关系。

### 4.2 正弦绝对位置编码：把多频率坐标加到输入

把上一节的频率定义代入 `base=10000`，原始 Transformer 使用固定正弦位置编码：

$$
PE(pos,2i)=\sin(pos/10000^{2i/d})
$$

$$
PE(pos,2i+1)=\cos(pos/10000^{2i/d})
$$

它被加到 token embedding 上，因此位置信息进入后续所有线性层。它的优点是简单、不增加参数；缺点是位置是“混入输入表示”的，后续层很难显式控制位置如何影响 QK 点积。

核心实现就是先构造几何频率，再把偶数维填成 sin、奇数维填成 cos：

```python
def sinusoidal_positions(seq_len: int, dim: int, base: float = 10000.0):
    positions = torch.arange(seq_len, dtype=torch.float32)[:, None]
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions * inv_freq[None, :]

    table = torch.zeros(seq_len, dim)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


x = token_embedding + sinusoidal_positions(seq_len, d_model)
```

### 4.3 正弦绝对位置编码的外推问题

正弦位置编码有一个容易被误解的点：公式可以计算任意位置，不等于模型自然具备可靠的长上下文外推能力。沿用前面的频率定义：

$$
\omega_i=10000^{-2i/d}
$$

则绝对位置编码的角度是：

$$
\theta_i(m)=m\omega_i
$$

位置向量由 $\sin\theta_i(m)$ 和 $\cos\theta_i(m)$ 组成。数学上 $m$ 可以继续增大，但训练时模型只见过 $0\le m\lt L$ 的相位组合。外推时的具体问题分两类：

- 高频维度在训练长度内可能已经绕过很多圈，继续增加上下文会带来更多周期性重复；同一个频率上，相距一个周期的两个位置会非常相似。
- 低频维度在训练长度内可能只走过很短一段弧，模型学到的是这一段局部变化；推理时如果进入更远位置，它会看到训练中没覆盖过的新相位区域。

更关键的是，绝对位置编码先加到输入：

$$
\tilde{x}_m=e_m+PE(m)
$$

再进入 Q/K 投影。单层 attention score 中位置项会混在：

$$
s_{mn}=(e_m+PE(m))W_QW_K^\top(e_n+PE(n))^\top
$$

展开后包含：

$$
e_m A e_n^\top
+e_m A PE(n)^\top
+PE(m)A e_n^\top
+PE(m)A PE(n)^\top,\quad A=W_QW_K^\top
$$

这里没有天然保证 score 只依赖 $n-m$。模型可以学到某些绝对位置相位组合和任务模式的关联；当推理长度超过训练长度，新的 $PE(m)$ 虽然仍然有界，但它和内容项、投影矩阵 $A$ 的组合没有在训练中被约束过。问题不在“公式算不出来”，而在“模型没有学过如何解释这些相位组合”。

这不等于“正弦绝对位置编码完全没有相对位置信息”。在纯位置向量之间，三角函数本身确实包含相对位移结构：

$$
PE(m)PE(n)^\top
=\sum_i \cos((m-n)\omega_i)
$$

并且固定偏移 $k$ 时，$PE(m+k)$ 可以由 $PE(m)$ 在线性空间中表示。因此模型理论上可以从绝对位置编码中推导相对关系。区别在于：这种相对关系是隐式可学习/可利用的，不是像 RoPE 那样直接把 $R_{n-m}$ 放进 QK 点积，也不是像 ALiBi 那样直接给 score 加一个只依赖距离的 bias。

从这个公式看，正弦绝对位置编码的扩展思路大致有三类：

- 改角度映射：把 $\theta_i(m)=m\omega_i$ 改成 $\theta_i(f(m))=f(m)\omega_i$，或重新选择 $\omega_i$，让目标长度内的位置落到更合适的相位范围。但因为 $PE(m)$ 已经进入所有后续投影和 FFN 表示，这类修改通常需要继续训练或微调来适配。
- 改位置进入模型的位置：不用把绝对位置直接加进输入，而改成 RoPE 这类只作用在 Q/K 的机制，或 ALiBi 这类直接作用在 score 上的机制，让外推问题更集中地出现在 attention score。
- 改训练分布：直接在更长上下文上训练或微调，让模型见过新的绝对位置组合。

### 4.4 RoPE 的核心：位置进入 QK 点积

参考：https://zhuanlan.zhihu.com/p/647109286

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

其中 $m$ 是位置。关键点在两个旋转向量点积时：

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k
$$

点积自然依赖相对位移 $n-m$。这就是“RoPE 用绝对位置旋转 Q/K，却在 score 中体现相对位置”的核心。

和正弦绝对位置编码相比，RoPE 的位置不再先污染整个 hidden state，而是只在 Q/K 点积前进入：

$$
s_{mn}=(R_m q_m)^\top(R_n k_n)=q_m^\top R_{n-m}k_n
$$

因此 RoPE 的长上下文问题更集中：训练长度 $L$ 限制了模型见过的相对距离 $\lvert n-m\rvert$ 和相对旋转 $R_{n-m}$；扩展上下文时，主要要处理的是位置到旋转角度的映射，而不是所有层如何解释一个新的输入位置向量。

### 4.5 多实现视角：公式等价和布局等价是两件事

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

然后再用 `rotate_half`。

把几个实现压缩到核心代码，就是下面这组对照：

```python
def apply_rope_interleaved_pair(x, cos, sin):
    # interleaved layout: (x0, x1), (x2, x3), ...
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def rotate_every_pair_interleaved(x):
    # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1)
    return rotated.reshape_as(x)


def apply_rope_interleaved_vector(x, cos, sin):
    # 和直接公式相同，只是写成 x*cos + rotate(x)*sin。
    cos_full = cos.repeat_interleave(2, dim=-1)
    sin_full = sin.repeat_interleave(2, dim=-1)
    return x * cos_full + rotate_every_pair_interleaved(x) * sin_full


def apply_rope_interleaved_complex(x, cos, sin):
    # 把 (x0, x1) 看作复数 x0 + i*x1，再乘以 e^{i theta}。
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
    )
    phase = torch.complex(cos.float(), sin.float())
    return torch.view_as_real(x_complex * phase).flatten(-2).to(x.dtype)


def rotate_half(x):
    # split-half layout: [x_left, x_right] -> [-x_right, x_left]
    x_left, x_right = x.chunk(2, dim=-1)
    return torch.cat([-x_right, x_left], dim=-1)


def apply_rope_split_half(x, cos, sin):
    # 这个写法要求输入已经是 [x0, x2, ..., x1, x3, ...] 的布局。
    return x * cos + rotate_half(x) * sin
```

`apply_rope_interleaved_pair`、`apply_rope_interleaved_vector`、`apply_rope_interleaved_complex` 的数学旋转对都是相邻维度；`apply_rope_split_half` 的旋转对跨越前后半维。把 split-half 代码直接喂 interleaved tensor，输出不同是预期现象。

### 4.6 ALiBi 的位置观

ALiBi 不把位置向量加到 embedding，也不旋转 Q/K，而是直接给 attention score 加距离惩罚：

$$
score_{h,i,j}=\frac{q_{h,i}k_{h,j}^{\top}}{\sqrt{d_h}}-m_h(i-j)
$$

它的优势是外推直觉简单：训练短上下文时，模型已经学到“越远惩罚越大”的结构；推理长上下文时继续沿用。代价是表达形式更受约束，不像 RoPE 那样在 Q/K 子空间里保留更丰富的相对相位关系。

对应实验：`examples/positional_encoding.py`。

### 4.7 长上下文位置扩展：角度映射和外推边界

RoPE 的每个旋转维度都有一个频率：

$$
\omega_i=10000^{-2i/d}
$$

位置 $m$ 到角度的原始映射是：

$$
\theta_i(m)=m\omega_i
$$

两个位置 $m,n$ 在第 $i$ 个旋转维度上的相对相位差是：

$$
\theta_i(n)-\theta_i(m)=(n-m)\omega_i
$$

这解释了 RoPE 的两个性质。第一，attention score 能看到相对位移，因为点积里出现的是 $n-m$。第二，长上下文外推会出问题：如果训练上下文长度是 $L$，模型主要在 $\lvert n-m\rvert\lt L$ 对应的相对相位组合上学习；推理长度变成 $L'\gg L$ 时，直接使用 $\theta_i(m)=m\omega_i$ 会引入训练中没有覆盖过的更大距离和相位组合。

Position Interpolation 改的是 RoPE 的位置函数 $f(m)$：

$$
\theta_i^{PI}(m)=f(m)\omega_i,\quad
f(m)=m\frac{L}{L'}
$$

这样 $m=L'$ 附近的位置会被压回训练长度 $L$ 附近的角度范围。代价也直接从公式里看得出来：真实距离 $\Delta=m-n$ 会变成：

$$
\Delta'=\Delta\frac{L}{L'}
$$

如果 $L'=8L$，长上下文里的 8 个 token 间隔在 RoPE 角度里只相当于训练时的 1 个 token 间隔。它减少了训练外相位，但也压缩了局部分辨率。

YaRN / LongRoPE 这类方法会进一步避免所有频率统一乘 $L/L'$。更抽象地写，可以让每个频率维度使用自己的缩放：

$$
\theta_i^{scaled}(m)=m\alpha_i\omega_i
$$

其中 $\alpha_i$ 可以随频率维度变化，也可以区分短上下文区域和长上下文区域。这样做的目的不是改变 attention 公式，而是在“短距离分辨率”和“长距离相位不越界”之间做更细的分配。

这些 $\alpha_i$ 通常不是把一组可学习参数丢进模型、靠普通反向传播从零学出来。更常见的做法是：

- YaRN：根据目标扩展倍数、频率波长和若干超参数构造一个分段/平滑 ramp。短波长、高频维度更偏向插值缩放，长波长、低频维度更多保留原始外推；再配合少量继续训练让模型适应新的相位分布。
- LongRoPE：把不同维度、不同位置范围的缩放因子作为搜索对象，用搜索得到的非均匀缩放策略兼顾短上下文保真和长上下文扩展；搜索后仍通常需要继续训练或微调。

所以 PI 是“所有频率共享一个缩放因子”，YaRN / LongRoPE 则是“缩放因子随频率、位置区域或搜索策略变化”。这就是它们能更细地分配短距离分辨率和长距离相位范围的原因。

对应核心代码如下：

```python
def rope_inv_freq(dim: int, base: float = 10000.0):
    return base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)


def rope_angles(positions: torch.Tensor, inv_freq: torch.Tensor):
    # 原始 RoPE: theta_i(m) = m * omega_i
    return positions.float()[:, None] * inv_freq[None, :]


def position_interpolation_positions(positions, train_context, target_context):
    # f(m) = m * L / L'
    return positions.float() * (train_context / target_context)


def scaled_rope_angles(positions, inv_freq, per_frequency_scale):
    # YaRN / LongRoPE 类方法可以理解为给不同频率不同 alpha_i。
    return positions.float()[:, None] * (inv_freq * per_frequency_scale)[None, :]
```

最容易混淆的几个方向：

- 直接外推：不改 RoPE，位置继续增长。实现简单，但高频维度相位可能快速转到训练外区域。
- Position Interpolation：把新上下文位置线性压缩回训练上下文范围。例如训练长度 $L$、目标长度 $L'$，位置 $m$ 使用 $m\cdot L/L'$。
- YaRN 类方法：不是所有频率都用同一个缩放，而是按频率波长和扩展倍数构造分段/平滑缩放，并配合少量微调。
- LongRoPE 类方法：把短上下文保真和长上下文扩展作为联合目标，通过搜索得到更细粒度的位置缩放策略。

这几类方法都在改 RoPE 的位置映射，而不是改 attention 的 Q/K/V head 组织，也不是 FlashAttention 那种 IO 优化。它们通常可以和 GQA、MLA、FlashAttention 同时出现。

对应实验：`examples/long_context_position.py`。它展示了原始 RoPE、Position Interpolation、YaRN-like 频率缩放在相同位置上的角度变化。实验目标是把“为什么长上下文要改角度映射”这件事跑出来，而不是复现完整论文训练 recipe。

## 5. MLA：从 KV cache 压缩到矩阵吸收

### 5.1 MLA 在压缩什么

MLA 的目标不是把 attention matrix 低秩近似掉，而是让 K/V 的生成经过一个低维 latent cache。以 DeepSeek-V2 风格记号表示：

先固定本节的符号约定：MLA 小节采用更贴近 PyTorch 的 row-vector 写法，hidden state 和 latent 都看成行向量，线性投影写成 $hW$。很多论文会用 column-vector 写成 $Wh$，两者互为转置。前面 RoPE 基础推导里写出的 $R_{n-m}$ 对应 column-vector 方向；本节把旋转矩阵放在右侧时会得到 $R_{m-n}$。相对距离符号反过来不是机制差异，只是矩阵放置方向不同。

设 $a$ 表示 attention head 下标，$H$ 表示 head 数。Query 路径：

$$
c_t^Q = \mathrm{RMSNorm}(h_t W^{DQ})
$$

$$
q_{t,a}^C=c_t^QW_a^{UQ},\quad \bar{q}_{t,a}^R=c_t^QW_a^{QR}
$$

KV 路径：

$$
c_t^{KV}=\mathrm{RMSNorm}(h_tW^{DKV})
$$

$$
k_{t,a}^C=c_t^{KV}W_a^{UK},\quad v_{t,a}=c_t^{KV}W_a^{UV}
$$

同时还有一条共享的 RoPE key 路径。横线表示还没有做位置旋转：

$$
\bar{k}_t^R=h_tW^{KR}
$$

它没有 head 下标。也就是说，$q^R$ 是每个 head 一份，$k^R$ 是所有 heads 共享一份；计算时可以临时 broadcast，但 cache 中不保存 $H$ 份。RoPE 旋转后：

$$
q_{t,a}^R=\bar{q}_{t,a}^RR_t,\quad k_t^R=\bar{k}_t^RR_t
$$

核心 shape 可以按下面理解：

$$
q^C:(B,T,H,d_{nope}),\quad q^R:(B,T,H,d_{rope})
$$

$$
k^C:(B,T,H,d_{nope}),\quad v:(B,T,H,d_v),\quad k^R:(B,T,d_{rope})
$$

attention score 拆成内容部分和位置部分：

$$
score_{m,n,a}
=
\frac{q_{m,a}^C(k_{n,a}^C)^\top + q_{m,a}^R(k_n^R)^\top}
{\sqrt{d_{nope}+d_{rope}}}
$$

对应到代码，关键点是 `k_rope` 的原始 shape 没有 head 维度，只有算位置 score 时才加一个单例 head 维度让它广播：

```python
# q_rope: (batch, seq_len, num_heads, d_rope)
# k_rope: (batch, seq_len, d_rope), shared by all heads
q_rope = apply_rope(q_rope, cos, sin)
k_rope = apply_rope(k_rope[:, :, None, :], cos, sin)  # (B, T, 1, d_rope)

# matmul broadcasts the singleton key-head dimension from 1 to num_heads.
score_rope = torch.matmul(
    q_rope.transpose(1, 2),                 # (B, H, T, d_rope)
    k_rope.transpose(1, 2).transpose(-2, -1) # (B, 1, d_rope, T)
)
```

普通 MHA 每个 token cache 约为：

$$
h(qk\_head\_dim + v\_head\_dim)
$$

DeepSeek-V2 典型配置里：

$$
128(128+128)=32768
$$

MLA cache 只保存 compressed KV 和一条共享 RoPE key：

$$
d_c + d_{rope}=512+64=576
$$

比例约为 56.9x。这个数字成立的前提就是 $k^R$ 不按 head 复制；如果把它误写成 $(B,T,H,d_{rope})$ 的 cache，MLA 的缓存优势会被错误估计。这个数字的意义不是“attention 计算少了 56.9x”，而是每个历史 token 要从 cache 里读出的 K/V 表示大幅减少了。

### 5.2 RMSNorm：和 LayerNorm 同轴，但去掉 centering

MLA 公式里有两处 RMSNorm：

$$
c_t^Q = \mathrm{RMSNorm}(h_t W^{DQ}),\quad
c_t^{KV}=\mathrm{RMSNorm}(h_tW^{DKV})
$$

它不是 MLA 压缩 cache 的核心技巧，但它会影响后续 latent 的尺度。理解 RMSNorm 前，先把常见 normalization 的“统计轴”分清楚。

设 Transformer hidden states 为：

$$
X\in\mathbb{R}^{B\times T\times d}
$$

BatchNorm 通常对每个 hidden feature 单独统计 batch 维和时间/空间维：

$$
\mu_j=\frac{1}{BT}\sum_{b,t}X_{b,t,j},\quad
\sigma_j^2=\frac{1}{BT}\sum_{b,t}(X_{b,t,j}-\mu_j)^2
$$

也就是说，同一个 batch 里不同样本、不同 token 位置会互相影响统计量。这个切分角度在 CNN 里常见，但对自回归 Transformer 不自然：当前 token 的归一化不应该依赖 future token，也不应该强依赖同 batch 里其他样本。

LayerNorm 换了统计轴。它对每个 token 独立地沿 hidden 维统计：

$$
\mu=\frac{1}{d}\mathbf{1}^{\top}x,\quad
\sigma=\sqrt{\frac{1}{d}\|x-\mu\mathbf{1}\|_2^2+\epsilon}
$$

$$
\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu\mathbf{1}}{\sigma}+\beta
$$

RMSNorm 和 LayerNorm 的切分角度一致：也是对单个 token 的 hidden 维做归一化。它删掉了减均值，只保留 root-mean-square 缩放：

$$
r=\sqrt{\frac{1}{d}\|x\|_2^2+\epsilon}
$$

$$
\mathrm{RMSNorm}(x)=g\odot\frac{x}{r}
$$

对应最小实现就是：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 token 独立沿 hidden 维计算 RMS，不混合 batch 或时间位置。
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms
```

这不是简单地“少算一个均值”而已。对不含 affine 的核心归一化 $y=x/r$，其 Jacobian 是：

$$
\frac{\partial y}{\partial x}
=
\frac{1}{r}I-\frac{1}{d r^3}xx^\top
$$

如果上游梯度是 $g=\partial L/\partial y$，则：

$$
\frac{\partial L}{\partial x}
=
\frac{g}{r}
-
\frac{x}{d r^3}(x^\top g)
$$

这个式子说明了 RMSNorm 的梯度流：和 $x$ 正交的梯度分量大致被 $1/r$ 缩放；沿着 $x$ 的径向分量会被第二项抵消。若忽略 $\epsilon$，有：

$$
\left(\frac{\partial y}{\partial x}\right)x=0
$$

这对应尺度不变性：$x$ 整体乘一个正数时，$x/r$ 不变，所以沿“整体放大/缩小”的方向不会改变输出。

LayerNorm 的 Jacobian 可以写成：

$$
\tilde{x}=x-\mu\mathbf{1},\quad
P=I-\frac{1}{d}\mathbf{1}\mathbf{1}^{\top}
$$

$$
\frac{\partial\,\mathrm{LN}(x)}{\partial x}
=
\frac{P}{\sigma}
-
\frac{1}{d\sigma^3}\tilde{x}\tilde{x}^{\top}
$$

这里 $P$ 是 centering 投影矩阵，所以：

$$
P\mathbf{1}=0
$$

这意味着 LayerNorm 对整体平移方向也不敏感：$x$ 加上 $c\mathbf{1}$，中心化后的 $\tilde{x}$ 不变。RMSNorm 没有这个投影：

$$
\left(\frac{\partial y}{\partial x}\right)\mathbf{1}
=
\frac{\mathbf{1}}{r}-\frac{x(x^\top\mathbf{1})}{d r^3}
$$

一般不等于 0。换句话说，RMSNorm 保留了 hidden 向量的均值/偏置方向对输出的影响，只消除整体尺度波动；LayerNorm 同时消除整体尺度和整体平移方向。RMSNorm 论文把这概括为：保留 re-scaling invariance，去掉 re-centering invariance。

这也解释了它在 MLA 里的位置。$hW^{DQ}$ 和 $hW^{DKV}$ 被压到低维 latent 后，后续 score 会包含：

$$
(c^QW^{UQ})(c^{KV}W^{UK})^\top
$$

如果 $c^Q$ 或 $c^{KV}$ 的范数随 token 大幅波动，score 尺度也会波动。RMSNorm 先把 latent 的 RMS 拉到稳定范围，再交给后续投影。它本身不能被吸收到固定矩阵里，因为 $1/r(x)$ 依赖当前 token 输入；但吸收发生在 RMSNorm 之后的线性矩阵之间，所以两者不冲突。

对应实验：`examples/norm_comparison.py`。它打印 BatchNorm/LayerNorm/RMSNorm 的统计轴差异，并验证 RMSNorm Jacobian 的解析式和 autograd 一致。

### 5.3 内容路径的矩阵吸收

下面先固定某个 attention head，省略 head 下标。内容 attention 展开为：

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

### 5.4 为什么 RoPE 不能直接并进同一个吸收矩阵

内容路径能吸收，是因为中间矩阵不依赖 token 位置。对 query 位置 $m$、key 位置 $n$：

$$
score^C_{mn}
=(c_m^QW^{UQ})(c_n^{KV}W^{UK})^\top
$$

可以改写为：

$$
score^C_{mn}
=c_m^Q\left(W^{UQ}(W^{UK})^\top\right)(c_n^{KV})^\top
$$

中间的：

$$
W^{QK}=W^{UQ}(W^{UK})^\top
$$

是固定矩阵，和 $m,n$ 无关，所以可以被预先吸收。

RoPE 路径多了位置旋转。为了和 5.1 的 row-vector 约定保持一致，下面把旋转矩阵放在右侧，并先写出真实 MLA 中的共享 key 路径：

$$
q^R_{m,a}=(c_m^QW_a^{QR})R_m,\quad k^R_n=(h_nW^{KR})R_n
$$

先忽略 MLA 真实实现中 $k^R$ 来自 $h_nW^{KR}$ 而不是 $c_n^{KV}$，假设我们想把它也写成某个 latent key 路径：

$$
q^R_{m,a}=(c_m^QW^Q_{R,a})R_m,\quad k^R_{n,a}=(c_n^{KV}W^K_{R,a})R_n
$$

那么 RoPE score 是：

$$
score^R_{mn,a}
=(c_m^QW^Q_{R,a})R_mR_n^\top(W^K_{R,a})^\top(c_n^{KV})^\top
$$

利用旋转矩阵正交性，$R_mR_n^\top$ 只由相对位置决定。记：

$$
R_mR_n^\top=R_{m-n}
$$

于是：

$$
score^R_{mn,a}
=c_m^Q\left(W^Q_{R,a} R_{m-n}(W^K_{R,a})^\top\right)(c_n^{KV})^\top
$$

问题就在括号里的中间矩阵：

$$
W^Q_{R,a} R_{m-n}(W^K_{R,a})^\top
$$

它依赖相对位置 $m-n$。内容路径只有一个固定 $W^{QK}$；RoPE 路径则每一种相对距离都对应一个不同的中间矩阵。要把它“吸收”为一个固定矩阵，就必须让同一个 $W^{QK}$ 同时等于所有 $W^Q_{R,a} R_{m-n}(W^K_{R,a})^\top$，这在非退化情况下不成立。若改用 column-vector 写法，中间会出现 $R_{n-m}$，但它仍然依赖相对位置，不能吸收为一个与 $m,n$ 无关的固定矩阵。

这也解释了实现上的边界：RoPE 可以通过“按位置旋转 query、按位置旋转 key”保持可分计算，但不能像内容路径那样把两个投影矩阵合成一个与位置无关的权重矩阵，然后直接在 latent cache 上做一次固定点积。

回到 MLA 的真实路径，还有第二层障碍：

- $k^R$ 来自 $hW^{KR}$ 的共享位置 key 路径，没有 head 下标，不是从 $c^{KV}$ 通过 $W^{UK}$ 恢复出来的内容 key。
- MLA cache 有意保存 $c^{KV}$ 和 $k^R$ 两部分：前者服务内容吸收，后者服务 RoPE 位置 score。

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

本项目的 `examples/flash_attention.py` 保留了这种差异，并打印 `output block writes`。在 `seq_len=64, block=16` 的例子里，v1 写 16 次，v2 写 4 次。

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

对应实验：`examples/sparse_linear_attention.py`。它展示 sliding-window sparse attention 的可见边数量变化，以及一个确定性 feature map 的 causal linear attention。实验中 sparse/linear 输出和 dense 输出有差异，这是预期现象，因为机制被改变了。

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

对应实验：`examples/moe_attention.py`。它保留 dense causal attention，然后把 FFN 换成 top-1 MoE，并打印每个专家收到的 token 数量。这个实验的重点是看清楚：attention 负责跨 token 混合，MoE 负责每个 token 后续走哪个 FFN 专家。

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

对应实验：`examples/discrete_gradient_estimators.py`。它展示 `logsumexp≈max`、softargmax、recursive soft top-k、soft F1 surrogate、hard argmax 无梯度，softmax relaxation、Gumbel-Softmax hard sample、straight-through argmax 都能让 router logits 获得梯度，并用 VQ-VAE codebook lookup 展示 encoder/codebook 的梯度路径。

## 9. 快速复写线索

快速复写时，优先抓住这些不变量：

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

本教程对应的代码都在 `examples/`：

- `examples/attention_family.py`：MHA/MQA/GQA 统一实现。
- `examples/transformer_usage.py`：encoder-decoder、decoder-only、KV cache 调用方式。
- `examples/positional_encoding.py`：sinusoidal、RoPE 多实现、ALiBi。
- `examples/long_context_position.py`：RoPE 长上下文位置缩放实验。
- `examples/norm_comparison.py`：BatchNorm、LayerNorm、RMSNorm 的统计轴和 Jacobian 对照。
- `examples/mla.py`：MLA 展开版与矩阵吸收版。
- `examples/flash_attention.py`：FlashAttention v1/v2 CPU 仿真。
- `examples/sparse_linear_attention.py`：Sparse/Linear Attention 机制对照。
- `examples/moe_attention.py`：Attention 后接 MoE-FFN 的路由实验。
- `examples/discrete_gradient_estimators.py`：MoE/VQ-VAE 中离散选择的替代梯度路径。

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
