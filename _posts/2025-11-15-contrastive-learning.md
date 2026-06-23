---
layout: post
title: 对比学习：从样本构造到表征学习
date: 2025-11-15 00:00:00
description: 从 Triplet、InfoNCE、NT-Xent 到 MoCo、SimCLR、DINO、CLIP 的统一视角。
tags: contrastive-learning representation-learning multimodal-learning machine-learning
categories: ai-notes
featured: true
---

本文重点内容如下：

- 对比学习的主问题不是“把相似样本拉近”这句口号，而是如何定义 anchor、positive、negative 和 dictionary。SimCLR、MoCo、CLIP 的 loss 形式很像，但正负样本和候选集合完全不同。
- NCE、InfoNCE、NT-Xent、CLIP symmetric loss 都可以写成候选集合上的交叉熵；Triplet loss 是显式 margin ranking；MINE 是互信息估计器，不是主流视觉对比预训练的训练目标。
- InstDisc、MoCo、SimCLR 的核心差别是字典机制：memory bank、queue、large batch 分别在负样本规模、特征一致性和显存成本之间取不同折中。
- BYOL、SimSiam、DINO 的重点是 negative-free 如何避免坍缩：predictor、stop-gradient、momentum teacher、centering、sharpening 都是在改变优化动力学。
- ArcFace/CosFace 不是自监督对比学习，但它们把特征和类别原型放到单位超球面上，用角度 margin 强化类内紧凑和类间分离，适合与对比学习放在同一套几何语言里理解。
- 健康生理信号应用属于个人兴趣预研，不作为对比学习通识路线贯穿全文；第 6 节单独讨论 ECG-PPG 这类多模态信号如何借鉴 CLIP 式对比目标。

## 0. 阅读地图

| 章节 | 解决的问题                               | 关键词                                                      |
| ---- | ---------------------------------------- | ----------------------------------------------------------- |
| 1    | 对比学习统一建模                         | anchor、positive、negative、dictionary、temperature         |
| 2    | loss 家族如何互相连接                    | Triplet、NCE、InfoNCE、NT-Xent、MINE、CLIP loss             |
| 3    | 代表论文如何定义输入、正负样本和训练损失 | InstDisc、CPC、CMC、MoCo、SimCLR、BYOL、SimSiam、DINO、CLIP |
| 4    | 精度提升方式背后的机制                   | augmentation、projection head、queue、batch、backbone       |
| 5    | 与人脸识别 loss 的连接                   | metric learning、CosFace、ArcFace、class prototype          |
| 6    | 健康生理信号应用预研                     | ECG、PPG、HR、HRV、多模态同步                               |

## 1. 统一视角：对比学习是候选集合上的判别任务

一个对比学习系统通常先把输入编码成表征：

$$
h = f_\theta(x),
\qquad
z = \frac{g_\phi(h)}{\|g_\phi(h)\|_2}.
$$

其中，`f` 是 backbone，`g` 是 projection head，`z` 常做 L2 normalize。两个样本之间常用点积或 cosine similarity：

$$
s(i,j)=z_i^\top z_j,
\qquad
\operatorname{logit}(i,j)=\frac{s(i,j)}{\tau}.
$$

`tau` 是 temperature。`tau` 越小，softmax 越尖锐，hard negative 权重越大；`tau` 太小会让训练更敏感。

每个算法都可以从四个对象入手：

| 对象         | 典型例子                                                | 它决定了什么             |
| ------------ | ------------------------------------------------------- | ------------------------ |
| anchor/query | 一张增强后的图片、一个上下文 latent、一段文本 embedding | 以谁为中心做匹配         |
| positive/key | 同图另一种增强、未来 latent、配对文本                   | 哪些因素应保持一致       |
| negative     | 其他实例、其他时间位置、其他图文对                      | 哪些因素应被区分         |
| dictionary   | memory bank、queue、当前 batch、类别 prototype          | 候选集合规模和特征一致性 |

同一个交叉熵形式，放在不同数据构造里会学到不同不变性。SimCLR 的 positive 来自同一图像的两种增强；CPC 的 positive 来自同一序列的未来 latent；CLIP 的 positive 来自同一条图文对。loss 只是最后一步，真正的建模选择藏在正负样本定义和采样策略里。

## 2. Loss 家族：从 margin 到 softmax，再到互信息

### 2.1 Triplet loss

Triplet loss 来自 metric learning，在 FaceNet 中用于学习人脸嵌入：

$$
\mathcal{L}_{triplet}
= \max\left(0, d(a,p)-d(a,n)+m\right).
$$

`a` 是 anchor，`p` 是 positive，`n` 是 negative，`m` 是 margin。它要求正样本距离至少比负样本近 `m`。

它的优点是直观，缺点也明显：只比较一个或少量 negative，强依赖 hard negative mining；当 triplet 已满足 margin 时梯度为 0；它没有候选集合分类的概率解释。大规模自监督视觉预训练更常用 softmax 型目标，因为一个 anchor 可以同时比较大量候选。

### 2.2 Softmax contrastive loss

如果 anchor `i` 的正样本是 `j`，候选集合为 `A(i)`，最常见的对比目标是：

$$
\mathcal{L}_i
=
-\log
\frac{\exp(s(i,j)/\tau)}
{\sum_{a\in A(i)} \exp(s(i,a)/\tau)}.
$$

这就是交叉熵：模型要在候选集合里把 `j` 分类为正确答案。这里的类别不是人工语义类，而是“这个 anchor 应该匹配哪个 key”。

令

$$
p_{ia}=
\frac{\exp(s(i,a)/\tau)}
{\sum_{b\in A(i)} \exp(s(i,b)/\tau)}.
$$

当相似度是下面这个点积时：

$$
s(i,a)=z_i^\top z_a,
$$

anchor 表征的梯度为：

$$
\frac{\partial \mathcal{L}_i}{\partial z_i}
=
\frac{1}{\tau}
\left(
\sum_{a\in A(i)} p_{ia}z_a - z_j
\right).
$$

梯度下降会把 anchor 拉向正样本 `z_j`，并按 softmax 概率把它推离候选集合中的其他 key。负样本越像正样本，softmax 概率越高，梯度贡献越大；这就是 hard negative 自动加权的来源。

### 2.3 NCE 与 InfoNCE

Noise-Contrastive Estimation 最初用于未归一化统计模型的估计。它把“来自真实数据”与“来自噪声分布”写成二分类问题。设模型给出的未归一化概率为 `p_m(i|v)`，噪声分布为 `p_n(i)`，每个正样本配 `m` 个噪声样本，则一个候选来自真实分布的后验为：

$$
P(D=1|i,v)
=
\frac{p_m(i|v)}
{p_m(i|v)+m p_n(i)}.
$$

NCE 损失为：

$$
\mathcal{L}_{NCE}
=
-\mathbb{E}_{p_d}\log P(D=1|i,v)
-m\mathbb{E}_{p_n}\log \left(1-P(D=1|i,v)\right).
$$

在 InstDisc 中，NCE 用来近似全训练集实例分类的巨大分母。严格说，NCE 与 full softmax 不是逐样本梯度完全相同；在噪声样本充分、模型族合适、归一化项处理正确时，NCE 的期望目标可以一致估计原始归一化模型。对比学习实现里更常把它理解为“大规模 softmax 的采样近似”。

InfoNCE 更常见于现代对比学习。给定一个正样本和 `N-1` 个负样本：

$$
\mathcal{L}_N
=
-\mathbb{E}
\log
\frac{f(x^+,c)}
{f(x^+,c)+\sum_{r=1}^{N-1} f(x_r^-,c)}.
$$

如果最优 critic 满足密度比形式：

$$
f^*(x,c)\propto \frac{p(x|c)}{p(x)},
$$

则有互信息下界：

$$
I(x;c)\ge \log N-\mathcal{L}_N.
$$

这条式子说明候选集合越大，理论上可表达的互信息下界越高；但负样本数不是越大越无条件好，false negative、batch 构成和优化难度都会改变实际效果。

### 2.4 NT-Xent

SimCLR 的 NT-Xent 是 normalized temperature-scaled cross entropy。对 batch 中 `N` 张图片各做两种增强，得到 `2N` 个 view。对 view `i`，其正样本 `j` 是同一原图的另一种增强，其余 `2N-2` 个 view 是负样本：

$$
\ell_{i,j}
=
-\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k=1}^{2N}\mathbf{1}_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

完整 loss 对两个方向都算：

$$
\mathcal{L}_{SimCLR}
=
\frac{1}{2N}\sum_{k=1}^{N}
\left(\ell_{2k-1,2k}+\ell_{2k,2k-1}\right).
$$

NT-Xent 可以看成 InfoNCE 在 SimCLR 批内采样方式下的名字。

### 2.5 MINE

互信息可以写成 KL 散度：

$$
I(X;Y)
=
D_{KL}(P_{XY}\|P_XP_Y).
$$

MINE 使用 Donsker-Varadhan 表示构造神经估计器：

$$
I(X;Y)
\ge
\mathbb{E}_{P_{XY}}[T_\theta(x,y)]
-
\log
\mathbb{E}_{P_XP_Y}\left[e^{T_\theta(x,y)}\right].
$$

它不是候选集合分类，而是直接估计互信息下界。MINE 更适合作为分析变量依赖或信息瓶颈的工具；在视觉自监督主线中，InfoNCE/NT-Xent 更常见，因为它们的 batch 训练更稳定。

### 2.6 Loss 关系速查

| Loss              | 正样本                            | 负样本                | 形式                     | 典型方法                 |
| ----------------- | --------------------------------- | --------------------- | ------------------------ | ------------------------ |
| Triplet           | 同身份/同语义样本                 | 不同身份/不同语义样本 | hinge ranking            | FaceNet、metric learning |
| NCE               | 真实样本                          | 噪声分布样本          | 二分类 logistic          | InstDisc 的大规模近似    |
| InfoNCE           | 一个正 key                        | 候选集合其他 key      | softmax CE               | CPC、MoCo、CMC           |
| NT-Xent           | 同图另一增强                      | batch 内其他 view     | symmetric softmax CE     | SimCLR                   |
| BYOL/SimSiam loss | 另一 view 的 stop-gradient target | 无显式负样本          | cosine regression        | BYOL、SimSiam            |
| DINO loss         | teacher 输出分布                  | 无显式负样本          | teacher-student CE       | DINO                     |
| CLIP loss         | 配对图文                          | batch 内其他图文      | bidirectional softmax CE | CLIP                     |
| MINE              | 联合分布样本                      | 边缘乘积分布样本      | DV lower bound           | MI 估计                  |
| ArcFace/CosFace   | 正确类别 prototype                | 其他类别 prototype    | margin softmax CE        | 人脸识别                 |

## 3. 代表论文：输入构造、正负样本、loss 与代码

### 3.1 时间线

| 时间      | 方法       | 核心机制                                         | 核心训练信号                                |
| --------- | ---------- | ------------------------------------------------ | ------------------------------------------- |
| 2018      | InstDisc   | instance discrimination、memory bank、NCE        | 每张图识别为自身实例                        |
| 2018      | CPC        | autoregressive context、future latent prediction | 上下文在候选集合中找未来 latent             |
| 2019      | CMC        | multiview coding                                 | 一个 view 在 batch 中匹配同图另一个 view    |
| 2019      | InvaSpread | augmentation invariance、instance spreading      | 同实例增强拉近，不同实例推远                |
| 2019/2020 | MoCo v1    | queue、momentum encoder                          | query 匹配正 key，排斥 queue 中负 key       |
| 2020      | SimCLR     | 强增强、MLP head、大 batch                       | 2N 个 view 上的 NT-Xent                     |
| 2020      | MoCo v2    | MLP head、更强增强、cosine schedule              | 仍是 MoCo InfoNCE，recipe 更接近 SimCLR     |
| 2020      | BYOL       | online/target、EMA、predictor                    | online prediction 对齐 stop-gradient target |
| 2020      | SimSiam    | predictor、stop-gradient                         | 无 EMA 的 Siamese cosine regression         |
| 2021      | MoCo v3    | ViT、momentum encoder、稳定性分析                | ViT 上的 contrastive objective              |
| 2021      | DINO       | self-distillation、momentum teacher、multi-crop  | student 预测 teacher 分布                   |
| 2021      | CLIP       | 图文双塔、双向 batch softmax                     | 图到文、文到图双向检索分类                  |

### 3.2 InstDisc：memory bank + NCE

InstDisc 把每张图片视作自己的类别。设训练集有 `n` 张图，memory bank 保存每个实例的历史特征 `v_1 ... v_n`，当前图 `x_i` 的特征为 `v = f_theta(x_i)`。完整 non-parametric softmax 是：

$$
p(i|v)
=
\frac{\exp(v_i^\top v/\tau)}
{\sum_{j=1}^n \exp(v_j^\top v/\tau)}.
$$

训练目标是让图片被识别为自身实例。由于分母要扫全数据集，论文使用 NCE 近似。

**Loss 归类与等价视角。** 上面的 non-parametric softmax 是“实例作为类别”的多分类 cross entropy：第 `i` 张图的 positive class 就是它自己的实例 id，其他实例都是 negative class。InstDisc 真正的工程难点是类别数等于数据集大小，full softmax 分母太大，所以论文用 NCE 把多分类归一化问题改写成 data-vs-noise 的二分类 logistic 目标。它不是另一个完全无关的 loss，而是对巨大实例 softmax 的采样估计路径；memory bank 改变的是 dictionary 的存储与采样方式，不改变“同实例为正、其他实例为负”的判别本质。

| 项目       | InstDisc 中的定义                       |
| ---------- | --------------------------------------- |
| 输入       | 一张图片及其增强 view                   |
| anchor     | 当前 encoder 输出 `v`                   |
| positive   | memory bank 中同一图片实例的 `v_i`      |
| negative   | 从 memory bank 按噪声分布采样的其他实例 |
| dictionary | 全数据 memory bank                      |
| loss       | NCE 近似的实例分类目标                  |

教学化代码如下：

```python
def instdisc_step(images, indices, encoder, memory_bank, tau, num_neg):
    # images: 当前 batch 的增强图片
    # indices: 每张图片在训练集中的实例 id
    # memory_bank: [num_images, dim]，保存历史特征，承担大字典角色
    q = l2_normalize(encoder(images), axis=1)             # [B, D]
    pos = memory_bank[indices]                            # [B, D]
    neg_ids = sample_uniform_except(indices, num_neg)      # [B, K]
    neg = memory_bank[neg_ids]                             # [B, K, D]

    # non-parametric softmax 的未归一化分数。
    # 原论文中的归一化常数 Z 可用估计值或常数近似，避免每步扫完整训练集。
    Z = estimate_partition_constant(memory_bank, tau)
    pos_score = exp(batch_dot(q, pos) / tau) / Z           # [B]
    neg_score = exp(batch_matmul(neg, q[:, :, None])[:, :, 0] / tau) / Z

    noise_prob = 1.0 / len(memory_bank)
    pos_posterior = pos_score / (pos_score + num_neg * noise_prob)
    neg_posterior = neg_score / (neg_score + num_neg * noise_prob)

    # NCE 是 data-vs-noise 二分类：positive 来自 data，negative 来自 noise。
    loss_pos = -log(pos_posterior)
    loss_neg = -sum(log(1.0 - neg_posterior), axis=1)
    loss = mean(loss_pos + loss_neg)

    # 当前 encoder 更新后，用新特征刷新对应 memory bank 条目。
    memory_bank[indices] = momentum_update(memory_bank[indices], q)
    return loss
```

memory bank 让负样本数量远大于 batch，但 bank 中条目来自不同训练时刻，特征一致性不足。MoCo 的动量 encoder 和 queue 正是沿着这个问题继续改。

### 3.3 CPC：InfoNCE 作为未来 latent 的候选分类

CPC 先把序列编码为 latent，再用 autoregressive context `c_t` 预测未来 `z_{t+k}`。它不直接重建原始输入，而是在 latent 空间里做判别预测。

| 项目       | CPC 中的定义                             |
| ---------- | ---------------------------------------- |
| 输入       | 图像 patch、音频、文本等序列             |
| anchor     | context vector `c_t`                     |
| positive   | 同一序列未来位置 `z_{t+k}`               |
| negative   | 其他位置或其他样本的 latent              |
| dictionary | 当前 batch 或采样集合中的 future latents |
| loss       | InfoNCE，多步预测时对多个 `k` 求和       |

单个预测步的 InfoNCE 可以写成：

$$
\mathcal{L}_{t,k}
=
-\log
\frac{\exp(z_{t+k}^{\top}W_k c_t)}
{\sum_{\tilde z\in \mathcal{N}_{t,k}}\exp(\tilde z^\top W_k c_t)}.
$$

**Loss 归类与等价视角。** 这是标准 InfoNCE：`c_t` 是 query，真实未来 latent `z_{t+k}` 是 positive，候选集合中的其他 latent 是 negative。分母不是为了做生成式重建，而是在候选集合中做 softmax 分类；当 critic 近似密度比时，最小化这个 cross entropy 等价于最大化互信息下界 `log N - L_N`。CPC 的特别之处在于 positive 来自“未来预测”，而不是同图增强或跨模态配对。

代码化实现：

```python
def cpc_step(sequence, encoder, autoregressive, W, tau):
    # sequence: [B, T, ...]
    # z: 每个时间位置的 latent；c: 聚合过去信息的 context
    z = encoder(sequence)                  # [B, T, D]
    c = autoregressive(z)                  # [B, T, C]

    losses = []
    for k, W_k in enumerate(W, start=1):
        query = matmul(c[:, :-k], W_k)      # [B, T-k, D]
        pos = z[:, k:]                      # [B, T-k, D]

        # 将 batch 和时间维摊平后，其他 future latents 都可作为候选。
        q = l2_normalize(flatten(query), axis=1)  # [M, D]
        keys = l2_normalize(flatten(pos), axis=1) # [M, D]
        logits = matmul(q, keys.T) / tau          # [M, M]
        labels = arange(len(q))                   # 对角线是同一序列同一预测步的正样本
        losses.append(cross_entropy(logits, labels))

    return mean(losses)
```

CPC 的关键不是“用了 RNN”，而是把预测未来改造成候选集合分类，并用 InfoNCE 连接互信息下界。

### 3.4 CMC：跨视图 InfoNCE

Contrastive Multiview Coding 把同一对象的不同 view 作为正样本，例如不同颜色空间、不同传感器或不同增强视图。目标是让一个 view 的表征能在候选集合中找到同一实例的另一个 view。

| 项目       | CMC 中的定义                            |
| ---------- | --------------------------------------- |
| 输入       | 同一图像或对象的两个 view               |
| anchor     | view 1 的 embedding                     |
| positive   | 同一实例的 view 2 embedding             |
| negative   | batch 中其他实例的 view 2 embedding     |
| dictionary | batch 或 memory bank 中的另一 view 表征 |
| loss       | 跨视图 InfoNCE，通常双向或多 view 求和  |

公式：

$$
\mathcal{L}_{v_1\to v_2}
=
-\frac{1}{N}\sum_i
\log
\frac{\exp(z_{i}^{v_1\top}z_i^{v_2}/\tau)}
{\sum_j \exp(z_i^{v_1\top}z_j^{v_2}/\tau)}.
$$

**Loss 归类与等价视角。** CMC 是跨视图版 InfoNCE。它和 SimCLR/CLIP 的矩阵形式同构：相似度矩阵的对角线是同一实例的跨视图 positive，非对角线是 batch 内 negative。区别只在 view 的来源：CMC 的 view 可以是颜色空间、模态或其他视图；SimCLR 的 view 是两种随机增强；CLIP 的 view 是图像和文本。因此 CMC 的 loss 可以归入“paired-view softmax contrastive loss”。

代码化实现：

```python
def cmc_step(view1, view2, encoder1, encoder2, tau):
    z1 = l2_normalize(encoder1(view1), axis=1)  # [N, D]
    z2 = l2_normalize(encoder2(view2), axis=1)  # [N, D]

    # logits[i, j] 表示第 i 个 view1 与第 j 个 view2 的相似度。
    # 对角线是同一实例的跨视图 positive，非对角线是负样本。
    logits = matmul(z1, z2.T) / tau             # [N, N]
    labels = arange(len(z1))

    loss_12 = cross_entropy(logits, labels)
    loss_21 = cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)
```

CMC 的思想后来在 CLIP 中变得更清楚：只要两个模态或视图存在配对关系，就能把对齐写成矩阵对角线分类。

### 3.5 InvaSpread：增强不变性与实例分散

InvaSpread 强调两件事：同一实例的增强 view 应该 invariant，不同实例的 embedding 应该 spreading。它和 SimCLR 的精神非常接近，只是 SimCLR 后来用更系统的 augmentation、projection head 和大 batch recipe 把效果推高。

| 项目       | InvaSpread 中的定义                     |
| ---------- | --------------------------------------- |
| 输入       | 同一图片的随机增强 view                 |
| anchor     | 一个增强 view 的 embedding              |
| positive   | 同一实例的另一个增强 view               |
| negative   | batch 中其他实例                        |
| dictionary | 当前 batch                              |
| loss       | instance-level softmax / InfoNCE 类目标 |

一个常见写法是：

$$
\mathcal{L}_{InvaSpread}
=
-\frac{1}{N}\sum_i
\log
\frac{\exp(z_i^\top z_i^+/\tau)}
{\sum_j \exp(z_i^\top z_j^+/\tau)}.
$$

**Loss 归类与等价视角。** 这个目标属于 instance discrimination softmax，也可以看成同实例增强构造下的 InfoNCE。`z_i^+` 是同一实例的增强 positive，其他 `z_j^+` 是 spreading 所需的 negative。它的“装饰”是强调 invariant + spreading 的解释语言；数学上仍是候选集合交叉熵。

代码化实现可以写成单向版本：

```python
def invaspread_step(x1, x2, encoder, tau):
    z1 = l2_normalize(encoder(x1), axis=1)
    z2 = l2_normalize(encoder(x2), axis=1)

    # 第 i 个 z1 应该匹配第 i 个 z2；其他列都是不同实例。
    logits = matmul(z1, z2.T) / tau
    labels = arange(len(z1))
    return cross_entropy(logits, labels)
```

这一类方法的风险是 false negative：同一 batch 中语义相近甚至同类的样本会被当作负样本。实例级自监督默认接受这个代价，换取无需标签的大规模训练。

### 3.6 MoCo：queue + momentum encoder 的 InfoNCE

MoCo 把对比学习写成 dictionary lookup。query 来自在线 encoder `f_q`，key 来自动量 encoder `f_k`。对一个 query，正 key 是同一图片的另一种增强，负 key 来自 queue。

$$
\mathcal{L}_{MoCo}
=
-\log
\frac{\exp(q^\top k^+/\tau)}
{\exp(q^\top k^+/\tau)+\sum_{k^-\in Q}\exp(q^\top k^-/\tau)}.
$$

**Loss 归类与等价视角。** MoCo 的 loss 是 InfoNCE 在 queue dictionary 上的实现。它与普通 softmax cross entropy 完全对齐：logits 的第 0 列是 positive，其余 `K` 列是 queue negative，label 恒为 0。MoCo 的新意主要不在 loss 公式，而在用 momentum encoder 让 queue 中的 key 更一致、用 FIFO queue 让字典规模与 batch size 解耦。

| 项目       | MoCo 中的定义                     |
| ---------- | --------------------------------- |
| 输入       | 同一图片的 query view 与 key view |
| anchor     | query encoder 输出 `q`            |
| positive   | momentum key encoder 输出 `k+`    |
| negative   | queue 中历史 mini-batch 的 key    |
| dictionary | FIFO queue                        |
| loss       | InfoNCE                           |

论文中的 PyTorch-like 思路可以写成：

```python
def moco_step(x_q, x_k, encoder_q, encoder_k, queue, tau, m):
    q = l2_normalize(encoder_q(x_q), axis=1)      # [B, D]

    # key 分支不反传；它由 query encoder 的 EMA 更新，保证 queue 中 key 更一致。
    with no_grad():
        encoder_k.params = m * encoder_k.params + (1 - m) * encoder_q.params
        k_pos = l2_normalize(encoder_k(x_k), axis=1)  # [B, D]

    pos_logits = sum(q * k_pos, axis=1, keepdims=True) / tau  # [B, 1]
    neg_logits = matmul(q, queue.T) / tau                     # [B, K]
    logits = concat([pos_logits, neg_logits], axis=1)
    labels = zeros(len(q), dtype=int)
    loss = cross_entropy(logits, labels)

    queue.enqueue(k_pos)
    queue.dequeue_oldest(len(k_pos))
    return loss
```

MoCo v1 的 Shuffle BN 用来降低 batch norm 造成的潜在信息泄漏。MoCo v2 保留 queue 与 momentum encoder，把 SimCLR 中有效的 MLP projection head、更强数据增强和 cosine learning schedule 引入 MoCo recipe。MoCo v3 关注 ViT 自监督训练稳定性，核心训练信号仍可理解为 momentum encoder 配合 contrastive objective。

### 3.7 SimCLR：Algorithm 1 与 NT-Xent

SimCLR 去掉 memory bank、queue 和 momentum encoder，只依赖大 batch 提供负样本。每个 batch 中 `N` 张图片各生成两种增强，得到 `2N` 个 view。

| 项目       | SimCLR 中的定义             |
| ---------- | --------------------------- |
| 输入       | 每张图片的两种随机增强      |
| anchor     | 任意一个 view               |
| positive   | 同一原图的另一个 view       |
| negative   | batch 中其余 `2N-2` 个 view |
| dictionary | 当前 batch                  |
| loss       | NT-Xent，双向求和           |

论文中的 NT-Xent 写作：

$$
\ell_{i,j}
=
-\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k=1}^{2N}\mathbf{1}_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

**Loss 归类与等价视角。** NT-Xent 是 SimCLR 对 InfoNCE 的命名版本：normalized 表示特征先做 L2 normalize，temperature-scaled 表示 logits 除以 `tau`，cross entropy 表示在 `2N-1` 个候选里分类 positive。它的“对称”来自每个正样本对会被用两次：`i -> j` 和 `j -> i`。矩阵实现里的 mask 只是排除自己和自己匹配，不是改变 loss 类别。

SimCLR Algorithm 1 的教学化写法：

```python
def simclr_step(images, encoder, projection_head, augment, tau):
    # images: [N, ...]
    views_1 = augment(images)
    views_2 = augment(images)
    views = concat([views_1, views_2], axis=0)          # [2N, ...]

    h = encoder(views)
    z = l2_normalize(projection_head(h), axis=1)        # [2N, D]

    logits = matmul(z, z.T) / tau                       # [2N, 2N]
    logits = mask_diagonal(logits, value=-inf)          # 自己不能当自己的候选

    # 若前 N 个是 view_1，后 N 个是 view_2：
    # i 的 positive 是 i+N；i+N 的 positive 是 i。
    labels = concat([arange(N, 2 * N), arange(0, N)])
    loss = cross_entropy(logits, labels)
    return loss
```

SimCLR 的重要实验结论：

- 随机裁剪和颜色扰动的组合非常关键，它们定义了模型应忽略的 nuisance factors。
- nonlinear projection head 提升预训练效果；下游通常使用 head 前的 `h`，而不是 loss 直接作用的 `z`。
- 大 batch 提供更多 in-batch negatives，因此可以不使用 memory bank 或 queue。

### 3.8 BYOL：online prediction 对齐 EMA target

BYOL 没有显式 negative。它有 online network 和 target network；target network 是 online network 的 exponential moving average。online 分支经过 predictor 后去预测 target 分支的 stop-gradient 表征。

| 项目       | BYOL 中的定义                                        |
| ---------- | ---------------------------------------------------- |
| 输入       | 同一图片的两种增强                                   |
| anchor     | online 分支的 prediction                             |
| target     | target 分支的 projection，stop-gradient              |
| negative   | 无显式负样本                                         |
| 防坍缩结构 | EMA target、predictor、stop-gradient、normalization  |
| loss       | normalized prediction 与 target 的 cosine regression |

常用写法是：

$$
\mathcal{L}_{BYOL}
=
2 - 2\cdot
\frac{q_\theta(z_\theta^{(1)})^\top \operatorname{sg}(z_\xi^{(2)})}
{\|q_\theta(z_\theta^{(1)})\|_2\|\operatorname{sg}(z_\xi^{(2)})\|_2},
$$

并对两个方向求和。

**Loss 归类与等价视角。** BYOL 不属于 softmax/InfoNCE 家族，因为它没有显式 negative，也没有候选集合分母。它是 normalized cosine regression：online predictor 的输出去拟合 target projection 的 stop-gradient 版本。它看起来像“只拉近 positive”，但 EMA target、predictor 和 stop-gradient 让两个分支的优化不对称，避免简单地把它等同于无约束的 L2/cosine matching。

代码化实现：

```python
def byol_step(x1, x2, online, target, predictor, m):
    # online 分支参与反传
    z1_online = online.project(online.encode(x1))
    z2_online = online.project(online.encode(x2))
    p1 = predictor(z1_online)
    p2 = predictor(z2_online)

    # target 分支不反传，由 online 参数的 EMA 更新。
    with no_grad():
        target.params = m * target.params + (1 - m) * online.params
        z1_target = target.project(target.encode(x1))
        z2_target = target.project(target.encode(x2))

    loss_12 = 2 - 2 * cosine(l2_normalize(p1), l2_normalize(stop_grad(z2_target)))
    loss_21 = 2 - 2 * cosine(l2_normalize(p2), l2_normalize(stop_grad(z1_target)))
    return mean(loss_12 + loss_21)
```

BYOL 的稳定性不应简单归因于 BatchNorm 泄漏。更稳妥的理解是：EMA target、predictor、stop-gradient、normalization 和优化 recipe 共同改变了 Siamese 网络的坍缩动力学。

### 3.9 SimSiam：去掉 momentum encoder 的 Siamese loss

SimSiam 进一步去掉 BYOL 的 momentum target，只保留共享 encoder、projection head、predictor 和 stop-gradient。

| 项目       | SimSiam 中的定义                         |
| ---------- | ---------------------------------------- |
| 输入       | 同一图片的两种增强                       |
| anchor     | 一个 view 的 predictor 输出              |
| target     | 另一个 view 的 projection，stop-gradient |
| negative   | 无显式负样本                             |
| 防坍缩结构 | predictor + stop-gradient                |
| loss       | negative cosine similarity               |

公式：

$$
\mathcal{L}_{SimSiam}
=
\frac{1}{2}D(p_1,\operatorname{sg}(z_2))
+
\frac{1}{2}D(p_2,\operatorname{sg}(z_1)),
$$

其中

$$
D(p,z)=-
\frac{p}{\|p\|_2}^{\top}
\frac{z}{\|z\|_2}.
$$

**Loss 归类与等价视角。** SimSiam 和 BYOL 同属 negative-free cosine regression。不同点是 SimSiam 去掉了 EMA target，靠共享 encoder、predictor 和 stop-gradient 保持不对称。它不是 InfoNCE 的特殊情况，因为没有 negative 分母；更准确的分类是“stop-gradient Siamese regression loss”。

代码化实现：

```python
def simsiam_step(x1, x2, encoder, projector, predictor):
    z1 = projector(encoder(x1))
    z2 = projector(encoder(x2))
    p1 = predictor(z1)
    p2 = predictor(z2)

    loss_12 = negative_cosine(p1, stop_grad(z2))
    loss_21 = negative_cosine(p2, stop_grad(z1))
    return 0.5 * (loss_12 + loss_21)
```

SimSiam 的核心实验信息是：stop-gradient 与 predictor 是避免坍缩的关键组件；没有显式负样本不意味着没有约束，只是约束从“排斥负样本”转成了“不对称优化结构”。

### 3.10 DINO：teacher-student self-distillation

DINO 使用 student network 预测 teacher network 的输出分布。teacher 是 student 的 momentum average；teacher 输出经过 centering 和 sharpening；multi-crop 让局部 view 对齐全局 view。

| 项目       | DINO 中的定义                                       |
| ---------- | --------------------------------------------------- |
| 输入       | 同一图片的 global crops 与 local crops              |
| anchor     | student 对各 view 的输出分布                        |
| target     | teacher 对 global view 的 sharpened distribution    |
| negative   | 无显式负样本                                        |
| 防坍缩结构 | centering、sharpening、momentum teacher、multi-crop |
| loss       | teacher-student cross entropy                       |

teacher 和 student 的概率分布可写成：

$$
P_s(x) = \operatorname{softmax}\left(g_{\theta_s}(x)/\tau_s\right),
$$

$$
P_t(x) = \operatorname{softmax}\left((g_{\theta_t}(x)-c)/\tau_t\right).
$$

每对 teacher view 与 student view 的 loss 是：

$$
\mathcal{L}_{DINO}
=
-P_t(x)^\top \log P_s(x').
$$

**Loss 归类与等价视角。** DINO 属于 teacher-student cross entropy，也可以看成 self-distillation loss。它有 softmax 和 cross entropy，但不是 contrastive softmax：分母在输出 prototype/class token 维度上，而不是 batch 中的 negative 样本上。centering 和 sharpening 是防坍缩装置：前者抑制单一维度长期占优，后者让 teacher target 更有信息量。

代码化实现：

```python
def dino_step(crops, student, teacher, center, tau_s, tau_t, m):
    # crops 包含 global crops 和 local crops；teacher 通常只看 global crops。
    student_logits = [student(crop) for crop in crops]

    with no_grad():
        teacher.params = m * teacher.params + (1 - m) * student.params
        teacher_logits = [teacher(crop) for crop in global_crops(crops)]
        teacher_probs = [
            softmax((logit - center) / tau_t, axis=1)
            for logit in teacher_logits
        ]

    student_probs = [
        log_softmax(logit / tau_s, axis=1)
        for logit in student_logits
    ]

    losses = []
    for t_id, t_prob in enumerate(teacher_probs):
        for s_id, s_logprob in enumerate(student_probs):
            if same_view(t_id, s_id):
                continue
            losses.append(cross_entropy_from_probs(t_prob, s_logprob))

    center = update_center(center, teacher_logits)
    return mean(losses), center
```

DINO 不是用负样本做排斥，而是用 teacher 分布提供目标；centering 防止某些维度长期占优，sharpening 防止 teacher 输出过于均匀。

### 3.11 CLIP：图文双塔的矩阵对角线分类

CLIP 的训练 batch 有 `N` 对图文。图像 encoder 和文本 encoder 分别输出 embedding，投影并 L2 normalize 后得到 `N x N` 相似度矩阵。矩阵对角线是配对图文 positive；非对角线是 batch 内 negative。

| 项目       | CLIP 中的定义                    |
| ---------- | -------------------------------- |
| 输入       | `N` 对图像-文本                  |
| anchor     | 图像 embedding 或文本 embedding  |
| positive   | 同一 pair 的另一模态 embedding   |
| negative   | batch 内其他图文                 |
| dictionary | 当前 batch 的另一模态 embeddings |
| loss       | 图到文、文到图双向 cross entropy |

公式：

$$
S_{ij}=z_i^{image\top}z_j^{text}\cdot \exp(t).
$$

图到文：

$$
\mathcal{L}_{I\to T}
=
\frac{1}{N}\sum_i
-\log
\frac{\exp(S_{ii})}
{\sum_j \exp(S_{ij})}.
$$

文到图：

$$
\mathcal{L}_{T\to I}
=
\frac{1}{N}\sum_i
-\log
\frac{\exp(S_{ii})}
{\sum_j \exp(S_{ji})}.
$$

最终：

$$
\mathcal{L}_{CLIP}
=
\frac{1}{2}
\left(
\mathcal{L}_{I\to T}
+
\mathcal{L}_{T\to I}
\right).
$$

**Loss 归类与等价视角。** CLIP 是双向 InfoNCE / symmetric cross entropy。图到文方向把每张图当 query，在 `N` 段文本中分类出配对文本；文到图方向反过来。相似度矩阵的对角线是 positive，非对角线是 batch 内 negative。和 SimCLR 的差别不是 loss 大类，而是 positive 的来源从“同图增强”变成“图文配对”，因此它学的是跨模态语义对齐。

论文伪代码的等价 NumPy 风格写法：

```python
def clip_step(images, texts, image_encoder, text_encoder, W_i, W_t, logit_scale):
    # images: [N, H, W, C]，texts: [N, L]，第 i 张图与第 i 段文本配对。
    image_features = image_encoder(images)          # [N, D_i]
    text_features = text_encoder(texts)             # [N, D_t]

    image_embed = l2_normalize(image_features @ W_i, axis=1)  # [N, D_e]
    text_embed = l2_normalize(text_features @ W_t, axis=1)    # [N, D_e]

    # logits[i, j] 表示第 i 张图与第 j 段文本的相似度。
    # 对角线 logits[i, i] 是 positive；非对角线都是 batch 内 negative。
    logits = (image_embed @ text_embed.T) * exp(logit_scale)  # [N, N]
    labels = arange(len(images))

    loss_i = cross_entropy(logits, labels)       # image -> text
    loss_t = cross_entropy(logits.T, labels)     # text -> image
    return 0.5 * (loss_i + loss_t)
```

这段代码里的 `loss_i` 和 `loss_t` 正好对应上面的两个方向；转置相似度矩阵不是换了另一种 loss，而是把检索方向从 image-to-text 换成 text-to-image。

## 4. 精度提升方式：涨点背后的机制

| 手段                 | 为什么有效                                               | 代价或风险                              |
| -------------------- | -------------------------------------------------------- | --------------------------------------- |
| 数据增强             | 定义模型应该忽略的 nuisance factor                       | 增强过强会破坏 positive 语义            |
| MLP projection head  | 让 contrastive loss 在 `z` 空间施压，保留 `h` 的下游信息 | head 深度、BN、维度都会影响稳定性       |
| 大 batch             | 提供更多 in-batch negatives                              | 显存和通信成本高，false negative 增多   |
| queue/memory bank    | 小 batch 也能维护大字典                                  | 特征 stale，需要 momentum 或刷新机制    |
| temperature          | 控制 hard negative 权重                                  | 太小不稳定，太大区分度不足              |
| 更强 backbone        | 提供更高容量和更好 inductive bias                        | 容量越大越依赖 recipe，ViT 稳定性更敏感 |
| momentum teacher     | 提供平滑目标，减少目标表征抖动                           | momentum 太大更新慢，太小目标噪声大     |
| stop-gradient        | 打破 Siamese 分支的同步坍缩路径                          | 依赖 predictor 等结构配合               |
| centering/sharpening | 控制输出分布，避免单维占优或全均匀                       | 超参敏感，常和 teacher-student 配套     |
| 线性评估/迁移评估    | 区分预训练表征和端到端分类器                             | 评估协议不同会导致论文间误读            |

数据增强不是装饰，而是 positive 的定义器。SimCLR 的 crop/color distortion 让模型忽略低级变化；MoCo v2 通过增强和 MLP head 吸收 SimCLR recipe；DINO 的 multi-crop 让局部视图对齐全局语义。判断一个增强是否合理，关键看它是否保留任务希望迁移的因素。

## 5. 和人脸识别的连接：ArcFace/CosFace 是监督版角度对比

人脸识别的目标是让同一身份类内紧、不同身份类间远。普通 softmax 的 logit 同时受角度和特征范数影响；CosFace 和 ArcFace 把特征和类别权重都 L2 normalize：

$$
\|x\|_2=1,\qquad
\|W_j\|_2=1,\qquad
W_j^\top x=\cos\theta_j.
$$

CosFace 在正确类别的 cosine 上减去 margin：

$$
\mathcal{L}_{CosFace}
=
-\log
\frac{e^{s(\cos\theta_y-m)}}
{e^{s(\cos\theta_y-m)}+\sum_{j\ne y}e^{s\cos\theta_j}}.
$$

ArcFace 在角度上加 margin：

$$
\mathcal{L}_{ArcFace}
=
-\log
\frac{e^{s\cos(\theta_y+m)}}
{e^{s\cos(\theta_y+m)}+\sum_{j\ne y}e^{s\cos\theta_j}}.
$$

**Loss 归类与等价视角。** CosFace/ArcFace 属于 supervised margin softmax。它们和 InfoNCE 一样使用 softmax cross entropy，也有 positive logit 和 negative logits；不同点是候选不是样本或 view，而是类别 prototype。CosFace 在 cosine 空间加 margin，ArcFace 在角度空间加 margin，本质上都是对正确类别 logit 做更严格的几何约束。

代码化看，它和“对正确候选做 cross entropy”同构：

```python
def arcface_step(features, labels, class_weights, scale, margin):
    x = l2_normalize(features, axis=1)
    W = l2_normalize(class_weights, axis=0)

    cos_theta = x @ W                              # [B, C]
    theta_y = arccos(gather(cos_theta, labels))
    target_logit = cos(theta_y + margin)

    logits = cos_theta.copy()
    logits[arange(len(labels)), labels] = target_logit
    logits = logits * scale
    return cross_entropy(logits, labels)
```

共同点：

- 都在单位超球面上比较相似度。
- 都通过 softmax 把 positive 和 negative 放进同一个竞争集合。
- 都把“类内紧凑、类间分离”变成可优化的几何约束。

差别也很重要：ArcFace/CosFace 的 positive 是人工身份标签对应的类别 prototype；SimCLR/MoCo 的 positive 是增强或同实例构造出来的 key；CLIP 的 positive 是图文配对。因此，人脸识别 loss 更准确地说是“监督、有 prototype 的对比几何”，不是自监督 contrastive pretraining。

## 6. 健康生理信号应用预研：ECG-PPG 可以借鉴对比学习，但不能照搬

这一节是个人兴趣方向的预研，不属于前面视觉/多模态对比学习技术路线的通识部分。

ECG 记录心脏电活动，PPG 记录外周血容量变化。两者共享心动周期信息，但并非同步到同一瞬间：ECG 的 R peak 到 PPG 脉搏峰之间存在 pulse transit time，受血压、血管状态、测量位置和个体差异影响。

一个可讨论的 ECG-PPG 对比学习设定：

| 组件          | 设计                                                                            |
| ------------- | ------------------------------------------------------------------------------- |
| encoder       | ECG encoder 与 PPG encoder，或共享部分时序 backbone                             |
| positive      | 同一受试者、同一时间附近、校正延迟后的 ECG-PPG 窗口                             |
| negative      | 不同时间、不同受试者、不同状态窗口                                              |
| hard negative | 同一受试者但错位窗口，用来逼迫模型关注同步生理状态                              |
| augmentation  | 带通滤波、幅值缩放、局部 dropout、轻微 jitter；不能破坏 R-R interval 或脉搏间期 |
| downstream    | HR、HRV、血压 proxy、睡眠/压力状态、异常检测                                    |

可以借鉴 CLIP 式双塔目标：

$$
S_{ij}=\frac{z_i^{ECG\top}z_j^{PPG}}{\tau},
$$

$$
\mathcal{L}
=
\frac{1}{2}
\left(
\operatorname{CE}(S,\operatorname{diag})
+
\operatorname{CE}(S^\top,\operatorname{diag})
\right).
$$

**Loss 归类与等价视角。** 这个 ECG-PPG 目标是 CLIP-style symmetric InfoNCE。ECG 到 PPG 的方向是在一批 PPG 窗口中找同步窗口；PPG 到 ECG 方向反过来。它的数学形式和 CLIP 一样，但 positive 的可靠性依赖时间同步、延迟校正和受试者级切分；这里的核心风险不是 loss 公式，而是生理数据构造是否把真实同步关系和身份/设备捷径区分开。

代码化实现：

```python
def ecg_ppg_contrast_step(ecg_window, ppg_window, ecg_encoder, ppg_encoder, tau):
    z_ecg = l2_normalize(ecg_encoder(ecg_window), axis=1)
    z_ppg = l2_normalize(ppg_encoder(ppg_window), axis=1)

    # 第 i 段 ECG 与第 i 段 PPG 需要先做时间同步或延迟校正。
    # 对角线是同一生理窗口的跨模态 positive；非对角线是 batch 内 negative。
    logits = (z_ecg @ z_ppg.T) / tau
    labels = arange(len(ecg_window))

    loss_ep = cross_entropy(logits, labels)
    loss_pe = cross_entropy(logits.T, labels)
    return 0.5 * (loss_ep + loss_pe)
```

潜在价值：

- 不必为每个窗口标注 HRV 或健康状态，也能用同步信号学习表征。
- PPG 更容易穿戴采集，ECG 更接近电生理基准，跨模态对齐可能提升 PPG-only 下游任务。
- 多传感器 wearable 场景天然存在“同一生理事件的不同观测”。

主要风险：

- 身份泄漏：模型学会同一人的 ECG/PPG 风格，而不是窗口内生理状态。
- 设备泄漏：采样率、滤波器、传感器位置变成捷径。
- 时间错配：PPG 延迟未建模时，positive 本身含噪。
- 下游混淆：HR 相对容易，HRV 对 R-R 或 peak interval 精度要求更高，不能只看全局 embedding。
- 医学结论外推：对比学习能学表征，不自动证明某个生理指标可被可靠预测。

更稳的实验协议应按受试者划分 train/val/test，用同一受试者错位窗口构造 hard negative，并分别评估 HR、HRV、跨设备泛化和跨人群泛化。窗口级随机划分很可能导致虚高。

## 7. 面试复写线索

1. 对比学习的核心不是某个固定 loss，而是 anchor、positive、negative、dictionary 的定义。
2. Triplet loss 是 margin ranking；NCE 是大规模归一化近似；InfoNCE/NT-Xent/CLIP loss 是候选集合 softmax；MINE 是互信息估计器。
3. InstDisc 用 memory bank 做大规模实例分类，但历史特征不一致；MoCo 用 queue 控制字典规模，用 momentum encoder 提升一致性；SimCLR 用大 batch 替代显式字典结构。
4. BYOL、SimSiam、DINO 说明显式 negative 不是唯一解，关键是用 stop-gradient、predictor、EMA teacher、centering 等机制避免坍缩。
5. CLIP 把对比学习从同图增强推广到图文配对，本质上是相似度矩阵对角线的双向分类。
6. ArcFace/CosFace 体现了同一套几何语言：单位球面、角度 margin、类内紧凑、类间分离。

## 8. 理解检查

- 为什么说 SimCLR、CMC、CLIP 的矩阵对角线都是 positive，但它们学到的不变性不同？
- MoCo 的 queue 为什么能比 batch 大很多？为什么又需要 momentum encoder？
- InfoNCE 的互信息下界为什么受候选集合大小影响？false negative 会怎样改变这个解释？
- projection head 为什么常常训练时保留、下游评估时丢掉？
- BYOL 和 SimSiam 没有显式负样本，分别依赖哪些结构避免坍缩？
- CLIP 代码里 `logits[i, j]` 的行 softmax 和列 softmax 分别对应什么检索方向？
- ECG-PPG 对比学习中，为什么窗口级随机划分可能导致虚高？

## 9. 主要出处

主线论文：

- InstDisc: [Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978), 2018.
- CPC: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748), 2018.
- CMC: [Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849), 2019.
- InvaSpread: [Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://arxiv.org/abs/1904.03436), 2019.
- MoCo: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722), 2019/2020.
- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), 2020.
- MoCo v2: [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297), 2020.
- BYOL: [Bootstrap your own latent](https://arxiv.org/abs/2006.07733), 2020.
- SimSiam: [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566), 2020/2021.
- MoCo v3: [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057), 2021.
- DINO: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), 2021.
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.

Loss 与相关连接：

- FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832), 2015.
- NCE: [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a.html), 2010.
- MINE: [Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062), 2018.
- CosFace: [Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414), 2018.
- ArcFace: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698), 2018.

中文二级阅读材料可用于补充直觉和表达，但公式、机制和时间线以上述论文为准：

- [知乎：对比学习相关整理](https://zhuanlan.zhihu.com/p/346686467)
- [知乎：对比学习综述笔记](https://zhuanlan.zhihu.com/p/555359995)
- [Bilibili 笔记：对比学习相关内容](https://www.bilibili.com/h5/note-app/view?cvid=14700928&pagefrom=comment)
