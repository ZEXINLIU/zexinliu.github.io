---
layout: post
title: "Contrastive Learning: Objectives, Dictionaries, Momentum Encoders, and Multimodal Alignment"
date: 2025-11-15 00:00:00
description: 从 anchor/positive/negative、dictionary 与 temperature 出发，梳理 Triplet、NCE/InfoNCE、NT-Xent、MoCo/SimCLR、BYOL/SimSiam/DINO、CLIP 以及 ArcFace/CosFace 的候选集合与表征几何。
tags: contrastive-learning representation-learning multimodal-learning machine-learning
categories: ai-notes
featured: true
---

对比学习可以按一条因果链理解：任务语义决定 view 与 positive 的构造；positive、negative 或 target 的来源决定候选集合；候选集合的规模、新鲜度和采样成本决定 memory bank、queue、large batch 或 prototype；loss 将这些设计转成梯度；encoder 更新、teacher、stop-gradient 和 centering 等机制决定训练信号是否稳定。

评价一个对比学习方法，可以先问五个问题：

- 语义假设：哪种关系被定义为 positive，哪些变化被要求不影响表征。SimCLR 的同图增强、CPC 的未来 latent、CLIP 的图文配对对应不同语义假设。
- 候选集合：negative、target 或 class prototype 来自 batch、memory bank、queue、teacher 还是分类头；这一步决定计算成本、特征一致性和 false negative 风险。
- 训练目标：Triplet、NCE、InfoNCE、NT-Xent、CLIP symmetric loss 和 margin softmax 都在刻画正候选相对其他候选的优势，但归一化范围、采样近似和监督来源不同。
- 更新机制：momentum encoder 主要缓解 queue/memory bank key 与当前 query encoder 的特征不一致；BYOL、SimSiam、DINO 的 predictor、stop-gradient、EMA teacher、centering、sharpening 主要处理 negative-free 设定下的目标稳定和坍缩风险。
- 几何解释：自监督对比学习、人脸识别 margin softmax 和 CLIP 式跨模态对齐都可以放到单位球面上的候选竞争框架中比较，但它们的 positive、dictionary 和监督信号不同。

## 0. 目录与读法

| 跳转                      | 本节回答的问题                           | 阅读重点                                                    |
| ------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| [1. 统一视角](#sec-1)     | 如何把不同算法放进同一套对象定义         | anchor、positive、negative、dictionary、temperature、动机链 |
| [2. Loss 家族](#sec-2)    | 公式相似的 loss 如何归类和区分           | Triplet、NCE、InfoNCE、NT-Xent、MINE、CLIP loss             |
| [3. 代表论文](#sec-3)     | 每篇论文如何定义输入、正负样本和训练目标 | InstDisc、CPC、CMC、MoCo、SimCLR、BYOL、SimSiam、DINO、CLIP |
| [4. 工程 recipe](#sec-4)  | 涨点手段对应什么机制和风险               | augmentation、projection head、queue、batch、backbone       |
| [5. 人脸识别连接](#sec-5) | margin softmax 如何提供监督版几何对照    | normalized softmax、SphereFace、AM-Softmax/CosFace、ArcFace |
| [6. 理解检查](#sec-6)     | 如何确认自己读懂关键机制                 | 公式、候选集合、更新机制、几何解释                          |
| [7. 主要出处](#sec-7)     | 公式和算法细节来自哪些论文               | 主线论文、loss 论文、人脸识别 margin softmax                |

<a id="sec-1"></a>

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

同一个交叉熵形式，放在不同数据构造里会学到不同不变性。SimCLR 的 positive 来自同一图像的两种增强；CPC 的 positive 来自同一序列的未来 latent；CLIP 的 positive 来自同一条图文对。loss 位于链条末端，前面的正负样本定义、候选集合和采样策略决定了训练信号的含义。

### 1.1 从任务定义到工程结构的动机链

多数对比学习方法可以按下面的决策顺序拆解：

```text
任务希望保留什么语义
-> 如何构造 view、anchor、positive
-> negative 或 target 从哪里来
-> 候选集合用 batch、memory bank、queue 还是 prototype
-> loss 如何提高正候选相对其他候选的 logit 或相似度
-> 编码器、projection head、teacher、stop-gradient 如何让这个目标可优化
```

这条链对应的技术问题如下：

| 环节         | 核心问题                             | 常见设计                                                               | 代价或失败模式                                                     |
| ------------ | ------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 任务与不变性 | 哪些变化应该被忽略，哪些差异必须保留 | 数据增强、跨视图配对、未来预测、图文配对、类别 prototype               | positive 定义错会学到错误不变性                                    |
| 候选集合     | 一个 anchor 要和多少候选竞争         | in-batch negatives、memory bank、queue、全类别 prototype               | 候选少则信号弱，候选多则计算、通信或 false negative 风险上升       |
| 字典结构     | 如何在小 batch 下得到大 dictionary   | memory bank、FIFO queue、large batch                                   | memory bank/queue 会引入 stale feature；large batch 显存和通信昂贵 |
| 编码器更新   | 候选特征如何保持一致                 | momentum encoder、EMA teacher、stop-gradient                           | 更新太慢目标滞后，更新太快目标抖动                                 |
| 训练目标     | 如何把正候选的相对优势写成可优化目标 | NCE、InfoNCE、NT-Xent、symmetric CE、margin softmax、cosine regression | softmax 依赖负样本质量；negative-free 需要额外防坍缩结构           |

需要区分两类失败模式：momentum encoder 主要缓解 queue/memory bank 中 key 表征与当前 query encoder 的不一致；BYOL、SimSiam、DINO 在 negative-free 设定下用 predictor、stop-gradient、EMA teacher、centering 或 sharpening 维持目标稳定并降低坍缩风险。二者都服务于训练信号稳定性，但对应的问题不同。

<a id="sec-2"></a>

## 2. Loss 家族：从 margin 到 softmax，再到互信息

### 2.1 Triplet loss

要点：Triplet loss 用 margin 约束 positive 与 negative 的相对距离。

Triplet loss 来自 metric learning，在 FaceNet 中用于学习人脸嵌入：

$$
\mathcal{L}_{triplet}
= \max\left(0, d(a,p)-d(a,n)+m\right).
$$

`a` 是 anchor，`p` 是 positive，`n` 是 negative，`m` 是 margin。它要求正样本距离至少比负样本近 `m`。

它的优点是直观，缺点也明显：只比较一个或少量 negative，强依赖 hard negative mining；当 triplet 已满足 margin 时梯度为 0；它没有候选集合分类的概率解释。大规模自监督视觉预训练更常用 softmax 型目标，因为一个 anchor 可以同时比较大量候选。

### 2.2 Softmax contrastive loss

要点：Softmax contrastive loss 是以 anchor 为 query、以候选集合为类别空间的交叉熵。

如果 anchor `i` 的正样本是 `j`，候选集合为 `A(i)`，最常见的对比目标是：

$$
\mathcal{L}_i
=
-\log
\frac{\exp(s(i,j)/\tau)}
{\sum_{a\in A(i)} \exp(s(i,a)/\tau)}.
$$

这个目标等价于交叉熵：模型在候选集合中将 `j` 作为正确类别。这里的类别指“这个 anchor 应该匹配哪个 key”，不对应人工语义类。

下面把 anchor 表征的梯度完整推一遍。为避免把 query 角色和 key 角色混在一起，先采用最常见的矩阵实现视角：当前行的 query 是 `z_i`，候选集合里的 key 是 `{z_a}`，且求这一行 loss 对 query 表征的梯度时，把 key 表征看成另一个分支输出。SimCLR/CLIP 的对称 loss 会让同一个样本在别的行里再作为 key 承担梯度，这部分稍后单独说明。

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

则 loss 可以展开成：

$$
\mathcal{L}_i
=
-\frac{1}{\tau}z_i^\top z_j
+
\log
\sum_{a\in A(i)}
\exp\left(\frac{z_i^\top z_a}{\tau}\right).
$$

第一项的梯度是：

$$
\frac{\partial}{\partial z_i}
\left(
-\frac{1}{\tau}z_i^\top z_j
\right)
=
-\frac{1}{\tau}z_j.
$$

第二项令

$$
Z_i
=
\sum_{a\in A(i)}
\exp\left(\frac{z_i^\top z_a}{\tau}\right),
$$

则

$$
\begin{aligned}
\frac{\partial}{\partial z_i}\log Z_i
&=
\frac{1}{Z_i}
\frac{\partial Z_i}{\partial z_i}\\
&=
\frac{1}{Z_i}
\sum_{a\in A(i)}
\exp\left(\frac{z_i^\top z_a}{\tau}\right)
\frac{1}{\tau}z_a\\
&=
\frac{1}{\tau}
\sum_{a\in A(i)}
p_{ia}z_a.
\end{aligned}
$$

两项相加，得到 query-role 下 anchor 表征的梯度：

$$
\frac{\partial \mathcal{L}_i}{\partial z_i}
=
\frac{1}{\tau}
\left(
\sum_{a\in A(i)} p_{ia}z_a - z_j
\right).
$$

梯度下降更新是 `z_i <- z_i - eta * gradient`，所以 `-z_j` 这一项会把 anchor 拉向正样本；`\sum_a p_{ia}z_a` 这一项会把 anchor 从 softmax 概率加权的候选均值处推开。越像 anchor 的 negative 会得到更大的 `p_{ia}`，因此贡献更大的排斥梯度，这就是 hard negative 自动加权的来源。

如果看某个 key `z_a` 在这一行里的梯度，则有：

$$
\frac{\partial \mathcal{L}_i}{\partial z_a}
=
\frac{1}{\tau}
\left(p_{ia}-\mathbf{1}_{a=j}\right)z_i.
$$

这说明 positive key 的系数是 `p_ij - 1 < 0`，梯度下降会把它拉向 query；negative key 的系数是 `p_ia > 0`，梯度下降会把它推离 query。对称 loss 中，同一个样本既在自己的行里做 query，也在其他样本的行里做 key，总梯度就是这些角色贡献的和。

如果候选集合显式包含 `a=i`，也就是自己作为 key 参与分母，那么 `z_i` 还会通过 self-key 的位置额外进入 `s(i,i)=z_i^\top z_i`。此时 query-role 梯度之外还要加上 self-key 梯度：

$$
\frac{\partial \mathcal{L}_i}{\partial z_i}
=
\frac{1}{\tau}
\left(
\sum_{a\in A(i)}p_{ia}z_a-z_j
\right)
+
\frac{1}{\tau}
\left(p_{ii}-\mathbf{1}_{i=j}\right)z_i.
$$

SimCLR 的 NT-Xent 分母用 `k != i` 的 mask 排除了 self-key，因此不会出现这一项。许多实现还会先做 L2 normalize，令 `z_i = u_i / ||u_i||`。这时上面的梯度是对 normalized embedding 的梯度，传回未归一化向量 `u_i` 时还要乘 normalize 的 Jacobian：

$$
\frac{\partial \mathcal{L}_i}{\partial u_i}
=
\frac{1}{\|u_i\|}
\left(I-z_i z_i^\top\right)
\frac{\partial \mathcal{L}_i}{\partial z_i}.
$$

这一步会把梯度投影到单位球的切空间，所以 normalized contrastive learning 更强调角度相似度，而不是简单增大向量范数。

### 2.3 NCE 与 InfoNCE

要点：NCE 用 data-vs-noise 二分类近似未归一化模型估计；InfoNCE 将候选集合分类和互信息下界联系起来。

Noise-Contrastive Estimation 最初用于未归一化统计模型的估计。所谓未归一化统计模型，是指模型容易给出一个能量或打分函数，却很难计算全空间归一化常数。例如：

$$
\tilde p_\theta(x)=\exp(f_\theta(x)),
\qquad
p_\theta(x)
=
\frac{\tilde p_\theta(x)}{Z_\theta},
\qquad
Z_\theta=\int \tilde p_\theta(x)dx.
$$

最大似然需要：

$$
\log p_\theta(x)
=
f_\theta(x)-\log Z_\theta.
$$

难点在 `Z_theta`。离散大词表、全训练集实例分类或连续空间能量模型里，`Z_theta` 可能意味着遍历巨大类别空间或做高维积分。它的梯度也不好算：

$$
\nabla_\theta \log Z_\theta
=
\mathbb{E}_{x\sim p_\theta}
\left[\nabla_\theta f_\theta(x)\right],
$$

这要求从当前模型分布采样或枚举全空间。NCE 的动机就是绕开这个 full normalization：不直接问“这个样本在全空间的归一化概率是多少”，而是问“这个样本更像真实数据，还是更像一个已知噪声分布生成的样本”。

NCE 构造一个二分类混合数据集：每个真实样本配 `m` 个噪声样本。令 `D=1` 表示样本来自真实数据，`D=0` 表示样本来自噪声。先验为：

$$
P(D=1)=\frac{1}{m+1},
\qquad
P(D=0)=\frac{m}{m+1}.
$$

条件密度为：

$$
p(x|D=1)=p_\theta(x),
\qquad
p(x|D=0)=p_n(x),
$$

其中 `p_n` 是已知噪声分布。由 Bayes 公式：

$$
\begin{aligned}
P(D=1|x)
&=
\frac{p(x|D=1)P(D=1)}
{p(x|D=1)P(D=1)+p(x|D=0)P(D=0)}\\
&=
\frac{p_\theta(x)\frac{1}{m+1}}
{p_\theta(x)\frac{1}{m+1}+p_n(x)\frac{m}{m+1}}\\
&=
\frac{p_\theta(x)}
{p_\theta(x)+m p_n(x)}.
\end{aligned}
$$

这就是“真实分布的后验公式”的来源。InstDisc 论文里把样本写成实例 id `i` 和当前表征 `v`，于是对应为：

$$
P(D=1|i,v)
=
\frac{p_m(i|v)}
{p_m(i|v)+m p_n(i)}.
$$

NCE 损失就是这个二分类问题的 logistic loss：

$$
\mathcal{L}_{NCE}
=
-\mathbb{E}_{p_d}\log P(D=1|i,v)
-m\mathbb{E}_{p_n}\log \left(1-P(D=1|i,v)\right).
$$

NCE 为什么能被说成大规模 softmax 的采样近似？先看 full softmax。如果每个实例都是一个类别，正确类别为 `i`，logit 为 `s_a`：

$$
\mathcal{L}_{full}
=
-\log
\frac{\exp(s_i)}
{\sum_{a=1}^{M}\exp(s_a)}.
$$

它对每个 logit 的梯度是：

$$
\frac{\partial \mathcal{L}_{full}}{\partial s_a}
=
p_a-\mathbf{1}_{a=i},
\qquad
p_a=
\frac{\exp(s_a)}{\sum_b\exp(s_b)}.
$$

这要求知道所有 `M` 个类别的归一化概率 `p_a`。NCE 只采样一个 positive 和 `m` 个 noise negative。对采样到的候选 `a`，可以写二分类 logit：

$$
h_a
=
\log p_m(a|v)-\log\left(m p_n(a)\right).
$$

binary logistic 的梯度形态是：

$$
\frac{\partial \mathcal{L}_{NCE}}{\partial h_a}
=
\sigma(h_a)-y_a,
$$

其中 `y_a=1` 表示 data sample，`y_a=0` 表示 noise sample。它和 full softmax 的 `p_a - one_hot` 很像，但不是逐样本相同：full softmax 的 `p_a` 来自全类别归一化；NCE 的 `sigma(h_a)` 来自“data vs noise”的局部二分类后验。只有在噪声样本充分、噪声分布覆盖合理、模型归一化项处理正确时，NCE 的期望目标才会逼近原始归一化模型估计。工程上说它是大规模 softmax 的采样近似，强调的是它用少量 noise negative 替代了全分母枚举，而不是说每一步梯度完全等价。

InfoNCE 的动机和 NCE 有亲缘关系，但目标发生了变化。NCE 关心估计一个概率模型 `p_theta(x)`，所以它显式使用噪声分布 `p_n`，并做 data-vs-noise 二分类。InfoNCE 更关心表示学习：给定上下文 `c`，从一个 positive 和 `N-1` 个 negative 组成的候选集合中找出哪个 `x` 来自条件分布 `p(x|c)`。

InfoNCE 的常见写法是：

$$
\mathcal{L}_N
=
-\mathbb{E}
\log
\frac{f(x^+,c)}
{f(x^+,c)+\sum_{r=1}^{N-1} f(x_r^-,c)}.
$$

也可以把候选集合写成 `X={x_1,...,x_N}`，其中一个 index `d` 是 positive。构造过程是：先均匀采样 `d`，再从 `p(x|c)` 采样 `x_d`，其余 `x_l` 从边缘分布 `p(x)` 采样。于是：

$$
p(X|d=i,c)
=
p(x_i|c)\prod_{l\ne i}p(x_l).
$$

由 Bayes 公式，并且 `P(d=i)=1/N`：

$$
\begin{aligned}
P(d=i|X,c)
&=
\frac{p(X|d=i,c)}
{\sum_{j=1}^{N}p(X|d=j,c)}\\
&=
\frac{p(x_i|c)\prod_{l\ne i}p(x_l)}
{\sum_{j=1}^{N}p(x_j|c)\prod_{l\ne j}p(x_l)}\\
&=
\frac{\frac{p(x_i|c)}{p(x_i)}}
{\sum_{j=1}^{N}\frac{p(x_j|c)}{p(x_j)}}.
\end{aligned}
$$

所以最优 critic 应该学习条件分布与边缘分布的密度比：

$$
f^*(x,c)\propto \frac{p(x|c)}{p(x)},
$$

这就是 InfoNCE 和互信息连接起来的核心原因，因为互信息本身就是这个密度比的对数期望：

$$
I(x;c)
=
\mathbb{E}_{p(x,c)}
\log
\frac{p(x|c)}{p(x)}.
$$

在这个构造下，InfoNCE 有互信息下界：

$$
I(x;c)\ge \log N-\mathcal{L}_N.
$$

直观证明可以这样理解：positive index `d` 在 `N` 个位置上均匀分布，所以先验熵是 `H(d)=log N`。模型用 `X,c` 去预测 `d`，交叉熵越小，说明候选集合越能揭示 `x` 和 `c` 的依赖。最优分类器的交叉熵对应 `H(d|X,c)`，因此 `log N - L_N` 可以看作从候选集合中识别 positive 得到的信息量；这个信息量不会超过真实变量间的互信息，所以形成下界。

从 NCE 到 InfoNCE，优化目标从“估计未归一化概率模型”转成“学习能区分 positive 的表示”。它弱化了显式噪声分布建模和归一化常数估计，把问题整理成一个稳定的 softmax cross entropy；代价是这个下界依赖候选集合大小与采样方式。候选集合越大，理论上可表达的互信息下界越高；但 false negative、batch 构成和优化难度都会改变实际效果。

### 2.4 NT-Xent

要点：NT-Xent 是 SimCLR 批内两视图采样设定下的 InfoNCE/softmax contrastive loss。

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

这句话里的重点有三层：normalized 指 embedding 做 L2 normalize；temperature-scaled 指相似度除以 `tau`；cross entropy 指每一行都在 `2N-1` 个候选中把同图另一 view 分类出来。它和一般 InfoNCE 的差别不在数学大类，而在 SimCLR 如何构造 batch、positive 和 mask。

### 2.5 MINE

要点：MINE 直接用 variational bound 估计互信息，不使用候选集合分类目标。

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

MINE 直接估计互信息下界，不使用候选集合分类目标。它更适合作为分析变量依赖或信息瓶颈的工具；在视觉自监督主线中，InfoNCE/NT-Xent 更常见，因为它们的 batch 训练更稳定。

### 2.6 Loss 关系速查

| Loss                | 核心定位                                                | 正样本/目标         | 负样本/对照           | 形式                     | 典型方法                                |
| ------------------- | ------------------------------------------------------- | ------------------- | --------------------- | ------------------------ | --------------------------------------- |
| Triplet             | 用 margin 规定“positive 必须比 negative 更近”           | 同身份/同语义样本   | 不同身份/不同语义样本 | hinge ranking            | FaceNet、metric learning                |
| Softmax contrastive | 在候选集合里把 positive 分类出来                        | 一个正 key          | 候选集合其他 key      | softmax CE               | 对比学习通用骨架                        |
| NCE                 | 把全空间归一化改成 data-vs-noise 二分类                 | 真实样本            | 噪声分布样本          | binary logistic          | InstDisc 的大规模近似                   |
| InfoNCE             | 在 `N` 个候选中找 positive，并学习密度比                | 条件分布样本        | 边缘分布样本          | softmax CE + MI bound    | CPC、MoCo、CMC                          |
| NT-Xent             | InfoNCE 在 SimCLR 批内两视图采样下的名字                | 同图另一增强        | batch 内其他 view     | symmetric softmax CE     | SimCLR                                  |
| BYOL/SimSiam loss   | 不做 negative 分类，而是预测 stop-gradient target       | 另一 view 的 target | 无显式负样本          | cosine regression        | BYOL、SimSiam                           |
| DINO loss           | student 预测 momentum teacher 的分布                    | teacher 输出分布    | 无显式负样本          | teacher-student CE       | DINO                                    |
| CLIP loss           | 双向 InfoNCE：图找文、文找图                            | 配对图文            | batch 内其他图文      | bidirectional softmax CE | CLIP                                    |
| MINE                | 直接估计互信息的 variational lower bound                | 联合分布样本        | 边缘乘积分布样本      | DV lower bound           | MI 估计                                 |
| Margin softmax      | 监督分类中只惩罚正确类别 logit，让它必须赢出一个 margin | 正确类别 prototype  | 其他类别 prototype    | margin softmax CE        | SphereFace、AM-Softmax/CosFace、ArcFace |

<a id="sec-3"></a>

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

**Loss 归类与等价视角。** 上面的 non-parametric softmax 是“实例作为类别”的多分类 cross entropy：第 `i` 张图的 positive class 是它自己的实例 id，其他实例是 negative class。InstDisc 的工程约束来自类别数等于数据集大小，full softmax 分母需要遍历全训练集；NCE 将这个多分类归一化问题改写成 data-vs-noise 的二分类 logistic 估计。memory bank 改变 dictionary 的存储与采样方式，判别目标仍是“同实例为正、其他实例为负”。

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

**Loss 归类与等价视角。** 这是标准 InfoNCE：`c_t` 是 query，真实未来 latent `z_{t+k}` 是 positive，候选集合中的其他 latent 是 negative。分母用于候选集合 softmax 分类，不用于生成式重建；当 critic 近似密度比时，最小化这个 cross entropy 等价于最大化互信息下界 `log N - L_N`。CPC 的 positive 来自未来预测，而不是同图增强或跨模态配对。

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

CPC 将未来预测写成候选集合分类，并通过 InfoNCE 连接互信息下界。

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

InvaSpread 强调两件事：同一实例的增强 view 应该 invariant，不同实例的 embedding 应该 spreading。它的目标接近 SimCLR；SimCLR 后来用更系统的 augmentation、projection head 和大 batch recipe 提升效果。

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

**Loss 归类与等价视角。** 这个目标属于 instance discrimination softmax，也可以看成同实例增强构造下的 InfoNCE。`z_i^+` 是同一实例的增强 positive，其他 `z_j^+` 是 spreading 所需的 negative。invariant + spreading 是对目标效果的描述，数学形式仍是候选集合交叉熵。

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

**Loss 归类与等价视角。** NT-Xent 是 SimCLR 对 InfoNCE 的命名版本：normalized 表示特征先做 L2 normalize，temperature-scaled 表示 logits 除以 `tau`，cross entropy 表示在 `2N-1` 个候选里分类 positive。对称性来自每个正样本对会被用两次：`i -> j` 和 `j -> i`。矩阵实现里的 mask 用于排除 self-match，不改变 loss 类别。

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

SimSiam 的核心实验信息是：stop-gradient 与 predictor 是避免坍缩的关键组件；没有显式负样本时，约束从“排斥负样本”转成“不对称优化结构”。

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

DINO 用 teacher 分布提供训练目标；centering 降低少数维度长期占优的风险，sharpening 降低 teacher 输出过于均匀的风险。

### 3.11 CLIP：图文双塔、双向对比训练与 zero-shot 分类

CLIP 的历史定位需要拆开看。图文多模态和视觉-语义 embedding 在 CLIP 之前已经存在，例如 DeViSE 用文本语义信息支持未见类别预测，Visual N-Grams 从 web 图文数据学习视觉短语。CLIP 论文报告使用 4 亿互联网图文对训练，将大规模自然语言监督、双塔对比训练和 prompt 化 zero-shot 推理组织成一个统一流程。

训练阶段，batch 中有 `N` 对图文。图像 encoder 和文本 encoder 分别输出 embedding，投影并 L2 normalize 后得到 `N x N` 相似度矩阵。矩阵对角线是配对图文 positive；非对角线是 batch 内 negative。

| 项目           | CLIP 中的定义                                 |
| -------------- | --------------------------------------------- |
| 输入           | `N` 对图像-文本                               |
| anchor         | 图像 embedding 或文本 embedding               |
| positive       | 同一 pair 的另一模态 embedding                |
| negative       | batch 内其他图文                              |
| dictionary     | 当前 batch 的另一模态 embeddings              |
| loss           | 图到文、文到图双向 cross entropy              |
| zero-shot 推理 | 用类别 prompt 的文本 embedding 作为分类器权重 |

训练相似度矩阵：

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

**Loss 归类与等价视角。** CLIP 是双向 InfoNCE / symmetric cross entropy。图到文方向把每张图当 query，在 `N` 段文本中分类出配对文本；文到图方向反过来。相似度矩阵的对角线是 positive，非对角线是 batch 内 negative。与 SimCLR 相比，主要差异是 positive 从“同图增强”变成“图文配对”，因此训练信号对应跨模态语义对齐。

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

这段代码里的 `loss_i` 和 `loss_t` 对应两个检索方向；转置相似度矩阵只改变 query/key 方向，不改变 loss 类型。

CLIP 的 zero-shot 分类发生在推理阶段。给定下游类别名集合 `C`，先把类别名写进自然语言模板，例如 `"a photo of a {label}"`。每个类别的多个 prompt 经过文本 encoder 后做平均和归一化：

$$
\bar z_c^{text}
=
\operatorname{normalize}
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\operatorname{normalize}
\left(f_{text}(T_k(c))\right)
\right).
$$

图像经过图像 encoder 得到 `z_x^{image}`，类别概率由图像 embedding 与各类别文本 embedding 的相似度给出：

$$
p(y=c|x)
=
\operatorname{softmax}_{c\in C}
\left(
\exp(t)\,
z_x^{image\top}\bar z_c^{text}
\right).
$$

这相当于用文本 encoder 生成一个数据集相关的线性分类头；下游数据集只需要类别名和 prompt，不需要在该数据集上重新训练分类器。

```python
def clip_zero_shot_predict(image, class_names, templates, image_encoder, text_encoder, logit_scale):
    image_embed = l2_normalize(image_encoder(image[None]), axis=1)  # [1, D]

    class_embeds = []
    for name in class_names:
        prompts = [template.format(name) for template in templates]
        text_embed = l2_normalize(text_encoder(prompts), axis=1)    # [K, D]
        class_embed = l2_normalize(mean(text_embed, axis=0), axis=0)
        class_embeds.append(class_embed)

    class_embeds = stack(class_embeds, axis=0)                      # [C, D]
    logits = exp(logit_scale) * (image_embed @ class_embeds.T)       # [1, C]
    return softmax(logits, axis=1)
```

zero-shot 能成立的条件来自训练任务本身：图像 encoder 学习把图像投到语言描述附近，文本 encoder 学习把自然语言短语投到同一空间。类别名 prompt 将新的分类任务转成图文匹配任务。它的限制也来自同一处：类别名是否能被 prompt 准确表达、训练图文分布是否覆盖该概念、相似类别之间的文本提示是否足够区分，都会影响 zero-shot 结果。

<a id="sec-4"></a>

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

数据增强直接参与 positive 定义。SimCLR 的 crop/color distortion 让模型忽略低级变化；MoCo v2 通过增强和 MLP head 吸收 SimCLR recipe；DINO 的 multi-crop 让局部视图对齐全局语义。增强是否合适，取决于它是否保留目标任务需要迁移的因素。

<a id="sec-5"></a>

## 5. 和人脸识别的连接：margin softmax 是有标签的原型竞争

人脸识别训练通常把身份当作分类标签，但最终要用 embedding 做 verification 或 retrieval。训练目标同时要求同一身份的特征聚合、不同身份的特征在角度上分离。FaceNet 用 triplet loss 直接约束样本对/三元组；SphereFace、CosFace、ArcFace 将这类约束放进 softmax 分类框架，让每个类别权重向量成为一个 class prototype。

### 5.1 Normalized softmax：prototype 版候选集合

先把特征和类别权重都放到单位球面上：

$$
\|x\|_2=1,\qquad
\|W_j\|_2=1,\qquad
W_j^\top x=\cos\theta_j.
$$

这里 `x` 是样本 embedding，`W_j` 是第 `j` 个身份类别的 prototype，`theta_j` 是样本到该 prototype 的夹角。为了让归一化后的 `[-1,1]` logit 仍有足够梯度，通常引入 scale `s`：

$$
\mathcal{L}_{norm}
=
-\log
\frac{e^{s\cos\theta_y}}
{e^{s\cos\theta_y}+\sum_{j\ne y}e^{s\cos\theta_j}}.
$$

忽略 L2 normalize 的 Jacobian，只看 normalized embedding `x` 上的 softmax CE 梯度：

$$
\frac{\partial \mathcal{L}_{norm}}{\partial x}
=
s\left(
\sum_j p_j W_j - W_y
\right)
=
s\left(
(p_y-1)W_y+\sum_{j\ne y}p_j W_j
\right).
$$

梯度下降会把 `x` 拉向自己的 prototype `W_y`，并按 `p_j` 的大小远离其他 prototype。越容易混淆的负类，`p_j` 越大，排斥项越强。类内聚合和类间分离来自同一个归一化候选集合中的 prototype 竞争。

普通 normalized softmax 的不足是判别边界太宽松：对两个类别 `y` 和 `k`，只要 `cos(theta_y) > cos(theta_k)` 就能把样本判给 `y`。它要求正确 prototype 赢，但没有要求赢出明确角度间隔。

### 5.2 Margin softmax：只让正确类别变难

margin softmax 的统一写法是：负类 logit 仍然是 `s cos(theta_j)`，只把正确类别 logit 从 `s cos(theta_y)` 改成更严格的 `s phi(theta_y)`：

$$
\mathcal{L}_{margin}
=
-\log
\frac{e^{s\phi(\theta_y)}}
{e^{s\phi(\theta_y)}+\sum_{j\ne y}e^{s\cos\theta_j}}.
$$

不同方法的差异主要在 `phi(theta_y)`：

| 方法                   | margin 类型        | 正确类别 logit                                  | 直觉                                   |
| ---------------------- | ------------------ | ----------------------------------------------- | -------------------------------------- |
| SphereFace / A-Softmax | 乘性角度 margin    | `s psi(theta_y)`, 近似看作 `s cos(m * theta_y)` | 让正确类别角度按倍数变严格             |
| AM-Softmax / CosFace   | 加性 cosine margin | `s(cos(theta_y) - m)`                           | 要求正确类别 cosine 比负类高出固定间隔 |
| ArcFace                | 加性角度 margin    | `s cos(theta_y + m)`                            | 直接要求正确类别角度小出固定间隔       |

SphereFace 的原始实现为了保证 `theta` 区间上的单调性，会使用分段函数 `psi(theta)`，而不是裸用 `cos(m theta)`。CosFace 和 AM-Softmax 的核心形式都是在 target cosine 上减去 additive margin；ArcFace 则把 margin 加到角度上，再取 cosine。

决策边界能说明三者的差异。二分类时，normalized softmax 只要求 `cos(theta_y) > cos(theta_k)`；CosFace 要求 `cos(theta_y) - m > cos(theta_k)`；ArcFace 要求 `cos(theta_y + m) > cos(theta_k)`，在 `0` 到 `pi` 的单调区间里近似等价于 `theta_y + m < theta_k`。这就是 ArcFace 被称为 additive angular margin 的原因：它直接在角度空间规定间隔。CosFace 的间隔发生在 cosine 空间，映射到角度空间后不是常数；SphereFace 是乘性角度约束，优化更硬，早期训练更依赖 annealing 和实现细节。

### 5.3 为什么 margin 会同时增强类内和类间

margin 只改正确类别 logit，看起来像只在拉近类内，实际会同时影响两边。

首先，`phi(theta_y)` 比 `cos(theta_y)` 更小，同一个样本在原本已经分类正确时仍可能得到较高 loss。为了降低 loss，模型必须继续减小 `theta_y`，也就是把样本压向自己的 prototype，类内更紧。

其次，softmax 分母中的负类没有消失。只要某个负类 prototype 与样本夹角小，它的 `p_j` 就会大，梯度中的 `p_j W_j` 会继续把样本从该负类方向推开。类别权重 `W_j` 本身也被反向传播更新：本类样本拉动自己的 prototype，其他类样本通过负类概率对它形成排斥。长期训练后，prototype 之间也会形成更大的角度间隔。

因此，人脸识别里的 margin softmax 可以看成“监督、有 class prototype 的对比学习几何”：anchor 是样本特征，positive 是正确身份 prototype，negative 是其他身份 prototype，dictionary 是分类头的全部类别权重。

### 5.4 代码化实现

下面的伪代码把 normalized softmax、SphereFace、CosFace/AM-Softmax、ArcFace 放进同一个框架。核心动作只有一个：先算完整 `B x C` cosine 矩阵，再替换正确类别那一列的 target logit。

```python
def sphereface_phi(theta, integer_margin):
    # SphereFace/A-Softmax 的教学化写法。
    # 原论文使用分段 psi(theta) 保证目标 logit 随 theta 单调下降。
    k = floor(integer_margin * theta / pi)
    return ((-1) ** k) * cos(integer_margin * theta) - 2 * k


def margin_softmax_step(features, labels, class_weights, scale, margin, kind):
    # features: [B, D]，样本 embedding。
    # class_weights: [D, C]，每一列是一个身份类别 prototype。
    x = l2_normalize(features, axis=1)
    W = l2_normalize(class_weights, axis=0)

    # cos_theta[b, c] 是第 b 个样本和第 c 个类别 prototype 的角度相似度。
    cos_theta = clip(x @ W, -1 + 1e-7, 1 - 1e-7)  # [B, C]
    target_cos = gather(cos_theta, labels)         # [B]
    target_theta = arccos(target_cos)

    if kind == "normalized":
        target_phi = target_cos
    elif kind == "sphereface":
        target_phi = sphereface_phi(target_theta, integer_margin=margin)
    elif kind == "cosface":
        target_phi = target_cos - margin
    elif kind == "arcface":
        target_phi = cos(target_theta + margin)
    else:
        raise ValueError(kind)

    logits = cos_theta.copy()
    logits[arange(len(labels)), labels] = target_phi
    logits = logits * scale
    return cross_entropy(logits, labels)
```

**Loss 归类与等价视角。** 这些方法仍然是 softmax cross entropy，差异集中在正确类别 logit 的 margin 变换上：SphereFace 做乘性角度 margin，CosFace/AM-Softmax 做加性 cosine margin，ArcFace 做加性角度 margin。和 SimCLR/MoCo/CLIP 的差别在 dictionary：这里的候选是有监督类别 prototype，而不是 batch view、queue key 或图文配对。

<a id="sec-6"></a>

## 6. 理解检查

- 为什么说 SimCLR、CMC、CLIP 的矩阵对角线都是 positive，但它们学到的不变性不同？
- MoCo 的 queue 为什么能比 batch 大很多？为什么又需要 momentum encoder？
- InfoNCE 的互信息下界为什么受候选集合大小影响？false negative 会怎样改变这个解释？
- projection head 为什么常常训练时保留、下游评估时丢掉？
- BYOL 和 SimSiam 没有显式负样本，分别依赖哪些结构避免坍缩？
- CLIP 代码里 `logits[i, j]` 的行 softmax 和列 softmax 分别对应什么检索方向？
- 为什么 CosFace/ArcFace 只改正确类别 logit，却能同时增强类内紧凑和类间分离？

<a id="sec-7"></a>

## 7. 主要出处

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
- DeViSE: [DeViSE: A Deep Visual-Semantic Embedding Model](https://papers.nips.cc/paper_files/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html), 2013.
- Visual N-Grams: [Learning Visual N-Grams from Web Data](https://arxiv.org/abs/1612.09161), 2016/2017.
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.

Loss 与相关连接：

- FaceNet: [A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832), 2015.
- NCE: [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a.html), 2010.
- MINE: [Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062), 2018.
- SphereFace: [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063), 2017.
- AM-Softmax: [Additive Margin Softmax for Face Verification](https://arxiv.org/abs/1801.05599), 2018.
- CosFace: [Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414), 2018.
- ArcFace: [Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698), 2018.
