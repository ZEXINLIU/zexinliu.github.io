---
layout: post
title: "Flow-Based Generative Models: Normalizing Flow, Flow Matching, Rectified Flow, and MeanFlow"
date: 2026-01-18 00:00:00
description: 从 Normalizing Flow、Flow Matching 到 Rectified Flow、MeanFlow 的生成模型主线。
tags: flow-matching normalizing-flows generative-models diffusion
categories: ai-notes
---

本文的重点如下：

- **Normalizing Flow / CNF 的难点是“密度怎么算”。** 标准 NF 要把网络限制成可逆且 log-det 可算的结构；CNF 放开网络结构后，又把代价转移到 ODE 积分和散度估计上。
- **Flow Matching 的难点是“边缘速度场不可见”。** 训练代码只回归 `u_t(x|z)`，但真正生成数据的是边缘速度场 `u_t(x)`；连续性方程、边缘化定理和 CFM/FM 等价性就是为了证明这件事。
- **Gaussian path、OT path、CondOT path 的难点是“路径设计决定 target 和采样难度”。** Gaussian path 解释 diffusion/noise schedule；OT displacement interpolation 解释最短动能路径；CondOT 把每个样本条件路径做成直线，但它不是说独立配对样本已经构成全局最优传输。
- **Stochastic Interpolants 的难点是“路径”和“采样过程”被拆开了。** 同一条插值密度 `rho_t` 可以由 probability flow ODE 生成，也可以由带可调噪声的 SDE 生成；为此不仅要学速度 `b_t`，还要在随机采样时学 score `s_t`。
- **Rectified Flow 的难点不是一句“走直线”能讲清。** 训练目标确实来自直线插值速度 `X_1-X_0`，但模型学到的是条件期望速度场；1-Rectified、2-Rectified/reflow 的核心是改造 coupling，让 ODE 轨迹更少交叉、更接近低步数可解，而不是记住每个配对点的一条直线。
- **MeanFlow 的难点是“预测对象换了”。** 它不再让模型输出瞬时速度，而是学习区间平均速度；JVP 训练目标和 stop-gradient 是理解一步生成的关键。
- **条件生成的难点是“条件如何进入同一个速度场”。** 类别、文本或图像条件本质上把 `v_theta(x,t)` 改成 `v_theta(x,t,c)`；classifier-free guidance 则在采样时组合有条件/无条件速度，改变贴合条件与多样性的权衡。

## 目录

- [统一视角](#统一视角)
- [Normalizing Flow 与 Continuous Normalizing Flow](#normalizing-flow-与-continuous-normalizing-flow)
- [连续性方程和两个核心定理](#连续性方程和两个核心定理)
- [Flow Matching](#flow-matching)
- [概率路径设计：Gaussian、OT 与 CondOT](#概率路径设计gaussianot-与-condot)
- [Stochastic Interpolants](#stochastic-interpolants)
- [Rectified Flow](#rectified-flow)
- [MeanFlow](#meanflow)
- [条件生成与 Classifier-Free Guidance](#条件生成与-classifier-free-guidance)
- [推理、加速和工程取舍](#推理加速和工程取舍)
- [方法对照表](#方法对照表)
- [理解检查](#理解检查)
- [来源](#来源)

## 统一视角

生成模型要做的事可以写成同一句话：从容易采样的源分布 `p_0` 出发，构造一个变换或过程，把样本送到数据分布 `p_1`。不同方法的分歧在于三件事：

| 方法                    | 学什么                                         | 训练时是否解 ODE | 推理时怎么采样               | 主要代价                                    |
| ----------------------- | ---------------------------------------------- | ---------------- | ---------------------------- | ------------------------------------------- |
| Normalizing Flow        | 显式可逆映射 `f_theta`                         | 否               | `z -> f_theta(z)` 一次前向   | 网络结构必须可逆，Jacobian 要可算           |
| CNF                     | 连续速度场 `v_theta(x,t)` 和密度变化           | 通常要解 ODE     | 从噪声积分 ODE 到数据        | 训练和似然评估要数值积分、算散度            |
| Flow Matching           | 生成指定概率路径的速度场                       | 否               | 从噪声积分 ODE 到数据        | 训练轻，推理仍有 NFE 成本                   |
| Stochastic Interpolants | 插值密度的速度 `b_t`，随机采样还需 score `s_t` | 否               | 同一 `rho_t` 可用 ODE 或 SDE | 框架统一但对象更多，score/denoiser 也要对齐 |
| Rectified Flow          | 尽量沿直线搬运耦合样本的速度场                 | 否               | ODE，可用很粗步长            | 需要 reflow 才能持续拉直                    |
| MeanFlow                | 区间平均速度 `u(z_t,r,t)`                      | 否，但训练要 JVP | `z_0 = z_1 - u(z_1,0,1)`     | 训练目标更复杂，依赖正确 JVP                |

这里统一采用 Flow Matching 常见的 **噪声到数据** 方向：

$$
X_0 \sim p_0,\quad X_1 \sim p_\mathrm{data}.
$$

MeanFlow 原论文采用 **数据到噪声** 记号，后文会单独标注；只要把时间反过来，两种写法表达的是同一条插值路径。

## Normalizing Flow 与 Continuous Normalizing Flow

### 标准 Normalizing Flow

设 `z ~ p_Z` 是容易采样和计算密度的基分布，`x = f_theta(z)` 是可逆变换。换元公式给出：

$$
\log p_X(x)
= \log p_Z(z) - \log \left|\det \frac{\partial f_\theta(z)}{\partial z}\right|,
\quad z=f_\theta^{-1}(x).
$$

这也是标准 Normalizing Flow 的核心约束：`f_theta` 不只要表达力强，还必须可逆，并且 Jacobian determinant 不能太贵。RealNVP、Glow 这类模型的工程设计，大量精力都花在 coupling layer、可逆卷积、逐层 log-det 累加上。

训练最大化数据似然：

$$
\max_\theta\ \mathbb{E}_{x\sim p_\mathrm{data}}
\left[
\log p_Z(f_\theta^{-1}(x))
+ \log \left|\det \frac{\partial f_\theta^{-1}(x)}{\partial x}\right|
\right].
$$

采样则很直接：

```python
def sample_normalizing_flow(base_dist, flow, batch_size):
    # z 是标准正态等简单分布的样本；flow.forward 是显式可逆映射的正向方向。
    z = base_dist.sample((batch_size,))
    x = flow.forward(z)
    return x
```

标准 NF 的优势是精确密度和快速采样，短板是可逆结构限制了网络设计自由度。

### Continuous Normalizing Flow

CNF 把离散可逆层的复合写成连续时间 ODE：

$$
\frac{dX_t}{dt}=v_\theta(X_t,t),\quad X_0\sim p_0.
$$

只要速度场满足足够的正则条件，ODE 解映射 `psi_t` 是可逆流。密度变化由 instantaneous change of variables 给出：

$$
\frac{d}{dt}\log p_t(X_t)
= -\operatorname{div} v_\theta(X_t,t)
= -\operatorname{Tr}\left(\frac{\partial v_\theta}{\partial x}(X_t,t)\right).
$$

所以从 `t=0` 积分到 `t=1`：

$$
\log p_1(X_1)
= \log p_0(X_0)
- \int_0^1 \operatorname{div} v_\theta(X_t,t)\,dt.
$$

代码化时，CNF 往往把状态扩展成 `(x_t, logp_t)` 一起积分：

```python
def cnf_augmented_dynamics(t, state, velocity_model):
    x_t, logp_t = state

    # velocity_model 输出 ODE 右端项 v_theta(x_t, t)。
    v_t = velocity_model(x_t, t)

    # divergence 是 CNF 的密度修正项。高维图像里直接求完整 Jacobian 太贵，
    # 常用 Hutchinson trace estimator 估计 Tr(dv/dx)。
    div_v = estimate_divergence_with_hutchinson(v_t, x_t)

    # 沿样本轨迹的 log-density 变化率是 -div(v)。
    dlogp_dt = -div_v
    return v_t, dlogp_dt
```

CNF 放松了显式可逆网络的结构要求，但训练和似然评估都要数值积分；Flow Matching 后续的动机之一，就是保留 ODE 生成能力，同时把训练改成 simulation-free 的监督回归。

## 连续性方程和两个核心定理

### 速度场如何生成概率路径

给定速度场 `u_t(x)`，样本轨迹满足：

$$
\frac{dX_t}{dt}=u_t(X_t).
$$

如果 `X_t` 的分布是 `p_t`，则 `u_t` 和 `p_t` 必须满足连续性方程：

$$
\partial_t p_t(x) + \nabla\cdot\left(p_t(x)u_t(x)\right)=0.
$$

弱形式证明更直观。取任意平滑测试函数 `phi`：

$$
\frac{d}{dt}\mathbb{E}[\phi(X_t)]
= \mathbb{E}\left[\nabla\phi(X_t)\cdot u_t(X_t)\right].
$$

把期望写成积分并分部积分：

$$
\frac{d}{dt}\int \phi(x)p_t(x)\,dx
= \int \nabla\phi(x)\cdot u_t(x)p_t(x)\,dx
= -\int \phi(x)\nabla\cdot(p_tu_t)(x)\,dx.
$$

由于这对任意 `phi` 成立，就得到连续性方程。它的含义不是“某个粒子守恒”，而是“概率质量沿速度场流动时不会凭空产生或消失”。

### 条件路径边缘化定理

Flow Matching 实际可操作的是条件路径。给定数据点 `z ~ q(z)`，构造条件概率路径 `p_t(x|z)` 和生成它的条件速度场 `u_t(x|z)`。边缘路径定义为：

$$
p_t(x)=\int p_t(x|z)q(z)\,dz.
$$

边缘速度场定义为条件速度场的后验加权平均：

$$
u_t(x)
= \int u_t(x|z)\frac{p_t(x|z)q(z)}{p_t(x)}\,dz.
$$

这个公式里的权重就是 `z` 在给定 `x_t=x` 后的后验密度。把它代回连续性方程：

$$
\begin{aligned}
\partial_t p_t(x)
&= \int \partial_t p_t(x|z)q(z)\,dz \\
&= -\int \nabla\cdot\left(p_t(x|z)u_t(x|z)\right)q(z)\,dz \\
&= -\nabla\cdot\left(
\int p_t(x|z)u_t(x|z)q(z)\,dz
\right) \\
&= -\nabla\cdot\left(p_t(x)u_t(x)\right).
\end{aligned}
$$

所以条件速度场虽然只知道“从噪声到某个数据点”的局部目标，边缘化后仍能生成整体数据路径。

### CFM 和 FM 为什么等价

理想 Flow Matching loss 是：

$$
\mathcal{L}_\mathrm{FM}(\theta)
= \mathbb{E}_{t,x\sim p_t}
\left[
\left\|v_\theta(x,t)-u_t(x)\right\|^2
\right].
$$

问题是 `u_t(x)` 包含上面的后验积分，通常不可计算。Conditional Flow Matching 改用条件速度场：

$$
\mathcal{L}_\mathrm{CFM}(\theta)
= \mathbb{E}_{t,z\sim q,x\sim p_t(\cdot|z)}
\left[
\left\|v_\theta(x,t)-u_t(x|z)\right\|^2
\right].
$$

令随机变量 `U = u_t(X_t|Z)`，则边缘速度场满足：

$$
u_t(x)=\mathbb{E}[U|X_t=x].
$$

展开平方项：

$$
\begin{aligned}
\mathcal{L}_\mathrm{CFM}
&= \mathbb{E}\|v_\theta(X_t,t)\|^2
-2\mathbb{E}\left[v_\theta(X_t,t)\cdot U\right]
+\mathbb{E}\|U\|^2, \\
\mathcal{L}_\mathrm{FM}
&= \mathbb{E}\|v_\theta(X_t,t)\|^2
-2\mathbb{E}\left[v_\theta(X_t,t)\cdot u_t(X_t)\right]
+\mathbb{E}\|u_t(X_t)\|^2.
\end{aligned}
$$

中间交叉项相同，因为：

$$
\mathbb{E}\left[v_\theta(X_t,t)\cdot U\right]
= \mathbb{E}\left[
v_\theta(X_t,t)\cdot \mathbb{E}[U|X_t]
\right].
$$

剩下的差别不依赖 `theta`：

$$
\mathcal{L}_\mathrm{CFM}(\theta)-\mathcal{L}_\mathrm{FM}(\theta)
= \mathbb{E}\|U\|^2-\mathbb{E}\|u_t(X_t)\|^2.
$$

因此两者梯度相同。训练时回归条件速度场，最优解却是边缘速度场。

## Flow Matching

### Gaussian 条件路径

Flow Matching 原论文把一大类路径写成 Gaussian 条件概率路径：

$$
p_t(x|z)=\mathcal{N}\left(x\mid \alpha_t z,\ \beta_t^2 I\right),
$$

其中：

$$
\alpha_0=0,\quad \beta_0=1,\quad \alpha_1=1,\quad \beta_1\approx 0.
$$

采样写成重参数化形式：

$$
x_t=\alpha_t z+\beta_t\epsilon,\quad \epsilon\sim\mathcal{N}(0,I).
$$

条件速度场由 `x_t` 对时间求导得到：

$$
u_t(x_t|z)=\dot{\alpha}_t z+\dot{\beta}_t\epsilon.
$$

如果要写成 `x` 和 `z` 的函数，利用 `epsilon=(x-\alpha_t z)/\beta_t`：

$$
u_t(x|z)
= \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z
+ \frac{\dot{\beta}_t}{\beta_t}x.
$$

这一行是教程和代码对齐的关键：代码里最稳定的写法通常不是先还原上式，而是直接用重参数化得到 `target = alpha_dot * z + beta_dot * eps`。

### CondOT 线性路径

最常用的 CondOT 路径取：

$$
\alpha_t=t,\quad \beta_t=1-t.
$$

于是：

$$
x_t=(1-t)\epsilon+t z,\quad
u_t(x_t|z)=z-\epsilon.
$$

等价地，若只给定 `x_t`：

$$
u_t(x|z)=\frac{z-x}{1-t},\quad t<1.
$$

这两个式子不要混用错：`x_t=(1-t)eps+t z` 对应的方差是 `(1-t)^2I`，不是 `(1-t^2)I`。

### 训练目标和代码对照

CondOT CFM loss：

$$
\mathcal{L}_\mathrm{CFM}(\theta)
=
\mathbb{E}_{t,z,\epsilon}
\left[
\left\|
v_\theta((1-t)\epsilon+tz,t)-(z-\epsilon)
\right\|^2
\right].
$$

教程级代码如下：

```python
def condot_flow_matching_loss(model, data_batch):
    # data_batch: z ~ p_data，形状可以是 [B, C, H, W] 或 [B, D]。
    z = data_batch

    # epsilon 是源分布样本。CondOT 线性路径从 epsilon 出发，走向数据 z。
    eps = torch.randn_like(z)

    # 每个样本单独抽时间，避免模型只学习少数固定时间切片。
    # view_shape 用来把 [B] 的时间广播到图像/向量维度。
    t = torch.rand(z.shape[0], device=z.device)
    view_shape = (z.shape[0],) + (1,) * (z.ndim - 1)
    t_view = t.view(view_shape)

    # 条件路径样本 x_t = (1-t) eps + t z。
    # 这个对象在公式里服从 p_t(.|z)，在代码里就是模型的 noisy input。
    x_t = (1.0 - t_view) * eps + t_view * z

    # 条件速度场是路径对时间的导数：d/dt [(1-t)eps + t z] = z - eps。
    # 梯度不需要流向 target；监督信号来自手工构造的概率路径。
    target_velocity = z - eps

    pred_velocity = model(x_t, t)
    return torch.mean((pred_velocity - target_velocity) ** 2)
```

`Flow Matching Guide and Code` 里的库接口也是同一件事：`path.sample(t, x_0, x_1)` 返回 `x_t` 和 `dx_t`，然后用 `MSE(velocity_model(x_t,t), dx_t)`。`dx_t` 就是公式里的 `dot psi_t`。

### 推理

训练完成后，模型近似边缘速度场。采样从 `x_0 ~ N(0,I)` 开始，解 ODE：

$$
\frac{dX_t}{dt}=v_\theta(X_t,t),\quad t:0\to 1.
$$

Euler：

$$
X_{t+h}=X_t+h\,v_\theta(X_t,t).
$$

Heun：

$$
\tilde{X}_{t+h}=X_t+h\,v_\theta(X_t,t),
$$

$$
X_{t+h}=X_t+\frac{h}{2}
\left[
v_\theta(X_t,t)+v_\theta(\tilde{X}_{t+h},t+h)
\right].
$$

推理代码骨架：

```python
@torch.no_grad()
def sample_flow_matching(model, shape, steps):
    x = torch.randn(shape)
    times = torch.linspace(0.0, 1.0, steps + 1, device=x.device)

    for t0, t1 in zip(times[:-1], times[1:]):
        h = t1 - t0
        t_batch = t0.expand(shape[0])

        # Euler 是一阶 ODE 求解器；路径越直、速度场越平滑，少步数误差越小。
        v = model(x, t_batch)
        x = x + h * v

    return x
```

FM 的“simulation-free”只指训练时不需要 rollout ODE；生成时仍要积分。Rectified Flow 和 MeanFlow 主要就是围绕这一点做推理加速。

## 概率路径设计：Gaussian、OT 与 CondOT

概率路径设计属于 Flow Matching 的核心问题：先规定 `p_t` 应该怎样从噪声走到数据，再推导生成它的速度场和训练 target。Normalizing Flow 关注的是可逆变换的密度；Flow Matching 关注的是“我想让概率质量沿哪条路径走”。

### Gaussian path：把 diffusion 写成概率路径

Gaussian 条件路径统一写成：

$$
p_t(x|z)=\mathcal{N}\left(x\mid \alpha_t z,\ \beta_t^2 I\right).
$$

它的训练 target 是：

$$
u_t(x_t|z)=\dot{\alpha}_t z+\dot{\beta}_t\epsilon,\quad
x_t=\alpha_t z+\beta_t\epsilon.
$$

这里的重点不只是“加噪”。`alpha_t` 决定数据成分保留多少，`beta_t` 决定噪声尺度；不同 diffusion noise schedule 可以被看成不同 Gaussian probability path。Flow Matching 的好处是可以绕开 SDE 反向推导，直接指定路径并回归对应速度。

常见 VP/VE diffusion path 和 FM 的关系可以这样看：

- VP path：数据系数随时间衰减，方差保持在受控范围内，适合和 score-based diffusion 的概率流 ODE 对齐。
- VE path：均值常以数据点为中心，噪声方差从大到小变化，强调从大噪声尺度逐步去噪。
- CondOT path：取 `alpha_t=t, beta_t=1-t`，条件路径是直线，target 退化成最简单的 `z-eps`。

### OT path：最小动能的边缘概率路径

动态最优传输可以写成 Benamou-Brenier 形式：

$$
\min_{p_t,u_t}
\int_0^1\int \|u_t(x)\|^2p_t(x)\,dx\,dt
$$

约束是：

$$
p_0=p,\quad p_1=q,\quad
\partial_t p_t+\nabla\cdot(p_tu_t)=0.
$$

如果存在从 `p_0` 到 `p_1` 的 OT map `phi`，对应的 displacement interpolation 是：

$$
\psi_t^\star(x)=(1-t)x+t\phi(x).
$$

沿这条路径，每个源点的速度是常数：

$$
\frac{d}{dt}\psi_t^\star(x)=\phi(x)-x.
$$

这解释了为什么 OT 路径常被认为更适合少步 ODE 采样：在理想 OT map 下，样本轨迹是直线，Euler 一步就能精确到达终点。难点是，真实数据分布下的全局 OT map 通常不可得；训练集里也不知道哪个噪声点应该配哪个数据点。

### CondOT path：每个条件路径直，不等于全局 OT 已解决

Flow Matching 原论文里的 OT 条件路径常写成：

$$
\psi_t(x_0|z)=(1-t)x_0+tz,
$$

其中 `x_0~N(0,I)`，`z~p_data`。这条条件路径对固定 `z` 来说是“从噪声到该数据点”的线性缩放高斯：

$$
p_t(x|z)=\mathcal{N}\left(x\mid tz,\ (1-t)^2I\right).
$$

它被称为 conditional OT / CondOT，是因为在“源分布到单点 Dirac delta”的条件问题里，线性收缩就是最自然的 OT 路径。它带来的代码 target 很干净：

$$
x_t=(1-t)\epsilon+tz,\quad
\dot{x}_t=z-\epsilon.
$$

但它不是在说“随机抽一个噪声和随机抽一个数据，二者已经是全局最优传输配对”。独立 coupling 下的 `epsilon -> z` 只是一个训练用条件构造；边缘速度场会把所有条件速度做后验平均。Rectified Flow 的 reflow 正是沿着这个缺口继续推进：先训练一个流，再用流生成的新 coupling 重新训练，让耦合更接近确定性传输。

## Stochastic Interpolants

Stochastic Interpolants 可以看成 Flow Matching 的更宽统一框架：先构造一个随机插值 `I_t`，它在 `t=0` 和 `t=1` 分别服从两个端点分布；再从这个插值里读出速度、score、ODE 和 SDE。它和 Flow Matching 的共同点是 simulation-free 回归；差别是它同时把 deterministic flow 和 stochastic diffusion 放进同一条密度路径里。

### 插值对象

设：

$$
X_0\sim \rho_0,\quad X_1\sim \rho_1,\quad Z\sim\mathcal{N}(0,I),
$$

其中 `(X_0,X_1)` 可以来自任意 coupling。典型线性 stochastic interpolant 写成：

$$
I_t=\alpha_tX_0+\beta_tX_1+\gamma_tZ.
$$

边界条件是：

$$
\alpha_0=1,\quad \beta_0=0,\quad \gamma_0=0,
$$

$$
\alpha_1=0,\quad \beta_1=1,\quad \gamma_1=0.
$$

于是 `I_0=X_0`，`I_1=X_1`。当 `gamma_t>0` 时，中间路径带有额外高斯桥噪声；当 `gamma_t=0` 时，它退化成 deterministic interpolation，Rectified Flow 的直线插值就是特殊情况：

$$
I_t=(1-t)X_0+tX_1.
$$

### 速度场：和 Flow Matching 同一个回归骨架

令 `rho_t` 是 `I_t` 的边缘密度。Stochastic Interpolants 定义 transport velocity：

$$
b(x,t)=\mathbb{E}\left[\dot{I}_t\mid I_t=x\right].
$$

它满足连续性方程：

$$
\partial_t\rho_t+\nabla\cdot(\rho_tb_t)=0.
$$

训练目标是：

$$
\mathcal{L}_b(\theta)
=
\mathbb{E}
\left[
\left\|
b_\theta(I_t,t)-\dot{I}_t
\right\|^2
\right].
$$

这和 CFM/RF 的 MSE 形式完全同构。区别在于 target 由更一般的 `I_t` 决定：

$$
\dot{I}_t
=\dot{\alpha}_tX_0+\dot{\beta}_tX_1+\dot{\gamma}_tZ.
$$

代码化：

```python
def stochastic_interpolant_velocity_loss(model_b, x0, x1):
    # x0 和 x1 来自某个 coupling。独立 coupling、OT coupling、reflow coupling 都可以。
    z = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    view = (x0.shape[0],) + (1,) * (x0.ndim - 1)

    alpha, beta, gamma = schedule_values(t, view)
    alpha_dot, beta_dot, gamma_dot = schedule_derivatives(t, view)

    # I_t 是主对象：所有 loss 都在 I_t 的边缘密度 rho_t 上训练。
    i_t = alpha * x0 + beta * x1 + gamma * z

    # 速度监督来自插值对时间的导数，不需要 rollout ODE。
    target_b = alpha_dot * x0 + beta_dot * x1 + gamma_dot * z

    pred_b = model_b(i_t, t)
    return torch.mean((pred_b - target_b) ** 2)
```

若取 `alpha_t=1-t, beta_t=t, gamma_t=0`，并令 `x0=eps, x1=data`，这个 loss 就回到 Rectified Flow / CondOT 的直线速度回归。

### Score：随机采样时多出来的对象

当 `gamma_t>0` 时，可以从插值里的高斯变量得到 score：

$$
s(x,t)=\nabla_x\log\rho_t(x)
=-\frac{1}{\gamma_t}\mathbb{E}[Z\mid I_t=x].
$$

因此 score 网络可以用 denoising 形式训练：

$$
\mathcal{L}_s(\theta)
=
\mathbb{E}
\left[
\left\|
s_\theta(I_t,t)+\frac{Z}{\gamma_t}
\right\|^2
\right].
$$

代码化时要避开 `gamma_t` 接近 0 的端点，或对时间采样区间做截断：

```python
def stochastic_interpolant_score_loss(model_s, x0, x1, eps=1e-4):
    z = torch.randn_like(x0)

    # score target 里有 1/gamma_t，通常不要在端点采样。
    t = eps + (1.0 - 2.0 * eps) * torch.rand(x0.shape[0], device=x0.device)
    view = (x0.shape[0],) + (1,) * (x0.ndim - 1)

    alpha, beta, gamma = schedule_values(t, view)
    i_t = alpha * x0 + beta * x1 + gamma * z

    target_s = -z / gamma
    pred_s = model_s(i_t, t)
    return torch.mean((pred_s - target_s) ** 2)
```

这个 score 不是 Flow Matching 必需的；只有当采样器使用 SDE 或需要 diffusion-style 反向过程时，它才成为主角。

### 同一条路径，ODE 和 SDE 都能走

如果只用速度场，采样可以走 probability flow ODE：

$$
dY_t=b(Y_t,t)\,dt.
$$

如果还学了 score，同一条边缘密度路径 `rho_t` 也可以由一族 SDE 生成。取任意非负扩散强度 `a_t`：

$$
dY_t=\left[b(Y_t,t)+a_t s(Y_t,t)\right]dt+\sqrt{2a_t}\,dW_t.
$$

它的 Fokker-Planck 方程会抵消 score 项带来的扩散修正，仍然得到：

$$
\partial_t\rho_t+\nabla\cdot(\rho_tb_t)=0.
$$

这就是 Stochastic Interpolants 的统一意义：`I_t` 定义密度路径，`b_t` 负责概率质量的确定性搬运，`s_t` 让同一条路径也能用带噪声的扩散过程采样。Flow Matching 更偏向只学 `b_t` 的 ODE 视角；score-based diffusion 更强调 `s_t`；Stochastic Interpolants 把两者放在同一个插值对象下面。

### 和主线方法的对应关系

| 选择                               | 回到哪类方法                     | 解释                                    |
| ---------------------------------- | -------------------------------- | --------------------------------------- |
| `gamma_t=0`，线性 `alpha,beta`     | Rectified Flow / CondOT 速度回归 | target 是 `X_1-X_0`                     |
| 固定 `X_1=z`，`X_0~N(0,I)`         | Flow Matching 的条件路径         | 条件路径边缘化得到整体速度              |
| `gamma_t>0`，同时学 `b_t` 和 `s_t` | diffusion / score-based 采样视角 | ODE 与 SDE 可共享同一 `rho_t`           |
| 改变 `(X_0,X_1)` coupling          | OT、mini-batch OT、reflow 等     | coupling 改变速度 target 和轨迹交叉程度 |

Stochastic Interpolants 因此适合作为主线的“桥”：它解释了为什么 FM、RF、diffusion 的训练代码都像 MSE，却分别在学习速度、score 或平均速度的不同对象。

## Rectified Flow

Rectified Flow 的入口不是先指定一个漂亮的边缘概率路径，而是先拿到一个 coupling：

$$
(X_0,X_1)\sim \pi,\quad X_0\sim \pi_0,\quad X_1\sim \pi_1.
$$

在生成任务里，最常见的初始 coupling 是独立配对：`X_0` 是高斯噪声，`X_1` 是数据样本。给定一对样本，定义直线插值：

$$
X_t=(1-t)X_0+tX_1.
$$

直线的瞬时速度是常数：

$$
\frac{dX_t}{dt}=X_1-X_0.
$$

Rectified Flow 的训练目标是：

$$
\min_\theta
\int_0^1
\mathbb{E}
\left[
\left\|v_\theta(X_t,t)-(X_1-X_0)\right\|^2
\right]dt.
$$

最优速度场满足：

$$
v^\star(x,t)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

这个条件期望是理解 Rectified Flow 的第一道门槛：模型不是拿着隐藏的 `(X_0,X_1)` 配对做查询，也不是记住每个训练 pair 的直线；在同一个空间点 `x` 和同一时间 `t` 上，如果许多直线插值交叉，模型只能输出这些候选速度的平均结果。

### 动机：把非因果直线插值变成可采样的 ODE 流

直线插值 `X_t=(1-t)X_0+tX_1` 很诱人，因为它一步就知道终点；但它本身不是一个可直接采样的 deterministic flow。原因是不同 pair 的直线可能在中途交叉：同一个 `x_t` 可能对应多个不同终点和多个不同速度。ODE 速度场必须是单值函数 `v(x,t)`，不能在同一个状态同时走向多个方向。

Rectified Flow 的训练目标做的事，是把这些“非因果”的 pairwise 直线速度投影成一个单值速度场：

$$
v^\star(x,t)=\mathbb{E}[X_1-X_0\mid X_t=x].
$$

这个速度场生成的 ODE：

$$
\frac{dZ_t}{dt}=v^\star(Z_t,t),\quad Z_0\sim \pi_0,
$$

在合适正则条件下仍然生成从 `pi_0` 到 `pi_1` 的边缘分布路径。直观上，它把“每个 pair 自己画一条线”的训练信号，整理成“任意位置只允许一个速度”的生成器。

### 误区：不是简单拟合配对点，也不是保证每条轨迹都直

把 Rectified Flow 理解成“让模型学习从噪声点到数据点走直线”只说对了一半。更准确的说法是：

- 训练监督来自配对点直线的导数 `X_1-X_0`。
- 模型最优解是条件期望速度 `E[X_1-X_0|X_t=x]`。
- 学到的 ODE 轨迹 `Z_t` 由单值速度场决定，不再携带原始 pair 的身份。
- 第一轮独立 coupling 下，轨迹可能仍然弯，也可能因为交叉处速度平均而偏离任意单条训练直线。

所以“直线”在 Rectified Flow 里更像训练信号和优化方向，不是第一轮模型自动获得的逐样本几何保证。真正让低步数采样变稳的，是后面的 reflow：用已经学到的 ODE 重新生成 coupling，再训练下一轮。

### 1-Rectified Flow：第一轮从独立 coupling 开始

如果 `X_0=epsilon~N(0,I)`、`X_1=z~p_data` 且采用独立耦合，第一轮训练常被称为 1-Rectified Flow。它和 CondOT CFM 在代码形态上非常接近：

```python
def rectified_flow_loss(model, x1):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    t_view = t.view((x1.shape[0],) + (1,) * (x1.ndim - 1))

    # 线性插值路径。和 CondOT 的 x_t=(1-t)eps+t z 是同一个对象。
    x_t = (1.0 - t_view) * x0 + t_view * x1

    # Rectified Flow 强调的是耦合对的直线速度。
    # 若耦合改变，target 也随之改变。
    target = x1 - x0
    pred = model(x_t, t)
    return torch.mean((pred - target) ** 2)
```

这段代码和 CondOT 的区别主要在解释重心：CondOT 说“这是条件概率路径的速度”；Rectified Flow 说“这是当前 coupling 的直线运输信号，下一步我要把 coupling rectified”。

### 2-Rectified Flow / Reflow：重新构造更好的 coupling

第一轮用独立耦合训练后，ODE 产生的是一个确定性映射：

$$
Z_1=\operatorname{ODEsolve}(X_0;v_\theta),\quad X_0\sim\pi_0.
$$

Reflow 用新的 coupling `(X_0,Z_1)` 再训练同样的直线目标：

$$
X_t^\mathrm{new}=(1-t)X_0+tZ_1,\quad
target=Z_1-X_0.
$$

教程级 reflow 伪代码：

```python
@torch.no_grad()
def build_reflow_pairs(flow_model, x0_batch, ode_steps):
    # x0_batch 来自源分布。第一轮通常是标准高斯噪声。
    x0 = x0_batch

    # 用 1-Rectified Flow 的 ODE 采样终点。
    # 这个 z1 不是原数据集中随机拿来的 x1，而是当前 flow 从同一个 x0 送到的终点。
    z1 = ode_solve(flow_model, x0, t0=0.0, t1=1.0, steps=ode_steps)
    return x0, z1

def reflow_loss(student_model, x0, z1):
    t = torch.rand(x0.shape[0], device=x0.device)
    t_view = t.view((x0.shape[0],) + (1,) * (x0.ndim - 1))

    # 2-RF 重新拟合 teacher flow 诱导出的 pair。
    x_t = (1.0 - t_view) * x0 + t_view * z1
    target = z1 - x0
    return torch.mean((student_model(x_t, t) - target) ** 2)
```

论文证明 rectification 会把任意 coupling 变成确定性 coupling，并且凸 transport cost 不增加。递归 reflow 的目标是让新 coupling 更接近“同一个起点直接到同一个终点”的 causal transport；轨迹交叉减少后，ODE 速度场更接近常向量场。

工程上这意味着同样的生成质量可能用更少 ODE 步数达到，极端情况下可以用单步 Euler：

$$
X_1\approx X_0+v_\theta(X_0,0).
$$

这不是因为 ODE 不存在了，而是因为 reflow 后的速度场更接近“从起点直接指向终点”的常速度场。

### 和 Flow Matching 的关系

两者可以这样区分：

- Flow Matching 的核心是 **先指定概率路径，再用条件速度场训练边缘速度场**；它自然容纳 Gaussian diffusion path、CondOT path、流形/离散扩展。
- Rectified Flow 的核心是 **给定 coupling 后沿直线回归，并通过 reflow 改善 coupling**；它最关心轨迹交叉能否减少、ODE 是否能用少步甚至一步近似。
- CondOT CFM 和单轮 Rectified Flow 在常见噪声-数据独立耦合下有相同的代码形态，但论文问题意识不同。

## MeanFlow

MeanFlow 的论文记号采用数据到噪声方向：

$$
z_t=(1-t)x+t\epsilon,\quad x\sim p_\mathrm{data},\quad \epsilon\sim\mathcal{N}(0,I).
$$

条件瞬时速度是：

$$
v_t=\epsilon-x.
$$

Flow Matching 学的是瞬时速度 `v(z_t,t)`。MeanFlow 改学区间 `[r,t]` 上的平均速度：

$$
u(z_t,r,t)
=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau,\quad r<t.
$$

平均速度直接给出区间更新：

$$
z_r=z_t-(t-r)u(z_t,r,t).
$$

一阶采样时，取 `t=1,r=0,z_1=epsilon`：

$$
z_0=z_1-u(z_1,0,1).
$$

### MeanFlow identity

平均速度定义等价于：

$$
(t-r)u(z_t,r,t)=\int_r^t v(z_\tau,\tau)\,d\tau.
$$

对 `t` 求全导数，左边得到：

$$
\frac{d}{dt}\left[(t-r)u(z_t,r,t)\right]
=u(z_t,r,t)+(t-r)\frac{d}{dt}u(z_t,r,t).
$$

右边由微积分基本定理得到 `v(z_t,t)`，因此：

$$
v(z_t,t)=u(z_t,r,t)+(t-r)\frac{d}{dt}u(z_t,r,t).
$$

也就是：

$$
u(z_t,r,t)=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t).
$$

其中全导数为：

$$
\frac{d}{dt}u(z_t,r,t)
=
\partial_z u(z_t,r,t)\,v(z_t,t)
+\partial_t u(z_t,r,t),
$$

因为 `r` 固定，所以 `dr/dt=0`。这正是 JVP 的切向量：

$$
(\dot{z}_t,\dot{r},\dot{t})=(v_t,0,1).
$$

### 训练目标和 JVP 代码

MeanFlow 参数化 `u_theta(z,r,t)`，把上面的恒等式改成有效回归目标：

$$
u_\mathrm{tgt}
=v_t-(t-r)
\left(
v_t\partial_z u_\theta+\partial_tu_\theta
\right).
$$

训练 loss：

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\left\|
u_\theta(z_t,r,t)-\operatorname{sg}(u_\mathrm{tgt})
\right\|^2
\right].
$$

`sg` 是 stop-gradient。它让 JVP 出现在 target 里，但优化时不需要对 JVP 再做二阶反传。

```python
def meanflow_loss(mean_velocity_model, x):
    # MeanFlow 论文约定：t=0 是数据，t=1 是噪声。
    eps = torch.randn_like(x)

    # sample_t_r 需要保证 0 <= r <= t <= 1。
    # 实践中论文会混合 r=t 和 r<t；r=t 时退化为标准 Flow Matching。
    r, t = sample_ordered_times(batch_size=x.shape[0], device=x.device)
    t_view = t.view((x.shape[0],) + (1,) * (x.ndim - 1))
    r_view = r.view((x.shape[0],) + (1,) * (x.ndim - 1))

    z_t = (1.0 - t_view) * x + t_view * eps
    v_t = eps - x

    def fn(z_arg, r_arg, t_arg):
        return mean_velocity_model(z_arg, r_arg, t_arg)

    # JVP 计算 d/dt u_theta(z_t,r,t)。
    # tangent=(v_t, 0, 1) 对应 dz_t/dt=v_t, dr/dt=0, dt/dt=1。
    u_pred, du_dt = torch.func.jvp(
        fn,
        (z_t, r, t),
        (v_t, torch.zeros_like(r), torch.ones_like(t)),
    )

    u_tgt = v_t - (t_view - r_view) * du_dt

    # target stop-gradient 避免二阶优化；梯度只更新 u_pred 这一侧。
    return torch.mean((u_pred - u_tgt.detach()) ** 2)
```

采样：

```python
@torch.no_grad()
def sample_meanflow(mean_velocity_model, shape):
    z_1 = torch.randn(shape)
    batch = shape[0]
    r = torch.zeros(batch, device=z_1.device)
    t = torch.ones(batch, device=z_1.device)

    # 一次函数评估把噪声端 z_1 推到数据端 z_0。
    return z_1 - mean_velocity_model(z_1, r, t)
```

MeanFlow 的关键不是“把 FM 的 ODE solver 换成一步 Euler”，而是模型输出的对象已经变了：它预测的是跨时间区间的平均速度，所以 `z_t -> z_r` 的积分被折叠进一个网络调用。

## 条件生成与 Classifier-Free Guidance

条件生成不改变 Flow Matching 的数学骨架，只是把数据分布从 `p_data(x)` 换成条件分布 `p_data(x|c)`。条件 `c` 可以是类别标签、文本 embedding、低分辨率图像、视频首帧、语音特征或任意上下文。

速度场从：

$$
v_\theta(x,t)
$$

变成：

$$
v_\theta(x,t,c).
$$

CondOT 条件生成训练目标为：

$$
\mathcal{L}_\mathrm{cond}(\theta)
=
\mathbb{E}_{(z,c),\epsilon,t}
\left[
\left\|
v_\theta((1-t)\epsilon+tz,t,c)-(z-\epsilon)
\right\|^2
\right].
$$

代码结构只多了一个条件输入：

```python
def conditional_flow_matching_loss(model, image, condition):
    # image 是数据 z，condition 可以是类别 id、文本 embedding 或其他条件 token。
    eps = torch.randn_like(image)
    t = torch.rand(image.shape[0], device=image.device)
    t_view = t.view((image.shape[0],) + (1,) * (image.ndim - 1))

    x_t = (1.0 - t_view) * eps + t_view * image
    target = image - eps

    pred = model(x_t, t, condition)
    return torch.mean((pred - target) ** 2)
```

### 条件进入网络的位置

不同条件类型对应不同接口，但本质都是让网络在估计同一个时间切片速度时看到上下文：

- 类别条件：把 class id 变成 embedding，加到 time embedding 或作为 AdaLN/FiLM 调制项。
- 文本条件：用文本编码器得到 token embedding，通过 cross-attention 或 joint attention 注入 DiT/U-Net。
- 图像/视频条件：把低分辨率帧、mask、首帧或参考图编码成额外 token/channel，和 noisy latent 一起送入网络。

条件生成的关键边界是：条件 `c` 不参与 `x_t=(1-t)eps+tz` 的几何插值；它改变的是速度场估计，即“在这个条件下，当前 noisy state 应该往哪类数据流动”。

### Classifier-Free Guidance

Classifier-Free Guidance 训练时随机丢弃条件，让同一个模型同时学有条件速度和无条件速度：

```python
def cfg_training_loss(model, image, condition, null_condition, drop_prob=0.1):
    eps = torch.randn_like(image)
    t = torch.rand(image.shape[0], device=image.device)
    t_view = t.view((image.shape[0],) + (1,) * (image.ndim - 1))

    x_t = (1.0 - t_view) * eps + t_view * image
    target = image - eps

    # mask=True 的样本使用空条件，迫使模型也会估计无条件速度场。
    drop = torch.rand(image.shape[0], device=image.device) < drop_prob
    mixed_condition = replace_condition(condition, null_condition, drop)

    pred = model(x_t, t, mixed_condition)
    return torch.mean((pred - target) ** 2)
```

采样时同时计算有条件和无条件速度：

$$
v_\mathrm{cond}=v_\theta(x,t,c),\quad
v_\mathrm{uncond}=v_\theta(x,t,\varnothing).
$$

CFG 组合为：

$$
v_\mathrm{cfg}
=v_\mathrm{uncond}
+s\left(v_\mathrm{cond}-v_\mathrm{uncond}\right).
$$

当 `s=1` 时就是普通条件速度；`s>1` 会更强地贴近条件，但也可能降低多样性或引入过饱和、过锐化等问题。

```python
@torch.no_grad()
def guided_velocity(model, x_t, t, condition, null_condition, guidance_scale):
    v_cond = model(x_t, t, condition)
    v_uncond = model(x_t, t, null_condition)
    return v_uncond + guidance_scale * (v_cond - v_uncond)
```

Flow Matching、Rectified Flow 和 MeanFlow 都可以做条件生成；差别只是模型输出对象不同。FM/RF 的 CFG 组合的是瞬时速度，MeanFlow 的 CFG 组合的是平均速度。

## 推理、加速和工程取舍

### ODE 求解器

Flow Matching 和 Rectified Flow 推理时都在解 ODE。常见选择：

- Euler：一阶，NFE 等于步数；路径直时很好用，路径弯时误差明显。
- Heun / midpoint：二阶，每步通常 2 NFE；少步采样时常比 Euler 稳。
- 自适应 ODE solver：能控制误差，但在大规模图像生成中调度和吞吐未必划算。

### 为什么 OT/直线路径有利于少步采样

CondOT 路径的条件流是直线：

$$
\psi_t(x_0|z)=(1-t)x_0+tz.
$$

如果边缘速度场也足够接近直线，Euler 的局部线性假设就更准。Flow Matching 原论文报告 OT 路径相比 diffusion path 更利于少 NFE 采样；Rectified Flow 进一步把“变直”作为训练-再训练机制；MeanFlow 则直接学习区间平均速度，绕过多步积分。

### 训练目标和推理目标的错位

Flow Matching 训练的是每个时间切片上的瞬时速度回归。即使训练 loss 很低，推理仍会累积 ODE 离散误差。常见补救路径：

- 用更好的 solver 或时间网格，减少数值误差。
- 用 Rectified Flow / reflow 让轨迹更直。
- 用 distillation / consistency / shortcut 类方法把多步模型压成少步模型。
- 用 MeanFlow 直接学习跨区间平均速度。

这些路线都在处理同一个工程矛盾：训练时希望监督信号简单，推理时希望函数评估次数少。

## 方法对照表

| 主题                    | 关键公式                               | 代码里的 target                             | 推理成本                       |
| ----------------------- | -------------------------------------- | ------------------------------------------- | ------------------------------ |
| Standard NF             | `log p_X = log p_Z - log det J_f`      | 最大似然，不是速度回归                      | 一次可逆前向                   |
| CNF                     | `d log p / dt = -div v`                | 最大似然或其他路径目标                      | ODE 积分                       |
| Gaussian path           | `x_t=alpha_t z+beta_t eps`             | `alpha_dot*z+beta_dot*eps`                  | 取决于路径弯曲和 solver        |
| Stochastic Interpolants | `I_t=alpha_t X_0+beta_t X_1+gamma_t Z` | `dot I_t`；SDE 还要 `-Z/gamma_t` 训练 score | 同一密度路径可用 ODE 或 SDE    |
| FM / CondOT             | `x_t=(1-t)eps+tz`                      | `z - eps`                                   | 多步 ODE                       |
| Rectified Flow          | `X_t=(1-t)X_0+tX_1`                    | `X_1-X_0`                                   | reflow 后可少步                |
| MeanFlow                | `z_r=z_t-(t-r)u(z_t,r,t)`              | `v_t-(t-r)JVP(u_theta)`                     | 目标是一阶                     |
| 条件生成 / CFG          | `v_cfg=v_uncond+s(v_cond-v_uncond)`    | 同主模型 target，额外输入条件               | guidance scale 影响质量/多样性 |

## 理解检查

1. **为什么 CFM 可以替代 FM？**

   因为条件速度场 `U=u_t(X_t|Z)` 的条件期望就是边缘速度场 `u_t(X_t)`。平方损失展开后，与参数有关的二次项和交叉项相同，差别只剩与 `theta` 无关的常数。

2. **CondOT 的 target 为什么是 `z - eps`？**

   对 `x_t=(1-t)eps+tz` 关于 `t` 求导即可得到 `z-eps`。如果把路径方差误写成 `(1-t^2)I`，这个 target 就不再对应同一条路径。

3. **Stochastic Interpolants 和 Flow Matching 的关系是什么？**

   两者都用 MSE 回归插值路径的速度；Stochastic Interpolants 允许更一般的 coupling 和额外高斯桥噪声，并且显式把 score 加进框架，从而同时覆盖 ODE flow 和 SDE diffusion。

4. **Rectified Flow 和 CondOT Flow Matching 是不是同一个东西？**

   在独立噪声-数据耦合和线性路径下，训练 loss 形式相同；但 Rectified Flow 的核心是 coupling 和 reflow，Flow Matching 的核心是条件路径、边缘化定理和更一般的概率路径设计。

5. **CondOT 为什么不等于全局 OT 已经解决？**

   CondOT 只是在固定数据点的条件问题中使用线性收缩路径；独立噪声-数据 pair 不一定是全局最优传输配对。真正的全局 OT path 需要源点到数据点的 OT map。

6. **MeanFlow 为什么可以一阶采样？**

   它预测的是 `[r,t]` 区间平均速度。若 `u(z_t,r,t)` 学准了，`z_r=z_t-(t-r)u(z_t,r,t)` 本身就是积分结果，不需要把区间拆成很多瞬时速度步。

7. **条件生成里的条件变量改变了什么？**

   条件 `c` 不改变 `x_t=(1-t)eps+tz` 的插值公式；它改变速度场估计 `v_theta(x,t,c)`。CFG 则在采样时组合无条件速度和有条件速度。

8. **连续性方程在整条主线里承担什么角色？**

   它把“样本沿 ODE 走”翻译成“密度随时间怎么变”。条件路径边缘化、边缘速度场生成边缘路径、CNF 的 log-density 更新，本质都在使用这个守恒关系。

## 来源

本教程优先依据项目内三份 PDF 和对应论文页面：

- `papers/2023_flow_matching_generative_modeling.pdf`：Yaron Lipman et al., [_Flow Matching for Generative Modeling_](https://arxiv.org/abs/2210.02747), arXiv:2210.02747, 2022/2023。
- `papers/flow_matching_guide_code.pdf`：Yaron Lipman et al., [_Flow Matching Guide and Code_](https://arxiv.org/abs/2412.06264), arXiv:2412.06264, 2024。
- `papers/intro_flow_matching_diffusion_models.pdf`：Peter Holderrieth and Ezra Erives, [_An Introduction to Flow Matching and Diffusion Models_](https://arxiv.org/abs/2506.02070), arXiv:2506.02070v3, 2026-03-18。
- Xingchao Liu, Chengyue Gong, Qiang Liu, [_Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow_](https://arxiv.org/abs/2209.03003), arXiv:2209.03003, 2022。
- Zhengyang Geng et al., [_Mean Flows for One-step Generative Modeling_](https://arxiv.org/abs/2505.13447), arXiv:2505.13447, 2025 tech report。
- Michael S. Albergo, Mark Goldstein, Nicholas M. Boffi, Rajesh Ranganath, Eric Vanden-Eijnden, [_Stochastic Interpolants: A Unifying Framework for Flows and Diffusions_](https://arxiv.org/abs/2303.08797), arXiv:2303.08797, 2023。
- Danilo Jimenez Rezende and Shakir Mohamed, [_Variational Inference with Normalizing Flows_](https://arxiv.org/abs/1505.05770), arXiv:1505.05770, 2015/2016。
- Laurent Dinh et al., [_Density estimation using Real NVP_](https://arxiv.org/abs/1605.08803), arXiv:1605.08803, 2016/2017。
- Ricky T. Q. Chen et al., [_Neural Ordinary Differential Equations_](https://arxiv.org/abs/1806.07366), arXiv:1806.07366, 2018/2019。
- Will Grathwohl et al., [_FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models_](https://arxiv.org/abs/1810.01367), arXiv:1810.01367, 2018。
- TongTong313, [_rectified-flow_](https://github.com/TongTong313/rectified-flow), GitHub reference implementation, used as auxiliary code-reading material for 1-Rectified/2-Rectified and condition-guided training intuition.
- 用户补充知乎资料 `https://zhuanlan.zhihu.com/p/11686643707`：作为辅助直觉来源处理；公式和算法事实仍以论文为准。
