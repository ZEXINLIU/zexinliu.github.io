---
layout: post
title: "Score-Based Diffusion Models: From Denoising Score Matching to SDE/ODE Samplers"
date: 2025-12-08 00:00:00
description: 从 score matching、DDPM 到 SDE/ODE 采样的扩散模型笔记。
tags: diffusion score-matching generative-models stochastic-processes
categories: ai-notes
---

本文包含的重点内容如下：

- **扩散模型不是 DDPM 的同义词。** DDPM 是 VP SDE 在离散时间、噪声预测参数化下的一种写法；NCSN/SMLD 更接近 VE SDE 的离散噪声层。难点不在记住模型名，而在看出它们都在学习一族带噪边缘分布的 score。
- **训练头和采样所需对象不是一回事。** `score_sde_pytorch` 的 legacy DDPM loss 拟合 `epsilon`，但采样器只认 `score_fn`；`epsilon`、`x0`、`v` 三种预测头必须能换回同一个 VP 条件 score，公式和源码接口才能对齐。
- **`marginal_prob` 是训练链路的核心接口。** 连续 SDE loss 不是先手写某个 DDPM timestep，而是由 `marginal_prob` 生成 `x_t = mean + std * z`，再把 DSM 目标写成 `score * std + z`。VPSDE 的 `log_mean_coeff` 正是线性 SDE 的积分解。
- **离散和连续之间要双向解释。** DDPM 到 VPSDE 需要把 `beta_i` 看成 `beta(t_i) Delta t`；SMLD 到 VESDE 需要把相邻噪声层方差差分看成 `d sigma^2(t)`。源码里的 `discretize` 则反过来把连续 SDE 落成采样器能执行的一步更新。
- **reverse SDE、PC sampler、DDPM ancestral、DDIM、probability flow ODE 解决的是不同推理取舍。** 它们共享已训练的 score 或噪声预测，但在随机性、步数、样本质量、是否支持 likelihood 上差异很大，不能混成“都是反向扩散”。
- **条件生成和 Latent Diffusion 改的是条件与空间，不是扩散数学的主干。** Classifier guidance/CFG 本质上是在采样时改 score 或网络输出；latent diffusion 则把同一套噪声预测搬到 autoencoder latent 空间，Stable Diffusion 的高效生成正来自这个空间迁移。
- **probability flow ODE 的价值不只是确定性采样。** 它和 reverse SDE 共享边缘分布，因此可以沿确定性轨迹做 latent encoding 和 likelihood/bits-per-dim 评估；代价是 ODE solver 和 divergence 估计比普通训练 loss 贵得多。

## 目录

- [0. 读代码前先固定符号](#0-读代码前先固定符号)
- [1. 参考脉络](#1-参考脉络)
- [2. 从 score matching 到 NCSN](#2-从-score-matching-到-ncsn)
- [3. DDPM 是 VP 离散链的一种写法](#3-ddpm-是-vp-离散链的一种写法)
- [4. SDE 统一框架](#4-sde-统一框架)
- [5. 训练调用链](#5-训练调用链)
- [6. Reverse SDE、PC sampling 和 ODE](#6-reverse-sdepc-sampling-和-ode)
- [7. 条件生成与 Latent Diffusion](#7-条件生成与-latent-diffusion)
- [8. 数学夹层：Ito、Fokker-Planck 和 likelihood](#8-数学夹层itofokker-planck-和-likelihood)
- [9. 理解检查](#9-理解检查)

## 0. 读代码前先固定符号

本文统一用数据到噪声的方向描述前向过程：

- `x_0 ~ p_data` 是真实数据。
- `x_t` 是第 `t` 个噪声水平的样本，连续时间里 `t in [0, T]`。
- `p_t(x)` 是 `x_t` 的边缘分布。
- `s_theta(x, t)` 目标是近似 `nabla_x log p_t(x)`。
- VP/DDPM 常写 `x_t = alpha_t x_0 + sigma_t epsilon`，其中 `alpha_t = sqrt(bar_alpha_t)`，`sigma_t = sqrt(1 - bar_alpha_t)`。
- VE/NCSN 常写 `x_t = x_0 + sigma_t epsilon`。

早期 NCSN 论文经常把采样方向写成“从大噪声到小噪声”，而 DDPM 习惯把训练前向写成“从数据到噪声”。看代码时要特别注意时间标签是否被翻转：`models/utils.py` 中 VE 离散模型用 `labels = sde.T - t`，因为旧 NCSN label 0 对应最大噪声。

## 1. 参考脉络

| 时间 | 工作                                                                                                                                               | 在本文中的角色                                                            |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 2005 | [Hyvarinen, Score Matching](https://www.jmlr.org/papers/v6/hyvarinen05a.html)                                                                      | 不求归一化常数，直接拟合 `nabla_x log p(x)`                               |
| 2011 | [Vincent, Denoising Score Matching](https://direct.mit.edu/neco/article-abstract/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising) | 用加噪条件分布的解析 score 避免 Hessian trace                             |
| 2015 | [Sohl-Dickstein et al., nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585)                                                           | 逐步加噪再反向生成，DDPM 的重要前身                                       |
| 2019 | [Song & Ermon, NCSN](https://arxiv.org/abs/1907.05600)                                                                                             | 多噪声 DSM + annealed Langevin dynamics                                   |
| 2020 | [Song & Ermon, NCSNv2](https://arxiv.org/abs/2006.09011)                                                                                           | 改进噪声尺度、网络与采样技巧                                              |
| 2020 | [Ho et al., DDPM](https://arxiv.org/abs/2006.11239)                                                                                                | VP 离散链、噪声预测目标、ancestral sampling                               |
| 2020 | [Song et al., DDIM](https://arxiv.org/abs/2010.02502)                                                                                              | 用非马尔可夫/确定性路径加速 DDPM 采样                                     |
| 2021 | [Song et al., Score SDE](https://arxiv.org/abs/2011.13456)                                                                                         | 用 SDE 统一 NCSN、DDPM、ODE likelihood 和 PC sampling                     |
| 2021 | [Dhariwal & Nichol, Guided Diffusion](https://arxiv.org/abs/2105.05233)                                                                            | 用 classifier guidance 把类别梯度加入采样 score                           |
| 2021 | [Rombach et al., Latent Diffusion](https://arxiv.org/abs/2112.10752)                                                                               | 把扩散过程迁移到 autoencoder latent 空间，Stable Diffusion 的基础路线之一 |
| 2022 | [Ho & Salimans, Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)                                                                        | 不额外训练 classifier，用条件/无条件输出差值做 guidance                   |

本项目的源码基准是 `score_sde_pytorch/`。最值得对照的文件是：

- `score_sde_pytorch/sde_lib.py`：`VPSDE`、`VESDE`、`subVPSDE`、reverse SDE/ODE。
- `score_sde_pytorch/losses.py`：连续 SDE DSM loss、legacy SMLD loss、legacy DDPM loss。
- `score_sde_pytorch/models/utils.py`：把网络输出包装成真实 score 的地方。
- `score_sde_pytorch/sampling.py`：PC sampler、ODE sampler、predictor/corrector 注册表。
- `score_sde_pytorch/likelihood.py`：probability flow ODE 的 likelihood 计算。
- `score_sde_pytorch/controllable_generation.py`：inpainting/colorization 的观测约束投影；它是可控生成示例，但不是 classifier guidance 或 CFG 实现。

源码文件本身保持为外部基准：教程只引用和解释这些代码块，不把教学注释写回 `score_sde_pytorch/`。如果某段源码变量名容易误导，例如 `losses.py` 中 legacy DDPM loss 把网络输出临时命名为 `score`，解释会写在教程正文里，而不是改动基准代码。

### 1.1 一张等价关系表

这些方法的等价性要限定在同一条概率路径、同一套噪声参数化和相同边缘分布上理解。它们不是逐行 loss 完全相同，而是“学到的对象可以互相换算”。

| 路线                 | 前向扰动                                                  | 网络直接监督                  | 采样时真正需要的量                             | 源码锚点                                                   |
| -------------------- | --------------------------------------------------------- | ----------------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| DSM                  | `x_t = x_0 + sigma epsilon`                               | 条件 score `-epsilon / sigma` | score                                          | `get_smld_loss_fn` 的 `target = -noise / sigma^2`          |
| NCSN / SMLD          | 多个离散 `sigma_i` 的 DSM                                 | 每个噪声层的 score            | annealed Langevin corrector                    | `VESDE.discretize`、`AnnealedLangevinDynamics`             |
| DDPM                 | `x_t = sqrt(bar_alpha_t)x_0 + sqrt(1-bar_alpha_t)epsilon` | 噪声 `epsilon`                | `score = -epsilon_theta / sqrt(1-bar_alpha_t)` | `get_ddpm_loss_fn`、`get_score_fn(VPSDE)`                  |
| VP SDE               | DDPM 的连续极限                                           | 连续时间 DSM                  | reverse SDE 或 ODE drift                       | `VPSDE.sde`、`VPSDE.marginal_prob`                         |
| VE SDE               | SMLD/NCSN 的连续极限                                      | 连续时间 DSM                  | reverse SDE、PC、ODE                           | `VESDE.sde`、`VESDE.marginal_prob`                         |
| Probability flow ODE | 与任意前向 SDE 共享边缘分布                               | 仍使用同一个 score 模型       | 确定性 drift、likelihood 轨迹散度              | `SDE.reverse(..., probability_flow=True)`、`likelihood.py` |

最短的换算链是：

$$
\epsilon_\theta(x_t,t) \approx \epsilon
\quad\Longleftrightarrow\quad
s_\theta(x_t,t) \approx -\frac{\epsilon_\theta(x_t,t)}{\sigma_t}
\quad\Longleftrightarrow\quad
\hat{x}_0 = \frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}.
$$

其中 VP/DDPM 使用 `alpha_t^2 + sigma_t^2 = 1`；VE/NCSN 使用 `alpha_t = 1`、`sigma_t = sigma(t)`。

### 1.2 训练、推理和源码对照总览

这张表是全文的索引：每一行都回答同一个问题，训练到底监督什么，推理到底调用什么，和其他模型是什么关系。

| 模型或算法                | 训练核心                                                                                          | 推理/评估核心                                                                          | 关系和边界                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 原始 score matching       | 拟合 `nabla_x log p_data(x)`，理论目标需要 `tr(nabla_x s_theta)`                                  | 理论上可接 Langevin dynamics；本仓库不直接实现这一原始高维目标                         | 是 DSM/NCSN 的出发点，主要价值是说明为什么归一化常数会从 score 里消失                                 |
| DSM / NCSN / SMLD         | `losses.get_smld_loss_fn` 拟合 `target = -noise / sigma^2`，网络输出直接解释为 score              | `AnnealedLangevinDynamics` 或 VE 路线的 corrector，从大噪声层逐级降噪                  | 是 VE SDE 的离散噪声层特例；`VESDE.discretize` 把连续 VE 重新落回相邻 sigma 差分                      |
| DDPM                      | `losses.get_ddpm_loss_fn` 拟合前向噪声 `epsilon`，再由 `models/utils.py::get_score_fn` 转成 score | `AncestralSamplingPredictor.vpsde_update_fn` 对应 DDPM 祖先采样                        | 是 VP SDE 的离散特例；本仓库实现的是 `epsilon` prediction，`x0/v` 是同一路径上的替代参数化            |
| VP SDE                    | `VPSDE.marginal_prob` 给出 `mean/std`，`get_sde_loss_fn` 拟合连续时间 DSM                         | `EulerMaruyamaPredictor`、`ReverseDiffusionPredictor` 或 ODE sampler                   | 连续化 DDPM；`beta_i ≈ beta(t_i) Delta t` 是离散链转连续 SDE 的关键                                   |
| VE SDE                    | `VESDE.marginal_prob` 生成 `x_t = x_0 + sigma(t) z`，目标仍是 `-z/std`                            | PC sampler、Langevin corrector、VE reverse diffusion                                   | 连续化 NCSN/SMLD；均值不收缩，噪声方差随时间增大                                                      |
| sub-VP SDE                | 与 VP 一样走连续 SDE loss，但扩散强度为 likelihood 友好的特殊形式                                 | 主要用于 likelihood/ODE 评估，也可进入通用 sampler                                     | 不是 DDPM 的普通离散链；它保留 VP drift，调整 diffusion 和边缘标准差                                  |
| Probability flow ODE      | 不单独训练新模型，复用任意 SDE 已训练的 score                                                     | `sampling.get_ode_sampler` 做确定性采样；`likelihood.py` 积分 divergence 得到 bits/dim | 是与 reverse SDE 共享边缘分布的确定性动力系统；优势是 likelihood 和编码，代价是 ODE NFE 与 trace 估计 |
| DDIM                      | 不改变 DDPM 训练目标，仍复用噪声预测或等价 score                                                  | 本仓库没有 `DDIMSampler`；教程按非马尔可夫反向族和跳步采样解释                         | 加速来自选择稀疏时间子序列和可设 `eta=0` 的确定性路径；它不是本仓库 ODE 类                            |
| Classifier guidance / CFG | 条件训练或条件 dropout 后得到条件/无条件 score 或噪声预测                                         | 采样时改 score：外部 classifier 梯度或条件/无条件输出差值                              | 属于条件生成接口；本仓库的 controllable generation 是观测投影，不是这两种 guidance                    |
| Latent Diffusion          | 在 autoencoder latent `z_0` 上训练同样的 score/noise/v prediction                                 | 在 latent 中采样，再用 decoder 回到像素空间                                            | 改变数据空间和条件注入方式，不改变 DDPM/SDE 的核心噪声路径                                            |

## 2. 从 score matching 到 NCSN

### 2.1 为什么要学 score

能量模型可以写成

$$
p_\theta(x) = \frac{\exp(f_\theta(x))}{Z_\theta}, \quad
Z_\theta = \int \exp(f_\theta(x)) dx.
$$

直接做极大似然会遇到 `Z_theta`，但 score 会把这个常数消掉：

$$
\nabla_x \log p_\theta(x)
= \nabla_x f_\theta(x) - \nabla_x \log Z_\theta
= \nabla_x f_\theta(x).
$$

Score matching 的目标是最小化 Fisher divergence：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)} \left[ ||s_\theta(x) - \nabla_x \log p_{\mathrm{data}}(x)||_2^2 \right].
$$

经过分部积分，可以把未知的 `nabla_x log p_data(x)` 去掉，得到只依赖样本和模型导数的目标：

$$
\mathbb{E}_{p_{\mathrm{data}}(x)} \left[
\operatorname{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} ||s_\theta(x)||_2^2
\right] + const.
$$

这个目标避开了归一化常数，但引入了 `tr(nabla_x s_theta)`，高维图像上代价很高。DSM 的关键就是用加噪条件分布的解析 score 替代这个 trace。

### 2.2 DSM：把 score 变成去噪方向

对数据加高斯噪声：

$$
\tilde{x} = x + \sigma \epsilon,\quad \epsilon \sim \mathcal{N}(0, I).
$$

条件分布是

$$
q_\sigma(\tilde{x} | x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I),
$$

它的条件 score 有解析式：

$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x)
= - \frac{\tilde{x} - x}{\sigma^2}
= - \frac{\epsilon}{\sigma}.
$$

DSM 训练目标：

$$
\mathbb{E}_{x, \tilde{x}} \left[
||s_\theta(\tilde{x}, \sigma) + \frac{\tilde{x} - x}{\sigma^2}||_2^2
\right].
$$

直觉上，score 指向 log density 增大的方向；对一个被高斯噪声扰动的样本，它的高密度方向就是回到干净样本附近。这个“去噪方向”不是经验比喻，而是条件高斯 score 的解析形式。

### 2.3 NCSN：一个网络学多个噪声层

单个很小的 `sigma` 只能覆盖数据流形附近，低密度区域 score 学不好；单个很大的 `sigma` 又会破坏数据结构。NCSN 用几何噪声序列

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_L > 0
$$

训练一个条件网络 `s_theta(x, sigma_i)`，目标是多尺度 DSM：

$$
L(\theta)
= \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i)
\mathbb{E}_{x, \epsilon} \left[
||s_\theta(x + \sigma_i \epsilon, \sigma_i) + \frac{\epsilon}{\sigma_i}||_2^2
\right].
$$

常见权重 `lambda(sigma_i)=sigma_i^2` 的作用是平衡不同噪声层的量级。因为目标 score 的范数约为 `1 / sigma_i`，乘上 `sigma_i^2` 后，各层 loss 不会天然被小噪声主导。

源码里的 legacy SMLD loss 正对应这一段：

```python
# score_sde_pytorch/losses.py
sigmas = smld_sigma_array[labels]
noise = torch.randn_like(batch) * sigmas[:, None, None, None]
perturbed_data = noise + batch
target = -noise / (sigmas ** 2)[:, None, None, None]
losses = torch.square(score - target) * sigmas ** 2
```

注意这里的 `target` 是真实 score，网络输出直接当 score 用。`smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))` 说明旧 SMLD 配置使用降序 sigma label。

### 2.4 Annealed Langevin dynamics

如果已知某个分布的 score，Langevin dynamics 可以只靠 score 采样：

$$
x_{k+1}
= x_k + \eta s_\theta(x_k) + \sqrt{2 \eta} z_k,\quad z_k \sim \mathcal{N}(0, I).
$$

NCSN 的 annealed Langevin dynamics 从最大噪声层开始采样，再逐层降低噪声：

$$
x_{k+1}
= x_k + \eta_i s_\theta(x_k, \sigma_i) + \sqrt{2 \eta_i} z_k.
$$

大噪声层的分布更接近简单高斯，低密度空洞少；相邻噪声层差距小，因此上一层的结果是下一层的好初值。`sampling.py::AnnealedLangevinDynamics` 用

$$
\eta_i = 2 (\operatorname{SNR} \cdot \operatorname{std}_t)^2 \alpha_t
$$

来设置步长；`LangevinCorrector` 则用当前 batch 的 `noise_norm / grad_norm` 自适应保持目标 SNR。

## 3. DDPM 是 VP 离散链的一种写法

### 3.1 前向加噪和闭式采样

DDPM 前向过程：

$$
q(x_t | x_{t-1})
= \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I).
$$

令 `alpha_t = 1 - beta_t`，`bar_alpha_t = prod_{s=1}^t alpha_s`，可以一步采样任意时间：

$$
q(x_t | x_0)
= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I),
$$

也就是

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon.
$$

这个闭式形式直接体现在 `losses.py::get_ddpm_loss_fn`：

```python
perturbed_data = sqrt_alphas_cumprod[labels] * batch \
  + sqrt_1m_alphas_cumprod[labels] * noise
losses = torch.square(model_output - noise)
```

源码变量名 `score = model_fn(...)` 容易误导：在 legacy DDPM loss 里它实际是网络预测的 `epsilon_theta`，不是最终采样使用的 score。

### 3.2 三种预测头：epsilon、x0 和 v

写成统一形式：

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon,\quad
\alpha_t^2 + \sigma_t^2 = 1.
$$

三种预测方式不是三种不同扩散过程，而是同一条带噪路径上选择不同的监督目标。给定其中任意一个，都可以代数恢复另外两个，并进一步换成 score。

| 预测头               | 训练目标                                         | 恢复关系                                                                                 | score 换算                                      |
| -------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `epsilon` prediction | 预测前向加噪里的标准高斯噪声 `epsilon`           | `hat{x}_0 = (x_t - sigma_t hat{epsilon}) / alpha_t`                                      | `s_theta = -hat{epsilon}/sigma_t`               |
| `x0` prediction      | 直接预测干净样本 `x_0`                           | `hat{epsilon} = (x_t - alpha_t hat{x}_0) / sigma_t`                                      | `s_theta = -(x_t - alpha_t hat{x}_0)/sigma_t^2` |
| `v` prediction       | 预测旋转坐标 `v = alpha_t epsilon - sigma_t x_0` | `hat{x}_0 = alpha_t x_t - sigma_t hat{v}`，`hat{epsilon} = sigma_t x_t + alpha_t hat{v}` | 先恢复 `hat{epsilon}`，再除以 `-sigma_t`        |

**epsilon prediction** 是 DDPM 论文和本仓库 legacy DDPM 路线采用的方式：

$$
\hat{x}_0 = \frac{x_t - \sigma_t \hat{\epsilon}_\theta(x_t,t)}{\alpha_t},
\quad
\hat{s}_\theta(x_t,t) = - \frac{\hat{\epsilon}_\theta(x_t,t)}{\sigma_t}.
$$

**x0 prediction** 让网络直接输出干净样本估计，采样时再换回噪声或 score：

$$
\hat{\epsilon}_\theta = \frac{x_t - \alpha_t \hat{x}_{0,\theta}}{\sigma_t}.
$$

把它代入 VP 条件 score 公式，得到

$$
\hat{s}_\theta(x_t,t)
= -\frac{x_t - \alpha_t \hat{x}_{0,\theta}(x_t,t)}{\sigma_t^2}.
$$

**v prediction** 常用定义是

$$
v = \alpha_t \epsilon - \sigma_t x_0.
$$

由于

$$
\begin{bmatrix}x_t \\ v\end{bmatrix}
=
\begin{bmatrix}\alpha_t & \sigma_t \\ -\sigma_t & \alpha_t\end{bmatrix}
\begin{bmatrix}x_0 \\ \epsilon\end{bmatrix},
$$

反解得到

$$
x_0 = \alpha_t x_t - \sigma_t v,\quad
\epsilon = \sigma_t x_t + \alpha_t v.
$$

`v` 是对 `(x_0, epsilon)` 的正交旋转。它的好处不是创造了新的目标信息，而是改变了不同噪声水平上的目标尺度和优化条件；Progressive Distillation 等少步采样/蒸馏工作中常用这种参数化来稳定学生模型训练。Stable Diffusion 的主线来自 latent diffusion；不同版本和 scheduler 可以选择 `epsilon` 或 `v` 参数化，因此教程里应把它作为现代扩散实现的重要接口，而不是只停在 DDPM 的 `epsilon` 预测。

本仓库的 legacy DDPM 路线采用噪声预测；`x0` 和 `v` prediction 没有作为独立训练目标实现。教程里讨论三者，是为了把 DDPM 公式、现代 scheduler 接口和 score-based 采样所需的 `score_fn` 对齐。

### 3.3 DDPM 和 score 的接口转换

采样器需要的是 `nabla_x log p_t(x)`，DDPM 网络输出的是 `epsilon_theta`。转换发生在 `score_sde_pytorch/models/utils.py`：

```python
std = sde.sqrt_1m_alphas_cumprod[labels.long()]
score = -model_output / std[:, None, None, None]
```

这解释了为什么 DDPM 可以放进 score-based 采样框架：不是训练目标本身叫 score matching，而是它学到的噪声预测和 VP 条件 score 只差一个 `-1 / std` 的缩放。

## 4. SDE 统一框架

连续时间前向 SDE 写作

$$
dX_t = f(X_t,t)dt + g(t)dW_t.
$$

其中 `f` 是 drift，`g` 是 diffusion；后文所有 SDE 公式和源码对照都采用这个约定。

`score_sde_pytorch` 的抽象类要求每个 SDE 实现五件事：

- `sde(x,t)`：返回 drift 和 diffusion。
- `marginal_prob(x,t)`：返回 `p_{0t}(x_t | x_0)` 的均值和标准差。
- `prior_sampling(shape)`：从 `p_T` 采样。
- `prior_logp(z)`：算 `p_T` 的 log density，用于 likelihood。
- `discretize(x,t)`：给 predictor/reverse diffusion 使用的离散步。

### 4.1 离散链和连续 SDE 怎么互相转

从连续 SDE 到离散模拟，默认的一阶方法是 Euler-Maruyama：

$$
X_{t+\Delta t}
= X_t + f(X_t,t)\Delta t + g(t)\sqrt{\Delta t}\,z,
\quad z\sim \mathcal{N}(0,I).
$$

反过来，从 DDPM 离散链看 VP SDE，要把 `beta_i` 看成连续 `beta(t_i)` 在一个小时间步上的积分：

$$
\beta_i \approx \beta(t_i)\Delta t.
$$

DDPM 的一步前向为

$$
x_{i+1}=\sqrt{1-\beta_i}x_i+\sqrt{\beta_i}z_i.
$$

当 `Delta t` 很小时，

$$
\sqrt{1-\beta(t_i)\Delta t}
\approx 1-\frac{1}{2}\beta(t_i)\Delta t,
\quad
\sqrt{\beta_i}z_i
\approx \sqrt{\beta(t_i)}\sqrt{\Delta t}\,z_i.
$$

因此

$$
x_{i+1}-x_i
\approx -\frac{1}{2}\beta(t_i)x_i\Delta t
+\sqrt{\beta(t_i)}\sqrt{\Delta t}\,z_i,
$$

极限就是 VP SDE：

$$
dX_t = -\frac{1}{2}\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t.
$$

SMLD/NCSN 到 VE SDE 的逻辑类似。离散噪声层满足

$$
x_i=x_{i-1}+\sqrt{\sigma_i^2-\sigma_{i-1}^2}z_i.
$$

如果 `sigma_i = sigma(t_i)`，则

$$
\sigma_i^2-\sigma_{i-1}^2
\approx \frac{d\sigma^2(t)}{dt}\Delta t,
$$

所以连续极限为

$$
dX_t = \sqrt{\frac{d\sigma^2(t)}{dt}}\,dW_t.
$$

源码中的 `discretize` 是反方向：给定连续 SDE 类，返回一个有限步采样器实际使用的 `f, G`。基类 `SDE.discretize` 是 Euler-Maruyama；`VPSDE.discretize` 覆写为 DDPM 风格的 `sqrt(alpha)x - x` 与 `sqrt(beta)`；`VESDE.discretize` 覆写为 SMLD/NCSN 风格的 `sqrt(sigma_i^2-sigma_{i-1}^2)`。这就是“统一框架覆盖旧模型”的具体接口。

### 4.2 VPSDE：DDPM 的连续化

VP SDE：

$$
dX_t = -\frac{1}{2} \beta(t) X_t dt + \sqrt{\beta(t)} dW_t.
$$

代码：

```python
beta_t = beta_0 + t * (beta_1 - beta_0)
drift = -0.5 * beta_t[:, None, None, None] * x
diffusion = torch.sqrt(beta_t)
```

线性 SDE 可用积分因子解出。令

$$
B(t)=\int_0^t \beta(s)ds.
$$

则

$$
X_t = \exp(-\frac{1}{2}B(t))X_0
+ \int_0^t \exp(-\frac{1}{2}(B(t)-B(s)))\sqrt{\beta(s)}dW_s.
$$

均值：

$$
\mathbb{E}[X_t | X_0] = \exp(-\frac{1}{2}B(t))X_0.
$$

方差项用 Ito isometry：

$$
\operatorname{Var}[X_t | X_0]
= \int_0^t \exp(-(B(t)-B(s))) \beta(s) ds
= 1 - \exp(-B(t)).
$$

源码采用线性 schedule：

$$
\beta(t) = \beta_0 + t(\beta_1 - \beta_0),
$$

所以

$$
B(t)= \beta_0t + \frac{1}{2}(\beta_1-\beta_0)t^2.
$$

于是

$$
\operatorname{log\_mean\_coeff}
= -\frac{1}{2}B(t)
= -\frac{1}{2}\beta_0t - \frac{1}{4}(\beta_1-\beta_0)t^2.
$$

这正是 `VPSDE.marginal_prob`：

```python
log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
mean = exp(log_mean_coeff) * x
std = sqrt(1 - exp(2 * log_mean_coeff))
```

`VPSDE.discretize` 则回到 DDPM 风格：

$$
x_{i+1} = \sqrt{\alpha_i}x_i + \sqrt{\beta_i}z_i,
$$

代码里返回的是增量形式 `f = sqrt(alpha) * x - x` 和 `G = sqrt(beta)`。

### 4.3 VESDE：NCSN/SMLD 的连续化

VE SDE 没有收缩 drift：

$$
dX_t = \sqrt{\frac{d \sigma^2(t)}{dt}} dW_t.
$$

如果

$$
\sigma(t)=\sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t,
$$

则

$$
\frac{d \sigma^2(t)}{dt}
= 2 \log\left(\frac{\sigma_{max}}{\sigma_{min}}\right) \sigma^2(t),
$$

因此

$$
g(t)=\sigma(t)\sqrt{2\log(\sigma_{max}/\sigma_{min})}.
$$

源码：

```python
sigma = sigma_min * (sigma_max / sigma_min) ** t
drift = torch.zeros_like(x)
diffusion = sigma * sqrt(2 * (log(sigma_max) - log(sigma_min)))
mean = x
std = sigma
```

VE 的边缘分布保持均值不变、方差随时间爆炸：

$$
X_t | X_0 \sim \mathcal{N}(X_0, \sigma^2(t) I).
$$

离散化时：

$$
x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} z_i,
$$

这就是 `VESDE.discretize` 中的 `G = sqrt(sigma ** 2 - adjacent_sigma ** 2)`。

### 4.4 subVPSDE：为 likelihood 调整扩散强度

sub-VP 保留 VP 的 drift：

$$
f(x,t) = -\frac{1}{2} \beta(t)x,
$$

但把 diffusion 改成

$$
g(t)=\sqrt{\beta(t)(1 - \exp(-2B(t)))}.
$$

其中 `B(t)=int_0^t beta(s)ds`。这样均值仍是 `exp(-B(t)/2)x_0`，但边缘标准差变成

$$
\operatorname{std}(t)=1-\exp(-B(t)).
$$

源码里：

```python
discount = 1. - exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
diffusion = sqrt(beta_t * discount)
std = 1 - exp(2 * log_mean_coeff)
```

这里 `2 * log_mean_coeff = -B(t)`，所以 `std = 1 - exp(-B(t))`。这行看起来像“少了 sqrt”，但它对应的是 sub-VP 的特殊边缘标准差，不是 VP 的方差写错。

## 5. 训练调用链

训练主链路不要放进数学块；它是代码调用关系，不是公式：

```text
config
  -> run_lib.train / run_lib.evaluate
  -> sde_lib.{VPSDE, VESDE, subVPSDE}
  -> losses.get_step_fn(...)
  -> losses.get_sde_loss_fn(...)    # continuous=True
     or get_smld_loss_fn(...)       # VESDE + continuous=False
     or get_ddpm_loss_fn(...)       # VPSDE + continuous=False
  -> models.utils.get_score_fn(...)
  -> model(x_t, labels)
```

`run_lib.py` 依据 `config.training.sde` 创建 SDE：

- `vpsde`：`VPSDE(beta_min, beta_max, N=num_scales)`。
- `vesde`：`VESDE(sigma_min, sigma_max, N=num_scales)`。
- `subvpsde`：`subVPSDE(beta_min, beta_max, N=num_scales)`。

`losses.get_step_fn` 依据 `continuous` 分两路：

| 设置                        | 使用的 loss        | 对应算法               |
| --------------------------- | ------------------ | ---------------------- |
| `continuous=True`           | `get_sde_loss_fn`  | Score SDE 统一连续训练 |
| `continuous=False`, `VESDE` | `get_smld_loss_fn` | legacy SMLD/NCSN       |
| `continuous=False`, `VPSDE` | `get_ddpm_loss_fn` | legacy DDPM            |

连续 SDE loss 的核心步骤：

```python
# 1. 每个样本独立抽一个连续时间 t，训练目标覆盖整条噪声路径。
t = Uniform(eps, T)

# 2. z 是重参数化噪声；条件 score 的解析目标会写成 -z / std。
z = randn_like(batch)

# 3. marginal_prob 给出 p_{0t}(x_t | x_0) 的均值和标准差。
mean, std = sde.marginal_prob(batch, t)
perturbed_data = mean + std * z

# 4. get_score_fn 把模型输出包装成真正的 score。
#    VP/DDPM 路线会把 epsilon prediction 转成 -epsilon / std。
score = score_fn(perturbed_data, t)

# 5. 因为目标 score 是 -z/std，所以 score * std + z 应该接近 0。
loss = ||score * std + z||^2
```

这就是 DSM：

$$
s^*(x_t,t) = \nabla_{x_t} \log p_{0t}(x_t|x_0)
= - \frac{z}{\operatorname{std}_t}.
$$

所以

$$
s_\theta(x_t,t)\operatorname{std}_t + z \approx 0.
$$

如果 `likelihood_weighting=True`，loss 改为

$$
g^2(t)\left\|s_\theta(x_t,t) + \frac{z}{\operatorname{std}_t}\right\|^2,
$$

用于更贴近 likelihood 权重的训练；默认 CIFAR-10 配置里 `likelihood_weighting=False`。

三条训练分支的等价关系可以这样读：

- `get_smld_loss_fn`：直接拟合 `target = -noise / sigma^2`，这是 VE 离散层的 score。
- `get_ddpm_loss_fn`：拟合 `noise`，再由 `get_score_fn(VPSDE)` 转成 `-noise / sqrt(1 - bar_alpha_t)`。
- `get_sde_loss_fn`：先用 `marginal_prob` 生成任意连续时间的 `x_t`，再统一拟合 `-z / std`。

所以 DDPM 和 score matching 的关系不是“DDPM loss 字面上等于 Fisher divergence”，而是 DDPM 的噪声预测目标在 VP 高斯扰动路径下等价于学习该噪声边缘的 score。

## 6. Reverse SDE、PC sampling 和 ODE

### 6.1 Reverse-time SDE

给定前向 SDE：

$$
dX_t = f(X_t,t)dt + g(t)dW_t,
$$

反向时间 SDE 为

$$
dX_t =
[f(X_t,t) - g^2(t)\nabla_x \log p_t(X_t)]dt
+ g(t)d\bar{W}_t,
$$

这里的 `dt` 沿反向积分理解，源码通过 `dt = -1/N` 或 `x_mean = x - f` 实现从 `T` 到 `eps` 的步进。

`sde_lib.SDE.reverse` 是统一入口：

```python
drift = drift - diffusion ** 2 * score * (0.5 if probability_flow else 1.)
diffusion = 0. if probability_flow else diffusion
```

当 `probability_flow=False`，系数是 `1`，得到 reverse SDE；当为 `True`，系数是 `1/2` 且 diffusion 设为 0，得到 probability flow ODE。

### 6.2 PC sampler：corrector 修局部，predictor 推时间

`sampling.get_sampling_fn` 只分两大类：

- `method='pc'`：创建 predictor-corrector sampler。
- `method='ode'`：创建 probability flow ODE sampler。

PC sampler 的循环是：

```python
x = sde.prior_sampling(shape)
for t in linspace(T, eps, N):
    x, x_mean = corrector_update_fn(x, t)
    x, x_mean = predictor_update_fn(x, t)
return inverse_scaler(x_mean if denoise else x)
```

corrector 不改变时间，只在当前 `p_t` 上用 Langevin dynamics 把样本推向更高密度区域；predictor 才沿反向时间推进一步。

常见组合：

| 配置          | predictor            | corrector  | 含义                                   |
| ------------- | -------------------- | ---------- | -------------------------------------- |
| legacy NCSN   | `none`               | `ald`      | 只做 annealed Langevin dynamics        |
| legacy DDPM   | `ancestral_sampling` | `none`     | 标准 DDPM 祖先采样                     |
| VE continuous | `reverse_diffusion`  | `langevin` | reverse diffusion + Langevin corrector |
| VP continuous | `euler_maruyama`     | `none`     | Euler-Maruyama 反向积分                |

`ReverseDiffusionPredictor` 调 `rsde.discretize`，因此会使用具体 SDE 覆写后的离散规则；`EulerMaruyamaPredictor` 调 `rsde.sde`，更接近通用数值 SDE 积分。

方法优劣可以从“随机性、质量、速度、likelihood”四个维度看：

| 方法                    | 优势                                                     | 代价                                                              | 源码位置                                              |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------- |
| Reverse SDE + predictor | 保留随机扩散过程，和理论反向 SDE 直接对应                | 有随机方差，步数通常多                                            | `EulerMaruyamaPredictor`、`ReverseDiffusionPredictor` |
| Predictor-Corrector     | Corrector 在每个噪声层做 Langevin 修正，样本质量通常更强 | 每个时间步多次 score eval，慢                                     | `get_pc_sampler`、`LangevinCorrector`                 |
| DDPM ancestral sampling | 与离散 DDPM 训练完全贴合，公式直观                       | 随机多步采样，速度慢                                              | `AncestralSamplingPredictor.vpsde_update_fn`          |
| DDIM                    | 可做确定性或低随机采样，步数可减少                       | 本仓库无直接实现；离散 DDPM 特化，不提供 likelihood 计算          | 教程解释，源码未实现                                  |
| Probability flow ODE    | 确定性、可反向编码、可计算 likelihood/bits-per-dim       | ODE solver NFE 不固定，散度估计有额外成本；采样 FID 不一定优于 PC | `get_ode_sampler`、`likelihood.py`                    |

### 6.3 DDPM ancestral sampling

VP 的 `AncestralSamplingPredictor`：

$$
x_{mean} = \frac{x + \beta_t s_\theta(x,t)}{\sqrt{1-\beta_t}},
\quad
x_{t-1} = x_{mean} + \sqrt{\beta_t}z.
$$

若 `s_theta = -epsilon_theta / sqrt{1 - bar_alpha_t}`，可化回 DDPM 论文里用噪声预测写出的均值形式：

$$
\mu_\theta(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right).
$$

这说明 DDPM ancestral sampling 和 score-based reverse diffusion 在 VP 离散设置下是一件事的两种写法。

### 6.4 DDIM：为什么能跳步加速

DDIM 的关键不是重新训练一个模型，而是保留 DDPM 已经训练好的边缘分布

$$
q(x_t|x_0)=\mathcal{N}\left(\sqrt{\bar{\alpha}_t}x_0,\,(1-\bar{\alpha}_t)I\right),
$$

然后重新设计反向采样过程。DDPM ancestral sampling 逐步采 `x_T -> x_{T-1} -> ... -> x_0`，每一步都走相邻 timestep；DDIM 允许选择一个稀疏时间子序列

$$
\tau_S > \tau_{S-1} > \cdots > \tau_1 > \tau_0 = 0,\quad S \ll T,
$$

只在这些时间点之间跳转，所以网络评估次数从 `T` 次降到 `S` 次。加速来自少做 score/epsilon network forward，不来自更小的网络或新的训练目标。

给定当前噪声样本 `x_t`，DDPM 噪声预测头先给出

$$
\hat{x}_0(x_t,t)
= \frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}},
\quad
\hat{\epsilon}(x_t,t)=\epsilon_\theta(x_t,t).
$$

DDIM 构造的非马尔可夫反向条件分布可以写成

$$
q_\sigma(x_s|x_t,x_0)
= \mathcal{N}\left(
\sqrt{\bar{\alpha}_s}x_0
+ \sqrt{1-\bar{\alpha}_s-\sigma_t^2}
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}},
\sigma_t^2 I
\right),
$$

其中 `s < t` 可以是相邻 timestep，也可以是跳过很多步后的前一个采样点。把未知的 `x_0` 换成 `hat{x}_0`，把真实噪声方向换成 `hat{epsilon}`，得到实际采样式：

$$
x_s
= \sqrt{\bar{\alpha}_s}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_s-\sigma_t^2}\hat{\epsilon}
+ \sigma_t z,\quad z\sim\mathcal{N}(0,I).
$$

噪声强度通常写成

$$
\sigma_t(\eta)
= \eta
\sqrt{\frac{1-\bar{\alpha}_s}{1-\bar{\alpha}_t}}
\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_s}}.
$$

`eta=1` 时接近 DDPM 风格的随机 ancestral 更新；`eta=0` 时，

$$
x_s
= \sqrt{\bar{\alpha}_s}\hat{x}_0
+ \sqrt{1-\bar{\alpha}_s}\hat{\epsilon},
$$

采样变成确定性：同一个初始噪声 `x_T` 会生成同一个样本。这个式子的直觉很重要：模型在当前 `x_t` 估计一份干净图 `hat{x}_0` 和一份噪声方向 `hat{epsilon}`，DDIM 直接把它们重新组合到更低噪声水平 `s` 上，而不是必须经过所有中间噪声层。

跳步为什么可行？因为训练时模型学的是任意 timestep 的 `epsilon_theta(x_t,t)`，而 DDPM 闭式边缘 `q(x_t|x_0)` 直接给出了每个噪声水平和 `x_0` 的关系。只要模型在稀疏时间点上仍能给出足够好的 `hat{x}_0/hat{epsilon}`，就可以从 `t` 跳到 `s`。代价也在这里：步子越大，当前估计误差越容易被带到后续所有步骤；因此 DDIM 的加速通常伴随质量、细节或多样性的取舍。

它和 probability flow ODE 的共同点是：二者都可以用同一个已训练 score/噪声模型走确定性采样路径。边界也必须分清：DDIM 是 DDPM 离散边缘分布上的非马尔可夫反向族；probability flow ODE 是连续 SDE 框架里由 Fokker-Planck/连续性方程推出的 ODE，并天然支持 likelihood 积分。本地 `score_sde_pytorch` PyTorch 版没有单独 `DDIMSampler` 类；如果本项目以后补 DDIM 示例，应作为 DDPM 离散采样器实现，而不是挂到现有 `get_ode_sampler` 上伪装成 probability flow ODE。

### 6.5 Probability flow ODE 为什么去掉噪声仍然有效

前向 SDE 的 Fokker-Planck 方程：

$$
\partial_t p_t(x)
= -\nabla \cdot (f(x,t)p_t(x))
+ \frac{1}{2} g^2(t) \Delta p_t(x).
$$

考虑 ODE：

$$
dX_t = \left[f(X_t,t) - \frac{1}{2}g^2(t)\nabla_x \log p_t(X_t)\right]dt.
$$

ODE 的连续性方程：

$$
\partial_t p_t(x)
= -\nabla \cdot (p_t v_t)(x),
$$

其中

$$
v_t = f - \frac{1}{2}g^2 \nabla \log p_t.
$$

代入：

$$
-\nabla \cdot(p_t v_t)
= -\nabla \cdot(p_t f)
+ \frac{1}{2}g^2 \nabla \cdot(p_t \nabla \log p_t).
$$

又因为

$$
p_t \nabla \log p_t = \nabla p_t,
$$

所以

$$
\nabla \cdot(p_t \nabla \log p_t) = \Delta p_t.
$$

ODE 的边缘分布演化方程与 SDE 的 Fokker-Planck 方程一致。因此 ODE 没有随机项，仍能生成同一族边缘分布；随机性只来自初始 `x_T ~ p_T`。

`sampling.py::get_ode_sampler` 用 `scipy.integrate.solve_ivp` 从 `T` 积到 `eps`。`likelihood.py` 也沿 probability flow ODE 走，但额外积分 divergence：

$$
\frac{d \log p_t(x_t)}{dt} = -\nabla \cdot v_t(x_t),
$$

并用 Hutchinson-Skilling estimator 估计 trace，避免显式构造 Jacobian。

### 6.6 Probability flow ODE 的 likelihood 链路

Probability flow ODE 的最大工程优势之一是 likelihood 评估。它给出从数据 `x_0` 到 latent `z=x_T` 的确定性可逆轨迹，因此可以沿 ODE 同时积分样本和 log-density 变化：

```python
# score_sde_pytorch/likelihood.py
init = concat([flatten(data), zeros(batch_size)])
solution = solve_ivp(ode_func, (eps, sde.T), init)
z = solution.y[:, -1][:-batch_size]
delta_logp = solution.y[:, -1][-batch_size:]
prior_logp = sde.prior_logp(z)
bpd = -(prior_logp + delta_logp) / log(2) / num_dims
```

`ode_func` 里有两部分：

```python
drift = drift_fn(model, sample, vec_t)
logp_grad = div_fn(model, sample, vec_t, epsilon)
```

第一部分推进 probability flow ODE 的状态；第二部分用 Hutchinson-Skilling estimator 估计 `div(drift)`，也就是连续 change-of-variables 里的 trace 项。源码默认 Rademacher 噪声，因此单次估计是无偏但有方差的；评估时可以通过更多样本或重复降低波动。

在 `run_lib.evaluate` 中，`config.eval.enable_bpd=True` 时会构建 `likelihood_fn`，对 `bpd_dataset` 指定的数据集逐 checkpoint 计算 bits/dim 并保存：

```python
if config.eval.enable_bpd:
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

bpd = likelihood_fn(score_model, eval_batch)[0]
```

这说明 likelihood 更适合作为 checkpoint/evaluation 指标来监控训练，而不是每个训练 step 的轻量 loss。原因很直接：它要跑 ODE solver，还要估计 divergence，比普通 DSM/DDPM loss 昂贵得多。

和采样指标相比，bits/dim 衡量的是模型对数据分布的密度解释能力；FID/IS 更偏向视觉样本质量。Score SDE 统一框架的价值之一，就是同一个 score 模型既能用 reverse SDE/PC 追求样本质量，也能用 probability flow ODE 做 likelihood 和 latent encoding。

## 7. 条件生成与 Latent Diffusion

### 7.1 条件生成：采样时改 score

无条件 score 模型学习的是

$$
s_\theta(x_t,t)\approx \nabla_{x_t}\log p_t(x_t).
$$

条件生成需要的是

$$
\nabla_{x_t}\log p_t(x_t|y)
= \nabla_{x_t}\log p_t(x_t)
+ \nabla_{x_t}\log p_t(y|x_t).
$$

这条式子说明了 classifier guidance 的核心：先训练一个无条件扩散模型，再训练一个能看噪声图 `x_t` 和时间 `t` 的分类器 `p_phi(y|x_t,t)`，采样时把分类器梯度加到 score 上：

$$
s_{\mathrm{guided}}(x_t,t,y)
= s_\theta(x_t,t)
+ w\nabla_{x_t}\log p_\phi(y|x_t,t).
$$

`w` 是 guidance scale。它越大，样本越被推向分类器认为属于 `y` 的区域，类别一致性和视觉锐度可能更强；但过大时会牺牲多样性，甚至把样本推到过饱和或分类器偏见很强的区域。若模型输出是 DDPM 的噪声预测，因为 `s=-epsilon/sigma_t`，同一个修正写成噪声参数化就是

$$
\epsilon_{\mathrm{guided}}
= \epsilon_\theta
- w\sigma_t\nabla_{x_t}\log p_\phi(y|x_t,t).
$$

Classifier-free guidance 不再额外训练分类器，而是在训练扩散模型时随机丢掉条件，让同一个网络同时学会有条件和无条件两种输出。采样时做线性外推：

$$
s_{\mathrm{cfg}}
= s_{\mathrm{uncond}}
+ w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

如果使用噪声预测头，等价写法是

$$
\epsilon_{\mathrm{cfg}}
= \epsilon_{\mathrm{uncond}}
+ w(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
$$

这里 `s_cond - s_uncond` 或 `epsilon_cond - epsilon_uncond` 可以理解为“条件给采样方向增加了什么”。CFG 的工程优势是少训练一个 noisy classifier，文本到图像模型也更容易把文本 encoder 的条件向量直接送进 U-Net；代价是需要在训练中做 conditional dropout，并在采样时通常要跑 conditional/unconditional 两次网络输出或做等价 batch 拼接。

### 7.2 本仓库的 controllable generation 属于观测投影

`score_sde_pytorch/controllable_generation.py` 里有 inpainting 和 colorization，但它们不是 classifier guidance 或 CFG。它们的逻辑是在每个 predictor/corrector 更新后，把已知观测重新投影回当前噪声水平。

Inpainting 的关键代码是：

```python
x, x_mean = update_fn(x, vec_t, model=model)
masked_data_mean, std = sde.marginal_prob(data, vec_t)
masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
x = x * (1. - mask) + masked_data * mask
x_mean = x * (1. - mask) + masked_data_mean * mask
```

这段代码没有引入 `nabla_x log p(y|x_t)`，也没有组合条件/无条件网络输出。它做的是数据一致性：未知区域由 score sampler 更新，已知区域则替换成和观测 `data` 在同一噪声水平下的带噪版本。Colorization 也是同一个思想，只是先把 RGB 解耦到 luminance/chrominance 结构，再固定灰度信息对应的通道。

因此本项目里可以把条件生成分成三类：

| 类型                      | 条件如何进入采样                       | 是否在本仓库直接实现                |
| ------------------------- | -------------------------------------- | ----------------------------------- | ---- |
| Classifier guidance       | 额外分类器提供 `nabla_x log p(y        | x_t)`                               | 没有 |
| Classifier-free guidance  | 条件/无条件扩散输出线性组合            | 没有                                |
| Inpainting / colorization | predictor/corrector 后做观测一致性投影 | 有，见 `controllable_generation.py` |

### 7.3 Latent Diffusion：把扩散搬到压缩空间

Latent Diffusion 解决的不是“score 要不要学”的问题，而是“在哪个空间学”。像素空间扩散直接在 `H x W x 3` 图像上反复跑 U-Net，计算和显存都很重；latent diffusion 先训练一个 autoencoder：

```text
image x
  -> encoder E
  -> latent z_0
  -> diffusion U-Net in latent space
  -> denoised latent
  -> decoder D
  -> image
```

扩散训练仍然可以写成同一套 DDPM/VP 形式：

$$
z_t = \alpha_t z_0 + \sigma_t \epsilon,
\quad
\epsilon_\theta(z_t,t,c)\approx\epsilon.
$$

区别在于 `z_0 = E(x_0)`，网络不再直接处理像素，而是在压缩 latent 上做噪声预测。采样结束后，先得到 `hat{z}_0`，再用 decoder 还原：

$$
\hat{x}_0 = D(\hat{z}_0).
$$

Stable Diffusion 的关键工程收益就在这里：latent 的空间分辨率通常比像素图小很多，U-Net 的注意力和卷积开销显著下降，高分辨率生成才变得可承受。条件文本通常通过 text encoder 得到 token embedding，再用 cross-attention 注入 U-Net；CFG 则在文本条件输出和空条件输出之间外推。

Latent Diffusion 的边界也要说清：它没有推翻 DDPM、Score SDE、`epsilon/x0/v` prediction 这些核心接口，而是改变数据表示和条件注入方式。新的瓶颈变成 autoencoder 的压缩率、decoder 还原质量、latent 空间中 score 的语义是否足够贴近像素感知质量。

## 8. 数学夹层：Ito、Fokker-Planck 和 likelihood

### 8.1 Ito's lemma 负责把随机路径变成密度演化

对 SDE

$$
dX_t = f(X_t,t)dt + g(t)dW_t
$$

和光滑测试函数 `phi(x,t)`，Ito's lemma 给出：

$$
d\phi(X_t,t)
= \left(\partial_t \phi + f \cdot \nabla \phi
+ \frac{1}{2}g^2 \Delta \phi \right)dt
+ g \nabla \phi \cdot dW_t.
$$

两边取期望，随机积分项期望为 0：

$$
\frac{d}{dt}\mathbb{E}[\phi(X_t,t)]
= \mathbb{E}[\partial_t \phi + f \cdot \nabla \phi + \frac{1}{2}g^2 \Delta \phi].
$$

把期望写成 `int phi(x,t)p_t(x)dx`，再对 drift 项和 Laplacian 项分部积分，就得到 Fokker-Planck 方程。这个方程是 reverse SDE 与 probability flow ODE 能共享边缘分布的底层理由。

### 8.2 ODE likelihood 的 change of variables

普通 normalizing flow 用

$$
\frac{d \log p_t(x_t)}{dt}
= -\operatorname{tr}\left(\frac{\partial v_t}{\partial x_t}\right)
= -\nabla \cdot v_t(x_t).
$$

Probability flow ODE 给了一个连续可逆映射：从数据 `x_0` 积到 latent `z=x_T`，再加上路径上的 divergence，就能得到

$$
\log p_0(x_0)
= \log p_T(z) + \int_0^T \nabla \cdot v_t(x_t) dt.
$$

`likelihood.py` 把样本和 `delta_logp` 拼成一个扩展 ODE 状态，一起交给 `solve_ivp`。divergence 用 Hutchinson estimator：

$$
\operatorname{tr}(J) = \mathbb{E}_\epsilon[\epsilon^T J \epsilon],
$$

其中 `epsilon` 可取 Rademacher 或 Gaussian 噪声。

## 9. 理解检查

1. 为什么 legacy DDPM loss 里网络输出可以叫 `noise`，但采样器仍然需要 `score_fn`？
2. VPSDE 的 `log_mean_coeff` 为什么是 `-0.25 * t^2 * (beta_1-beta_0) - 0.5 * beta_0 * t`？
3. VE SDE 的 `marginal_prob` 为什么均值不变，而 VP SDE 的均值会指数衰减？
4. 从 DDPM 离散链推出 VPSDE 时，`beta_i` 为什么要理解成 `beta(t_i) Delta t`，而不是直接等于连续的 `beta(t_i)`？
5. `LangevinCorrector` 和 `EulerMaruyamaPredictor` 都会调用 score，它们对时间变量的作用有什么不同？
6. DDIM 为什么可以用少量时间点加速采样？`eta=0` 时确定性路径具体在重组哪两个估计量？
7. 如果要在本仓库加入 DDIM，应该挂在 `sampling.py` 的 predictor 体系、ODE sampler 体系，还是单独作为 DDPM 离散采样器？理由是什么？
8. Probability flow ODE 没有扩散项，为什么仍然能和 reverse SDE 拥有相同的边缘分布？
9. Classifier guidance 和 classifier-free guidance 分别从哪里得到条件方向？为什么 guidance scale 太大可能牺牲多样性？
10. Latent Diffusion 改变的是扩散过程本身，还是扩散过程所在的数据空间？它为什么能支撑 Stable Diffusion 的高分辨率生成？
11. 为什么 bits/dim 更适合作为 checkpoint/evaluation 指标，而不是每个训练 step 都计算的训练 loss？
