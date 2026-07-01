---
layout: post
title: "Orthogonal Polynomials for Uncertainty Quantification: Recurrence Algorithms, PCE, and Biomedical Simulation"
date: 2025-11-25 00:00:00
description: A research-level map of orthogonal polynomial recurrence algorithms, polynomial chaos expansion, and noninvasive uncertainty quantification for biomedical simulations.
tags: mathematics scientific-computing orthogonal-polynomials uncertainty-quantification polynomial-chaos
categories: research-notes
featured: true
_styles: |
  #markdown-content pre code,
  #markdown-content figure.highlight pre code,
  #markdown-content div.highlighter-rouge pre code {
    white-space: pre;
    word-wrap: normal;
    overflow-wrap: normal;
  }
---

本文梳理的是我博士期间围绕正交多项式、递推系数、不确定性量化和生物医学仿真的研究主线。旧版本文章主要是面向非专业读者的正交多项式科普；这里保留“为什么正交多项式重要”的入口，但把重点推进到更核心的问题：如何针对非经典测度稳定构造正交基，并把它们用于 polynomial chaos expansion (PCE) 和真实医学/工程仿真的 uncertainty quantification (UQ)。

本文的重点如下：

- 给定概率测度 $\mu$，关键不是写出某个闭式多项式，而是构造与 $\mu$ 匹配的正交基，并用递推系数作为稳定求值、求积、逼近和 PCE 的计算接口。
- 递推系数不是细枝末节。它们连接 Gaussian quadrature、正交基求值、最小二乘逼近、Christoffel 函数、weighted sampling 和 polynomial chaos expansion。
- 一维问题已有成熟理论，但非经典测度会让 Hankel 行列式、矩矩阵和直接正交化迅速病态。我的一条研究线是把“已知矩信息”转化为稳定的三项递推系数。
- 多维问题不是变量多一点，而是标量递推变成矩阵递推。正交基按总次数分块后，每个坐标乘法都对应一组递推矩阵。
- 生物医学仿真中的 UQ 仍然落在同一组数学对象上：输入参数的概率分布、正交多项式基、采样点、外部正向模型、PCE 系数，以及由系数读出的均值、方差、分位数和敏感性。

## 目录

1. [从 Gaussian Quadrature 看到问题入口](#sec-1)
2. [研究动机：把不确定输入变成可计算输出分布](#sec-2)
3. [文献、论文和源码锚点](#sec-3)
4. [正交多项式的计算接口：三项递推](#sec-4)
5. [一维算法：从矩信息到递推系数](#sec-5)
6. [多维算法：矩阵递推才是真正的推广](#sec-6)
7. [PCE：正交基如何变成 UQ 工具](#sec-7)
8. [UncertainSCI：非侵入式 UQ 的软件接口](#sec-8)
9. [生物医学应用：数学对象如何进入真实项目](#sec-9)
10. [与 AI 时代的连接](#sec-10)
11. [一条可复述的研究路线](#sec-11)
12. [理解检查](#sec-12)

<a id="sec-1"></a>

## 1. 从 Gaussian Quadrature 看到问题入口

正交多项式出现在 approximation theory、quadrature、special functions、differential equations 和 uncertainty quantification 中。这个说法很宽泛，不容易抓住重点。一个更直观的入口是：如何高效、准确地计算积分？

如果被积函数足够光滑，最朴素的数值积分方法是 midpoint rule：

$$
\int_a^b f(x)\,dx \approx (b-a) f\left(\frac{a+b}{2}\right).
$$

再往前走，可以用通过区间端点的线性函数得到 trapezoidal rule：

$$
\int_a^b f(x)\,dx \approx (b-a)\frac{f(a)+f(b)}{2}.
$$

Simpson's rule 则用二次多项式插值：

$$
\int_a^b f(x)\,dx \approx \frac{b-a}{6}
\left[
f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)
\right].
$$

这些 Newton-Cotes 型公式通常使用等距节点，直观、方便，也容易复用函数值。但在同样的函数评估次数下，Gaussian quadrature 往往能对光滑函数给出更高精度。一个经典结论是：$n$ 点 Gaussian quadrature 可以精确积分次数不超过 $2n-1$ 的多项式。

以 Gauss-Legendre quadrature 为例：

$$
\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i f(x_i).
$$

这里的节点 $x_i$ 是 Legendre 多项式的根，权重 $w_i$ 也由正交结构决定。如果权函数变成 Jacobi 型：

$$
f(x) = (1-x)^{\alpha}(1+x)^{\beta}g(x),
$$

更自然的选择就会变成 Jacobi 多项式诱导出的节点和权重。

这就是正交多项式进入数值计算的第一层原因：它们不是抽象装饰，而是在告诉我们“应该在哪里采样、怎样加权、如何稳定地计算基函数”。真正困难的问题随后出现：如果输入分布不是 Legendre、Hermite、Laguerre、Jacobi 这类经典测度，甚至是来自应用问题的非经典测度，我们还能稳定构造这些多项式和 quadrature rule 吗？

<a id="sec-2"></a>

## 2. 研究动机：把不确定输入变成可计算输出分布

科学计算里经常遇到同一个问题：模型 $f$ 很复杂，输入参数 $\Xi$ 又不是确定值。组织电导率、心脏位置、血管材料参数、脑刺激电极位置都可能有误差或个体差异。于是输出

$$
Y = f(\Xi)
$$

也应该被看成随机变量，而不是一个固定曲线或固定图像。

最直接的办法是 Monte Carlo：抽很多组参数，反复运行仿真，统计输出。但生物医学仿真往往很贵，一次有限元或电场计算可能已经不便宜，更不用说上万次。不确定性量化的数学替代思路是构造代理模型：

$$
f(\Xi) \approx \sum_{\alpha \in \Lambda} c_{\alpha}\psi_{\alpha}(\Xi),
$$

其中 $\{\psi_{\alpha}\}$ 是相对于输入分布正交的多项式基，$\Lambda$ 是选定的多指标集合。这就是 polynomial chaos expansion。若正交基足够稳定，PCE 的系数 $c_{\alpha}$ 不只可以预测输出，还可以直接给出均值、方差、分位数和 Sobol sensitivity。

所以整条路线从一个底层数学问题开始：

> 如何针对一般概率测度，稳定构造可用于求值、积分、逼近和不确定性量化的正交多项式？

<a id="sec-3"></a>

## 3. 文献、论文和源码锚点

核心材料可以按数学对象组织：

| 材料                                                                                                                                                                      | 数学对象                                    | 对技术路线的作用                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------- |
| Ph.D. thesis, _On the Computation of Recurrence Coefficients for Orthogonal Polynomials with Applications_                                                                | 递推系数、多维递推矩阵、PCE                 | 总览整条研究路线：一维算法、多维算法、UncertainSCI 应用。                 |
| [On the computation of recurrence coefficients for univariate orthogonal polynomials](https://doi.org/10.1007/s10915-021-01586-w)                                         | 一维三项递推系数                            | 提出 predictor-corrector Lanczos (PCL) 思路，并比较多种递推系数计算方法。 |
| [A Stieltjes algorithm for generating multivariate orthogonal polynomials](https://arxiv.org/abs/2202.04843)                                                              | 多维正交多项式矩阵递推                      | 提出 Multivariate Stieltjes (MS) algorithm，用矩信息构造递推矩阵。        |
| [UncertainSCI: A Python Package for Noninvasive Parametric Uncertainty Quantification of Simulation Pipelines](https://doi.org/10.21105/joss.04249)                       | PCE 软件接口                                | 说明 UncertainSCI 的非侵入式设计、API 和应用定位。                        |
| [UncertainSCI: Uncertainty quantification for computational models in biomedicine and bioengineering](https://doi.org/10.1016/j.compbiomed.2022.106407)                   | PCE 代理模型、weighted sampling、敏感性分析 | 展示心脏和脑刺激模型中的统计量、标准差和敏感性分析。                      |
| [Using UncertainSCI to Quantify Uncertainty in Cardiac Simulations](https://doi.org/10.22489/cinc.2020.275)                                                               | 心脏仿真的 UQ                               | 展示心脏位置、电导率变化如何进入不确定性量化流程。                        |
| [Uncertainty Quantification in Brain Stimulation Using UncertainSCI](https://www.sciencedirect.com/science/article/pii/S1935861X21004721)                                 | 脑刺激 UQ                                   | 展示 ECoG、tCS、TMS 等模型中的参数不确定性传播。                          |
| [Influence of Material Parameter Variability on the Predicted Coronary Artery Biomechanical Environment via Uncertainty Quantification](https://arxiv.org/abs/2401.15047) | 有限元 UQ、材料参数敏感性                   | 展示冠状动脉材料参数不确定性如何传播到应力场。                            |

源码对照：

- [Uni_ttr_examples](https://github.com/ZEXINLIU/Uni_ttr_examples)：一维 predictor-corrector / predictor-corrector Lanczos 与已有算法的实验入口。
- [Multi_ttr_examples](https://github.com/ZEXINLIU/Multi_ttr_examples)：多维 Multivariate Stieltjes 与矩方法的稳定性对照。
- [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI)：不确定性量化软件。
- [UncertainSCI/ttr.py](https://github.com/SCIInstitute/UncertainSCI/blob/master/UncertainSCI/ttr.py)：一维递推系数计算。
- [UncertainSCI/pce.py](https://github.com/SCIInstitute/UncertainSCI/blob/master/UncertainSCI/pce.py)：PCE 构造、采样、加权最小二乘、统计量和敏感性分析。
- [UncertainSCI 官方示例](https://uncertainsci.readthedocs.io/en/latest/tutorials/index.html)：官方小模型教程和可运行示例。

<a id="sec-4"></a>

## 4. 正交多项式的计算接口：三项递推

设 $\mu$ 是实线上的正测度，$\{p_n\}_{n \ge 0}$ 是相对于 $\mu$ 的标准正交多项式：

$$
\int_{\mathbb{R}} p_n(x)p_m(x)\,d\mu(x) = \delta_{mn}.
$$

在一维中，它们满足三项递推：

$$
x p_n(x)
= b_n p_{n-1}(x) + a_{n+1}p_n(x) + b_{n+1}p_{n+1}(x),
\quad p_{-1}=0.
$$

这条公式的计算含义比形式更重要。只要知道 $\{a_n,b_n\}$，就可以用局部递推稳定计算任意阶多项式的取值，而不需要把 $p_n$ 展开成单项式基。单项式系数在高阶时容易放大舍入误差；递推只在相邻阶之间传递信息。

同一组系数还定义 Jacobi matrix：

$$
J_n =
\begin{pmatrix}
a_1 & b_1 \\
b_1 & a_2 & b_2 \\
& \ddots & \ddots & \ddots \\
& & b_{n-1} & a_n
\end{pmatrix}.
$$

$J_n$ 的特征值给出 $n$ 点 Gaussian quadrature 的节点。因此递推系数同时服务三件事：多项式求值、构造积分公式、支持 PCE 中的正交基。

教学版求值代码如下：

```python
def evaluate_orthogonal_basis(x, ab, max_degree):
    # ab[n, 0] stores diagonal coefficients; ab[n, 1] stores off-diagonal coefficients.
    # This is only the recurrence interface. Production code also handles vectorization,
    # normalization, and boundary cases.
    p_minus_1 = 0.0
    p_n = 1.0 / ab[0, 1]
    values = [p_n]

    for n in range(max_degree):
        a_next = ab[n + 1, 0]
        b_n = ab[n, 1]
        b_next = ab[n + 1, 1]

        p_next = ((x - a_next) * p_n - b_n * p_minus_1) / b_next
        values.append(p_next)
        p_minus_1, p_n = p_n, p_next

    return values
```

<a id="sec-5"></a>

## 5. 一维算法：从矩信息到递推系数

若 $\mu$ 是 Legendre、Hermite、Laguerre 等经典测度，递推系数可以查表。但科学计算中的输入分布常常不是经典测度：可能分段、偏斜、有奇异点、有离散质量，或者来自经验数据。

此时自然会想到用矩：

$$
m_k = \int x^k\,d\mu(x).
$$

但直接从矩矩阵做正交化会遇到病态。Hankel 矩阵

$$
H_n = (m_{i+j})_{i,j=0}^{n}
$$

在高阶时条件数很快变大。直接 Gram-Schmidt、Hankel 行列式和单项式基方法概念简单，却会把矩计算误差放大成多项式系数误差。

[On the computation of recurrence coefficients for univariate orthogonal polynomials](https://doi.org/10.1007/s10915-021-01586-w) 对比了 discrete Painleve equations、Hankel determinants、arbitrary polynomial chaos、modified Chebyshev、Stieltjes procedure 和 Lanczos-type algorithm，并提出 PC/PCL。这里 PC 是 predictor-corrector，PCL 是 predictor-corrector Lanczos。

PC 的思想是利用相邻递推系数的连续性：先用上一阶预测下一阶，再用正交性残差修正。

```python
def predictor_corrector_step(mu_moment, p_nm1, p_n, a_n, b_n):
    # mu_moment(q) means integral q(x) dmu(x).
    # p_nm1 and p_n are two already constructed neighboring orthonormal polynomials.

    a_guess = a_n
    b_guess = b_n

    def p_np1_tilde(x):
        return ((x - a_guess) * p_n(x) - b_n * p_nm1(x)) / b_guess

    # Correct orthogonality and normalization using moment-based inner products.
    G_n_np1 = mu_moment(lambda x: p_n(x) * p_np1_tilde(x))
    a_corrected = a_guess + b_n * G_n_np1

    G_np1_np1 = mu_moment(lambda x: p_np1_tilde(x) ** 2)
    b_corrected = b_guess * G_np1_np1**0.5

    return a_corrected, b_corrected
```

PCL 再把 predictor-corrector 和 stabilized Lanczos 结合，覆盖连续和离散混合测度。`UncertainSCI/ttr.py` 里的 `predict_correct_bounded`、`predict_correct_unbounded`、`predict_correct_discrete`、`stieltjes_bounded` 等函数，正是这条思路的代码入口。

这里的数学贡献可以概括为：把“对测度求正交多项式”转化成逐阶、局部、可修正的递推系数估计问题，避开直接在全局矩矩阵上硬解高阶正交基。

<a id="sec-6"></a>

## 6. 多维算法：矩阵递推才是真正的推广

多维正交多项式的关键变化是：固定总次数后，$n$ 阶对象不是单个多项式，而是一组多项式向量

$$
\mathbb{P}_n(x)
= \left(P_{\alpha}(x)\right)_{|\alpha|=n}.
$$

对第 $i$ 个坐标，乘法算子 $x_i$ 满足块三项递推：

$$
x_i \mathbb{P}_n(x)
= A_{n,i}\mathbb{P}_{n+1}(x)
+ B_{n,i}\mathbb{P}_n(x)
+ A_{n-1,i}^{T}\mathbb{P}_{n-1}(x).
$$

这里 $A_{n,i}$ 和 $B_{n,i}$ 是矩阵，不是标量。它们描述的是“坐标乘法”在不同总次数多项式空间之间的投影。多维问题的难点也在这里：不同坐标方向的矩阵必须彼此一致，否则递推出来的对象不能代表同一个多维正交系统。

[A Stieltjes algorithm for generating multivariate orthogonal polynomials](https://arxiv.org/abs/2202.04843) 的 Multivariate Stieltjes algorithm 解决的是这个问题：假设能计算矩信息，就构造递推矩阵，使多维正交多项式可以稳定求值。

简化的算法视角如下：

```python
def multivariate_stieltjes(moment, degree_max, dim):
    # moment(alpha) returns integral x**alpha dmu(x).
    # The output is a set of recurrence matrix blocks by degree and coordinate.
    basis_blocks = initialize_degree_zero_basis()
    recurrence = {}

    for n in range(degree_max):
        for i in range(dim):
            recurrence[n, i] = solve_recurrence_matrix_block(
                moment=moment,
                current_basis=basis_blocks[n],
                coordinate=i,
            )

        basis_blocks[n + 1] = build_next_degree_block(recurrence, n)

    return recurrence
```

MS 在二维和三维中基本可以显式推进；更高维会出现非凸优化。论文和 [Multi_ttr_examples](https://github.com/ZEXINLIU/Multi_ttr_examples) 的实验重点是稳定性：在低维低阶时，矩方法和 Gram-Schmidt 类方法可能还能工作；阶数增大后，MS 的条件数和求值误差更可控。

这一步把问题从一维系数推进到多维矩阵。它仍然遵循同一个原则：不要直接在病态基底上求显式多项式，而是寻找稳定的递推接口。

<a id="sec-7"></a>

## 7. PCE：正交基如何变成 UQ 工具

UQ 中的正向问题是把输入分布推到输出分布。PCE 的做法是在输入随机变量 $\Xi$ 的概率空间里建立正交基：

$$
\mathbb{E}[\psi_{\alpha}(\Xi)\psi_{\beta}(\Xi)] = \delta_{\alpha\beta}.
$$

然后近似

$$
f(\Xi) \approx \sum_{\alpha \in \Lambda} c_{\alpha}\psi_{\alpha}(\Xi).
$$

一旦 $\{\psi_\alpha\}$ 正交，系数就有直接统计意义。若常数项是 $c_0$，则

$$
\mathbb{E}[f(\Xi)] \approx c_0,
\quad
\mathrm{Var}[f(\Xi)] \approx \sum_{\alpha \ne 0} c_{\alpha}^2.
$$

Sobol 敏感性也可以从系数分组得到：只含某个参数或某组参数的基函数项对应相应的方差贡献。这就是为什么正交基的稳定构造会影响 UQ 的可靠性。

从数学对象到软件对象的对应关系如下：

| 数学对象                  | UncertainSCI 对象                                                 |
| ------------------------- | ----------------------------------------------------------------- |
| 输入随机变量 $\Xi$ 的分布 | `BetaDistribution`、`NormalDistribution`、`TensorialDistribution` |
| 多指标集合 $\Lambda$      | `TotalDegreeSet` 等指标集合                                       |
| 正交基 $\psi_{\alpha}$    | 分布对象内部的正交多项式族                                        |
| 样本点 $\{\xi_j\}$        | `pce.generate_samples()`                                          |
| 仿真输出 $f(\xi_j)$       | `model_output`                                                    |
| PCE 系数 $c_\alpha$       | `pce.build()` 后得到的 `pce.coefficients`                         |
| 输出统计                  | `pce.mean()`、`pce.stdev()`、`pce.quantile()`                     |
| 敏感性                    | `pce.total_sensitivity()`、`pce.global_sensitivity()`             |

代码骨架如下：

```python
from UncertainSCI.pce import PolynomialChaosExpansion


def build_pce_uq(model, distribution, order):
    pce = PolynomialChaosExpansion(
        distribution=distribution,
        order=order,
        sampling="greedy-induced",
        training="wlsq",
    )

    pce.generate_samples()
    pce.build(model=model)

    return pce.mean(), pce.stdev(), pce.total_sensitivity()
```

<a id="sec-8"></a>

## 8. UncertainSCI：非侵入式 UQ 的软件接口

UncertainSCI 的价值在于把上面的数学对象包装成非侵入式工作流：外部模拟器只需要接受参数样本并返回输出；PCE、采样、统计和敏感性分析由数学层处理。

官方示例很适合理解这一点。

### 8.1 三参数扩散方程示例

[PCE Statistics 示例](https://uncertainsci.readthedocs.io/en/latest/tutorials/demos/pce_statistics.html) 和 [demos/build_pce.py](https://github.com/SCIInstitute/UncertainSCI/blob/master/demos/build_pce.py) 使用三参数一维扩散方程：

$$
-\frac{d}{dx}
\left(a(x,p)\frac{d}{dx}u(x,p)\right)
= f(x),
\quad x \in [-1,1].
$$

这个例子的数学含义是：

- 输入分布决定正交基和采样位置。
- 多项式阶数决定 $\Lambda$ 的大小，也决定要跑多少次模型。
- 输出 $u(x,p)$ 是随空间 $x$ 变化的函数，所以每个空间离散自由度都有一组 PCE 系数。
- 均值、标准差和敏感性都可以画成空间函数。

这和心脏或脑刺激仿真的区别只在正向模型更复杂：小示例里的 $u(x,p)$ 是偏微分方程的解，真实应用中的 $u$ 可能是躯干电势、电场强度、激活时间或应力场。

### 8.2 正弦模型示例：看懂参数交互

[demos/basic_uq_example.py](https://github.com/SCIInstitute/UncertainSCI/blob/master/demos/basic_uq_example.py) 用四参数函数

$$
f(x;p)=p_0\sin(p_1x+p_2)+p_3.
$$

四个参数分别控制振幅、频率、相位和平移量。这个例子很小，但它说明了 Sobol 型敏感性的意义：如果只看均值曲线，看不到哪个参数导致输出摆动；把 PCE 系数按参数子集分组后，可以区分单变量效应和变量交互效应。

### 8.3 随机扩散系数示例：从简化 PDE 到生物医学模型

[demos/sensitivity_measures.py](https://github.com/SCIInstitute/UncertainSCI/blob/master/demos/sensitivity_measures.py) 使用随机扩散系数：

$$
a(x,p)=\bar a(x)+\sum_{j=1}^{d}\lambda_jY_j\phi_j(x),
$$

其中 $(\lambda_j,\phi_j)$ 来自指数协方差核的 Karhunen-Loeve 展开。组织属性、纤维方向、材料刚度都可以被看成空间相关的随机结构；PCE 在这里做的事情，是把随机场参数 $p_j$ 映射到解 $u(x,p)$ 的统计结构，并比较总敏感性、全局敏感性和基于导数的敏感性。

<a id="sec-9"></a>

## 9. 生物医学应用：数学对象如何进入真实项目

应用部分可以按同一条数学链读：

$$
\text{参数分布}
\rightarrow
\text{正交基和样本}
\rightarrow
\text{外部仿真}
\rightarrow
\text{PCE 代理模型}
\rightarrow
\text{统计和敏感性}.
$$

### 9.1 心脏电场与躯干电势

[Using UncertainSCI to Quantify Uncertainty in Cardiac Simulations](https://doi.org/10.22489/cinc.2020.275) 处理两个问题：

- 心脏位置的变化如何影响 boundary element method 计算的躯干电势。
- 躯干和肺部电导率的变化如何影响 finite element method 计算的躯干电势。

从数学上看，心脏位置和组织电导率就是随机参数 $\Xi$。SCIRun 或 MATLAB 中的正向模型负责计算 $f(\xi_j)$，UncertainSCI 负责选择样本 $\xi_j$、拟合 PCE，并输出均值、标准差和敏感性。论文讨论中强调，Python 封装层可以接入已有 MATLAB 和 SCIRun 流程，不需要把仿真器内部改成 UQ 专用代码。

### 9.2 脑刺激模型

[Uncertainty Quantification in Brain Stimulation Using UncertainSCI](https://www.sciencedirect.com/science/article/pii/S1935861X21004721) 涉及 ECoG、tCS 和 TMS。它们的共同点是：电极位置、组织电导率、几何分割误差等参数会影响电场分布。

如果只使用一组名义参数，模型只给出一张电场图；UQ 流程会给出一组互相配套的结果：

- 均值场：平均预测。
- 标准差场：哪些区域最受输入不确定性影响。
- 敏感性图：哪个参数或参数交互主导方差。

数学上，标准差场来自 PCE 非常数项系数的平方和，敏感性图来自按参数子集分组的系数平方和。这说明应用图像不是经验后处理，而是正交展开系数的直接结果。

### 9.3 心脏纤维方向和激活序列

[UncertainSCI: Uncertainty quantification for computational models in biomedicine and bioengineering](https://doi.org/10.1016/j.compbiomed.2022.106407) 展示心肌纤维方向对激活序列的影响。纤维方向的变化不是简单标量误差，它会改变心室中的电传播路径。

在 UQ 语言里，纤维方向被参数化为随机输入，激活时间是空间场输出。PCE 代理模型在每个空间点上都有系数，因此可以得到激活时间均值、标准差，以及心外膜和心内膜纤维方向对不同区域的敏感性贡献。

### 9.4 冠状动脉有限元应力

[Influence of Material Parameter Variability on the Predicted Coronary Artery Biomechanical Environment via Uncertainty Quantification](https://arxiv.org/abs/2401.15047) 将 UncertainSCI 与有限元框架结合，研究冠状动脉材料参数如何影响血管壁应力。

这里的数学对象是 strain energy function 中的材料参数。论文的主要结论是：材料参数不确定性会传播成跨壁应力场的显著方差，外膜层中的应力离散程度最大，各向异性部分相关参数对应力变化更敏感。

这给出一个很实际的解释：PCE 阶数不是随便选的超参数。多项式空间的表达能力会直接变成工程中的仿真预算。

<a id="sec-10"></a>

## 10. 与 AI 时代的连接

这条路线和 AI 的连接点不是简单地“用神经网络替代仿真”。更准确的连接有三层。

第一，PCE 代理模型具有可解释结构。神经网络可以拟合复杂函数，但 PCE 的正交结构让均值、方差和 Sobol 敏感性可以从系数中读出。对于医学和工程模型，这种可审计性很重要。

第二，递推系数提供了依赖输入分布的表示方式。许多 AI 工作流默认把输入归一化后交给通用模型；这里的数学强调基函数应该匹配输入测度。不同参数分布对应不同正交基、不同采样策略和不同误差结构。

第三，UncertainSCI 的工作流可以被看成仿真代理系统的骨架。未来的智能体可以帮助用户从文献或数据中定义参数分布，选择多项式阶数和采样策略，调用外部仿真器，检查代理模型误差，再把敏感性分析转换成下一轮实验建议。

<a id="sec-11"></a>

## 11. 一条可复述的研究路线

这条研究路线可以压缩为六句话：

1. 底层问题是如何针对一般测度稳定构造正交多项式。
2. 一维时，重点是把矩信息转换成可靠递推系数，避免直接对矩矩阵做正交化带来的病态。
3. 多维时，重点是把标量递推推广为递推矩阵，并维持不同坐标乘法之间的一致性。
4. 稳定正交基进入 PCE 后，就能把昂贵仿真输出近似为可分析的多项式代理模型。
5. UncertainSCI 把这套数学封装成非侵入式软件，让 MATLAB、SCIRun、FEBio 等外部模型都能接入 UQ 流程。
6. 在医学和工程仿真里，这条路线的价值是从单点预测推进到带有方差、分位数和敏感性解释的可信预测。

<a id="sec-12"></a>

## 12. 理解检查

- 为什么递推系数比显式单项式系数更适合作为正交多项式的数值接口？
- PC/PCL 修正的是递推系数、正交性残差，还是求积节点？
- 多维递推矩阵为什么必须满足不同坐标方向的一致性？
- PCE 的均值、方差和 Sobol 敏感性分别对应哪些系数组合？
- `build_pce.py` 中的一维扩散方程示例与心脏、脑刺激应用在数学结构上哪里相同？
- 为什么非侵入式不确定性量化对已有生物医学仿真软件特别重要？
