---
layout: post
title: "Modern Reinforcement Learning Objectives: Bellman Targets, PPO/TRPO, GRPO, DAPO, CISPO, and GSPO"
date: 2026-06-25 00:00:00
description: A technical map of Bellman targets, policy gradients, trust-region optimization, PPO clipping, RLHF, DPO, and token- versus sequence-level objectives in modern LLM reinforcement learning.
tags: reinforcement-learning policy-gradient llm-rl ppo rlhf
categories: technical-notes
featured: true
toc:
  sidebar: left
---

# Bellman Target、Trust Region、PPO Clip 与 LLM Token/Sequence Ratio：强化学习目标函数的细节主线

本文重点包括：

- **Bellman 递推把“未来未知”变成可迭代的 bootstrap target。** MC、TD、Sarsa、Q-learning、DQN 的差别不只是采样方式不同，而是 target 中用了实际下一动作、贪心下一动作、目标网络还是整段回报；这个选择决定了偏差、方差、过估计和 off-policy 风险。
- **策略梯度真正难的不是写出 $\nabla \log \pi_\theta$，而是把整段回报拆成低方差、可归因的训练信号。** REINFORCE 用 trajectory return，baseline/A2C 把状态价值当 control variate，GAE 在 MC 与 TD 之间调偏差-方差，PPO 再用旧策略比值把多轮 minibatch 更新压回可信范围。
- **TRPO 和 PPO 的关系是“显式约束”到“一阶可优化代理目标”的工程折中。** TRPO 用平均 KL trust region、自然梯度方向和 line search 控制策略步长；PPO 把同一保守思想压进 clipped surrogate，用 `min` 构造 pessimistic bound，换来多轮 minibatch SGD 的易用性。
- **LLM RL 的核心分歧是 token-level 还是 sequence-level。** PPO 依靠 critic 把整段奖励拆成 token advantage；GRPO 用组内相对奖励让一条 response 的 token 共享 advantage；DAPO 改成 token-level loss 避免长回答被样本均值稀释；CISPO 改裁剪对象以保留 token 梯度；GSPO 把 ratio 提到 sequence level 以降低 token 级重要性权重的方差。
- **实际训练问题往往藏在目标函数的聚合方式里。** all-correct/all-wrong prompt group 会让 GRPO advantage 消失，低概率但高价值 token 容易被 clip 过早截断，长 CoT 会诱发过长回答和奖励噪声，KL/reference 约束既能防漂移也可能压制从 base model 到 reasoning policy 的必要迁移。
- **每个关键目标都配自包含代码块，而不是只列本地脚本路径。** 公式、mask、logprob、ratio、advantage、stop-gradient 和聚合维度会在正文中直接对应；事实来源以论文和公开实现为准，本地教学代码只作为内部理解参考，不作为公开源码引用。

## 目录

- [0. 阅读和来源说明](#0-阅读和来源说明)
- [Basic concepts](#basic-concepts)
- [Basic algorithms](#basic-algorithms)
  - [Value-based RL](#value-based-rl)
  - [Policy-based RL](#policy-based-rl)
- [Modern algorithms](#modern-algorithms)
  - [TRPO -> PPO](#trpo--ppo-从约束优化到裁剪代理目标)
  - [PPO](#proximal-policy-optimization-ppo)
  - [GRPO](#grpo-from-deepseekmath)
  - [DAPO](#dapo-seed-2025)
  - [CISPO](#cispo-minimax)
  - [GSPO](#gspo-qwen-2025)
  - [Reward Model](#reward-model-bradley-terry-偏好建模)
  - [RLHF](#reinforcement-learning-from-human-feedback-rlhf)
  - [DPO](#direct-preference-optimization-dpo)
- [Training-Inference Mismatch](#training-inference-mismatch)
- [Mathematical tools behind algorithms](#mathematical-tools-behind-algorithms)
  - [Bellman expectation equation 推导](#bellman-expectation-equation-推导)
  - [Bellman optimality contraction](#bellman-optimality-contraction)
  - [策略梯度定理证明](#策略梯度定理证明)
  - [TRPO trust-region 二阶近似](#trpo-trust-region-二阶近似)

## 0. 阅读和来源说明

阅读顺序建议：

1. **Basic concepts**：先建立 $v_\pi(s)$、 $q_\pi(s,a)$、Bellman expectation equation 和 Bellman optimality equation。
2. **Value-based RL**：看 MC、TD、Sarsa、Q-learning、DQN 如何把 Bellman 递推变成采样 target。
3. **Policy-based RL**：看策略梯度定理、REINFORCE、baseline、QAC/A2C 和 off-policy actor-critic 如何直接优化 $\pi_\theta$。
4. **Modern algorithms**：比较 PPO、GRPO、DAPO、CISPO、GSPO、Reward Model、RLHF、DPO 的目标函数、粒度选择和训练稳定性问题。
5. **Mathematical tools**：回查 Bellman 推导、策略梯度定理、TRPO 二阶近似、Bradley-Terry、KL 非对称性和 KL estimator。

来源分层如下：

- **论文是公式和时间线基准**：[DQN](https://arxiv.org/abs/1312.5602) (2013)、[Double DQN](https://arxiv.org/abs/1509.06461) (2015)、[Dueling DQN](https://arxiv.org/abs/1511.06581) (2015)、[TRPO](https://arxiv.org/abs/1502.05477) (2015)、[GAE](https://arxiv.org/abs/1506.02438) (2015)、[Scheduled Sampling](https://arxiv.org/abs/1506.03099) (2015)、[PPO](https://arxiv.org/abs/1707.06347) (2017)、[RLHF summarization](https://arxiv.org/abs/2009.01325) (2020)、[InstructGPT](https://arxiv.org/abs/2203.02155) (2022)、[DPO](https://arxiv.org/abs/2305.18290) (2023)、[GRPO/DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024)、[DAPO](https://arxiv.org/abs/2503.14476) (2025)、[CISPO/MiniMax-M1](https://arxiv.org/abs/2506.13585) (2025)、[GSPO](https://arxiv.org/abs/2507.18071) (2025)。模仿学习中的分布偏移和数据聚合参考 [DAgger](https://proceedings.mlr.press/v15/ross11a.html)。
- **公开实现用于核对工程对象**：[OpenAI Baselines PPO2](https://github.com/openai/baselines/tree/master/baselines/ppo2) 对应 PPO 的 rollout、advantage、clip、value loss；[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) 对应 policy gradient / actor-critic 推导；[Hugging Face TRL](https://github.com/huggingface/trl) 对应 DPO/GRPO 等 trainer 的公开实现；[DAPO 项目页](https://dapo-sia.github.io/) 和 [DAPO 代码](https://github.com/BytedTsinghua-SIA/DAPO) 对应 DAPO 的开源训练 recipe；[MiniMax-M1](https://github.com/MiniMax-AI/MiniMax-M1) 对应 CISPO 的公开发布。
- **正文代码块是教学化重写**：它们服务于公式理解，保留输入、mask、advantage、ratio 和梯度流向，不等同于某个框架的完整训练脚本。外部博客和中文材料只作为直觉辅助，不作为唯一依据。

# Basic concepts

本节采用 Sutton & Barto 常见记号：时刻 $t$，智能体处于状态 $S_t$，按策略 $\pi$ 选择动作 $A_t$，环境转移到 $S_{t+1}$ 并返回即时奖励 $R_{t+1}$。大写字母表示随机变量，小写字母表示随机变量的具体取值。

- trajectory: a state-action-reward chain $S_t \xrightarrow{A_t} S_{t+1}, R_{t+1} \xrightarrow{A_{t+1}} S_{t+2}, R_{t+2} \cdots$
- discounted return along a trajectory
- definition: $G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^{k} R_{t+1+k}$
- recursive formula: $G_t = R_{t+1} + \gamma G_{t+1}$
- system model
- state transition probability $p(s'|s,a)$
- reward probability $p(r|s,a)$
- state value $v_\pi(s)$
- definition:

$$
v_\pi(s) = \mathbb{E}[G_t|S_t=s] = \mathbb{E}[R_{t+1}|S_t=s] + \gamma\mathbb{E}[G_{t+1}|S_t=s]
$$

- Bellman expectation equation (element-wise / matrix / expectation form):

$$
\begin{aligned}
v_\pi(s) &= \sum_{a\in\mathcal{A}}\pi(a|s)\left[\sum_r p(r|s,a)r + \gamma \sum_{s'\in \mathcal{S}}p(s'|s,a) v_\pi(s')\right] \\
v_\pi(s) &= r_\pi(s) + \gamma\sum_{s'\in\mathcal{S}} p_\pi(s'|s) v_\pi(s') \\
v_\pi &= r_\pi+\gamma P_\pi v_\pi \\
v_\pi(s) &= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s].
\end{aligned}
$$

解读：当前状态价值 = 即时奖励期望 + 下一个状态价值期望。元素形式、矩阵形式和 expectation form 的展开证明见 [Bellman expectation equation 推导](#bellman-expectation-equation-推导)。

- Bellman equation (matrix-vector form): suppose states are indexed as $s_i$ for $i=1,\dots,n$, where $n=|\mathcal{S}|$.

$$
\begin{aligned}
v_\pi &= r_\pi+\gamma P_\pi v_\pi,\\
\left[P_\pi\right]_{i,j} &= p_\pi(s_j|s_i),\\
v_\pi(s_i) &= r_\pi(s_i)+\gamma\sum_{s_j\in\mathcal{S}}p_\pi(s_j|s_i)v_\pi(s_j).
\end{aligned}
$$

Here $P_\pi$ is a nonnegative stochastic matrix, so each row sums to 1.

- close-form solution: $v_\pi = (I -\gamma P_\pi)^{-1} r_\pi$, where invertible can be proved by Gershgorin circle theorem
- iterative solution: $v_{k+1} = r_\pi + \gamma P_\pi v_k,\quad k = 0, 1, 2, \dots$, convergence proved by showing error $v_k - v_\pi$ converge to 0
- Bellman optimality equation (BOE) (element-wise form)

$$
v(s)=\max_{\pi(s)\in\Pi(s)}\sum_{a\in\mathcal{A}}\pi(a|s)q(s,a)
$$

- Bellman optimality equation (BOE) (matrix-vector form)

$$
\begin{aligned}
v &= \max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v)=f(v),\\
\left[r_\pi\right]_s
&= \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}p(r|s,a)r,\\
\left[P_\pi\right]_{s,s'}
&= \sum_{a\in\mathcal{A}}\pi(a|s)p(s'|s,a).
\end{aligned}
$$

- contraction property: the nonlinear $v = f(v)$ has a fixed point solution $v^{\ast}$ by contraction mapping theorem. The contraction proof can be checked in [Bellman optimality contraction](#bellman-optimality-contraction)。
- BOE solution solved by value iteration:

$$
\begin{aligned}
v^{\ast}
&=
\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v^{\ast}),\\
v_{k+1}
&=
f(v_k)=\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v_k),
\quad k=0,1,2,\dots
\end{aligned}
$$

- The optimal policy after solving $v^{\ast}$:

$$
\pi^{\ast} = \mathrm{argmax}_{\pi\in\Pi}(r_\pi+\gamma P_\pi v^{\ast})
$$

- action value $q_\pi(s,a)$
- definition: $q_\pi(s,a) = \mathbb{E}[G_t|S_t=s,A_t=a]$
- relation with state value: $v_\pi(s) = \sum_{a\in\mathcal{A}}\pi(a|s)q_\pi(s,a)$
- Bellman equation (element-wise form):

$$
q_\pi(s,a) = \sum_{r\in\mathcal{R}}p(r|s,a)r + \gamma\sum_{s'\in\mathcal{S}}p(s'|s,a)v_\pi(s')
$$

总结动作价值函数的多种表达形式：

$$
\begin{aligned}
q_\pi(s,a) &= \sum_{r\in\mathcal{R}}p(r|s,a)r + \gamma\sum_{s'\in\mathcal{S}}p(s'|s,a) (\sum_{a'\in\mathcal{A}}\pi(a'|s')q_\pi(s',a')) \\
q_\pi(s,a) &= \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a]
\end{aligned}
$$

The equivalence can be proved by $p(s',a'|s,a) = p(s'|s,a)\pi(a'|s')$ by conditional independence.

解读：采取动作 $a_t$ 的价值 = $a_t$ 带来的即时奖励 + $a_t$ 导致的下一状态的平均价值

- Bellman equation (matrix-vector form): replace $v_\pi(s')$ by $q_\pi(s,a)$.

$$
\begin{aligned}
q_\pi(s,a) = \sum_{r\in\mathcal{R}}p(r|s,a)r + \gamma\sum_{s'\in\mathcal{S}}p(s'|s,a) (\sum_{a'\in\mathcal{A}}\pi(a'|s')q_\pi(s',a'))
\end{aligned}
$$

The same relation can be written as a vector equation over state-action pairs:

$$
\begin{aligned}
q_\pi &= \tilde r+\gamma P\Pi q_\pi,\\
\left[q_\pi\right]_{(s,a)} &= q_\pi(s,a),\\
\left[\tilde r\right]_{(s,a)}
&= \sum_{r\in\mathcal{R}}p(r|s,a)r,\\
\left[P\right]_{(s,a),s'} &= p(s'|s,a),\\
\Pi_{s',a'} &= \pi(a'|s').
\end{aligned}
$$

- Bellman optimality equation (BOE)

$$
\begin{aligned}
q^{\ast}(s,a) &= r(s,a) + \gamma\sum_{s'\in\mathcal{S}} p(s'|s,a) \max_{a'} q^{\ast}(s',a') \\
q^{\ast}(s,a) &= \mathbb{E}[R_{t+1} + \gamma \max_{a'} q^{\ast}(S_{t+1},a')|S_t=s,A_t=a]
\end{aligned}
$$

# Basic algorithms

## Value-based RL

### Dynamic Programming Algorithm

解读：已知环境模型，即奖励概率 $p(r|s,a)$ 和状态转移概率 $p(s'|s,a)$，其中奖励概率进一步得到 $r(s,a) = \sum_r p(r|s,a)r$，分布代表在状态 $s$ 做动作 $a$ 后，会以什么概率到达各个下一状态 $s'$，以及平均会拿多少奖励

- Value Iteration (1-step truncated policy iteration)

实际上在做迭代算法求解 BOE: $v_{k+1} = f(v_k) = \max_{\pi\in\Pi}(r_\pi + \gamma P_\pi v_k)$

1. policy update

- matrix form: $\pi_{k+1} = \mathrm{argmax}(r_\pi + \gamma P_\pi v_k)$
- element-wise form: $\pi_{k+1}(a|s) = \mathrm{argmax}\sum_a \pi(a|s) q_k(s,a)$

2. value update

- matrix form: $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$
- element-wise form: $v_{k+1} = \sum_a \pi_{k+1}(a|s) q_k(s,a)$

summary: start with $v_0$, $v_k(s) \rightarrow q_k(s,a) \rightarrow a^{\ast}(s) \rightarrow \pi_{k+1}(a|s) \rightarrow v_{k+1}(s)$ until a certain criterion of $v_k$ is satisfied

- Policy Iteration ($\infty$-step truncated policy iteration)

1. policy evaluation

- matrix form: $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$, i.e., solving Bellman equation, another iterative algorithm embedded $v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}$
- element-wise form: $v_{\pi_k}^{(j+1)}(s) = \sum_a \pi(a|s) q_{\pi_k}^{(j)}(s,a)$

2. policy improvement

- matrix form:

$$
\pi_{k+1} = \mathrm{argmax}_{\pi}(r_\pi+\gamma P_{\pi}v_{\pi_k})
$$

- element-wise form:

$$
\pi_{k+1} = \mathrm{argmax}_{\pi}\sum_a\pi(a|s)q_{\pi_k}(s,a)
$$

summary: start with $\pi_0$, $\pi_k \rightarrow v_{\pi_k} \rightarrow q_{\pi_k}(s,a) \rightarrow a^{\ast}(s) \rightarrow \pi_{k+1}$ until a certain criterion of $v_{\pi_k}^{(j)}$ is satisfied

动作价值 $q_\pi(s,a)$ 在 Policy Iteration 中的重要性：

- policy evaluation 先估计 $v_{\pi_k}(s,a)$，再通过 system model 得到 $q_{\pi_k}(s,a)$
- policy improvement 中机基于 $q_{\pi_k}(s,a)$ 得到新的策略 $\pi_{k+1}$

### model-free

- MC-based RL
- MC Basic (calculate action values from experience samples)

1. policy evaluation

- definition of action value $q_{\pi_k}(s,a) = \mathbb{E}[G_t|S_s=s,A_t=a]$
- starting from $(s,a)$, following $\pi_k$，obtain $n$ episodes and the return of the $i$-th episode is $g_{\pi_k}^{(i)}(s,a)$, then $q_{\pi_k}(s,a) \approx q_k(s,a) = \frac{1}{n}\sum_{i=1}^{n} g_{\pi_k}^{(i)}(s,a)$

2. policy improvement the same as policy iteration

- MC exploring Starts

1. utilize samples: first visit / every visit instead of initial visit
2. update policy: use return of single episode instead of all ones

- MC $\epsilon$-Greedy
- for soft policy
- note probability of $x\lt\epsilon$ for randomly selecting actions includes "may select the greedy action again"
- summary: non-bootstrapping, high estimation variance & low bias, non-incremental (must wait until an episode has been collected)

- TD(temporal difference) learning
- Basic TD learning (estimate state values)
- algorithm:

$$
v_{t+1}(s_t) = v_t(s_t) -\alpha_t(s_t) \left[ v_t(s_t) - \left(r_{t+1}+\gamma v_t(s_{t+1})\right) \right]
$$

For all $s\neq s_t$, the estimate remains unchanged. TD target and TD error are:

$$
\bar v_t = r_{t+1}+\gamma v_t(s_{t+1}), \quad \delta_t = v(s_t)-\bar v_t.
$$

- derivation: apply Robbins-Monro algorithm to solve Bellman expectation equation $v_\pi(s) = \mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]$
- summary:

1. the algorithm attempts to drive $v_t(s_t)$ to $\bar{v}_t$
2. TD error reflects discrepancy between estimate $v_t$ and true state value $v_\pi$, so TD error is zero in expectation sense when $v_t$ is accurate
3. bootstrapping, low estimation variance (due to fewer variables involved) & high bias, incremental (update once receiving an experience sample)

- Sarsa (estimate action values)
- algorithm: $q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t)[q_t(s_t,a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))]$, and remains unchanged for all $(s,a) \neq (s_t,a_t)$.
- derivation: apply Robbins-Monro algorithm to solve Bellman expectation equation in terms of action, $q_\pi(s,a) = \mathbb{E}[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a]$
- summary
- Sarsa is a stochastic approximation algorithm for solving for Bellman (expectation) equation of given policy

$$
\begin{aligned}
q_\pi(s,a) = \mathbb{E}[R + \gamma q_\pi(S',A'|s,a)]
\end{aligned}
$$

- 交替进行策略评估和策略改进 (更新的 target 和 交互环境的 behavior 策略都是 $\epsilon$-greedy 策略)
- on-policy:
- samples: $s_t \xrightarrow{\pi_b} a_t \xrightarrow{model} r_{t+1}, s_{t+1} \xrightarrow{\pi_b} a_{t+1}$
- evaluation of target policy $\pi_T$ relies on $r_{t+1}, s_{t+1}, a_{t+1}$, where $a_{t+1}$ is generated following behavior policy $\pi_b$，即生成下一个动作 $a_{t+1}$ 用于估计 q-value 的策略和与环境交互的策略是同一个
- Sarsa 估计的是当前行为策略的价值，因此会规避探索带来的风险

- Q-learning
- algorithm: $q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t)[q_t(s_t,a_t) - (r_{t+1} + \gamma \max_{a\in\mathcal{A}(s_{t+1})} q_t(s_{t+1}, a))]$, remains unchanged for all $(s,a) \neq (s_t,a_t)$.
- derivation: apply Robbins-Monro algorithm to solve Bellman optimality equation in terms of action, $q(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a\in\mathcal{A}(S_{t+1})} q(S_{t+1},a)|S_t=s,A_t=a]$
- summary:
- Q-learning is a stochastic approximation algorithm for solving for Bellman optimal equation, the optimal solution satisfies

$$
\begin{aligned}
q^{\ast}(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a\in\mathcal{A}(S_{t+1})} q^{\ast}(S_{t+1},a)|S_t=s,A_t=a]
\end{aligned}
$$

- 价值迭代(利用Bellman Optimality Equation 直接估计最优动作价值函数)是通过 greedy 策略，而与环境交互的是 $\epsilon$-greedy 策略
- off-policy (can be implemented on/off-policy):
- samples: $s_t \xrightarrow{\pi_b} a_t \xrightarrow{model} r_{t+1}, s_{t+1}$
- the estimation of $q(s,a)$ relies on $r_{t+1}, s_{t+1}$, which does not involve $\pi_b$，估计 q-value 时使用的策略是 greedy，但是与环境交互的是 $\epsilon$-greedy
- Q-learning 估计的是最优策略的价值，与当前行为策略的探索无关

- DQN (deep Q-learning)
- objective: the squared Bellman optimality error

$$
\begin{aligned}
J(w) = \mathbb{E}[(R + \gamma \max_{a'\in\mathcal{A}(S')} \hat{q}(S',a',w_T) - \hat{q}(S,A,w))^{2}]
\end{aligned}
$$

if the same parameters $w$ appear in both the target and prediction terms, the target moves at the same time as the prediction. DQN therefore uses $w$ for the online network and $w_T$ for the target network.

- gradient of $J$ is

$$
\begin{aligned}
\nabla_w J = -2\mathbb{E}[(R + \gamma \max_{a'\in\mathcal{A}(S')} \hat{q}(S',a',w_T) - \hat{q}(S,A,w)) \nabla_w \hat{q}(S,A,w)]
\end{aligned}
$$

- algorithm: $(r + \gamma \max_{a'\in\mathcal{A}(s')}\hat{q}(s',a',w_T) - \hat{q}(s,a,w))^{2}$

```python
        # batch 来自 replay buffer: (s, a, r, done, s_next)
        q_all_actions = online_q_net(states)  # [batch_size, num_actions]

        # gather 只取本次真实执行过的动作 a 的 Q 值，对应 \hat{q}(S,A,w)。
        q_sa = q_all_actions.gather(1, actions[:, None]).squeeze(1)

        with torch.no_grad():
            # target network 只生成 bootstrap target，不接收当前 loss 的梯度。
            next_q = target_q_net(next_states).amax(dim=-1)
            td_target = rewards + gamma * (1.0 - dones.float()) * next_q

        # Bellman optimality error: 让 online network 逼近固定一小段时间的 target。
        dqn_loss = torch.nn.functional.mse_loss(q_sa, td_target)
```

- $\max$ 过高估计问题：如果所有动作的估计都带有噪声，那么最大值更容易选中被噪声推高的动作。因此，即使每个单独的 Q 值估计在平均意义下没有偏差， $\max$ 之后的估计也可能偏高：

$$
\mathbb{E}[\max_a \hat{q}(s,a,w_T)] \ge \max_a\mathbb{E}[\hat{q}(s,a,w_T)]
$$

- “致命三元组”问题（The Deadly Triad）
- 函数逼近误差：对特定状态-动作对 $(s,a)$ 的价值估计进行更新时，由于共享参数，会隐含地改变其他（可能不相关）状态-动作对的估计值
- 离线策略数据分布： $\pi_b$ 可能与 $\pi_T$ 不同，使用 $\pi_b$ 收集的数据来更新策略用以趋向于从 $\pi_T$ 推导的目标，如 TD target 时，我们实际上是在根据目标策略在实际执行时可能不会遇到的状态和动作分布来评估它。函数逼近器可能因此被迫外推这些“离策略”动作的价值，导致时序差分目标中可能出现较大误差
- 自举更新（Bootstrapping）：更新目标包含了下一个状态的估计值 $\hat{q}(s',a,w)$，如果这个估计值不准确（由于函数逼近误差或离策略数据外推），误差就会被“自举”到 $\hat{q}(s,a,w)$ 中。Q-learning 的 $\max$ 寻找下一个状态的最大（可能被高估的）Q 值。如果在状态 $s'$ 函数逼近器错误地将高值分配给行为策略很少或从未采取过的动作 $a'$，那么这个被高估的值将被用于目标中，从而可能增加 $\hat{q}(s,a,w)$ 的价值估计。这会产生一个反馈循环，使得逼近误差和离策略外推被自举放大，可能导致发散
- 解决方案：
- two networks: online network $w$ updates in every gradient step, and target network $w_T$ copies $w$ every $C$ iterations
- experience replay: collect $\mathcal{B} = \lbrace(s,a,r,s')\rbrace$ as replay buffer, and sample transitions approximately uniformly to break the correlation between sequential samples generated by $\pi_b$
- Implementation: DQN 原始论文的做法：训练循环里，每执行一步动作就立即从回放池采样训练一次。一边收集数据边训练，最大化数据利用效率
- 其他优化：
- Prioritized Experience Replay（优先经验回放）: 给 TD Error 大的经验更高的采样概率，再用重要性权重修正非均匀采样带来的偏差。在标准的经验回放中，旧经验按照先进先出（FIFO）的方式被淘汰。确实有可能一条关键经验被淘汰，但由于训练初期回放池还没满，关键经验通常会被多次采样到
- Double DQN: 把“选择动作”和“评估动作”分开。它先用当前网络选择动作，再用目标网络评价这个动作：

$$
a^{\ast} = \mathrm{argmax}_{a\in\mathcal{A}(s')}\hat q(s',a,w), \quad y = r+\gamma\hat q(s',a^{\ast},w_T).
$$

两个网络的误差不再通过同一个最大值运算直接叠加，q-value 过高估计的问题得到缓解

- Dueling DQN: 改变网络架构，适用动作差异不明显的状态场景 $q(s,a) = v(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'\in\mathcal{A}} A(s,a')$, $v(s)$ is a scalar, $A(s,a)$ produces action dimension

```python
            features = backbone(states)
            values = value_head(features)          # shape: [batch_size, 1]
            advantages = advantage_head(features)  # shape: [batch_size, num_actions]
            q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
```

- 关于 on-policy & off-policy 的总结：
- 定义: target policy $\pi_T$ 将要评估或者改进的策略叫做目标策略；behavior policy $\pi_b$ 实际与环境交互用来决定动作的策略，二者相同代表on-policy
- Sarsa is on-policy: 更新的 target policy 是 $\epsilon$-greedy policy，和采取动作的 behavior policy 是相同的策略
- Q-learning is off-policy: 更新的 target policy 是 greedy policy，而采取动作的 behavior policy 是另一个不同的 $\epsilon$-greedy policy

## Policy-based RL

DQN 在训练好之后对同一个状态永远输出同一个动作（确定性策略）；策略网络对同一个状态有可能输出不同的动作（随机性策略），其输出的不是动作分数而是概率分布（离散动作：输出各个动作概率 by softmax；连续动作：输出高斯分布参数）。这种随机性不是缺陷，而是特性——它天然包含探索，不需要额外的 $\epsilon$-greedy

### 策略梯度定理

策略梯度把“提高好动作概率、降低坏动作概率”写成可反向传播的目标。主流可用形式是

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_t,a_t} \left[ \nabla_\theta\log\pi_\theta(a_t|s_t)A^{\pi}(s_t,a_t) \right], \quad A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s).
$$

完整证明见 [策略梯度定理证明](#策略梯度定理证明)。那里会说明 trajectory likelihood-ratio、reward-to-go、baseline/advantage 和 discounted occupancy 形式之间的关系。

在代码里只需要抓住两点：第一，采样数据应来自当前策略或带重要性采样修正；第二，advantage 是权重，通常不让 actor loss 反向更新 critic。

```python
# states/actions 来自当前策略采样的 rollout；old data 不能随便复用，否则分布已经变了。
dist = policy(states)
log_probs = dist.log_prob(actions)

# advantages 可以来自 MC return、TD error 或 GAE。
# detach 表示 advantage 是权重，不让 actor loss 反向更新 critic。
policy_loss = -(log_probs * advantages.detach()).mean()
policy_loss.backward()
```

- gradient-ascent algorithm:

$$
\theta_{t+1}=\theta_t+\alpha\nabla_\theta\log\pi_{\theta_t}(a_t|s_t)\hat{A}_t
$$

- gradient-descent loss:

$$
\mathcal{L}_{PG}(\theta)=-\log\pi_\theta(a_t|s_t)\hat{A}_t
$$

### Basic algorithms

- Monte Carlo policy gradient (REINFORCE)
- algorithm:

$$
\mathbb{E}_{s\sim S,A\sim\pi_\theta(S)}
\left[
\nabla_\theta \ln\pi_\theta(A|S)q_\pi(S,A)
\right]
$$

- implementation:

1. generate an episode $\lbrace s_0, a_0, r_1, \dots, s_{T-1}, a_{T-1}, r_T\rbrace$ following $\pi_\theta$
2. at each time step $t$

- obtain $q_t(s_t,a_t) = \sum_{k=t+1}^{T} \gamma^{k-t-1}r_k$ by MC estimation, this explain why needs to collect an episode before "every experience sample update" of $\theta$, a more efficent bootstep version:

```python
            for reward in reversed(rewards):
                G = reward + gamma * G
                returns.insert(0, G)
```

- $\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln\pi_{\theta_t}(a_t|s_t)q_t(s_t,a_t)$
- on-policy: sampling following $\pi_\theta$ 必须用当前策略产生的数据。策略一更新，旧数据就失效了
- cons: 高方差，MC estimation 从时刻 $t$ 到 episode 结束的累积回报，它包含了这段路径上的所有随机性。同一个动作，不同的采样轨迹可能给出截然不同的 $G_t$，其波动导致高方差

```python
    # 前向传播：获取每个状态下的动作概率
    probs = policy(states_tensor)  # [T, action_dim]
    # 计算所采取动作的对数概率 log π(a_t|s_t)
    # gather(1, actions) 选取每个状态对应动作的概率
    action_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    log_probs = torch.log(action_probs + 1e-8)  # 加小常数防止 log(0)
    # 策略梯度损失：-log π(a_t|s_t) * G_t
    loss = -(log_probs * returns_tensor).mean()
```

- REINFORCE with baseline
- algorithm:

$$
\mathbb{E}_{s\sim d^{\pi},A\sim\pi_\theta(\cdot|S)}
\left[
\nabla_\theta \ln\pi_\theta(A|S)(q_\pi(S,A)-b(S))
\right]
$$

The baseline term has zero expectation:

$$
\mathbb{E}_{s\sim S,A\sim\pi_\theta(S)} \left[ \nabla_\theta \ln\pi_\theta(A|S)b(S) \right] =0
$$

- choices of $b(s)$: reduce approximation variance

1. optimal: by minimize the trace of $var(X)$, complex in practice
2. suboptimal: choose the state value as baseline.

$$
b(s)=\mathbb{E}_{A\sim\pi}\left[q_\pi(s,A)\right]=v_\pi(s)
$$

- implementation: use MC estimation $g_t$ to approximate $q_t$.

$$
\theta_{t+1} = \theta_t+\alpha\nabla_\theta\ln\pi_{\theta_t}(a_t|s_t) \left(q_t(s_t,a_t)-v_\phi(s_t)\right)
$$

- 价值网络的训练：

$$
\mathcal{L}_V(\phi)=\left(v_\phi(s_t)-g_t\right)^{2}
$$

- 策略网络的训练：使用优势 $A_t = g_t - v_\phi(s_t)$

```python
    # 价值网络学习 V(s)
    values = value_net(states_t)
    value_loss = nn.MSELoss()(values, returns_t)
    # 用优势更新策略
    with torch.no_grad():
        values_pred = value_net(states_t)
    advantages = returns_t - values_pred
    policy_loss = -(log_probs * advantages).mean()
```

- Q Actor-critic (QAC)
- implementation: at each time step $t$ in each episode

1. generate $a_t$ following $\pi_\theta(a|s_t)$, observe $r_{t+1}, s_{t+1}$ and then generate $a_{t+1}$ following $\pi_\theta(a|s_{t+1})$
2. the policy update is

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln\pi_{\theta_t}(a_t|s_t)q(s_t,a_t, w_t)
$$

where $q_t(s_t,a_t)$ is approximated by TD learning, value update is

$$
w_{t+1} = w_t + \alpha_w[r_{t+1} + \gamma q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t)]\nabla_w q(s_t,a_t,w_t)
$$

- Advantage Actor-critic (A2C)
- algorithm:

$$
\theta_{t+1} = \theta_t+\alpha\nabla_\theta\ln\pi_{\theta_t}(a_t|s_t) \left[q_t(s_t,a_t)-v_t(s_t)\right]
$$

where $\delta_t(s_t,a_t)=q_t(s_t,a_t)-v_t(s_t)$ is the advantage function.

- Advantage function can be estimated by TD error

$$
q_t(s_t,a_t) - v_t(s_t) \approx r_{t+1} + \gamma v_t(s_{t+1}) - v_t(s_t),
$$

which can be proved by

$$
q_\pi(s_t,a_t) - v_\pi(s_t) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) - v_\pi(S_t)|S_t=s_t,A_t=a_t],
$$

- Summary:
- return estimator $q_t(s,a)$ requires a network for action value, suboptimal baseline $v_t(s)$ requires a network for state value, original advantage function needs to maintain TWO networks
- TD error estimation helps reduce to One network for state value
- implementation: at each time step $t$ in each episode

1. generate $a_t$ following $\pi_\theta(a|s_t)$, observe $r_{t+1}, s_{t+1}$
2. estimate advantage function by TD error $\delta_t = r_{t+1} + \gamma v(s_{t+1},w_t) - v(s_t,w_t)$
3. policy update by $\theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta\ln\pi_\theta(a_t|s_t)$, loss is calculated as $-\delta_t \log\pi_\theta(a_t|s_t)$
4. value update $w_{t+1} = w_t + \alpha_w \delta_t\nabla_w v(s_t, w_t)$, loss is calculated as $(\delta_t)^{2}$

```python
        # TD Error
        td_target = reward + gamma * next_value
        td_error = td_target - value
        # Actor 损失：策略梯度 × 优势
        actor_loss = -log_prob * td_error.detach()
        # Critic 损失：让 V(s) 接近 TD Target
        critic_loss = td_error.pow(2)
        # 总损失
        loss = actor_loss + critic_loss
```

- Off-policy actor-critic
- Off-policy policy gradient theorem

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\rho, A\sim\beta(\cdot|S)}[\frac{\pi_\theta(A|S)}{\beta(A|S)}\nabla_\theta\ln\pi_\theta(A|S)q_\pi(S,A)],
$$

where $\rho(s) = \sum_{s'\in\mathcal{S}} d_\beta(s') Pr_\pi(s|s')$ is the state distribution and $Pr_\pi(s|s')$ is the discounted total probability of transitioning.

- algorithm (with baseline and importance sampling)

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\rho, A\sim\beta(\cdot|S)}[\frac{\pi_\theta(A|S)}{\beta(A|S)}\nabla_\theta\ln\pi_\theta(A|S)(q_\pi(S,A) - v_\pi(S))]
$$

- implementation: given a behavior policy $\beta(a|s)$, at each time step $t$ in each episode

1. generate $a_t$ following $\beta(a|s_t)$, observe $r_{t+1}, s_{t+1}$
2. estimate advantage function by TD error $\delta_t = r_{t+1} + \gamma v(s_{t+1},w_t) - v(s_t,w_t)$
3. policy update by $\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} \delta_t \nabla_\theta\ln\pi_\theta(a_t|s_t)$, loss is calculated as $-\frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)}\delta_t \log\pi_\theta(a_t|s_t)$
4. value update $w_{t+1} = w_t + \alpha_w \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} \delta_t\nabla_w v(s_t, w_t)$, loss is calculated as $(\delta_t)^{2}$

### Modern algorithms

<a id="trpo--ppo-从约束优化到裁剪代理目标"></a>

#### TRPO -> PPO：从约束优化到裁剪代理目标

##### 目标与约束

TRPO 的目标不是“每次都更新得很小”，而是“在策略分布变化可控的区域内尽量更新得大”。它先最大化旧策略分布下的新策略 surrogate objective，再用平均 KL 约束限制新旧策略距离：

$$
\max_\theta
\mathbb{E}_{s\sim\rho_{\theta_{old}},a\sim\pi_{\theta_{old}}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
A_{\theta_{old}}(s,a)
\right]
$$

$$
\text{s.t.}\quad
\mathbb{E}_{s\sim\rho_{\theta_{old}}}
\left[
D_{KL}\left(\pi_{\theta_{old}}(\cdot|s) \Vert \pi_\theta(\cdot|s)\right)
\right]\le\delta.
$$

##### 机制直觉

如果只最大化 ratio-weighted advantage，策略可以把某些动作概率推得过猛，采样分布和更新后分布迅速脱节；KL trust region 把“策略还能相信这批旧 rollout 多久”变成一个约束。

##### 实现代价与 PPO 入口

TRPO 的工程代价来自解约束优化：线性化 surrogate objective，二阶近似 KL，使用 Fisher-vector product 和 conjugate gradient 近似自然梯度方向，再通过 line search 确认 surrogate 变好且 KL 未越界。推导见 [TRPO trust-region 二阶近似](#trpo-trust-region-二阶近似)。

```python
    # TRPO 的核心数据对象：旧策略 rollout、旧 logprob、advantage、KL 约束半径。
    ratio = torch.exp(new_logp - old_logp)
    surrogate = (ratio * advantages).mean()

    # 平均 KL 不是训练 loss 中的普通 penalty，而是本次 step 是否可接受的约束。
    mean_kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()

    # 真实 TRPO 会用 conjugate gradient 求近似自然梯度方向，
    # 再 line search 缩小 step，直到 surrogate 改善且 mean_kl <= target_kl。
    accept_step = (surrogate > old_surrogate) and (mean_kl <= target_kl)
```

PPO 的 clipped surrogate 是 TRPO 思想的一阶工程替代：不再显式求自然梯度和 line search，而是在目标函数里直接限制 ratio 的收益。它牺牲了 TRPO 的显式约束解法，换来普通 SGD/Adam、多 epoch minibatch 和更简单的实现。

这条关系也是理解后续 GRPO/DAPO/CISPO/GSPO 的入口：它们都在问同一个问题，旧策略采样得到的数据，更新几轮以后还能不能继续提供可信梯度；分歧只是“限制 token ratio、sequence ratio、IS 权重，还是直接 mask token”。

<a id="proximal-policy-optimization-ppo"></a>

#### Proximal Policy Optimization (PPO)

##### 目标函数：非 LLM 简化版本

$$
J_{PPO}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)] \\
r_t(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{old}(a|s)} = \exp(\log\pi_{\theta}(a|s) - \log\pi_{old}(a|s))
$$

##### 目标函数：LLM token-level 版本

$$
J_{PPO}(\theta) = \mathbb{E}_{q\sim \mathcal{D},o_i\sim \pi_{old}(\cdot|q)}[\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min(r_{i,t}\hat{A}_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta D_{KL}(\pi_\theta \Vert \pi_{ref})]
$$

其中 $o$ 代表模型生成的一条序列， $o_i$ 代表第 $i$ 条序列， $o_{i,t}$ 代表第 $i$ 条序列的第 $t$ 个 token。先对一条序列的每个 token 计算优势并累加，再除以序列长度进行归一化，这是防止长序列优势累加过大，在优化时过度重视长序列。

##### Token 级对象

Token 级优势和 ratio 分别是：

$$
\hat A_{i,t} \quad\text{and}\quad r_{i,t} = \frac{\pi_\theta(o_{i,t}|q,o_{i,\lt t})} {\pi_{old}(o_{i,t}|q,o_{i,\lt t})}.
$$

$\hat A_{i,t}$ 由奖励模型与批判模型共同计算，衡量第 $t$ 个 token 对最终回报的贡献差异； $r_{i,t}>1$ 表示新策略相对旧策略更倾向于输出该 token，反之则说明新策略正在降低该 token 的概率。

```python
    # old_logp 是 rollout 采样时旧策略给动作/token 的 log probability。
    # new_logp 是当前正在更新的策略重新评估同一批动作/token 得到的 log probability。
    ratio = torch.exp(new_logp - old_logp)

    # advantage 的正负决定 clip 真正限制哪一边：
    # A > 0 时，ratio 过大代表把好动作推得太猛，上界会被截住；
    # A < 0 时，ratio 过小代表把坏动作压得太猛，下界会被截住。
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = ratio * advantages
    clipped_surrogate = clipped_ratio * advantages

    # max objective 写成训练 loss 时要取负号；min 会选择更保守的一项。
    policy_loss = -torch.minimum(surrogate, clipped_surrogate)
    policy_loss = (policy_loss * action_or_token_mask).sum() / action_or_token_mask.sum()
```

##### Clip 与 Token-Masking

动机：将 current policy 中 prob 偏离 old policy 太远的 token mask 掉，确保每次更新不会“迈太大的步子”，从而避免训练崩溃。

为什么是单边裁剪机制？

- 场景一：优势 $A > 0$。当更新方向（即奖励方向）与优势方向相同，且超过 clip 上界时，认为本次更新方向正确但过于激烈， $min$ 选择了 clip 项，其对 $\theta$ 来说是一个常数，等同于直接去掉梯度不参与本次训练更新；当方向相反，且超过 clip 下界时，认为此时更新方向错误，需要大的惩罚修正更新方向， $min$ 使得忽略 clip，使用错误更新方向带来的大的惩罚梯度参与本次训练更新；所以 $A > 0$ 时单边裁剪的对象是 "Higher Area"。
- 场景二：优势 $A \lt 0$。当更新方向（即奖励方向）与优势方向相同，且超过 clip 下界时，认为本次更新方向正确但过于激烈， $min$ 选择了 clip 项，其对 $\theta$ 来说是一个常数，等同于直接去掉梯度不参与本次训练更新；当方向相反，且超过 clip 上界时，认为此时更新方向错误，需要大的惩罚修正更新方向， $min$ 使得忽略 clip，使用错误更新方向带来的大的惩罚梯度参与本次训练更新；所以 $A \lt 0$ 时单边裁剪的对象是 "Lower Area"。

actor objective 为什么要有 $\min$? 在更新方向（即奖励方向）与优势方向不同时，忽略 clip 的限制使用更大的惩罚让目标回到正轨：

1. $A_t > 0, r_t \lt 1 - \epsilon$, $J(\theta) = r_t A_t$, not $(1 - \epsilon)A_t$ by clip
2. $A_t \lt 0, r_t > 1 + \epsilon$, $J(\theta) = r_t A_t$, not $(1 + \epsilon)A_t$ by clip

##### Dual Clip

当更新方向与优势方向相反时，在 $A \lt 0$ 场景下，当 $\pi_{old}$ 很小而 $\pi_\theta$ 很大时，导致 $\frac{\pi_\theta}{\pi_{old}}$ 很大，巨大的权重会主导整个估计，导致方差爆炸，同样会影响训练的稳定性。直接把这部分 token 也 mask 掉不合理，所以具体实现时对这部分规定 loss 上限。

参考：https://zhuanlan.zhihu.com/p/1950988412405417622

##### GAE 与 advantage

GAE 本质上是 one-step TD error 和 MC 之间的平滑插值， $\lambda$ 平衡偏差与方差：

- $\lambda = 0$, $A_t = \sum_{k=0}^{\infty} 0^{k} \delta_{t+k} = \delta_t$, $\delta_t = r_t + V(s_{t+1}) - V(s_t)$ is one-step TD，低方差高偏差
- $\lambda = 1$, $A_t = \sum_{k=0}^{\infty} \gamma^{k} \delta_{t+k} = G_t - v(s_t)$, this can be proved by sum over $\delta_{t}$ expansion，低偏差、高方差
- $0\lt\lambda\lt1$, as $\lambda$ increase, 指数衰减的权重 $(\gamma \lambda)^{k}$ 让远处的 TD Error 贡献逐渐减小

##### 多轮更新与重要性采样

动机：消除采样策略(old policy)和当前策略的分布差异，保证回报期望计算的准确性，以使模型可以在同一份采样数据上进行多轮优化更新，提高采样数据的使用效率。

- on-policy or off-policy：PPO 算法整体属于 on-policy，但更新过程有 off-policy 的成分。
- 采样：on-policy rollout，每个输出都是根据当前 policy 生成的。
- 计算 log_prob：用当前 policy 和 ref policy 计算 log prob，然后计算权重 r。
- 多轮 actor 更新：每一轮更新后 policy 会变，但数据依然是一开始采样的。新策略在学旧策略生成的数据，产生 off-policy 成分，因此需要用重要性采样比值 $r_t(\theta)=\pi_\theta/\pi_{old}$ 做校正，并用 clip 限制校正后方差和策略漂移。

##### 非 LLM PPO 实现流程

for each episode, implement below:

1. collect a rollout $\lbrace s_t, a_t, \pi_\theta(a_t|s_t), s_{t+1}\rbrace$ for a given number of steps,  行为策略 $\pi_\theta$ update after each episode
2. compute Generalized Advantage Estimation (GAE) and return

- TD error $\delta_t = r_t + \gamma v(s_{t+1}) - v(s_t)$
- Advantage $A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^{k} \delta_{t+k}$，等价于递推关系 $A_t = \delta_t + (\gamma \lambda)A_{t+1}$, implemented in reverse order in practice, 需要归一化提高训练稳定性
- return $q(s_t,a_t) = A(s_t,a_t) + v(s_t)$, target for value

3. for each ppo epoch (model update after each epoch)

- 用新策略评估旧数据 evaluate rollout by updated model, 得到 $\pi_\theta(a_t|s_t), v_\theta(s_t), H(\theta)$
- 计算 ppo clip loss
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{old}(a_t|s_t)} = \exp(\log\pi_{\theta}(a_t|s_t) - \log\pi_{old}(a_t|s_t))$
- policy loss $-\min(r_t(\theta)A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)$
- value loss $(v_\theta(s_t) - q(s_t,a_t))^{2}$
- entropy bonus $H(\pi_\theta) = -\mathbb{E}[\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t)]$
- total loss = policy loss + value loss - entropy bonus
- 总结：同一批经验不要只用一次，可以多学几轮；但是每学一轮都要检查新策略相对旧策略变了多少。如果变化还小，就继续学习；如果某些动作的概率被推得太高或压得太低，就把这部分更新截住
- 参考：https://zhuanlan.zhihu.com/p/614115887

<a id="grpo-from-deepseekmath"></a>

#### GRPO (from DeepSeekMath)

##### 目标与对象定义

- 对于模型 $\pi_\theta$ 给定一个问题 $q$ 采样多个回答 $\lbrace o_i \rbrace_{i=1}^{G}$， $G$ 是 group 数量，每个回答有不同的长度 $|o_i|$
- $\pi_\theta(o_{i,t}|q, o_{i,\lt t})$ 是在 $q$ 的采样解答 $o_{i,t}$ 解码的第 $t$ 个词元的策略概率
- KL 约束 $\pi_\theta$ 和 $\pi_{ref}$ 分布差异，使用 k3 估计（无偏 & 方差小）

##### K3 estimator

$$
D_{KL}[\pi_\theta \Vert \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta}(o_{i,t}|q,o_{i,\lt t})} - \log \frac{\pi_{ref}(o_{i,t}|q, o_{i,\lt t})}{\pi_{\theta}(o_{i,t}|q,o_{i,\lt t})} - 1
$$

##### Group relative advantage

相较于 PPO 去掉了价值模型，将 Advantage 定义为相对于组中其他响应的输出奖励。 $\hat{A}_{i,t}$ 是组的相对优势，一条采样回答中的每个 token 的优势都是一样的（token level）。

$$
\hat{A}_{i,t} = \frac{r_i - mean(\lbrace R_k \rbrace_{k=1}^{G})}{std(\lbrace R_k \rbrace_{k=1}^{G})}
$$

$$
\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q,a\sim\mathcal{D},\lbrace o_i \rbrace_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|q)}[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}[\min(r_{i,t}\hat{A}_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}) - \beta D_{KL}(\pi_\theta \Vert \pi_{ref})]]
$$

##### 与 PPO 的差异

差异点：PPO 需要价值网络计算 Advantage；GRPO 去掉了价值模型，将 Advantage 定义为相对于组中其他响应的输出奖励。

共同的问题：PPO/GRPO 损失函数中的不当裁剪操作对性能有负面影响，不能有效促进 CoT 推理行为出现。具体来说，与反思行为相关的 token 通常是推理路径中的“分叉点”，在 Base 模型中出现频率低，且被赋予较低的概率。这些 token 往往具备较高的重要性采样权重 $r$，因此在第一次 on-policy 更新后就会被裁剪掉，无法参与后续的 off-policy 梯度更新。然而，这些低概率 token 往往对于稳定熵以及促进可扩展的强化学习至关重要（之后的 DAPO 尝试通过提高裁剪上界来缓解，CISPO 尝试设计明确避免丢弃 token，对较大更新的 token 通过内在机制将熵控制在合理范围）。

##### Loss 现象分析

训练 GRPO 的 Loss 为什么会有负值并且会上升？

- 观察：当 $\min(,) > KL > 0$ 时，损失为负数，由于 ratio 恒为正，所以要求优势 $\hat{A}_{i,t} > 0$
- 分析：当组 reward 的方差较大时，便容易出现较大的 $\hat{A}_{i,t}$，或者是 ratio 有较大的变化值时，都可能使得 $\min(,) > KL > 0$
- 回答：GRPO损失不保证为正数（如组 reward 的方差较大时）。训练过程优化策略远离ref，因为KL约束会产生较大的损失变化导致大梯度，Loss会上升

##### 实现流程

1. 对每个 prompt 在线采样多个回答，得到一组样本 (batch) : $x_j \rightarrow \lbrace y_{j,1}, \dots, y_{j,G}\rbrace$, where $x_j$ is $j$-th prompt, , $G$ is group size, $y_{j,i}$ is $i$-th response under $j$-th prompt，
2. 计算对应奖励 $r_{j,i} = R(x_j, y_{j,i})$（对于 GSM8K 数据集，每道题都有明确的数值答案，不需要训练任何 RM，直接用规则判断。RLVR 核心思想：可验证奖励）
3. 计算组内均值、标准差和 response-level advantage：

$$
\begin{aligned}
\bar r_j &= \frac{1}{G}\sum_{i=1}^{G}r_{j,i},\\
s_j &= \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_{j,i}-\bar r_j)^{2}},\\
\hat A_{j,i} &= \frac{r_{j,i}-\bar r_j}{s_j+\epsilon}.
\end{aligned}
$$

如果标准差 $s_j\approx 0$，说明这组回答暂时没有可学习的差异，需设置 $\hat A_{j,i}=0$。做组内归一化有效的原因：

- 难度归一化：不同题目的难度不同。简单题所有回答都正确（奖励均值很高），难题大部分回答都错误（奖励均值很低）。如果用绝对奖励，简单题的回答会获得更高的梯度信号，模型会把大部分精力花在简单题上。组内归一化消除了这种偏差，它只关注"这道题内部谁更好"，不受题目绝对难度的影响。
- 相对比较更稳定：人类偏好本质上也是比较式的（"A 比 B 好"），不是绝对的（"A 得 87 分"）。GRPO 的组内比较和人类的判断方式天然一致。
- 方差更低：同一组内的回答共享相同的 prompt，唯一的差异是模型生成的随机性。这种"控制变量"式的比较比跨样本的绝对评分更稳定。

4. GRPO training loss:

$$
\mathcal{L} = -J_{\text{GRPO}}^{\text{clip}}(\theta) +\beta_{KL}\hat D_{KL}
$$

1. policy loss:

$$
J_{\text{GRPO}}^{\text{clip}}(\theta) = \mathbb{E}_{j,i} \left[ \min\left( \rho_{j,i}(\theta)\hat A_{j,i}, \mathrm{clip}(\rho_{j,i}(\theta),1-\epsilon,1+\epsilon)\hat A_{j,i} \right) \right]
$$

- 新旧策略的概率比值:

$$
\rho_{j,i}(\theta) = \frac{\pi_\theta(y_{j,i}|x_j)} {\pi_{old}(y_{j,i}|x_j)} = \exp\left( \log\pi_\theta(y_{j,i}|x_j)-\log\pi_{old}(y_{j,i}|x_j) \right)
$$

- PPO 的裁剪机制 advantages $\hat{A}_{j,i}$ 来自组内比较，替代 Critic (PPO)

2. GRPO uses the k3 estimator for KL divergence:

$$
\hat D_{KL} = \exp(\Delta)-\Delta-1, \quad \Delta=\log\pi_{ref}(y|x)-\log\pi_\theta(y|x).
$$

```python
        # rewards: [num_prompts, group_size]，每行是一道题的多条 sampled responses。
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
        response_adv = (rewards - group_mean) / group_std

        # response_mask: [num_prompts * group_size, max_response_len]，只统计回答 token。
        # GRPO 的 advantage 是 response-level，同一条 response 内所有 token 共享同一个优势。
        adv_tokens = response_adv.reshape(-1, 1).expand_as(response_mask)

        token_ratio = torch.exp(new_token_logp - old_token_logp)
        clipped_ratio = token_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        token_objective = torch.minimum(token_ratio * adv_tokens, clipped_ratio * adv_tokens)

        # GRPO 原始形态通常先对每条 response 的 token 求平均，再对 response/group 求平均。
        per_response_objective = (token_objective * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1)

        # K3 estimator: delta = log pi_ref - log pi_theta, 以 token 方式估计 KL。
        delta = ref_token_logp - new_token_logp
        kl_k3 = torch.exp(delta) - delta - 1.0
        per_response_kl = (kl_k3 * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1)

        grpo_loss = -(per_response_objective - beta_kl * per_response_kl).mean()
```

##### 实际问题

- Response-Level 的 Advantage：对同一提示（prompt）采样多个响应/回答（response），再计算该组响应内的标准化得分，并将其作为该响应中所有 token 的统一 Advantage。也就是说，GRPO 中同一响应内的所有 token 共享相同的 Advantage，不存在区分性
- loss 计算方式是：先在每个样本内按 Token 对损失进行平均，然后在样本之间聚合损失，即每个样本在最终损失计算中被分配相同的权重（每个样本对 Loss 等权），这会导致较长回复中的 Token 可能对总体损失的贡献较低（影响被长样本稀释），所以就可能导致高质量的长样本学习的较差，而低质量的长文本又无法得到惩罚
- token-level 的 importance sampling，即对每个 token 分别乘以其对应的重要性权重以进行分布校正，可能导致方差大
- 参考
- https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo

<a id="dapo-seed-2025"></a>

#### DAPO (Seed, 2025)

##### 目标函数

$$
\mathcal{L}_{DAPO}(\theta) = -\mathbb{E}_{q,a\sim\mathcal{D},\lbrace o_i \rbrace_{i=1}^{G}\sim\pi_{\theta_{old}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min(r_{i,t}\hat{A}_{i,t}, clip(r_{i,t}, 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i,t})
\right]
$$

其中 $r_{i,t}=\frac{\pi_\theta(o_{i,t}|q,o_{i,\lt t})}{\pi_{old}(o_{i,t}|q,o_{i,\lt t})}$。DAPO 论文的核心改动是 Clip-Higher、动态采样、token-level policy gradient loss 和 overlong reward shaping；它移除了显式 KL penalty。

```python
    # DAPO 仍然可以使用 response-level/group-level advantage，
    # 但 loss 聚合改成 token-level：所有有效 token 进入同一个分母。
    token_ratio = torch.exp(new_token_logp - old_token_logp)
    clipped_ratio = token_ratio.clamp(1.0 - clip_eps_low, 1.0 + clip_eps_high)

    # Clip-Higher 常见设置是放宽正向上界，让低概率但高奖励 token 有更大上升空间。
    token_objective = torch.minimum(token_ratio * adv_tokens, clipped_ratio * adv_tokens)

    # response_mask 的 1 表示真实回答 token，0 表示 padding 或 prompt token。
    # 与 GRPO 的 per-response average 不同，这里直接按 token 总数归一化。
    dapo_loss = -(token_objective * response_mask).sum() / response_mask.sum().clamp_min(1)
```

s.t.

$$
0 \lt |\lbrace o_i \mid \text{equivalent}(a, o_i)\rbrace| \lt G
$$

##### 关键改动

DAPO 的优化角度可以拆成四个分支：Clip-Higher、动态采样、token-level policy gradient loss、Overlong Reward Shaping。

##### Clip-Higher

动机：使用 PPO 或 GRPO 训练时观察到的熵坍缩现象，就是 Policy 的熵迅速下降，导致某些组生成的结果几乎相同，限制了探索。分析：原有的上限截断阈值在一定程度上限制了低概率（但有潜力） token 的概率提升，从而可能抑制模型生成的多样性

- 如何理解抑制低概率 token 的增长和抑制模型多样性？以 $\epsilon=0.2$ 为例，低概率 token 的有效变化范围（即不会被 clip 影响）很小，这是因为对于同样的 clip range，高概率 token 的有效变化范围被大的概率放大了。模型更倾向于维持原有的高概率Token（高概率Token更容易保持高概率或继续增加），而不是探索新的Token，导致生成的内容缺乏多样性。
- 方案：仅对正样本，增大 clip 的上限（clip higher = 0.28），而负样本的 clip 区间保持不变，为低概率 token 的概率提升释放了更多空间
- 为什么主要放宽正样本上界：DAPO 想保护“低概率但高奖励”的探索 token，让它们能继续上升；负样本对应的是降低概率，继续放宽下界会加强惩罚和分布漂移，对保持熵与稳定性未必有同样收益。

##### Dynamic Sampling

动机：当某些提示词 Acc 等于1时的梯度消失问题。因为 Acc 为1时，GRPO一组输出的 Reward 都是1，Advantage等于0，此时 Policy 没有优化，因此降低了样本效率。根据经验，精度等于1的样本数量会在训练过程中持续增加

- 方案：使用过采样并过滤掉准确度等于1和0的提示词，目标中的 equivalent 的意思是样本是等效的（过滤掉无效的）。具体做法是：在训练前持续采样，直到批次被准确率既不为 0 也不为 1 的样本完全填充

##### Token-Level Policy Gradient Loss

动机：GRPO loss 计算方式是：先在每个样本内按 Token 对损失进行平均，然后在样本之间聚合损失，即每个样本在最终损失计算中被分配相同的权重（每个样本对 Loss 等权），这会导致较长回复中的 Token 可能对总体损失的贡献较低（影响被长样本稀释），所以就可能导致高质量的长样本学习的较差，而低质量的长文本又无法得到惩罚

- 方案：DAPO loss 计算方式是：直接对所有 token 级别做均值计算，使得较长的序列（相比短的）对整体Loss的影响更大。此外，从单个Token的角度来看，如果某种特定的生成模式能够导致奖励的增加或减少，那么无论该模式出现在多长的回复中，它都会被同等地促进或抑制

##### Overlong Reward Shaping

动机：GRPO 训练中常见的一个工程问题：回答长度失控。模型可能学会"写得越多越好"（因为更长的回答更容易包含正确推理），导致生成 2000+ token 的冗长回答。GRPO 的原始做法是设定最大长度，超过就截断并给惩罚。但截断是硬边界，回答 499 token 没事，501 token 就被惩罚，梯度信号不连续。截断样本的不当奖励塑造（一般提取不到答案，所以Reward为-1）会引入奖励噪声，并严重扰乱训练过程。因为一个合理的推理过程可能仅因长度过长而受到惩罚，使模型对其推理过程的有效性产生困惑

- 方案：DAPO 提出了一种基于长度的惩罚机制——软过长惩罚（Soft Overlong Punishment），用一个平滑的长度惩罚函数替代硬截断，让模型自然地学会控制回答长度。具体来说，当响应长度超过预设的最大值时，定义一个惩罚区间。在区间内，响应越长，受到的惩罚越大。该惩罚将被添加到原始的基于规则的正确性奖励中，从而引导模型避免生成过长的响应。论文中，预期的最大长度设为 16384 个 Token，并额外分配 4096 个 Token 作为软惩罚缓冲区。因此，生成的最大 Token 数被设定为 20480 个。软过长惩罚具体做法：预期长度之内，惩罚为0；缓冲区内，惩罚 $\frac{(L_{max} - L_{cache}) - |y|}{L_{cache}}$ 随长度增加惩罚线形从0加重至-1；超过最大长度，惩罚为-1
- KL散度项移除：在训练长思维链推理模型时，策略模型分布（actor model）可能与初始模型（reference model）显著偏离，即“Base 到 LongCoT 本来变化大”。因此这种限制是没有必要的
- 参考：
- https://zhuanlan.zhihu.com/p/1899235750018541074
- https://yam.gift/2025/08/14/NLP/LLM-Training/2025-08-14-Token-Level-GSPO-GMPO/

<a id="cispo-minimax"></a>

#### CISPO (Minimax)

##### 优化角度

CISPO (Clipped IS-weight Policy Optimization) 不再像 PPO/GRPO 那样通过 `min` 把某些 token update 的梯度变成 0，而是裁剪 importance sampling weight，并对该权重做 stop-gradient。这样所有 token 的 $\log \pi_\theta$ 梯度仍然参与训练，更新强度由 clipped IS weight 控制。

##### 从 IS REINFORCE 到 CISPO 目标

从带 IS 修正的 REINFORCE 看，off-policy minibatch 更新可以写成：

$$
J_{\text{REINFORCE}}(\theta)=
\mathbb{E}_{(q,a)\sim\mathcal{D},o_i\sim\pi_{\theta_{old}}(\cdot|q)}
\left[
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
sg(r_{i,t}(\theta))\hat{A}_{i,t}
\log\pi_\theta(o_{i,t}|q,o_{i,\lt t})
\right],
$$

其中

$$
r_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}|q,o_{i,\lt t})}
{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,\lt t})}.
$$

CISPO 在这个形式上引入 clipped IS weight，并采用 GRPO 的 group relative advantage 与 DAPO 的 token-level loss：

$$
J_{CISPO}(\theta) =
\mathbb{E}_{(q,a)\sim \mathcal{D},\lbrace o_i \rbrace_{i=1}^{G}\sim \pi_{\theta_{old}}(\cdot|q)}
\left[
\frac{1}{\sum_{i=1}^{G}|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
sg(\hat{r}_{i,t}(\theta))\hat{A}_{i,t}
\log \pi_\theta(o_{i,t}|q,o_{i,\lt t})
\right],
$$

$$
\hat{r}_{i,t}(\theta)=clip(r_{i,t}(\theta),1-\epsilon_{low}^{IS},1+\epsilon_{high}^{IS}).
$$

##### 代码实现

```python
    # response_mask 只统计回答 token；adv_tokens 通常来自 group relative advantage。
    token_ratio = torch.exp(new_token_logp - old_token_logp)

    # MiniMax-M1 报告中强调：裁剪 IS 权重，而不是裁剪 token update。
    # detach 对应公式中的 sg(.)，权重只作为数值重加权，不承接梯度。
    clipped_is_weight = token_ratio.clamp(1.0 - eps_low_is, 1.0 + eps_high_is).detach()

    # new_token_logp 始终保留梯度；没有 PPO 的 min，也没有把越界 token 直接丢掉。
    token_objective = clipped_is_weight * adv_tokens * new_token_logp
    cispo_loss = -(token_objective * response_mask).sum() / response_mask.sum().clamp_min(1)
```

##### 和 PPO/GRPO 的关键差异

1. PPO/GRPO 的 `min(unclipped, clipped)` 是 advantage-sign-dependent 的单边保守目标；当 token ratio 在“有利方向”越界时，对应 token 可能不再贡献梯度。
2. CISPO 的 $\hat r$ 被 `detach`，所以 $\nabla_\theta J$ 主要来自 $\nabla_\theta\log\pi_\theta$；越界 token 仍参与训练，只是梯度权重被压到边界。
3. 如果不裁剪 IS weight，CISPO 退化为带 stop-gradient IS 权重的标准 policy gradient；裁剪会引入轻微偏差，但换来低方差和长 response 中更高的 token 利用率。

##### 实际操作问题

- **上界更关键。** MiniMax-M1 报告提到实验中主要调 $\epsilon^{IS}_{high}$，低界可设得较宽；因为他们更关心高 ratio token 的方差和爆炸风险。
- **没有显式 KL penalty。** CISPO 和 DAPO/GSPO 一样，不把 reference KL 当主约束，稳定性更多依赖动态采样、长度惩罚、IS 权重范围和优化器设置。
- **可能保留过多错误方向梯度。** 当某些 token 已经在坏方向上显著偏离旧策略时，CISPO 仍保留其梯度，只是缩小权重；这比 PPO 更高效，但也要求更谨慎地监控 entropy、clip fraction、梯度范数和重复输出。

##### 统一 mask 视角

MiniMax-M1 还给出一个带 token-wise mask 的统一写法，用来表示“何时完全丢掉 token 梯度”。若 $M_{i,t}=1$ 恒成立，就是纯 CISPO；若 $M_{i,t}$ 模拟 PPO 的单边截断，就可以回到更保守的 token update 过滤。

$$
J_{\text{unify}}(\theta)=
\mathbb{E}\left[
\frac{1}{\sum_i |o_i|}
\sum_i\sum_t
sg(\hat r_{i,t})\hat A_{i,t}
\log\pi_\theta(o_{i,t}|q,o_{i,\lt t})M_{i,t}
\right]
$$

$$
M_{i,t}=
\begin{cases}
0,& \hat A_{i,t}>0\ \text{and}\ r_{i,t}(\theta)>1+\epsilon_{high}\\
0,& \hat A_{i,t}\lt0\ \text{and}\ r_{i,t}(\theta)\lt1-\epsilon_{low}\\
1,& \text{otherwise}
\end{cases}
$$

<a id="gspo-qwen-2025"></a>

#### GSPO (Qwen, 2025)

##### 优化角度

GSPO (Group Sequence Policy Optimization) 认为 GRPO 的 token-level importance ratio 用错了修正粒度：奖励是整条 response 的 sequence-level reward，但 GRPO 在每个 token 上分别做 off-policy correction。GSPO 把 ratio、clip、rewarding、optimization 都提升到 sequence level，让修正粒度和奖励粒度对齐。

##### 目标函数

对同一个 prompt $x$ 采样 $G$ 条 response $\lbrace y_i \rbrace_{i=1}^{G}$，先做组内优势：

$$
\hat A_i=
\frac{r(x,y_i)-mean(\lbrace r(x,y_j) \rbrace_{j=1}^{G})}
{std(\lbrace r(x,y_j) \rbrace_{j=1}^{G})}.
$$

sequence-level ratio 使用长度归一化的 sequence likelihood ratio：

$$
s_i(\theta)=\exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\log\frac{\pi_\theta(o_{i,t}|q,o_{i,\lt t})}{\pi_{old}(o_{i,t}|q,o_{i,\lt t})}\right)
$$

$$
J_{GSPO}(\theta)=\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min(s_i(\theta)\hat{A}_i, clip(s_i(\theta),1-\epsilon,1+\epsilon)\hat{A}_i)
\right]
$$

##### 长度归一化与代码实现

为什么要做长度归一化：完整 sequence likelihood 是所有 token 概率的乘积，长度越长越容易产生极端 ratio；取 $1/|o_i|$ 的几何平均后，不同长度 response 可以共享同一数量级的 clip range。

```python
    # token_log_ratio: [batch_responses, max_response_len]
    token_log_ratio = (new_token_logp - old_token_logp) * response_mask
    response_len = response_mask.sum(dim=-1).clamp_min(1)

    # GSPO 把 token log-ratio 先做长度归一化，再得到 sequence-level ratio。
    seq_log_ratio = token_log_ratio.sum(dim=-1) / response_len
    seq_ratio = torch.exp(seq_log_ratio)
    clipped_seq_ratio = seq_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)

    # response_adv 是组内 reward 标准化后的 sequence-level advantage。
    seq_objective = torch.minimum(seq_ratio * response_adv, clipped_seq_ratio * response_adv)

    # clip 判断发生在整条 response，而不是逐 token 判断。
    gspo_loss = -seq_objective.mean()
```

##### 梯度视角

忽略 clip 时，GSPO 梯度近似为：

$$
\nabla_\theta J_{GSPO}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
s_i(\theta)\hat A_i
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\nabla_\theta \log\pi_\theta(y_{i,t}|x,y_{i,\lt t})
\right].
$$

对比 GRPO：

$$
\nabla_\theta J_{GRPO}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\hat A_i
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,\lt t})}
\nabla_\theta \log\pi_\theta(y_{i,t}|x,y_{i,\lt t})
\right].
$$

关键区别：GRPO 每个 token 被自己的 ratio 加权，长 response 中少数极端 token 会持续放大梯度噪声；GSPO 用整条 response 的同一个 $s_i$ 加权所有 token，降低 token-level ratio 方差。

##### GSPO-token

论文还给出 token-level variant，用 sequence-level $s_i$ 的数值控制 clip，但通过 stop-gradient 构造 token-level 可反传项：

$$
s_{i,t}(\theta)=sg[s_i(\theta)]\cdot
\frac{\pi_\theta(y_{i,t}|x,y_{i,\lt t})}
{sg[\pi_\theta(y_{i,t}|x,y_{i,\lt t})]}.
$$

数值上 $s_{i,t}=s_i$，但梯度来自当前 token 的 logprob；当一条 response 内所有 token 共享同一个 $\hat A_i$ 时，GSPO-token 与 GSPO 在目标、clip 条件和理论梯度上等价。它的价值在于：如果未来有 step-level 或 token-level advantage，可以在不回到 GRPO token ratio 的情况下加入细粒度 credit assignment。

```python
    # seq_ratio.detach() 提供 sequence-level 数值权重；
    # exp(new_token_logp - new_token_logp.detach()) 数值为 1，但梯度等价于 new_token_logp。
    token_factor = torch.exp(new_token_logp - new_token_logp.detach())
    gspo_token_ratio = seq_ratio.detach()[:, None] * token_factor
    gspo_token_ratio = gspo_token_ratio * response_mask

    token_adv = response_adv[:, None].expand_as(response_mask)
    token_obj = torch.minimum(
        gspo_token_ratio * token_adv,
        gspo_token_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * token_adv,
    )
    gspo_token_loss = -(token_obj * response_mask).sum() / response_mask.sum().clamp_min(1)
```

##### 实际操作问题

- **MoE routing mismatch。** GSPO 论文指出 token-level ratio 对 MoE expert routing 变化很敏感；同一 response 在更新后可能激活不同 expert，导致 token ratio 波动。GSPO 只依赖 sequence likelihood，对单个 token likelihood 的敏感度更低，因此可以减少对 routing replay 的依赖。
- **推理/训练精度差异。** 如果 rollout 由 vLLM/SGLang 等推理引擎生成，训练引擎重新算 token logprob 可能和推理时不完全一致。GSPO 使用 sequence-level likelihood，对这种细粒度差异更宽容。
- **整段裁剪的取舍。** response-level clip 会把一整条 response 作为 off-policy 判断单位；这降低了 token ratio 噪声，但也意味着少数异常 token 可能影响整段样本是否被裁剪。实际监控应同时看 sequence clip fraction、长度分布、reward 分布和 entropy。

<a id="reward-model-bradley-terry-偏好建模"></a>

#### Reward Model：Bradley-Terry 偏好建模

##### 目标和公式

目的：预测两个竞争者结果的概率模型，常用于处理成对比较的数据。 $p_i$ 是正实数分数，可以写成指数分数函数 $p_i=e^{\beta_i}$。

$$
\begin{aligned}
p(i>j) &= \frac{p_i}{p_i + p_j} \\
&= \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} \\
&= \frac{1}{1 + e^{-(\beta_i - \beta_j)}} \\
&= \sigma(\beta_i - \beta_j)
\end{aligned}
$$

##### MLE 目标

$$
\mathrm{argmin}_{\beta} \sum_{i,j} -\log\sigma(\beta_i-\beta_j)
$$

该目标鼓励 $\beta_i\gg\beta_j$。

##### 训练数据和 BT likelihood

给定 prompt $x$ 根据人类偏好标注得到回答 $y_1 \succ y_2$，构建偏好数据集 $\mathcal{D} = \lbrace x^{(i)}, y_w^{(i)}, y_l^{(i)}\rbrace_{i=1}^{N}$，reward 模型需要预测出分数 $r^{\ast}(y, x)$。通过 BT 模型建模人类偏好分布：

$$
p^{\ast}(y_1 \succ y_2|x) = \frac{e^{r^{\ast}(x, y_1)}}{e^{r^{\ast}(x, y_1)} + e^{r^{\ast}(x, y_2)}} = \frac{1}{1 + e^{-(r^{\ast}(x, y_1) - r^{\ast}(x, y_2))}} = \sigma(r^{\ast}(x, y_1) - r^{\ast}(x, y_2))
$$

MLE loss:

$$
\mathcal{L}(r_\phi, \mathcal{D} ) = -\mathbb{E}_{x, y_w, y_l\sim \mathcal{D}}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

```python
        # reward_model 对 prompt+response 输出一个 sequence-level scalar reward。
        chosen_reward = reward_model(prompt_ids, chosen_response_ids)
        rejected_reward = reward_model(prompt_ids, rejected_response_ids)

        # Bradley-Terry: 只关心分数差，不关心两个 reward 的绝对平移。
        reward_margin = chosen_reward - rejected_reward
        reward_model_loss = -torch.nn.functional.logsigmoid(reward_margin).mean()
```

##### 奖励颗粒度

- sequence-level: 整个回答一个分数 (PPO, GRPO)
- token-level: 每个 token 独立分数，标注成本高，信用分配（解耦）难
- step-level: 按推理步骤分段，需要步骤分割器

##### 实际操作注意

- 数据切分：训练集和验证集共享同一个 prompt，甚至共享部分回答。这样得到的 eval accuracy 会偏乐观，更稳妥的做法是按 prompt 切分
- RM 分数尺度：RM 训练只关心分数差，不关心绝对尺度，但 PPO 阶段感受到的奖励尺度完全不同，PPO之前需要校准，常见做法是在固定校准集上做标准化
- reward hacking：PPO 的 actor 会主动搜索让 RM 给高分的输出分布。如果 RM 偏爱某种表面模式，Actor 会把这种模式推到极端，例如：偏爱长回答。Actor 可能学到：越写越长，尽管奖励不断上升，但是真实的信息密度下降。常见做法是在 PPO 前先做对抗性测试，如果“超长废话”比“正确但简短”分数高，先别跑 PPO

<a id="reinforcement-learning-from-human-feedback-rlhf"></a>

#### Reinforcement Learning from Human Feedback (RLHF)

##### SFT 目标

$$
\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y)\sim\mathcal{D}}[\log\pi_\theta(y|x)] \approx -\sum_{t=1}^{T} \log\pi_\theta(y_t|x,y_{\lt t})
$$

##### RLHF 目标

$$
\mathcal{J}_{RLHF} = \mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(\cdot|x)}[r_\phi(x,y)] - \beta D_{KL}(\pi_\theta(\cdot|x) \Vert \pi_{ref}(\cdot|x))
$$

解释：让当前模型自己生成回答，用 RM 打分，再用 PPO 提高高分回答的概率，追求偏好奖励，同时别偏离 SFT 太远，用裁剪、优势估计和 KL 约束（k2 estimation）来稳定更新。

##### LLM-RLHF 的 token-level reward

Reward Model 往往只在完整回答 $y$ 结束后给一个分数 $r(x,y)$，为了让 PPO 能在 token 级别更新，工程上会把奖励拆成两部分：1. 每个 token 都有 KL 惩罚，防止偏离 reference; 2. 最后一个 token 或 EOS 位置加上 RM 的整段奖励。

$$
\begin{aligned}
r_t^{token} &= -\beta(\log\pi_\theta(y_t|s_t) - \log\pi_{ref}(y_t|s_t)), \quad t \lt T \\
&= r_\phi(x,y) - \beta(\log\pi_\theta(y_t|s_t) - \log\pi_{ref}(y_t|s_t)), \quad t = T
\end{aligned}
$$

##### 梯度颗粒度

- token-level: 每个 token 有独立的梯度信号，虽然奖励仍然集中在末尾，但通过 Critic 估计每个位置的 value，计算出每个 token 独立的 advantage
- sequence-level: 整条回答的所有 token 共用一个梯度信号，但其中真正决定质量的往往只有几个。如果所有 token 被平均更新，梯度信号就被大量无关 token 稀释了，模型需要把学习精力集中在真正重要的 token 上

<a id="direct-preference-optimization-dpo"></a>

#### Direct Preference Optimization (DPO)

##### 目标函数

$$
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{x,y_w,y_l\sim \mathcal{D}}[\log \sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]
$$

```python
    # chosen/rejected 都是整条回答的 log probability，
    # 通常由 token logprob 乘 response mask 后求和得到。
    chosen_logp = (chosen_token_logp * chosen_mask).sum(dim=-1)
    rejected_logp = (rejected_token_logp * rejected_mask).sum(dim=-1)
    ref_chosen_logp = (ref_chosen_token_logp * chosen_mask).sum(dim=-1)
    ref_rejected_logp = (ref_rejected_token_logp * rejected_mask).sum(dim=-1)

    chosen_logratio = chosen_logp - ref_chosen_logp
    rejected_logratio = rejected_logp - ref_rejected_logp

    # DPO 只看 chosen 相对 rejected 的 log-ratio 差值，不需要在线 rollout 或显式 RM。
    logits = beta * (chosen_logratio - rejected_logratio)
    dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()
```

##### 目标推导

从 RLHF 的目标开始：

$$
\begin{aligned}
\max_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(\cdot|x)}&[r_\phi(x,y)] - \beta D_{KL}(\pi_\theta(\cdot|x) \Vert \pi_{ref}(\cdot|x)) \\
&= \max_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(y|x)}[r(x,y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}] \\
&= \min_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta}r(x,y)] \\
&= \min_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \log\frac{1}{Z(x)} + \log\frac{1}{Z(x)} - \log e^{ \frac{1}{\beta}r(x,y)} + \log e^{ \frac{1}{\beta}r(x,y)} - \frac{1}{\beta}r(x,y)] \\
&= \min_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\frac{1}{Z(x)}\pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}} + \log\frac{1}{Z(x)}] \\
&= \min_\theta\mathbb{E}_{x\sim \mathcal{D}, y\sim\pi_\theta(y|x)}[D_{KL}(\pi(y|x) \Vert \pi^{\ast}(y|x)) - \log Z(x)]
\end{aligned}
$$

##### 推导动机

找到一个配分函数 $Z(x) = \sum_y \pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}$ 作为一个有效的概率分布的归一化系数：

$$
\pi^{\ast}(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x)e^{\frac{1}{\beta}r(x,y)}
$$

这样使得目标的第一项 $\log$ 的分母是一个有效的概率分布，将优化目标表达成 KL 散度。奖励函数 RM 可以用策略概率的比值来表示，不需要额外的 RM，策略模型自己就蕴含了奖励信号。推导结果：最小化 KL 项，得到求解策略 $\pi_\theta(y|x) = \pi^{\ast}(y|x)$。

替代目标中的奖励函数 $r(x,y) = \beta\log\frac{\pi^{\ast}(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$ 带入 BT model：

$$
\begin{aligned}
p^{\ast}(y_1 \succ y_2|x) &= \frac{e^{r^{\ast}(x, y_1)}}{e^{r^{\ast}(x, y_1)} + e^{r^{\ast}(x, y_2)}} = \sigma(r^{\ast}(x, y_1) - r^{\ast}(x, y_2)) \\
&= \sigma(\beta \log\frac{\pi^{\ast}(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log\frac{\pi^{\ast}(y_2|x)}{\pi_{ref}(y_2|x)})
\end{aligned}
$$

##### 为什么标准 RLHF(PPO) 更鲁棒？

- DPO的训练目标会导致过拟合，因为 rejected token 策略 $\pi_\theta(y_l|x)$ 会快速收敛到 0，导致 DPO sigmoid 概率不断接近 1，损失一直降低，但是没有对齐偏好。解决方案可以是 IPO，它把偏好概率拟合到一个固定 margin：

$$
\log\frac{\pi^{\ast}(y_1|x)}{\pi_{ref}(y_1|x)} - \log\frac{\pi^{\ast}(y_2|x)}{\pi_{ref}(y_2|x)} \rightarrow \frac{\tau^{-1}}{2}
$$

- 在DPO的推导中，最优策略是基于BT-Model形式下能得到最大的reward，在非DPO的优化中，存在其他的策略能够使得DPO Loss更低
- 参考：https://zhuanlan.zhihu.com/p/692991235

# Training-Inference Mismatch

训练-推理不一致指训练阶段优化的分布、状态、奖励或系统实现，和最终推理/评测时模型真实面对的条件不一致。它不是一个单独 bug，而是一组会互相放大的错位。

## 发生场景

| 场景                          | 训练时看到什么                              | 推理/评测时面对什么                                | 典型后果                                                         |
| ----------------------------- | ------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| SFT teacher forcing           | 每一步条件是 gold prefix $y^{\ast}_{\lt t}$ | 每一步条件是模型自己生成的 prefix $\hat y_{\lt t}$ | 早期错误改变后续状态分布，形成 exposure bias                     |
| Offline preference / DPO      | 固定偏好对来自人工数据或旧策略              | 当前策略会生成新的错误类型                         | 拒绝样本概率被压得很低但真实偏好未必提升，容易过拟合偏好数据     |
| Online RLHF / RLVR            | 当前策略 rollout + RM/verifier 打分         | 部署时面对更宽 prompt 分布和不同采样设置           | 能发现当前策略错误，但也可能 reward hacking 或 verifier hacking  |
| Token loss vs sequence reward | loss 按 token 聚合                          | 奖励通常只评价整段 response 或最终答案             | credit assignment 粗糙，关键 token 被平均稀释或被 ratio 噪声放大 |
| 训练长度 vs 推理长度          | 固定 max length、截断或长度惩罚             | 用户可能给更长上下文、要求更长 CoT                 | 过长推理、重复循环、答案抽取失败、成本失控                       |
| 推理引擎 vs 训练引擎          | rollout 可能由 vLLM/SGLang 生成             | 训练端重算 logprob、MoE routing 或数值精度不同     | token ratio 不稳定，clip fraction 异常，MoE 训练可能崩           |

SFT 的条件分布错位可以直接写成：

$$
\mathcal{L}_{SFT} =-\sum_t\log\pi_\theta(y_t^{\ast}|x,y_{\lt t}^{\ast}), \quad \text{inference: } \hat y_t\sim\pi_\theta(\cdot|x,\hat y_{\lt t}).
$$

训练目标从未让模型在 $\hat y_{\lt t}$ 这个“自己犯错后的状态”上恢复；一旦早期 token 偏了，后面的状态分布就不是训练分布。这也是为什么只看 teacher-forced token accuracy 不能代表真实生成质量。

## 问题分析

- **状态分布错位会递归放大。** 在 RL 记号里，策略改变会改变状态访问分布 $d^{\pi}(s)$；语言模型里状态就是 prefix。SFT/DPO 如果只在静态数据上训练，就很难覆盖当前策略生成的新 prefix。
- **奖励粒度和梯度粒度不一致。** 整段 reward 只告诉模型“这条 response 好不好”，但 token-level update 要决定“哪些 token 该升/降”。PPO 用 critic/GAE 拆 token advantage，GRPO 让 response 内 token 共享 advantage，DAPO 改 token-level reduction，GSPO 改 sequence-level ratio，本质都在修这个粒度错位。
- **评测协议也是目标的一部分。** 温度、top-p、max tokens、答案抽取规则、verifier 宽严度都会改变最终分数。如果训练时 reward extractor 和评测时 answer parser 不一致，模型可能优化到一个评测不认的输出格式。
- **工程实现会制造隐形 off-policy。** 用推理引擎生成 rollout，再用训练引擎重算 logprob；或者 MoE 模型更新后 routing 改变，都会让“同一个 token 的新旧概率比”不再只反映策略分布变化。GSPO 对 sequence likelihood 的依赖较低敏感度，就是在缓解这一类错位。
- **越强的在线优化越需要防 reward hacking。** RLHF/RLVR 能让当前策略暴露真实错误，但策略也会主动搜索 reward/verifier 的漏洞。长回答偏好、格式投机、重复推理和“最终答案抽取”漏洞都属于这个范畴。

## 改善方向

- **SFT 阶段：让训练格式贴近推理格式。** 保持 chat template、system prompt、工具调用格式、EOS/stop token 和推理时一致；必要时加入模型生成前缀上的纠错数据或 DAgger/scheduled-sampling 风格的数据聚合。
- **偏好学习阶段：区分 offline alignment 和 online correction。** DPO 适合利用高质量固定偏好对，但要监控 rejected logprob 过快塌缩；PPO/GRPO/RLVR 适合暴露当前策略错误，但必须加入 KL/clip、长度约束、verifier 对抗测试和 prompt-level dynamic sampling。
- **RL 阶段：让 reward、advantage、ratio 的粒度一致。** 若奖励是 sequence-level，GRPO/GSPO 的 response-level advantage 更自然；若有 step-level verifier 或过程奖励，可以考虑 token/step advantage，但要避免 token-level ratio 方差主导训练。
- **长度控制：不要只靠硬截断。** 硬截断会把接近边界的合理推理变成噪声样本。DAPO 的 soft overlong punishment 是更平滑的做法；实际训练还应监控 response length 分位数、重复 n-gram、EOS 率和答案抽取失败率。
- **评测阶段：固定协议并复用 verifier。** 同一模型在不同采样温度、max tokens、答案抽取规则下可能得到完全不同结论。训练日志和最终报告必须记录这些配置。

```python
# 评测协议应该被当成实验对象保存，而不是散落在命令行参数里。
eval_protocol = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 32768,
    "stop": ["</answer>"],
    "answer_extractor": "same_as_training_verifier",
}

for prompt in eval_prompts:
    response = policy.generate(prompt, **eval_protocol)
    answer = extract_answer(response, mode=eval_protocol["answer_extractor"])
    score = verifier(prompt, answer)

    log_eval_case(
        prompt_id=prompt.id,
        response_len=count_response_tokens(response),
        extracted_answer=answer,
        verifier_score=score,
    )
```

一个实用判断：如果训练 loss 变好但评测分数不动，先不要急着调学习率。优先检查这四件事：rollout prompt 分布是否覆盖评测分布；reward/verifier 是否和评测 extractor 一致；长度分布是否漂移；新旧 logprob/ratio 是否受推理-训练引擎差异污染。

# Mathematical tools behind algorithms

### Bellman expectation equation 推导

状态价值从定义出发：

$$
v_\pi(s)=\mathbb{E}[G_t|S_t=s] =\mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_t=s].
$$

即时奖励项：

$$
\begin{aligned}
\mathbb{E}[R_{t+1}|S_t=s]
&= \sum_r r p(r|s)\\
&= \sum_r r\sum_a p(r|s,a)p(a|s)\\
&= \sum_a \pi(a|s)\sum_r p(r|s,a)r.
\end{aligned}
$$

未来回报项用全期望公式和 Markov property 展开：

$$
\begin{aligned}
\mathbb{E}[G_{t+1}|S_t=s]
&= \sum_{s'}\mathbb{E}[G_{t+1}|S_t=s,S_{t+1}=s']p(s'|s)\\
&= \sum_{s'}\mathbb{E}[G_{t+1}|S_{t+1}=s']p(s'|s)\\
&= \sum_{s'}v_\pi(s')\sum_a\pi(a|s)p(s'|s,a).
\end{aligned}
$$

代回得到 element-wise Bellman expectation equation：

$$
v_\pi(s)=
\sum_a\pi(a|s)
\left[
\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')
\right].
$$

若定义

$$
r_\pi(s)=\sum_a\pi(a|s)\sum_r p(r|s,a)r,
\quad
p_\pi(s'|s)=\sum_a\pi(a|s)p(s'|s,a),
$$

则有矩阵形式：

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi.
$$

当 $\gamma\lt1$ 且 $P_\pi$ 为随机矩阵时， $I-\gamma P_\pi$ 可逆，因此

$$
v_\pi=(I-\gamma P_\pi)^{-1}r_\pi.
$$

### Bellman optimality contraction

Bellman optimality operator 定义为

$$
(Tv)(s)=\max_a\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)v(s')\right].
$$

对任意两个 value function $u,v$：

$$
\begin{aligned}
|(Tu)(s)-(Tv)(s)|
&\le
\max_a
\left|
\gamma\sum_{s'}p(s'|s,a)(u(s')-v(s'))
\right|\\
&\le
\gamma\max_a\sum_{s'}p(s'|s,a)\|u-v\|_\infty\\
&=
\gamma\|u-v\|_\infty.
\end{aligned}
$$

因此

$$
\|Tu-Tv\|_\infty\le\gamma\|u-v\|_\infty.
$$

当 $\gamma\lt1$ 时， $T$ 是 contraction mapping，所以存在唯一不动点：

$$
v^{\ast}
$$

value iteration $v_{k+1}=Tv_k$ 会收敛到这个不动点。

### 策略梯度定理证明

最常用的证明从整条轨迹 $\tau=(s_0,a_0,r_1,\dots,s_T)$ 的似然比开始。有限时域目标写作

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)].
$$

对参数求导：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \int p_\theta(\tau)R(\tau)d\tau \\
&= \int \nabla_\theta p_\theta(\tau)R(\tau)d\tau \\
&= \int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)R(\tau)d\tau \\
&= \mathbb{E}_{\tau\sim p_\theta(\tau)}
\left[\nabla_\theta \log p_\theta(\tau)R(\tau)\right].
\end{aligned}
$$

轨迹概率分解为

$$
p_\theta(\tau)=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)p(s_{t+1},r_{t+1}|s_t,a_t).
$$

环境转移和奖励分布不含 $\theta$，所以

$$
\nabla_\theta\log p_\theta(\tau) =\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t|s_t).
$$

得到 trajectory 形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t|s_t)R(\tau) \right].
$$

时刻 $t$ 的动作不影响过去奖励，所以可把整段回报换成 reward-to-go：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t|s_t)G_t \right], \quad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}.
$$

再把 reward-to-go 换成条件期望，得到 action-value 形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_t,a_t} \left[ \nabla_\theta\log\pi_\theta(a_t|s_t)Q^{\pi}(s_t,a_t) \right].
$$

baseline 不改变期望。对任意只依赖状态的 $b(s)$：

$$
\begin{aligned}
\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}
\left[\nabla_\theta\log\pi_\theta(a|s)b(s)\right]
&=
b(s)\sum_a\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)\\
&=
b(s)\sum_a\nabla_\theta\pi_\theta(a|s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

因此可把 $Q^{\pi}(s,a)$ 换成 advantage：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_t,a_t} \left[ \nabla_\theta\log\pi_\theta(a_t|s_t)A^{\pi}(s_t,a_t) \right].
$$

无限时域折扣设定常写成 discounted occupancy form：

$$
\nabla_\theta J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s\sim d_\gamma^{\pi},a\sim\pi_\theta(\cdot|s)} \left[ \nabla_\theta\log\pi_\theta(a|s)Q^{\pi}(s,a) \right],
$$

其中

$$
d_\gamma^{\pi}(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}P(S_t=s|\pi).
$$

trajectory 形式更适合解释 REINFORCE/PPO 代码；discounted occupancy 形式更适合和 Bellman、actor-critic 的状态分布对齐。不同资料省略 $\frac{1}{1-\gamma}$ 时，通常是把常数吸收到学习率或采用未归一化 occupancy measure。

### TRPO trust-region 二阶近似

TRPO 的约束优化写成：

$$
\max_\theta L_{\theta_{old}}(\theta)
\quad
\text{s.t.}
\quad
\bar D_{KL}(\theta_{old},\theta)\le\delta.
$$

在 $\theta_{old}$ 附近对目标做一阶近似、对 KL 做二阶近似：

$$
L_{\theta_{old}}(\theta)\approx
L_{\theta_{old}}(\theta_{old})+g^{T}(\theta-\theta_{old}),
$$

$$
\bar D_{KL}(\theta_{old},\theta)\approx
\frac{1}{2}(\theta-\theta_{old})^{T}A(\theta-\theta_{old}),
$$

其中 $A$ 是 KL Hessian，也就是 Fisher information matrix 的经验近似；一阶项为：

$$
g=
\left.
\nabla_\theta L_{\theta_{old}}(\theta)
\right|_{\theta=\theta_{old}}
$$

于是局部问题变成：

$$
\max_x g^{T}x
\quad
\text{s.t.}
\quad
\frac{1}{2}x^{T}Ax\le\delta.
$$

其方向与自然梯度一致：

$$
s\approx A^{-1}g.
$$

大模型里不会显式形成 $A^{-1}$，而是用 conjugate gradient 通过 Fisher-vector product 近似求解 $As=g$。步长由 KL 约束给出：

$$
\beta=\sqrt{\frac{2\delta}{s^{T}As}}.
$$

最后 TRPO 做 line search：从 $\theta_{old}+\beta s$ 开始逐步缩小步长，直到 surrogate objective 改善且真实平均 KL 不超过 $\delta$。没有这个 line search，二阶近似误差可能让一次更新跨出 trust region，造成性能崩塌。

- Bradley-Terry (BT) model (useful in DPO)
- 目的：model preference 人类偏好模型。它是一种用于预测两个竞争者（如个人或团队）结果的概率模型，常用于处理成对比较的数据，通常用来估计和比较个体或项目的相对能力

$$
Pr(i>j) = \frac{p_i}{p_i + p_j} = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \frac{1}{1 + e^{-(\beta_i - \beta_j)}} = \sigma(\beta_i - \beta_j),
$$

$p_i$ 是 $i$ 的正实数分数， $p_i = e^{\beta_i}$ 是对应的指数分数函数

- 核心：loss function $y(x) = -\log \sigma(x) = -\log \sigma(\beta_i - \beta_j)$
- if $\beta_i - \beta_j \rightarrow +\infty$, then $y \rightarrow 0$，即BT模型 $\beta_i$ 与 $\beta_j$ 分数差值越大，loss 越小
- if $\beta_i - \beta_j \rightarrow -\infty$, then $y \rightarrow +\infty$

- KL divergence 的非对称性
- 假定已知随机变量 $p$，求相对简单的随机变量 $q$，使得 $q$ 尽量接近 $p$
- 正向KL散度
- 权重 $p(x)$
- 在 $p(x)$ 大的地方，想让 KL 散度小，就需要 $q(x)$ 的值也尽量大；在 $p(x)$ 小的地方， $q(x)$ 对整体影响不大
- 要想使正向 KL 散度最小，则要求在 $p(x)$ 不为 0 的地方， $q(x)$ 也尽量不为 0，所以正向 KL 散度被称为是 zero avoiding
- 得到的分布 $q(x)$ 是一个比较 “宽” 的分布

$$
q^{\ast} = \mathrm{argmin}_q D_{KL}(p \Vert q) = \mathrm{argmin}_q p(x) \log\frac{p(x)}{q(x)}
$$

- 反向KL散度 (used in RLHF)
- 权重 $q(x)$
- 在 $p(x)$ 小的地方，想让 KL 散度小，就需要 $q(x)$ 的值也尽量小；在 $p(x)$ 大的地方，可以适当忽略
- 要想使反向 KL 散度最小，则要求在 $p(x)$ 为 0 的地方， $q(x)$ 也尽量为 0，所以反向 KL 散度被称为是 zero forcing
- 得到的分布 $q(x)$ 是一个比较 “窄” 的分布

$$
q^{\ast} = \mathrm{argmin}_q D_{KL}(q \Vert p) = \mathrm{argmin}_q q(x) \log\frac{q(x)}{p(x)}
$$

- 例子：令 $p$ 是两个高斯分布的混合，令 $q$ 为单个高斯分布。那么两种 KL 散度如何选择呢？
- 正向：更在意真实分布 $p$ 中的常见事件，也就是两峰，我们要优先确保它们在分布 $q$ 中不是特别罕见（信息长度不是特别长）。当 $p$ 具有多个峰时， $q$ 选择将这些峰模糊到一起，以便将高概率质量放到所有峰上
- 反向：更在意真实分布 $p$ 中的罕见事件，也就是谷底，我们要优先确保它们在分布 $q$ 中不是特别常见（信息长度特别长的那些事件）。当 $p$ 具有多个峰并且这些峰间隔很宽时，最小化 KL 散度会选择单个峰，以避免将概率质量放置在 $p$ 的多个峰之间的低概率区域中

- KL divergence estimation
- 考虑反向 KL 散度：

$$
D_{KL}(q \Vert p) = \sum_x q(x)\log\frac{q(x)}{p(x)} = \mathbb{E}_{x\sim q} \left[ \log\frac{q(x)}{p(x)} \right]
$$

- 目标

1. 最好是无偏的，即估计值的期望与真实值相等
2. 方差尽量小

- 估计（定义 $\delta(x) = \frac{p(x)}{q(x)}$）

1. $k_1 = -\log \delta(x)$

- 直接按照定义，无偏估计
- 一半的估计是负数，但 KL 是非负数，导致方差较大

2. $k_2 = \frac{1}{2}(\log \delta(x))^{2}$

- 考虑 $f$-divergence：

$$
D_f(p,q) = \mathbb{E}_{x\sim q} \left[ f\left(\frac{p(x)}{q(x)}\right) \right]
$$

When $f$ is differentiable and $q$ is close to $p$, f-divergences look like KL divergence up to second order. For a parametrized distribution $p_\theta$:

$$
D_f(p_0,p_\theta) = \frac{f''(1)}{2}\theta^{T}F\theta+O(\theta^{3}),
$$

where $F$ is the Fisher information matrix for $p_\theta$ evaluated at $p_\theta=p_0$.

- KL 散度对应 $f(x) = -\log(x)$， $k_2$ 对应 $f(x) = \frac{1}{2}(\log x)^{2}$
- 有偏（小偏差）的二阶展开为：

$$
f\left(\frac{q(x)}{p(x)}\right) = f(1+\epsilon) \approx f(1)+f'(1)\epsilon+\frac{1}{2}f''(1)\epsilon^{2}
$$

$k_1,k_2$ 的 $f''(1)=1$。

- 低方差（ $k_2$ 恒为正）

3. $k_3 = \delta(x) - 1 - \log \delta(x) = \delta(x) - 1 + k_1$

- 无偏：找到一个项，期望为 0 的，保证 $k_3$ 无偏

$$
\begin{aligned}
\mathbb{E}_q[\delta(x) - 1] &= \mathbb{E}_q[\frac{p(x)}{q(x)} - 1] \\
&= \int q(x) (\frac{p(x)}{q(x)} - 1) dx \\
&= \int p(x) dx - \int q(x) dx \\
&= 0
\end{aligned}
$$

所以对任意的 $\lambda$， $k_1 + \lambda (\delta(x) - 1)$ 都是无偏估计，选择一个 $\lambda$ 使得方差最小（最简单 $\lambda = 1$）

- 低方差：找到一个项，与 $k_1$ 负相关，拉低方差

$$
\delta(x) - 1 \propto \log\delta(x) = -k_1
$$

- 真实分布对应的参考分布 $\pi_{ref}$，非真实分布对应当前策略 $\pi_\theta$
- 参考
- http://joschu.net/blog/kl-approx.html
- https://zhuanlan.zhihu.com/p/1893782254100115959

- Statistics related
- Chain rule of conditional probability

$$
\begin{aligned}
p(x|a) &= \sum_b p(x, b|a) \quad \text{law of total (cond) prob}\\
&= \sum_b p(x|b,a)p(b|a) \quad \text{Def of cond prob}
\end{aligned}
$$

- Law of total expectation

$$
\begin{aligned}
\mathbb{E}_a[X|A=a]p(a) &= \sum_a \left[\sum_x p(x|a) x\right] p(a) \quad \text{Def of cond Exp}\\
&= \sum_x\sum_a p(x|a) p(a) x \\
&= \sum_x p(x) x \quad \text{law of total prob} \\
&= E[X]
\end{aligned}
$$

- Law of total (conditional) expectation

$$
\begin{aligned}
\mathbb{E}[X|A=a] &= \sum_x x p(x|a) \\
&= \sum_x x\left[\sum_b p(x|b,a)p(b|a)\right] \quad \text{chain rule of cond prob} \\
&= \sum_b \left[\sum_x xp(x|b,a)\right] p(b|a) \\
&= \sum_b \mathbb{E}[X|A=a,B=b] p(b|a) \quad \text{Def of cond Exp}
\end{aligned}
$$
