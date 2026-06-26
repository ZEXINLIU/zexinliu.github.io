---
layout: page
title: "SafeRunRL: PPO for Target-Heart-Rate Treadmill Control"
description: Research prototype for target-heart-rate treadmill control using wearable PPG/ACC states, a learned heart-rate response environment, and PPO.
img:
importance: 3
category: reinforcement-learning
---

SafeRunRL is a research-oriented reinforcement learning prototype for adaptive treadmill control from wearable physiological signals. The first scenario is deliberately focused: keep a runner's heart rate inside a target zone by adjusting treadmill speed and incline every few seconds.

The project is designed around data that can be collected in a laboratory treadmill setting. A wearable device records heart rate, PPG signal quality, and acceleration while the treadmill logs speed and incline. These open-loop sessions are used to fit a simple heart-rate response environment, so the reinforcement learning policy can be trained offline before any real-time treadmill interaction is considered.

The core MDP formulation is intentionally compact:

- **State.** Current heart rate, heart-rate trend, PPG signal quality, acceleration intensity, treadmill speed, treadmill incline, and the target heart-rate bounds.
- **Action.** A discrete PPO action that keeps the current setting or makes a small speed/incline adjustment.
- **Environment.** A learned or fitted heart-rate response model that maps the current physiological state and treadmill action to the next state.
- **Reward.** Positive reward for staying inside the target heart-rate zone, with penalties for overshoot, low-quality signal decisions, and unnecessary action changes.
- **Policy.** A lightweight actor-critic network trained with rollout collection, GAE advantage estimation, clipped PPO objective, and value-function regression.

This project is not presented as a deployed medical safety product. Its purpose is to demonstrate practical RL modeling judgment: how to define the environment, state, action space, reward, simulator, PPO update, evaluation metrics, and deployment boundary for a wearable-health control problem.
