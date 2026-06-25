---
layout: page
title: "SafeRunRL: Safe Reinforcement Learning for Adaptive Treadmill Control"
description: Simulation-based safe RL controller using wearable physiological signals, uncertainty-aware state estimation, and action shielding.
img:
importance: 3
category: reinforcement-learning
---

SafeRunRL is a research-oriented engineering prototype for adaptive treadmill control from wearable physiological signals. The goal is not to claim a deployed medical safety system, but to build a realistic safe-RL project that connects wearable sensing, uncertainty-aware state estimation, human-in-the-loop control, and constrained sequential decision making.

The system estimates a user's exercise state from PPG, ACC/IMU, heart-rate trend, signal quality, activity intensity, and user profile, then adjusts treadmill speed and incline under safety constraints. The policy is trained and evaluated in a personalized cardiovascular simulator before any hardware-level control is considered.

Core algorithmic themes:

- **Wearable state estimation.** Convert noisy PPG/ACC streams into heart-rate trend, recovery proxy, signal-quality index, OOD score, motion intensity, and user-state features.
- **Human digital twin.** Simulate individual cardiovascular response to speed, incline, fatigue, recovery, and sensor noise so that policies can be trained without unsafe real-world exploration.
- **Constrained MDP formulation.** Separate training goals such as target heart-rate tracking and comfort from safety costs such as overshoot, unsafe acceleration, low-SQI actions, and stop conditions.
- **Safe RL with action shield.** Compare rule-based, PID/MPC, PPO, and Lagrangian safe PPO controllers while projecting all actions through a hard safety supervisor.
- **Stress-test evaluation.** Evaluate time-in-zone, overshoot, action smoothness, safety violation rate, OOD behavior, and worst-case performance across simulated user profiles.

This project is designed to demonstrate practical reinforcement learning judgment: the RL policy is useful only when it is paired with baselines, uncertainty estimates, safety constraints, and interpretable fallback logic.
