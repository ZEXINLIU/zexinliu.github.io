---
layout: page
title: Continuous Non-Invasive Blood Pressure Monitoring
description: 24-hour wearable BP monitoring from physiological signals, calibration, and population-level robustness.
img:
importance: 1
category: health-ai
---

I led the algorithm development for the 24-hour continuous non-invasive blood pressure monitoring feature introduced on the HONOR Watch 5 Pro. The project sits at the intersection of wearable sensing, medical signal modeling, uncertainty-aware calibration, and deployable machine learning.

The technical problem is difficult because wearable PPG and related physiological signals are indirect, noisy, motion-sensitive proxies of vascular dynamics. A production BP system must therefore solve more than a regression task: it needs robust signal quality control, population-level calibration, temporal trend consistency, and stable behavior under real-world daily use.

Core algorithmic themes:

- **Physiological signal representation.** Extract signal morphology, pulse transit and pulse wave features, rhythm context, and quality indicators from wearable sensor streams.
- **Calibration-aware modeling.** Combine personal calibration with population priors so the model can adapt to user-specific vascular and sensor-contact characteristics.
- **Domain adaptation and feature alignment.** Reduce distribution shifts across age, skin tone, device wearing state, activity level, and acquisition environment.
- **Generative modeling for signal robustness.** Use conditional generative modeling and diffusion/flow-style thinking to reason about plausible signal variation, denoising, missing regions, and uncertainty around PPG regions of interest.
- **Medical AI productization.** Translate research prototypes into a stable device feature with constrained compute, interpretable failure modes, and user-facing trend outputs.

This project reflects the kind of algorithm work I value most: mathematically grounded modeling, careful medical-domain interpretation, and real product deployment.
