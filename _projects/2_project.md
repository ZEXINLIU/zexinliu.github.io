---
layout: page
title: "Deceleration Capacity Screening from Wearable Signals"
description: Wearable heart-risk screening using deceleration capacity theory, ECG/PPG representation learning, and long-horizon aggregation.
img:
importance: 2
category: health-ai
---

I led development of the sudden cardiac arrest screening feature based on heart-rate deceleration capacity (DC), introduced on the HONOR Watch 5 Ultra. The project connects classical cardiac signal theory with multimodal representation learning for wearable heart-health screening research.

ECG is the standard signal source for RR-interval-based DC estimation, but continuous daily monitoring is more natural on wearable PPG. The algorithmic challenge is to preserve clinically meaningful rhythm information while handling PPG quality variation in daily use.

Public-facing algorithm themes:

- **Deceleration capacity modeling.** Use established rhythm theory as the physiological anchor rather than treating the task as generic time-series regression.
- **ECG/PPG representation learning.** Use synchronized signals to align wearable PPG representations with ECG-derived rhythm structure.
- **Signal-quality awareness.** Account for motion, wearing state, missingness, and pulse-detection confidence during long-horizon monitoring.
- **24-hour aggregation.** Combine interpretable rhythm features and learned representations into stable long-period screening outputs.

This project shows how contrastive learning can serve as a medical-domain transfer mechanism: ECG provides the standard rhythm reference, while PPG provides long-term wearable accessibility.
