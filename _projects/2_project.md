---
layout: page
title: Heart-Rate Deceleration Capacity Screening
description: Wearable heart-risk screening built on deceleration capacity, ECG/PPG representation learning, and health-study deployment.
img:
importance: 2
category: health-ai
---

I led development of the sudden cardiac arrest screening feature based on heart-rate deceleration capacity (DC), introduced on the HONOR Watch 5 Ultra. The feature is part of HONOR's health-study platform and uses heart-rate deceleration related signals to support heart-health risk screening research.

Deceleration capacity is connected to autonomic regulation, especially the vagus nerve's ability to slow heart rate. In practice, the challenge is to bridge classical HRV/DC theory with large-scale wearable sensing: consumer devices collect long, noisy, behavior-dependent signals rather than controlled clinical recordings.

Core algorithmic themes:

- **Classical cardiac signal theory.** Use deceleration capacity, rhythm variability, and related time-series descriptors as physiologically meaningful anchors.
- **ECG/PPG representation learning.** Learn robust embeddings over physiological waveforms so the system can capture morphology and temporal context beyond hand-crafted features.
- **Contrastive and predictive learning.** Use multimodal and temporal contrastive objectives to align ECG/PPG segments, separate signal artifacts from physiological variation, and improve label efficiency.
- **Risk-oriented modeling.** Emphasize stable ranking, interpretable signal quality control, and clinically meaningful trend behavior rather than only point-estimate accuracy.
- **Deployment under real-world variability.** Handle wearing state, motion, sensor noise, long-horizon monitoring, and health-study data pipelines.

This project connects my interest in contrastive representation learning with medical-domain algorithm delivery: the model must learn useful structure from noisy longitudinal data, while the feature still needs to respect physiology and product constraints.
