---
layout: page
title: "24H Cuffless Blood Pressure Monitoring"
description: Wearable PPG blood pressure monitoring with personalization, calibration, and product-level robustness.
img:
importance: 1
category: health-ai
---

I led algorithm development for the 24-hour continuous non-invasive blood pressure monitoring feature introduced on the HONOR Watch 5 Pro. This project focuses on turning short wearable PPG windows into reliable blood pressure estimates under real-world sensing conditions.

The core challenge is personalization. A population-level model must handle large individual differences in vascular characteristics, sensor contact, daily physiology, and calibration quality, while the user can only provide a small amount of cuff-based reference data.

Public-facing algorithm themes:

- **Physiological signal modeling.** Extract morphology, rhythm, signal-quality, and context features from wearable PPG streams.
- **Personalized calibration.** Adapt the model to individual users from limited reference measurements without overfitting short calibration sessions.
- **Generative personalization.** Use conditional generative modeling as a data-efficiency tool for user-specific adaptation under label scarcity.
- **Robust deployment.** Maintain stable behavior across motion, wearing state, skin contact, time of day, and daily trend changes.

This project demonstrates practical generative modeling in a medical AI setting: the algorithmic value is not image-like generation, but improving personalization and robustness when gold-standard labels are expensive.
