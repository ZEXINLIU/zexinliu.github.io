---
layout: post
title: The 24-hour continuous non-invasive blood pressure monitoring feature I developed was introduced on the newly released HONOR Watch 5 Pro
date: 2025-10-15 19:00:00-0400
inline: false
related_posts: false
---

The HONOR Watch 5 Pro was released with a 24-hour continuous non-invasive blood pressure monitoring feature. It supports multiple measurement modes, including 40-second single measurement and trend tracking, and provides non-pharmacological suggestions for blood pressure management based on factors such as salt intake, sleep, and exercise.

Official feature page: [HONOR Watch 5 Pro](https://www.honor.com/cn/wearables/honor-watch-5-pro/).

As the lead developer, I worked on the core algorithm pipeline for calibrated cuffless blood pressure estimation from wearable physiological signals. The pipeline combines BP regression, cuff-based calibration labels from an upper-arm blood pressure monitor, personalization, signal-quality control, and product-level robustness. Techniques involved include `conditional generative modeling` around photoplethysmography representations, `feature alignment and clustering` across broad populations, and `domain adaptation` for wearable deployment.
