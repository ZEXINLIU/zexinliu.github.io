---
layout: page
title: Orthogonal Polynomials and Biomedical Uncertainty Quantification
description: Ph.D. research on recurrence algorithms, polynomial chaos expansion, UncertainSCI, and reliable biomedical simulation.
img:
importance: 1
category: research
github: https://github.com/SCIInstitute/UncertainSCI
related_publications: true
---

My Ph.D. research at the University of Utah focused on scientific computing and uncertainty quantification, advised by Professor Akil Narayan. The core mathematical problem was how to construct stable orthogonal polynomial bases for nonclassical measures, then use them in polynomial chaos expansion (PCE) for expensive biomedical simulations.

The research has three connected layers.

First, in one dimension, I studied how to compute three-term recurrence coefficients for generalized orthogonal polynomial families when closed-form classical formulas are unavailable. This led to predictor-corrector and predictor-corrector Lanczos style algorithms that convert moment information into stable recurrence coefficients.

Second, in multiple dimensions, the scalar recurrence becomes a matrix-valued recurrence. I worked on algorithms for generating multivariate orthogonal polynomials through recurrence matrices, filling an important gap between one-dimensional theory and practical multivariate approximation.

Third, these bases became computational tools for uncertainty quantification. Through [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI), the mathematics was packaged into a noninvasive Python toolkit that can connect to existing simulation pipelines and produce PCE-based statistics and sensitivities.

Representative outcomes:

- [Univariate recurrence examples](https://github.com/ZEXINLIU/Uni_ttr_examples)
- [Multivariate recurrence examples](https://github.com/ZEXINLIU/Multi_ttr_examples)
- [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI)
- Applications in cardiac simulation, brain stimulation, and coronary artery biomechanical modeling

This project is the foundation of my quantitative background: numerical analysis, approximation theory, uncertainty quantification, and software that brings mathematical reliability into real biomedical modeling workflows.
