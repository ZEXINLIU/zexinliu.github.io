---
layout: post
title: constrastive learning, a powerful technique for self-supervised learning
date: 2025-10-01 00:00:00
description: insights on constrastive learning
tags: formatting math
categories: sample-posts
featured: true
---

What is contrastive learning? Contrastive Learning is a Machine Learning paradigm where unlabeled data points are juxtaposed against each other to teach a model which points are similar and which are different. That is, as the name suggests, samples are contrasted against each other, and those belonging to the same distribution, or have some latent features in common are pushed towards each other in the embedding space. In contrast, those belonging to different distributions or no attributes in common can be learned are pulled against each other.

Vision AI is a good example to quickly illustrate how does Contrastive Learning work. Given a collection of animal pictures, one may not recognize some of the animals, but can infer which pictures show the same animals. Contrastive Learning mimics the way humans learn.

The basic contrastive learning framework consists of selecting a data sample, called “anchor,” a data point belonging to the same distribution as the anchor, called the “positive” sample, and another data point belonging to a different distribution called the “negative” sample. The SSL model tries to minimize the distance between the anchor and positive samples, i.e., the samples belonging to the same distribution, in the latent space, and at the same time maximize the distance between the anchor and the negative samples. This produces `Triplet loss`, which was concieved by Google researchers for their prominent [FaceNet](https://arxiv.org/abs/1503.03832) algorithm for face detection.

The following is a timeline of the development of Contrastive Learning in the field of Computer Vision.
- 2018, [InstDics](https://arxiv.org/pdf/1805.01978): Non-Parametric Instance Discrimination, Memory Bank, NCE loss
- 2019, [InvaSpread](https://arxiv.org/pdf/1904.03436): Augmentation, Siamese Network, Comparison within a Batch
- 2019, [MOCO-V1](https://arxiv.org/pdf/1911.05722): Momentum Encoder, Dictionary Look-up (Queue)
- 2020, [SimCLR-V1](https://arxiv.org/pdf/2002.05709): Large Batchsize, MLP head, Multi-crop
- 2020, [MOCO-V2](https://arxiv.org/pdf/2003.04297): MLP head, Multi-crop, Require NO Large Batchsize
- 2020, [BYOL](https://arxiv.org/pdf/2006.07733): Require NO Negative Samples, Suffer data leak by BN (Controversial)
- 2021, [Siam](https://arxiv.org/pdf/2011.10566): Siamese Network, Require NO Negative Samples nor Momentum Encoder
- 2021, [MOCO-v3](https://arxiv.org/pdf/2104.02057): ViT Backbone, Instability alleviated by Frozen Patch Projection Layer
- 2021, [DINO](https://arxiv.org/pdf/2104.14294): ViT Backbone, Momentum Encoder, Multi-crop, Self-distill

We firstly talk about the paper `InstDics` since in my personal opinion, this work was a pioneer for Constrastive Learninig in CV, for firstly thinking about how to apply ideas of negative sampling and NCE loss in CV and how to design data structure for the convenience of sampling. These had a profound impact on subsequent works, such as SimCLR, MOCO, etc.

The paper treats each image instance as a distinct class of its own and train a classifier to distinguish between individual instance classes. Under non-parametric softmax formulation, for image $$ x $$ with feature $$ v=f_\theta(x) $$, the probability of it being recognized as i-th example is
\begin{equation}
\label{eq:non-para}
p(i|v) = \frac{\exp(v_i^T v / \tau)}{\sum_{i=1}^n \exp(v_j^T v / \tau)}, \quad p(i|v_i) = p(i|f_\theta(x_i))
\end{equation}
The objective is $$ J(\theta) = -\sum_{i=1}^n \log p(i|f_\theta(x_i)) = -\sum_{i=1}^n \log p(i|v_i) $$. Instead of exhaustively computing features for all images every time, we maintain a feature memory bank V for storing these representations. During each iteration, these representations as well as $\theta$ are optimized and then updated to memory bank at the corresponding instance entry. The only problem is the computational cost in the denominator in \eqref{non-para}. Noise-Contrastive Estimation (NCE) is introduced to approximate this full softmax.

The basic idea is to cast the multi-class classification problem into a set of binary classification problems, where the binary classification task is to discriminate between data samples and noise samples. We formalize noise distribution as a uniform: $$ p_n = \frac{1}{n} $$ (denoted by $$ D=0 $$), data distribution $$ p_d $$ (denoted by $$ D=1 $$). Thus we have the likelihood $p(i,v|D=0) = p_n$ and $p(i,v|D=1) = p(i|v)$.

assume that noise samples are $$ m $$ times more frequent than data samples, e.g. number of data is $$ 1 $$ and noise samples is $$ m $$, i,e, $$ P(D=0) = \frac{m}{1+m} $$ and $$ P(D=1) = \frac{1}{1+m} $$. Then the posterior probability of sample $$ i $$ with feature $$ v $$ being from the data distribution is 
\begin{equation}
\label{eq:posterior}
p(D=1|i, v) = \frac{p(i,v|D=1) * p(D=1)}{p(i,v|D=1) p(D=1) + p(i,v|D=0) * p(D=0)} = \frac{p(i|v) * \frac{1}{1+m}}{p(i|v) * \frac{1}{1+m} + p_n * \frac{m}{1+m}} = \frac{p(i|v)}{p(i|v) + m p_n(i)}
\end{equation}

The training object (with negative log) is
\begin{equation}
\label{eq:obj}
J_{NCE}(\theta) = -E_{p_d}[\log p(D=1|i, v)] - mE_{p_n} (\log(1 - p(D=1|i, v))).
\end{equation}
Apparently minimize the objective helps binary classification, but how related to original classification formulated by non-parametric softmax? To show this rigorously, we need to prove the equivalence of gradient of NCE and softmax.

Then I want to talk about `MOCO` since the idea of Momentum Update and Queue dictionary for looking up are devised ingeniouly and enlighten many subsequent works.