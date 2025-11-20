---
layout: post
title: What are orthgonal polynomials, and why they are important?
date: 2022-05-02 11:12:00-0400
description: Insights on Orthogonal Polynomials
tags: Orthogonal Polynomials
categories: sample-posts
related_posts: true
---

The theory of orthogonal polynomials (OP) plays an important role in many branches of mathematics, such as approximation theory (best approximation, interpolation, quadrature), special functions, continued fractions, differential and integral equations.

This seems too general and abroad for people outside this field to understand. Let's consider a simple and common scenario: how to compute an integral numerically efficiently and accurately? One may use Trapezoidal Rule or Simpson’s Rule introduced in Calculus course, but they have accuracy issue comparing to [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).

Before discussing Gaussian quadrature, we first describe a picture of quadrature rules. If the integrand is reasonably well-behaved (i.e. piecewise continuous and of bounded variation), a simplest way that fits intuition is midpoint rule (rectangle rule), given by
\begin{equation}
\label{eq:rectangle_rule}
\int_a^b f(x) dx \approx (b-a) f(\frac{a+b}{2}).
\end{equation}
In this case, the interpolating functions are step functions (segmented polynomials of degree 0) passing through midpoints. We can replace it with an affine function (staight line, polynomials of degree 1) passing through segment interval's end points, which yields Trapezoidal Rule, given by
\begin{equation}
\label{eq:trapezoidal_rule}
\int_a^b f(x) dx \approx (b-a) \frac{f(a)+f(b)}{2}.
\end{equation}
Notice that intepolation with polynomials evaluated at equally spaced points yields the Newton-Cotes fomulas, of which the rectangle rule and the trapezoidal rule are examples. Simpson’s Rule is a Newton-Cotes fomula, but based on a polynomial of order 2, given by
\begin{equation}
\label{eq:simpson_rule}
\int_a^b f(x) dx \approx \frac{(b-a)}{6} [f(a) + 4f(\frac{a+b}2{}) f(b)].
\end{equation}

Equally spaced points are so convenient that integrand values can be re-used. But given the same number of function evaluations, especially for smooth integrands, Gaussian quadrature (GQ) rule is typically more accurate than Newton-Cotes. Why? A theorem proofs related answer is: GQ can integrate polynomials of degree $2n-1$ exactly with $n$ function evaluations while Newton-Cotes rules have lower degrees up to $n$ of precision. A more insighted quick answer is: GQ chooses optimal sampling points based on polynomial roots, such as Legendre polynomials. This means that points where the function is more significant have a larger influence on the computed integral, enhancing accuracy, while equal weights are employed in Newton-Cotes.

We will discuss these in details afterwards. For now, I'll just give some snaps of Gaussian quadrature examples and show how orthogonal polynomials are involved.  Gauss–Legendre quadrature rule, will be an accurate approximation to the integral above if $f(x)$ is well-approximated by a polynomial of degree $2n-1$ or less on $[−1, 1]$. It can be stated as
\begin{equation}
\label{eq:gq_lengendre}
\int_{-1}^1 f(x) dx \approx \sum_{i=1}^n w_i f(x_i).
\end{equation}
Instead, if the integrand can be written as
$$
f(x) = (1-x)^{\alpha} (1+x)^{\beta} g(x), \quad, \alpha, \beta > 1
$$
then another set of nodes and weights will usually give more accurate quadrature rules, known as Gauss–Jacobi quadrature rules, i.e.
\begin{equation}
\label{eq:gq_jacobi}
\int_{-1}^1 f(x) dx = \int_{-1}^1 (1-x)^{\alpha} (1+x)^{\beta} g(x) dx \approx \sum_{i=1}^n w_i f(x_i).
\end{equation}
The name "Legendre" and "Jacobi" are some classical orthogonal polynomials with various beautiful properties. Definitely there are more, e.g., Hermite, Chebyshev, Laguerre and e.t.c. Some may sound familiar since they are widely used in all walks of life.

It remains to answer: how to compute the nodes and weights mentioned above? This is NOT a trivial question. One has to know about how orthogonal polynomials are defined, how to numerically compute the moments related to given OP, and how to devise algorithms to compute the known three-term recurrence formula from unidimensional case to higher dimensions. More details can be found [here](https://epubs.siam.org/doi/abs/10.1137/22M1477131) if you are interested.

Integration is only a simple applications using OPs. Specifically, Many orthogonal polynomials are also used because they arise naturally in certain differential equations. E.g. Hermite polynomials in the quantum harmonic oscillator, Legendre polynomials for Laplace’s equation, Laguerre polynomials for the hydrogen atom, and so on.