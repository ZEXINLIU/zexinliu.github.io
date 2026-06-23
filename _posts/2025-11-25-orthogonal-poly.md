---
layout: post
title: What are orthogonal polynomials and why are they important?
date: 2025-11-25 00:00:00
description: A short intuition-first note on Gaussian quadrature, recurrence coefficients, and why orthogonal polynomials matter.
tags: mathematics scientific-computing orthogonal-polynomials quadrature
categories: research-notes
featured: true
---

The theory of orthogonal polynomials (OP) plays an important role in many branches of mathematics, such as approximation theory (best approximation, interpolation, quadrature), special functions, continued fractions, differential and integral equations.

This seems too general and broad for people outside this field to understand. Let's consider a simple and common scenario: how to compute an integral numerically, efficiently, and accurately? One may use the trapezoidal rule or Simpson's rule introduced in calculus, but they have accuracy issues compared to [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).

Before discussing Gaussian quadrature, we first describe a picture of quadrature rules. If the integrand is reasonably well-behaved (i.e. piecewise continuous and of bounded variation), a simplest way that fits intuition is midpoint rule (rectangle rule), given by
\begin{equation}
\label{eq:rectangle_rule}
\int_a^b f(x) dx \approx (b-a) f(\frac{a+b}{2}).
\end{equation}
In this case, the interpolating functions are step functions (segmented polynomials of degree 0) passing through midpoints. We can replace them with affine functions (straight lines, polynomials of degree 1) passing through each segment interval's endpoints, which yields the trapezoidal rule:
\begin{equation}
\label{eq:trapezoidal_rule}
\int_a^b f(x) dx \approx (b-a) \frac{f(a)+f(b)}{2}.
\end{equation}
Notice that interpolation with polynomials evaluated at equally spaced points yields the Newton-Cotes formulas, of which the rectangle rule and the trapezoidal rule are examples. Simpson's rule is a Newton-Cotes formula based on a polynomial of order 2:
\begin{equation}
\label{eq:simpson_rule}
\int_a^b f(x) dx \approx \frac{(b-a)}{6} [f(a) + 4f(\frac{a+b}{2}) + f(b)].
\end{equation}

Equally spaced points are convenient because integrand values can be reused. But given the same number of function evaluations, especially for smooth integrands, a Gaussian quadrature (GQ) rule is typically more accurate than Newton-Cotes. Why? A theorem-based answer is: GQ can integrate polynomials of degree $$ 2n-1 $$ exactly with $$ n $$ function evaluations, while Newton-Cotes rules have lower degrees of precision. A quick intuitive answer is: GQ chooses sampling points based on polynomial roots, such as the roots of Legendre polynomials.

We will discuss these in detail later. For now, I'll give a few snapshots of Gaussian quadrature examples and show how orthogonal polynomials are involved. A Gauss-Legendre quadrature rule will be an accurate approximation if $$ f(x) $$ is well-approximated by a polynomial of degree $$ 2n-1 $$ or less on $$ [-1, 1] $$. It can be stated as
\begin{equation}
\label{eq:gq*lengendre}
\int*{-1}^1 f(x) dx \approx \sum\_{i=1}^n w_i f(x_i).
\end{equation}
Instead, if the integrand can be written as

$$
f(x) = (1-x)^{\alpha} (1+x)^{\beta} g(x), \quad, \alpha, \beta > 1
$$

then another set of nodes and weights will usually give more accurate quadrature rules, known as Gauss–Jacobi quadrature rules, i.e.
\begin{equation}
\label{eq:gq*jacobi}
\int*{-1}^1 f(x) dx = \int*{-1}^1 (1-x)^{\alpha} (1+x)^{\beta} g(x) dx \approx \sum*{i=1}^n w_i f(x_i).
\end{equation}
The names "Legendre" and "Jacobi" refer to classical orthogonal polynomial families with many beautiful properties. There are many more, including Hermite, Chebyshev, and Laguerre polynomials. Some may sound familiar because they are widely used across mathematics, physics, and engineering.

It remains to answer: how do we compute the nodes and weights mentioned above? This is not a trivial question. One has to understand how orthogonal polynomials are defined, for example by applying Gram-Schmidt to monomials; how to numerically compute finite moments with respect to fairly general measures; and how to devise Stieltjes-type algorithms to compute three-term recurrence formulas from the univariate case to higher-dimensional tensorial measures. More details can be found [here for the univariate case](https://link.springer.com/article/10.1007/s10915-021-01586-w) and [here for the multivariate case](https://epubs.siam.org/doi/abs/10.1137/22M1477131).

Integration is only one application of OPs. Orthogonal polynomials also arise naturally in differential equations. Hermite polynomials appear in the quantum harmonic oscillator; Legendre polynomials appear when solving Laplace's equation in spherical coordinates; Laguerre polynomials appear in the radial part of hydrogen-like atom wave functions. Accurate evaluation of these polynomials and their variants is important in both theoretical analysis and scientific applications.
