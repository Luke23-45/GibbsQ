# References For The Deterministic And Stochastic Routes

This file records the external theorem framework relevant to the `z2` notes.
The references now split into two active groups:

- deterministic reflection / projected-gradient references for the reflected ODE
- queueing and Lyapunov references for the stochastic certification program

## Core Queueing-Stability References

1. **J. G. Dai (1995).**
   *On positive Harris recurrence of multiclass queueing networks: a unified
   approach via fluid limit models.*
   Annals of Applied Probability 5(1), 49-77.

   This is the standard route from stability of the associated fluid model to
   positive Harris recurrence of the underlying queueing process.

2. **J. G. Dai and S. P. Meyn (1995).**
   *Stability and convergence of moments for multiclass queueing networks via
   fluid limit models.*
   IEEE Transactions on Automatic Control 40(11), 1889-1904.

   This extends the fluid-limit methodology once the correct large-scale limit
   has been identified.

3. **H. Chen and H. Zhang (1997).**
   *Stability of multiclass queueing networks under FIFO service discipline.*
   Mathematics of Operations Research 22(3), 691-725.

   This is useful as a template for writing reflected fluid equations carefully
   when boundary behavior matters.

4. **M. Schoenlein (2015).**
   *A Lyapunov view on positive Harris recurrence of multiclass queueing
   networks.*
   Operations Research Letters 43(3), 299-303.

   This is useful for proof design because it highlights how a fluid Lyapunov
   function can sometimes be lifted to a Foster-Lyapunov function for the Markov
   process.

## Deterministic Reflection And Projected Dynamics

5. **J. M. Harrison and M. I. Reiman (1981).**
   *Reflected Brownian motion on an orthant.*
   Annals of Probability 9(2), 302-308.

   This is the classical orthant-reflection reference. In the present project it
   is relevant for the Skorokhod-map viewpoint behind the reflected dynamics.

6. **W. P. M. H. Heemels, J. M. Schumacher, and S. Weiland (2000).**
   *Projected dynamical systems in a complementarity formalism.*
   Operations Research Letters 27(2), 83-91.

   This is useful for translating between reflected dynamics and projected or
   complementarity formulations.

7. **J. C. Dunn (1980).**
   *Global and asymptotic convergence rate estimates for a class of projected
   gradient processes.*
   SIAM Journal on Control and Optimization 18(4), 368-400.

   This is a classical reference for viewing the deterministic reflected-UAS
   surrogate as a projected gradient flow.

## Convex Gradient-Flow References

8. **H. Brezis (1973).**
   *Operateurs maximaux monotones et semi-groupes de contractions dans les
   espaces de Hilbert.*
   North-Holland.

   This is the standard maximal-monotone / subgradient-flow reference behind the
   weighted convex-gradient interpretation.

9. **R. E. Bruck, Jr. (1975).**
   *Asymptotic convergence of nonlinear contraction semigroups in Hilbert
   space.*
   Journal of Functional Analysis 18(1), 15-26.

   This is a classical reference for asymptotic convergence of convex
   gradient-like flows in Hilbert spaces.

10. **A. Pazy (1978).**
    *On the asymptotic behavior of semigroups of nonlinear contractions in
    Hilbert space.*
    Journal of Functional Analysis 27(3), 292-307.

    This complements Bruck's convergence theorem and is useful when writing the
    abstract semigroup version of the deterministic argument.

## Interpretation For This Project

The `z2` package now uses these references in two different roles:

- the projected-dynamics and convex-flow references support the deterministic
  reflected-ODE analysis
- the queueing references describe the separate stochastic route that would be
  needed for a CTMC stability theorem

What remains package-sensitive is not the deterministic convergence theorem. It
is the correct stochastic certification route, whether that route is eventually
framed through a large-scale limit, a direct Foster-Lyapunov argument, or some
other valid comparison theorem.
