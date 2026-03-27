\documentclass[11pt, a4paper]{article}

% --- UNIVERSAL PREAMBLE BLOCK ---

\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}

\usepackage{fontspec}

\usepackage[english, bidi=basic, provide=*]{babel}

\babelprovide[import, onchar=ids fonts]{english}

% Set default/Latin font to Sans Serif in the main (rm) slot

\babelfont{rm}{Noto Sans}

% Math packages

\usepackage{amsmath, amssymb, amsthm}

\newtheorem{theorem}{Theorem}

\newtheorem{lemma}[theorem]{Lemma}

\title{\textbf{Absolute Stability of Softmax-Routed Queueing Networks via Entropy Bounds}}

\author{A Rigorous Formulation}

\date{}

\begin{document}

\maketitle

\begin{abstract}

We present a succinct and exact proof of positive Harris recurrence for a system of $N$ parallel heterogeneous queues where arrivals are routed according to a softmax (thermal soft-damping) policy. By leveraging the variational properties of the Gibbs free energy, we prove that the system is stable for any strictly positive inverse temperature $\alpha > 0$, provided the fundamental capacity condition is met.

\end{abstract}

\section{System Model and Main Result}

Consider $N \ge 1$ parallel servers. Jobs arrive according to a Poisson process with rate $\lambda > 0$. Each server $i \in \{1, \dots, N\}$ has an exponentially distributed service rate $\mu_i > 0$. Let $Q(t) = (Q_1(t), \dots, Q_N(t)) \in \mathbb{Z}_+^N$ denote the queue lengths at time $t$.

Upon arrival, a job is routed to server $i$ with probability:

\begin{equation} \label{eq:raw_q_softmax}

p_i(Q) = \frac{\exp\left(-\alpha Q_i\right)}{\sum_{j=1}^N \exp\left(-\alpha Q_j\right)},

\end{equation}

where $\alpha > 0$ is the routing temperature parameter (inverse temperature).

\begin{theorem}

Assume the strict capacity condition holds: $\Lambda := \sum_{i=1}^N \mu_i > \lambda$. For any $\alpha > 0$, the continuous-time Markov chain (CTMC) $Q(t)$ is non-explosive, irreducible, and positive Harris recurrent.

\end{theorem}

\section{Proof of the Theorem}

\begin{proof}

\textbf{Step 1: CTMC Regularity and Generator.}

The state space $\mathcal{X} = \mathbb{Z}_+^N$ is countable. The maximum transition rate out of any state is bounded by $\lambda + \Lambda$, ensuring the chain is non-explosive. Because $\lambda > 0$ and $\mu_i > 0$, any state can be reached from the origin and vice-versa, ensuring irreducibility.

Define the norm-like Lyapunov function $V(Q) = \frac{1}{2}\sum_{i=1}^N Q_i^2$. The exact action of the generator $\mathcal{L}$ on $V$ is given by the expected drift:

\begin{align}

\mathcal{L}V(Q) &= \lambda \sum_{i=1}^N p_i(Q) \left[ \frac{1}{2}(Q_i+1)^2 - \frac{1}{2}Q_i^2 \right] + \sum_{i=1}^N \mu_i \mathbf{1}_{Q_i>0} \left[ \frac{1}{2}(Q_i-1)^2 - \frac{1}{2}Q_i^2 \right] \nonumber \\

&= \lambda \sum_{i=1}^N p_i(Q) Q_i - \sum_{i=1}^N \mu_i Q_i + C(Q), \label{eq:generator}

\end{align}

where $C(Q) = \frac{\lambda}{2} + \frac{1}{2}\sum_{i=1}^N \mu_i \mathbf{1}_{Q_i>0} \le \frac{\lambda + \Lambda}{2} =: C_1$.

\vspace{0.3cm}

\noindent \textbf{Step 2: The Gibbs Free Energy Bound.}

We recognize $p(Q)$ as the Gibbs distribution minimizing the free energy over the probability simplex $\Delta_{N-1}$. For any distribution $q \in \Delta_{N-1}$, the free energy is defined as $F(q) = \sum q_i Q_i - \frac{1}{\alpha} \mathcal{H}(q)$, where $\mathcal{H}(q) = -\sum q_i \log q_i$ is the Shannon entropy. The minimum is achieved at $p(Q)$, yielding the log-sum-exp (softmin) function:

\begin{equation}

\sum_{i=1}^N p_i(Q) Q_i - \frac{1}{\alpha} \mathcal{H}(p(Q)) = -\frac{1}{\alpha} \log \sum_{j=1}^N \exp(-\alpha Q_j).

\end{equation}

Because $\sum_{j=1}^N \exp(-\alpha Q_j) \ge \exp(-\alpha \min_j Q_j)$, the right-hand side is strictly bounded above by $\min_j Q_j$. Furthermore, since the discrete entropy is bounded by $\mathcal{H}(p(Q)) \le \log N$, we obtain the fundamental inequality:

\begin{equation} \label{eq:softmax_bound}

\sum_{i=1}^N p_i(Q) Q_i \le \min_i Q_i + \frac{\log N}{\alpha}.

\end{equation}

\vspace{0.3cm}

\noindent \textbf{Step 3: Bounding the Service Rate.}

Let $Q_{\min} = \min_i Q_i$, and define the non-negative differences $\Delta_i = Q_i - Q_{\min} \ge 0$. The total service rate can be exactly decomposed as:

\begin{equation} \label{eq:service_bound}

\sum_{i=1}^N \mu_i Q_i = \sum_{i=1}^N \mu_i (Q_{\min} + \Delta_i) = \Lambda Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i.

\end{equation}

\vspace{0.3cm}

\noindent \textbf{Step 4: Foster-Lyapunov Drift Inequality.}

Substituting \eqref{eq:softmax_bound} and \eqref{eq:service_bound} into \eqref{eq:generator}, we obtain:

\begin{equation}

\mathcal{L}V(Q) \le \lambda \left(Q_{\min} + \frac{\log N}{\alpha}\right) - \left( \Lambda Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i \right) + C_1.

\end{equation}

Rearranging terms yields:

\begin{equation}

\mathcal{L}V(Q) \le -(\Lambda - \lambda) Q_{\min} - \sum_{i=1}^N \mu_i \Delta_i + R,

\end{equation}

where $R = \frac{\lambda \log N}{\alpha} + C_1$ is a strictly finite constant for any $\alpha > 0$.

\vspace{0.3cm}

\noindent \textbf{Step 5: Establishing Positive Recurrence.}

Observe that the $L_1$ norm of the state is $|Q|_1 = \sum_{i=1}^N Q_i = N Q_{\min} + \sum_{i=1}^N \Delta_i$. Define the strictly positive constant $\varepsilon = \min \left( \frac{\Lambda - \lambda}{N}, \min_i \mu_i \right) > 0$. We can construct the lower bound:

\begin{equation}

\varepsilon |Q|_1 = \varepsilon N Q_{\min} + \varepsilon \sum_{i=1}^N \Delta_i \le (\Lambda - \lambda)Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i.

\end{equation}

Therefore, our drift inequality simplifies to:

\begin{equation}

\mathcal{L}V(Q) \le -\varepsilon |Q|_1 + R.

\end{equation}

Let $C$ be the finite (and thus compact) set of states where $\varepsilon |Q|_1 \le R + 1$. Outside of $C$, we have $\mathcal{L}V(Q) \le -1$. By the continuous-time Foster-Lyapunov criteria (Theorem 4.2 in Meyn \& Tweedie, 1993), the existence of this norm-like Lyapunov function with strict negative drift outside a compact set guarantees that $Q(t)$ is positive Harris recurrent and possesses a unique stationary distribution.

\end{proof}

\section{The Dual-Policy Architecture: Theory vs. Application}

The GibbsQ framework maintains a dual-policy architecture to balance theoretical rigor and applied deep learning performance:

\textbf{1. The Theoretical Baseline: Raw-Queue Softmax}

Section 1 establishes the Raw-Queue Softmax ($p_i \propto \exp(-\alpha Q_i)$) as the mathematically proven, absolutely stable baseline. We utilize this formulation primarily for handling basic homogeneous servers and for establishing strict analytical Foster-Lyapunov drift bounds across the state space.

\textbf{2. The Applied Standard: Unified Archimedean Softmax (UAS)}

For advanced deep learning implementations and empirical analysis of heterogeneous server configurations, we employ the Unified Archimedean Softmax (UAS) policy. The UAS provides a capacity-weighted pre-multiplier:

\begin{equation}
p_i(Q,\mu) = \frac{\mu_i \exp\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}{\sum_{j=1}^N \mu_j \exp\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
\end{equation}

For neural-policy initialization via Behavioral Cloning (BC) or REINFORCE warm-starting, the UAS formulation is profoundly superior because its "look-ahead potential" $(Q_i+1)/\mu_i$ aligns perfectly with the heterogeneity-aware input features ingested by the neural network. Furthermore, the $\mu_i$ multiplier natively routes traffic to faster servers at the idle state ($p_i \propto \mu_i \exp(-\alpha/\mu_i)$).

By retaining both formulations, the project can seamlessly traverse between asserting absolute analytic guarantees (via Raw-Queue bounds) and optimizing real-world heterogeneous routing models (via UAS neural gradients).

\end{document}

