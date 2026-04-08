\documentclass[11pt, a4paper]{article}

\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}
\usepackage{fontspec}
\usepackage[english, bidi=basic, provide=*]{babel}
\babelprovide[import, onchar=ids fonts]{english}
\babelfont{rm}{Noto Sans}
\usepackage{amsmath, amssymb, amsthm}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}

\title{\textbf{Entropy-Regularized Stability of Parallel Queues under Softmax and Unified Archimedean Softmax Routing}}
\author{A Rigorous Formulation}
\date{}

\begin{document}

\maketitle

\begin{abstract}
We study a system of \(N\) parallel heterogeneous queues with Poisson arrivals and exponential service. We present two entropy-regularized routing laws within the GibbsQ framework. The first is the raw queue softmax policy, for which we give a Foster-Lyapunov proof of positive Harris recurrence under the load condition \(\lambda < \sum_i \mu_i\). The second is the Unified Archimedean Softmax (UAS) policy, a heterogeneous extension derived from a prior-weighted entropy-regularized variational objective and analyzed with a weighted Lyapunov function. For UAS, we prove the routing formula, the weighted variational identity, the exact generator identity, and a Foster-Lyapunov drift inequality yielding positive Harris recurrence under the same load condition. The learned policy N-GibbsQ is treated as an empirical neural extension benchmarked against these analytical baselines.
\end{abstract}

\section{Framework And Main Theorem Layers}

The GibbsQ framework is an entropy-regularized queue-routing framework with three layers:

\begin{enumerate}
\item the raw softmax routing baseline,
\item the heterogeneous UAS routing extension,
\item the learned neural policy N-GibbsQ.
\end{enumerate}

The theorem statements in this manuscript concern the two analytical routing laws. The neural policy is treated separately as an empirical approximation layer.

\section{Raw Softmax Model And Theorem}

Consider \(N \ge 1\) parallel servers. Jobs arrive according to a Poisson process with rate \(\lambda > 0\). Server \(i \in \{1,\dots,N\}\) has exponential service rate \(\mu_i > 0\). Let \(Q(t)=(Q_1(t),\dots,Q_N(t)) \in \mathbb{Z}_+^N\) denote the queue-length state.

Under raw softmax routing, an arrival is sent to server \(i\) with probability

\begin{equation}\label{eq:raw_q_softmax}
p_i(Q)=\frac{\exp(-\alpha Q_i)}{\sum_{j=1}^N \exp(-\alpha Q_j)},
\end{equation}

where \(\alpha > 0\).

\begin{theorem}[Raw softmax positive Harris recurrence]
Assume the strict load condition
\[
\Lambda := \sum_{i=1}^N \mu_i > \lambda.
\]
Then for every \(\alpha > 0\), the CTMC \(Q(t)\) under \eqref{eq:raw_q_softmax} is non-explosive, irreducible, and positive Harris recurrent.
\end{theorem}

\section{Proof For Raw Softmax}

\begin{proof}
\textbf{Step 1: CTMC regularity and generator.}

The state space \(\mathcal X = \mathbb{Z}_+^N\) is countable. The total jump rate from any state is bounded by \(\lambda + \Lambda\), so the chain is non-explosive. Because \(\lambda > 0\) and \(\mu_i > 0\), the chain is irreducible.

Use the Lyapunov function
\[
V(Q)=\frac12 \sum_{i=1}^N Q_i^2.
\]

Its generator drift is
\begin{align}
\mathcal LV(Q)
&=
\lambda \sum_{i=1}^N p_i(Q)\left[\frac12 (Q_i+1)^2 - \frac12 Q_i^2\right]
+
\sum_{i=1}^N \mu_i \mathbf 1_{Q_i>0}\left[\frac12 (Q_i-1)^2 - \frac12 Q_i^2\right] \nonumber \\
&=
\lambda \sum_{i=1}^N p_i(Q)Q_i
-
\sum_{i=1}^N \mu_i Q_i
+
C(Q),
\label{eq:generator}
\end{align}
where
\[
C(Q)=\frac{\lambda}{2}+\frac12 \sum_{i=1}^N \mu_i \mathbf 1_{Q_i>0}
\le \frac{\lambda+\Lambda}{2}
=: C_1.
\]

\textbf{Step 2: Entropy-regularized variational bound.}

The softmax law is the minimizer of the entropy-regularized objective over the simplex:
\[
q \mapsto \sum_{i=1}^N q_i Q_i + \frac{1}{\alpha}\sum_{i=1}^N q_i \log q_i.
\]
Equivalently,
\[
\sum_{i=1}^N p_i(Q)Q_i - \frac{1}{\alpha}\mathcal H(p(Q))
=
-\frac{1}{\alpha}\log \sum_{j=1}^N e^{-\alpha Q_j},
\]
where \(\mathcal H(p)=-\sum_i p_i \log p_i\).

Since \(\mathcal H(p(Q)) \le \log N\), we obtain
\begin{equation}\label{eq:softmax_bound}
\sum_{i=1}^N p_i(Q)Q_i \le \min_i Q_i + \frac{\log N}{\alpha}.
\end{equation}

\textbf{Step 3: Service decomposition.}

Let \(Q_{\min}=\min_i Q_i\) and define \(\Delta_i = Q_i - Q_{\min} \ge 0\). Then
\begin{equation}\label{eq:service_bound}
\sum_{i=1}^N \mu_i Q_i
=
\Lambda Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i.
\end{equation}

\textbf{Step 4: Foster-Lyapunov inequality.}

Substituting \eqref{eq:softmax_bound} and \eqref{eq:service_bound} into \eqref{eq:generator},
\[
\mathcal LV(Q)
\le
\lambda\left(Q_{\min} + \frac{\log N}{\alpha}\right)
-
\left(\Lambda Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i\right)
+
C_1.
\]
Hence
\[
\mathcal LV(Q)
\le
-(\Lambda-\lambda)Q_{\min}
-
\sum_{i=1}^N \mu_i \Delta_i
+
R,
\]
where
\[
R=\frac{\lambda \log N}{\alpha}+C_1.
\]

\textbf{Step 5: Positive recurrence.}

Since
\[
|Q|_1 = NQ_{\min} + \sum_{i=1}^N \Delta_i,
\]
we may choose
\[
\varepsilon = \min\left(\frac{\Lambda-\lambda}{N}, \min_i \mu_i\right) > 0
\]
so that
\[
\varepsilon |Q|_1
\le
(\Lambda-\lambda)Q_{\min} + \sum_{i=1}^N \mu_i \Delta_i.
\]
Therefore
\[
\mathcal LV(Q) \le -\varepsilon |Q|_1 + R.
\]
Outside the finite set \(\{Q : \varepsilon |Q|_1 \le R+1\}\), the drift is strictly negative. By the continuous-time Foster-Lyapunov criterion, the CTMC is positive Harris recurrent.
\end{proof}

\section{Unified Archimedean Softmax (UAS)}

We now introduce the heterogeneous extension used by the GibbsQ framework.

Define the UAS potential
\[
\Phi_i(Q,\mu)=\frac{Q_i+1}{\mu_i}-\frac{1}{\alpha}\log \mu_i.
\]

The corresponding routing law is
\begin{equation}\label{eq:uas_routing}
p_i(Q,\mu)=
\frac{\mu_i \exp\!\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}
{\sum_{j=1}^N \mu_j \exp\!\left(-\alpha \frac{Q_j+1}{\mu_j}\right)}.
\end{equation}

\begin{proposition}[UAS weighted variational form]
Let
\[
\Lambda=\sum_{i=1}^N \mu_i,
\qquad
r_i=\frac{\mu_i}{\Lambda},
\qquad
x_i(Q,\mu)=\frac{Q_i+1}{\mu_i}.
\]
Define the prior-weighted entropy-regularized objective
\[
\mathcal G_{\mathrm{UAS}}(p)
=
\sum_{i=1}^N p_i x_i(Q,\mu)
+
\frac{1}{\alpha}\mathrm{KL}(p\|r)
-\frac{1}{\alpha}\log \Lambda
\]
over \(\Delta_{N-1}\). Its unique minimizer is exactly the UAS routing law \eqref{eq:uas_routing}, and the minimum value is
\[
-\frac{1}{\alpha}\log\sum_{i=1}^N r_i e^{-\alpha x_i(Q,\mu)}.
\]
\end{proposition}

\begin{proof}
Since
\[
\mathrm{KL}(p\|r)=\sum_i p_i \log p_i - \sum_i p_i \log r_i
=
\sum_i p_i \log p_i - \sum_i p_i \log \mu_i + \log \Lambda,
\]
this objective is algebraically identical to the earlier UAS entropy-regularized objective.
The first-order optimality conditions on the simplex yield
\[
\log p_i = \text{constant} - \alpha x_i(Q,\mu) + \log r_i
=
\text{constant} - \alpha \frac{Q_i+1}{\mu_i} + \log \mu_i,
\]
which is equivalent to \eqref{eq:uas_routing} after normalization.
Substituting the minimizer back into the objective gives the stated weighted log-sum-exp value.
\end{proof}

\section{UAS Lyapunov Argument}

For the heterogeneous theorem path, use the weighted Lyapunov function
\[
V_{\mathrm{UAS}}(Q)=\frac12 \sum_{i=1}^N \frac{Q_i^2}{\mu_i}.
\]

\begin{lemma}[UAS drift identity]
Under the UAS routing law \eqref{eq:uas_routing}, the generator action on \(V_{\mathrm{UAS}}\) is
\begin{equation}\label{eq:uas_drift}
\mathcal LV_{\mathrm{UAS}}(Q)
=
\lambda \sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
-
\sum_i Q_i
+
\frac12 \sum_i \mathbf 1_{Q_i>0}.
\end{equation}
\end{lemma}

The prior-weighted variational structure of UAS controls the arrival term in \eqref{eq:uas_drift} through the service-rate prior \(r_i=\mu_i/\Lambda\).

\begin{lemma}[UAS arrival-term bound]
Let
\[
x_i(Q,\mu)=\frac{Q_i+1}{\mu_i},
\qquad
r_i=\frac{\mu_i}{\Lambda}.
\]
Then under the UAS routing law,
\begin{equation}\label{eq:uas_arrival_term_bound}
\sum_i p_i(Q,\mu)\frac{Q_i + 1/2}{\mu_i}
\le
\frac{|Q|_1+N}{\Lambda}.
\end{equation}
\end{lemma}

\begin{proof}
By the weighted variational identity,
\[
\sum_i p_i x_i(Q,\mu) + \frac{1}{\alpha}\mathrm{KL}(p\|r)
=
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}.
\]
Since \(\mathrm{KL}(p\|r)\ge 0\),
\[
\sum_i p_i x_i(Q,\mu)
\le
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}.
\]
Apply Jensen's inequality to the convex map \(u\mapsto e^{-\alpha u}\):
\[
\sum_i r_i e^{-\alpha x_i(Q,\mu)}
\ge
\exp\left(-\alpha\sum_i r_i x_i(Q,\mu)\right).
\]
Therefore
\[
-\frac{1}{\alpha}\log\sum_i r_i e^{-\alpha x_i(Q,\mu)}
\le
\sum_i r_i x_i(Q,\mu)
=
\frac{1}{\Lambda}\sum_i \mu_i \frac{Q_i+1}{\mu_i}
=
\frac{|Q|_1+N}{\Lambda}.
\]
Hence
\[
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i}\le \frac{|Q|_1+N}{\Lambda}.
\]
Finally,
\[
\sum_i p_i(Q,\mu)\frac{Q_i+1/2}{\mu_i}
=
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i}
-
\frac12\sum_i \frac{p_i(Q,\mu)}{\mu_i}
\le
\sum_i p_i(Q,\mu)\frac{Q_i+1}{\mu_i},
\]
which gives \eqref{eq:uas_arrival_term_bound}.
\end{proof}

\begin{theorem}[UAS positive Harris recurrence]
Assume \(\Lambda=\sum_i \mu_i > \lambda\). Under the UAS routing law \eqref{eq:uas_routing}, the CTMC is non-explosive, irreducible, and positive Harris recurrent.
\end{theorem}

\begin{proof}
Non-explosion and irreducibility follow exactly as in the raw-softmax case. The weighted Lyapunov function \(V_{\mathrm{UAS}}\) is norm-like on \(\mathbb Z_+^N\).

Define
\[
\varepsilon_{\mathrm{UAS}}=\frac{\Lambda-\lambda}{\Lambda}
\]
and
\[
R_{\mathrm{UAS}}=
\frac{\lambda N}{\Lambda}
+
\frac{N}{2}.
\]

Combining \eqref{eq:uas_drift} with \eqref{eq:uas_arrival_term_bound},
\[
\mathcal LV_{\mathrm{UAS}}(Q)
\le
\lambda\frac{|Q|_1+N}{\Lambda}
-
|Q|_1
+
\frac{N}{2}
=
-\varepsilon_{\mathrm{UAS}} |Q|_1 + R_{\mathrm{UAS}}.
\]
Therefore, for the compact set
\[
C_{\mathrm{UAS}}=
\left\{
Q \in \mathbb Z_+^N :
|Q|_1 \le \frac{R_{\mathrm{UAS}}+1}{\varepsilon_{\mathrm{UAS}}}
\right\},
\]
we have
\[
\mathcal LV_{\mathrm{UAS}}(Q)\le -1
\qquad \text{for all } Q \notin C_{\mathrm{UAS}}.
\]
By the continuous-time Foster-Lyapunov criterion, the CTMC is positive Harris recurrent.
\end{proof}

\section{Theory Layers And Applied Architecture}

The GibbsQ framework uses the following layered interpretation:

\textbf{1. Raw softmax baseline.}
This is the simplest theorem-backed entropy-regularized routing law and the cleanest baseline for the Foster-Lyapunov argument.

\textbf{2. UAS heterogeneous extension.}
This is the theorem-backed heterogeneous analytical extension used when queue pressure and service heterogeneity must both enter the routing law. Its proof uses the prior-weighted variational identity and the weighted drift inequality \eqref{eq:uas_arrival_term_bound}.

\textbf{3. N-GibbsQ.}
This is the learned neural policy trained by behavior cloning and REINFORCE against the theorem-backed analytical routing baselines. It is an empirical approximation layer rather than part of the theorem statements.

In this sense, `GibbsQ` denotes the framework, while raw softmax and UAS are the analytical policies and `N-GibbsQ` is the learned policy.

\end{document}
