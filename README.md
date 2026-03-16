# GibbsQ — Softmax-Routed Queueing Network Stability Verification

Empirical verification of the proof that softmax (Boltzmann) routing yields positive Harris recurrence for a system of $N$ parallel heterogeneous queues.


> **Hardware note:** The full default-config pipeline (especially the stress test with
> `stress.n_values` containing N ≥ 128, and the bias verification at ρ > 0.95) requires
> a CUDA-capable GPU (≥ 8 GB VRAM) for practical wall-clock runtime.
> For CPU-only validation use `--config-name small` or `debug=true`, which completes
> in under 5 minutes.
> The default config is calibrated for GPU execution and is required for paper results.

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v

# Run drift verification (N=2 grid)
python -m experiments.verification.drift_verification

# Run policy comparison
python -m experiments.evaluation.policy_comparison

# Run α/ρ stability sweep
python -m experiments.sweeps.stability_sweep
```

## Project Structure

```
GibbsQ/
├── configs/                  # Hydra YAML configurations
│   ├── default.yaml          # N=10 heterogeneous servers, ρ=0.8
│   └── small.yaml            # N=2 for quick validation
├── src/gibbsq/                 # Core library package
│   ├── core/                 # Config, policies, drift verification
│   ├── engines/              # NumPy, JAX, distributed, differentiable engines
│   ├── analysis/             # Metrics and plotting
│   └── utils/                # Exporting, logging, runtime setup
├── experiments/              # Hydra-driven experiment scripts
│   ├── verification/
│   │   └── drift_verification.py # Experiment 2: drift bound verification
│   ├── sweeps/
│   │   └── stability_sweep.py    # Experiment 1+3: α/ρ parameter sweep
│   ├── testing/
│   │   └── stress_test.py
│   ├── training/
│   │   └── train_dga.py
│   └── evaluation/
│       └── policy_comparison.py  # Experiment 4: baseline comparison
├── tests/                    # Unit tests
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py
│   ├── test_policies.py
│   ├── test_simulator.py
│   ├── test_drift.py
│   └── test_metrics.py
└── plan/                     # Research plan documents
    ├── idea/main_idea.md     # Proof specification (LaTeX)
    └── main/                 # Generated plan
        ├── 01_architecture.md
        ├── 02_implementation_plan.md
        ├── 03_experiment_design.md
        └── 04_open_questions.md
```

## Configuration

All experiments use [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
# Override α and simulation time
python -m experiments.verification.drift_verification system.alpha=5.0 simulation.ssa.sim_time=50000

# Use a different base config
python -m experiments.evaluation.policy_comparison --config-name small
```

## Key Theoretical Result

For $N$ parallel servers with Poisson arrivals at rate $\lambda$ and exponential service rates $\mu_i$, the softmax routing policy

$$p_i(Q) = \frac{\exp(-\alpha Q_i)}{\sum_j \exp(-\alpha Q_j)}$$

yields a positive Harris recurrent CTMC for any $\alpha > 0$, provided the strict capacity condition $\Lambda = \sum_i \mu_i > \lambda$ holds. The proof uses a quadratic Lyapunov function $V(Q) = \frac{1}{2}\|Q\|_2^2$ and the Gibbs free energy characterization of softmax to establish the Foster-Lyapunov drift condition.
