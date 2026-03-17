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

# Run gradient validation (Track 5)
./scripts/run_experiment.ps1 reinforce_check

# Run REINFORCE SSA training (Track 1)
./scripts/run_experiment.ps1 reinforce_train

# Run corrected policy benchmark (Track 4)
./scripts/run_experiment.ps1 corrected_policy
```

## Project Structure

```
GibbsQ/
├── configs/                  # Hydra YAML configurations
│   ├── default.yaml          # N=10 heterogeneous servers, ρ=0.8
│   └── small.yaml            # N=2 for quick validation
├── src/gibbsq/                 # Core library package
│   ├── core/                 # Config, policies, features (Track 2)
│   ├── engines/              # NumPy and JAX engines (SSA-based)
│   ├── analysis/             # Metrics and plotting (Gini, Sojourn)
│   └── utils/                # Exporting, logging, runtime setup
├── experiments/              # Hydra-driven experiment scripts
│   ├── n_gibbsq/
│   │   ├── train_reinforce.py # Track 1: REINFORCE SSA training
│   │   └── train_domain_randomized.py # Track 3: DR training
│   ├── testing/
│   │   └── reinforce_gradient_check.py # Track 5: Gradient validation
│   └── evaluation/
│       └── corrected_policy_comparison.py # Track 4: Tiered benchmarks
├── scripts/                  # Execution & utility scripts
│   ├── run_experiment.ps1     # Unified entry point (Windows)
│   └── run_paper_experiments.ps1 # Full paper reproduction pipeline
```

## Configuration

All experiments use [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
# Override α and simulation time
python -m experiments.verification.drift_verification system.alpha=5.0 simulation.ssa.sim_time=50000

# Use a different base config
python -m experiments.evaluation.corrected_policy_comparison --config-name small
```

## Key Theoretical Result

For $N$ parallel servers with Poisson arrivals at rate $\lambda$ and exponential service rates $\mu_i$, the softmax routing policy

$$p_i(Q) = \frac{\exp(-\alpha Q_i)}{\sum_j \exp(-\alpha Q_j)}$$

yields a positive Harris recurrent CTMC for any $\alpha > 0$, provided the strict capacity condition $\Lambda = \sum_i \mu_i > \lambda$ holds. The proof uses a quadratic Lyapunov function $V(Q) = \frac{1}{2}\|Q\|_2^2$ and the Gibbs free energy characterization of softmax to establish the Foster-Lyapunov drift condition.
