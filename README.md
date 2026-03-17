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
python scripts/execution/experiment_runner.py reinforce_check

# Run REINFORCE SSA training (Track 1)
python scripts/execution/experiment_runner.py reinforce_train

# Run corrected policy benchmark (Track 4)
python scripts/execution/experiment_runner.py policy
```

## Project Structure

```
GibbsQ/
├── configs/                  # Hydra YAML configurations
├── src/gibbsq/                 # Core library package
│   ├── core/
│   ├── engines/
│   │   └── deprecated/       # Quarantine for differentiable_engine.py
│   ├── analysis/
│   └── utils/
├── experiments/              # Hydra-driven research drivers
│   ├── training/             # Learning routines (REINFORCE, DR)
│   ├── evaluation/           # Master benchmarks
│   │   ├── baselines_comparison.py
│   │   └── n_gibbsq_evals/   # Track-specific deep dives (ablation, etc.)
│   ├── sweeps/               # Parameter explorations
│   ├── testing/              # Code & Gradient validations
│   └── verification/         # Theoretical drift checks
└── scripts/                  # Professional execution suite
    ├── execution/
    │   ├── experiment_runner.py     # Unified task engine
    │   └── reproduction_pipeline.py # Full paper reproduction
    └── verification/
        └── validation_suite.py     # Implementation verification
```

## Configuration

All experiments use [Hydra](https://hydra.cc/) for configuration management. Override any parameter from the command line:

```bash
# Override α and simulation time
python scripts/execution/experiment_runner.py drift system.alpha=5.0 simulation.ssa.sim_time=50000

# Use a different base config
python scripts/execution/experiment_runner.py policy --config-name small
```

## Key Theoretical Result

For $N$ parallel servers with Poisson arrivals at rate $\lambda$ and exponential service rates $\mu_i$, the softmax routing policy

$$p_i(Q) = \frac{\exp(-\alpha Q_i)}{\sum_j \exp(-\alpha Q_j)}$$

yields a positive Harris recurrent CTMC for any $\alpha > 0$, provided the strict capacity condition $\Lambda = \sum_i \mu_i > \lambda$ holds. The proof uses a quadratic Lyapunov function $V(Q) = \frac{1}{2}\|Q\|_2^2$ and the Gibbs free energy characterization of softmax to establish the Foster-Lyapunov drift condition.
