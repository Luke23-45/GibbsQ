# GibbsQ — Entropy-Regularized Queue Routing And Neural Approximation

GibbsQ is a research codebase for entropy-regularized routing in parallel queueing systems. It distinguishes three layers:

- `SoftmaxRouting`: the theorem-backed raw queue softmax baseline
- `UASRouting`: the theorem-backed heterogeneous Unified Archimedean Softmax extension
- `N-GibbsQ`: the learned neural policy trained and benchmarked against those analytical baselines


> **Hardware note:** The full default-config pipeline (especially the stress test with
> `stress.n_values` containing N ≥ 128, and the bias verification at ρ > 0.95) requires
> a CUDA-capable GPU (≥ 8 GB VRAM) for practical wall-clock runtime.
> For the fastest smoke checks use `--config-name debug`.
> For CPU-only validation use `--config-name small`, which completes
> in under 5 minutes.
> The default config is calibrated for GPU execution and is required for paper results.

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the public proof/publication verification suite
pytest tests/ -q

# Run the broader internal engineering regression suite
pytest development_tests/ -q

# Run gradient validation (Track 5)
python scripts/execution/experiment_runner.py reinforce_check

# Run REINFORCE SSA training (Track 1)
python scripts/execution/experiment_runner.py reinforce_train

# Run corrected policy benchmark (Track 4)
python scripts/execution/experiment_runner.py policy

python scripts/execution/reproduction_pipeline.py --config-name debug

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

All experiments use [Hydra](https://hydra.cc/) for configuration management.

The project now uses self-contained profile configs:
- `configs/debug.yaml`
- `configs/small.yaml`
- `configs/default.yaml`
- `configs/final_experiment.yaml`

Each profile file contains:
- shared top-level defaults
- an `experiments:` section with one block for each public experiment

At runtime the effective config is resolved as:

`selected profile root + experiments.<experiment_name> block + CLI overrides`

This means:
- `debug` is the smoke/test profile
- `small` is the CPU validation profile
- `default` is the research baseline profile
- `final_experiment` is the publication profile
- experiments do not rely on runtime `+experiment=...` overlays anymore

Override any parameter from the command line:

```bash
# Override α and simulation time
python scripts/execution/experiment_runner.py drift system.alpha=5.0 simulation.ssa.sim_time=50000

# Use a different base config
python scripts/execution/experiment_runner.py policy --config-name small
```

## Key Theoretical Results

For $N$ parallel servers with Poisson arrivals at rate $\lambda$ and exponential service rates $\mu_i$, the raw softmax routing policy

$$p_i(Q) = \frac{\exp(-\alpha Q_i)}{\sum_j \exp(-\alpha Q_j)}$$

yields a positive Harris recurrent CTMC for any $\alpha > 0$, provided the strict load condition $\Lambda = \sum_i \mu_i > \lambda$ holds. The proof uses a quadratic Lyapunov function and an entropy-regularized variational bound to establish the Foster-Lyapunov drift condition.

The framework also includes the heterogeneous UAS routing policy

$$p_i(Q,\mu)=\frac{\mu_i \exp\left(-\alpha \frac{Q_i+1}{\mu_i}\right)}{\sum_j \mu_j \exp\left(-\alpha \frac{Q_j+1}{\mu_j}\right)},$$

for which the repo gives a prior-weighted entropy-regularized variational derivation, an exact weighted drift identity, and a Foster-Lyapunov closure proving positive Harris recurrence under the same load condition. `N-GibbsQ` is the empirical neural layer trained against these theorem-backed analytical baselines.

## Testing

The published `tests/` directory is intentionally small and focused on the paper claims:

- raw softmax drift proof checks
- UAS weighted-Jensen drift closure checks
- theorem-consumer constant checks
- publication-run release-safety checks

The broader engineering regression suite lives in `development_tests/` for internal development and maintenance.
