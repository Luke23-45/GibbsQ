# Sources Consulted

FILE READ
  Path      : scripts/execution/experiment_runner.py
  Purpose   : Defines the 12 experiment aliases and the module entry points they execute.
  Read at   : pass 1, setup

FILE READ
  Path      : scripts/execution/reproduction_pipeline.py
  Purpose   : Orchestrates the publication pipeline and injects experiment-specific Hydra overrides.
  Read at   : pass 1, setup

FILE READ
  Path      : scripts/verification/validation_suite.py
  Purpose   : Runs the verification workflows that call the shared experiment runner.
  Read at   : pass 1, setup

FILE READ
  Path      : docs/gibbsq.md
  Purpose   : States the raw-queue theorem and the separate UAS applied-policy formulation used for comparison against implementations.
  Read at   : pass 1, setup

FILE READ
  Path      : configs/default.yaml
  Purpose   : Supplies the default experiment configuration checked by the pre-flight validator.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : configs/small.yaml
  Purpose   : Supplies the small-profile configuration checked by the pre-flight validator.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : configs/large.yaml
  Purpose   : Supplies the large-profile configuration checked by the pre-flight validator.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : configs/drift.yaml
  Purpose   : Defines the drift experiment profile that is present in the repository but not covered by check_configs.py.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : experiments/testing/check_configs.py
  Purpose   : Implements the configuration validation loop for the check_configs experiment.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : src/gibbsq/core/config.py
  Purpose   : Converts Hydra configs to typed dataclasses and enforces structural and theoretical validation.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : experiments/testing/reinforce_gradient_check.py
  Purpose   : Implements the REINFORCE-versus-finite-difference validation experiment and its pass/fail criterion.
  Read at   : pass 1, experiment 2

FILE READ
  Path      : src/gibbsq/core/features.py
  Purpose   : Defines the look-ahead potential and softmax helper formulas used by the neural routing stack.
  Read at   : pass 1, experiment 2

FILE READ
  Path      : src/gibbsq/core/neural_policies.py
  Purpose   : Defines the neural router forward path and the adaptive-temperature helper used in evaluation wrappers.
  Read at   : pass 1, experiment 2

FILE READ
  Path      : src/gibbsq/engines/jax_ssa.py
  Purpose   : Generates batched SSA trajectories and computes the discounted action returns consumed by the gradient check.
  Read at   : pass 1, experiment 2

FILE READ
  Path      : experiments/verification/drift_verification.py
  Purpose   : Runs grid or trajectory-based Foster-Lyapunov drift checks and aborts on reported violations.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/core/drift.py
  Purpose   : Implements the scalar and vectorized exact drift and analytical bound computations used by the drift experiment.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/core/builders.py
  Purpose   : Builds policy instances for drift trajectory evaluation and imports the routing registry at runtime.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/core/registry.py
  Purpose   : Dispatches policy names to their concrete routing classes.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/analysis/metrics.py
  Purpose   : Provides steady-state and stationarity metrics shared by later evaluation experiments.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/utils/exporter.py
  Purpose   : Persists experiment metrics and trajectories to JSONL and Parquet.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : src/gibbsq/utils/logging.py
  Purpose   : Creates run capsules and optional WandB logging for every experiment.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : experiments/sweeps/stability_sweep.py
  Purpose   : Sweeps rho and alpha across JAX and NumPy simulation backends and logs stationarity outcomes.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : src/gibbsq/engines/jax_engine.py
  Purpose   : Implements the JAX SSA backend used by the stability sweep, stress test, and evaluation experiments.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : src/gibbsq/analysis/plotting.py
  Purpose   : Generates the sweep and evaluation charts, including the alpha-sweep line plot used here.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : src/gibbsq/analysis/chart_styles.py
  Purpose   : Defines chart-type-specific visual styles used by the plotting module.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : src/gibbsq/analysis/theme.py
  Purpose   : Applies the publication theme used by the experiment reporting layer.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : experiments/testing/stress_test.py
  Purpose   : Runs the large-N, critical-load, and heterogeneity stress scenarios on the JAX backend.
  Read at   : pass 1, experiment 5

FILE READ
  Path      : src/gibbsq/engines/distributed.py
  Purpose   : Wraps the vmapped JAX replication engine used by the stress-test experiment.
  Read at   : pass 1, experiment 5

FILE READ
  Path      : experiments/evaluation/baselines_comparison.py
  Purpose   : Compares analytical routing baselines and optional neural checkpoints, then assigns parity tiers.
  Read at   : pass 1, experiment 6

FILE READ
  Path      : experiments/training/pretrain_bc.py
  Purpose   : Runs behavior-cloning warm-start training and updates the global model pointer used by later evaluation experiments.
  Read at   : pass 1, experiment 7

FILE READ
  Path      : src/gibbsq/core/pretraining.py
  Purpose   : Collects expert UAS targets and trains the BC actor and value warm-start models.
  Read at   : pass 1, experiment 7

FILE READ
  Path      : experiments/training/train_reinforce.py
  Purpose   : Implements SSA trajectory collection, advantage construction, policy optimization, and REINFORCE checkpoint publication.
  Read at   : pass 1, experiment 8

FILE READ
  Path      : experiments/evaluation/n_gibbsq_evals/stats_bench.py
  Purpose   : Runs the statistical GibbsQ-vs-neural benchmark and performs the significance analysis.
  Read at   : pass 1, experiment 9

FILE READ
  Path      : experiments/evaluation/n_gibbsq_evals/gen_sweep.py
  Purpose   : Evaluates zero-shot neural generalization across service-rate scaling and load-factor grids.
  Read at   : pass 1, experiment 10

FILE READ
  Path      : experiments/evaluation/n_gibbsq_evals/ablation_ssa.py
  Purpose   : Trains and evaluates ablated neural variants on the SSA benchmark path.
  Read at   : pass 1, experiment 11

FILE READ
  Path      : experiments/evaluation/n_gibbsq_evals/critical_load.py
  Purpose   : Compares neural and analytical routing near the critical load boundary.
  Read at   : pass 1, experiment 12

FILE READ
  Path      : src/gibbsq/utils/chart_exporter.py
  Purpose   : Saves charts and attached experiment data in multi-format export workflows.
  Read at   : pass 1, experiments 9-12

FILE READ
  Path      : src/gibbsq/utils/device.py
  Purpose   : Configures JAX platform and precision before experiment execution.
  Read at   : pass 1, experiments 9-12

Each entry:
FILE READ
  Path      : [exact path]
  Purpose   : [one sentence]
  Read at   : [iteration #, experiment #]
