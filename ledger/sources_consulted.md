# Sources Consulted

FILE READ
  Path      : scripts/verify_phase_iv.ps1
  Purpose   : Defines the Phase IV verification sequence and track names.
  Read at   : pass 1, inventory

FILE READ
  Path      : scripts/run_experiment.ps1
  Purpose   : Maps experiment aliases to Python modules for Windows launcher.
  Read at   : pass 1, inventory

FILE READ
  Path      : scripts/run_experiment.sh
  Purpose   : Maps experiment aliases to Python modules for Linux/macOS launcher.
  Read at   : pass 1, inventory

FILE READ
  Path      : experiments/testing/reinforce_gradient_check.py
  Purpose   : Track 5 gradient estimator validation entrypoint and logic.
  Read at   : pass 1, experiment 1

FILE READ
  Path      : experiments/n_gibbsq/train_reinforce.py
  Purpose   : Track 1 REINFORCE training and weight pointer emission.
  Read at   : pass 1, experiment 2

FILE READ
  Path      : experiments/n_gibbsq/train_domain_randomized.py
  Purpose   : Track 3 domain-randomized training and pointer emission.
  Read at   : pass 1, experiment 3

FILE READ
  Path      : experiments/evaluation/corrected_policy_comparison.py
  Purpose   : Track 4 policy benchmarking and neural weight pointer consumption.
  Read at   : pass 1, experiment 4

FILE READ
  Path      : src/gibbsq/core/config.py
  Purpose   : Typed config schema, output_dir definition, hydra conversion.
  Read at   : pass 1, shared config

FILE READ
  Path      : configs/default.yaml
  Purpose   : Default output_dir and simulation settings.
  Read at   : pass 1, shared config

FILE READ
  Path      : configs/small.yaml
  Purpose   : Small verification config with output_dir=outputs/small.
  Read at   : pass 1, shared config

FILE READ
  Path      : src/gibbsq/utils/logging.py
  Purpose   : run_dir/run_id resolution for experiment outputs.
  Read at   : pass 1, shared output handling

FILE READ
  Path      : pyproject.toml
  Purpose   : Package layout (src-based) establishing import path requirements.
  Read at   : pass 1, cross-check

FILE READ
  Path      : src/gibbsq/utils/exporter.py
  Purpose   : Metrics persistence utility used by corrected policy comparison.
  Read at   : pass 1, shared output handling

FILE READ
  Path      : src/gibbsq/core/neural_policies.py
  Purpose   : NeuralRouter model dimensions used in compatibility checks.
  Read at   : pass 1, dependency
