# Config Notes

This folder is the documentation and audit surface for the active four-profile
configuration system:

- `debug`
- `small`
- `default`
- `final_experiment`

The YAML source files live in `configs/`. The CSV files in this folder are
derived audit artifacts.

## Source Of Truth

The project does **not** run directly from raw YAML text alone. The effective
configuration surface is:

`profile YAML root -> runtime normalization in config.py -> schema defaults injected by config.py -> experiments.<name> override block -> CLI overrides`

That means a parameter can be active at runtime even when it is omitted from a
profile YAML file. Examples include schema-defaulted fields such as:

- `stability_sweep.*` when absent from a root profile
- `neural.capacity_bound`
- `wandb.entity`
- `wandb.group`
- `wandb.tags`

Additional runtime-normalized behavior to remember:

- `system.service_rates: [x]` with `system.num_servers > 1` is expanded by
  `_normalize_service_rates()` to `[x, x, ..., x]`
- this is runtime shorthand handling, not literal YAML inheritance
- current shipped profiles do not rely on it, but future audits must account
  for it if a shorthand profile is introduced

Important exception:

- `domain_randomization.phases` is **not** inherited by omission
- when YAML omits `phases`, `_build_dr_config()` in
  `src/gibbsq/core/config.py` intentionally sets `phases=[]`
- this suppresses the dataclass default curriculum and is a deliberate runtime
  behavior, not an accident
- this omission rule applies both to root profile loading and to experiment-
  resolved configs after override blocks are merged

Because of that, the strongest source of truth is:

1. `src/gibbsq/core/config.py`
2. the profile YAMLs in `configs/`
3. the experiment override blocks in `experiments.*`
4. CLI overrides used for a given run

This README is explanatory only.

## What The CSVs Now Mean

The generated CSV artifacts in this folder now follow runtime semantics:

- root-profile rows are built from the resolved runtime root config
- explicit `experiments.*` rows are built from the literal YAML override blocks
- `N/A` means "no explicit override for this profile surface", not "the runtime
  has no value"

So the CSVs are now suitable for audit work on active root parameters plus
explicit experiment-specific overrides.

## Profile Roles

- `debug`
  - smoke/test profile
  - shortest horizon and smallest budget
  - may enable JAX for quick execution
  - not paper-facing
- `small`
  - low-cost CPU validation profile
  - lightweight but still scientifically shaped
- `default`
  - research baseline
  - public runner defaults here
- `final_experiment`
  - publication profile
  - highest rigor and artifact retention

## Parameter Classes

Every parameter recorded in `parameter_freeze_ledger.csv` is classified into one
of five classes.

### `Invariant`

Definition:

- should stay fixed across profiles unless the scientific or implementation
  contract changes

Typical examples:

- `system.alpha`
- `policy.name`
- `policy.d`
- `verification.parity_threshold_percent`
- `verification.jacobian_rel_tol`
- `verification.alpha_significance`
- `verification.confidence_interval`
- `verification.stationarity_threshold`
- `verification.parity_z_score`
- `jax_engine.*`
- `simulation.seed`
- `simulation.burn_in_fraction`
- `neural.use_rho`
- `neural.use_service_rates`

### `Profile-Scaled`

Definition:

- changes that mainly represent runtime budget, rigor, or coverage

Typical examples:

- `simulation.num_replications`
- `simulation.ssa.sim_time`
- `simulation.dga.sim_steps`
- `train_epochs`
- `batch_size`
- `stress.n_values`
- `stress.critical_rhos`
- `stress.critical_load_max_sim_time`
- `generalization.*`
- `stability_sweep.*`
- `neural_training.eval_*`
- `neural_training.bc_num_steps`
- `verification.gradient_check_*` budget fields
- `drift.q_max`

Rule:

- these should usually scale upward from
  `debug -> small -> default -> final_experiment`

### `Workload-Defining`

Definition:

- changes that alter the modeled system, neural regime, feature contract, or
  training regime itself

Typical examples:

- `system.num_servers`
- `system.arrival_rate`
- `system.service_rates`
- `neural.hidden_size`
- `neural.preprocessing`
- `neural.init_type`
- `neural.actor_lr`
- `neural.critic_lr`
- `neural.rho_input_scale`
- `domain_randomization.*`

### `Experiment-Specific Override`

Definition:

- value belongs only to a specific experiment block and should not silently
  become the root default

Typical examples:

- `experiments.drift.system.*`
- `experiments.drift.simulation.*`
- `experiments.drift.neural.*`
- `experiments.policy.jax.*`
- `experiments.sweep.wandb.*`
- `experiments.stats.simulation.num_replications`

### `Metadata/Output`

Definition:

- operational/logging/output fields rather than scientific knobs

Typical examples:

- `output_dir`
- `log_dir`
- `wandb.project`
- `wandb.entity`
- `wandb.group`
- `wandb.tags`
- `wandb.mode`
- `wandb.run_name`

## Current High-Sensitivity Areas

### `debug` vs `small`

These profiles share the same queueing system:

- `system.num_servers = 2`
- `system.arrival_rate = 1.0`
- `system.service_rates = [1.0, 1.5]`

They do **not** share the same full training regime. Important root
differences include:

- `neural.hidden_size`
- `neural.actor_lr`
- `neural.critic_lr`
- `domain_randomization.rho_max`
- `jax.enabled`

So `debug` is not merely a smaller `small`; it is also a different optimization
regime.

### `default` vs `final_experiment`

These profiles share the same queueing system:

- `system.num_servers = 10`
- `system.arrival_rate = 11.2`
- same 10-service-rate vector

They are not pure budget variants. Important root differences include:

- `neural.hidden_size`
- `neural.preprocessing`
- `neural.init_type`
- `neural.actor_lr`
- `domain_randomization.enabled`
- `domain_randomization.rho_min`
- `jax.enabled`

So comparisons between them are not pure "same regime, more budget" claims.

### `experiments.drift.*`

This is still the most important override family.

In `default`, `experiments.drift.*` intentionally injects a theorem-oriented
heavy-traffic setup that differs from the baseline root profile, including:

- a 2-server system
- longer horizons
- stronger `q_max`
- drift-specific stress/generalization grids
- drift-specific neural/domain-randomization settings
- separate output locations

That is legitimate only if drift is intentionally a different experiment
contract.

## Validation Reality

Do not assume the README or CSVs are enough by themselves. Performance-critical
review must also inspect validator coverage.

`validate()` in `src/gibbsq/core/config.py` now guards not only the old core
fields but also high-impact budget and correctness knobs such as:

- root `train_epochs`
- root `batch_size`
- `neural_training.curriculum`
- `neural_training.eval_batches`
- `neural_training.eval_trajs_per_batch`
- `neural_training.bc_num_steps`
- `neural_training.checkpoint_freq`
- `stress.n_values`
- `stress.critical_load_n`
- `stress.critical_load_max_sim_time`
- `verification.parity_z_score`
- `verification.gradient_check_n_test`
- `verification.gradient_check_hidden_size`
- `verification.gradient_check_error_threshold`
- `wandb.mode`
- domain-randomization phase `horizon`

If a parameter can materially change runtime cost or scientific meaning, it
should either be validated or be explicitly reviewed as a deliberate escape
hatch.

## Audit Checklist

Use this loop before expensive runs:

1. Read `src/gibbsq/core/config.py`.
   - confirm the parameter exists in the runtime schema
   - confirm validator constraints and special runtime semantics
2. Read the profile YAMLs in `configs/`.
   - confirm explicit root values
   - confirm where YAML omission leads to schema defaults
   - confirm where omission has custom semantics such as
     `domain_randomization.phases`
3. Read the relevant `experiments.<name>` block.
   - confirm whether a specific experiment override exists
   - confirm the override is semantically justified
4. Regenerate and review the CSV artifacts in `configs/notes/`.
   - confirm the ledger classification matches the live runtime surface
   - confirm profile-scaled parameters still scale sensibly
5. Decide one of four outcomes:
   - `keep fixed`
   - `keep variable by profile`
   - `keep experiment-specific override`
   - `needs redesign`

Review families in this order:

- `system.*`
- `simulation.*`
- `policy.*`
- `drift.*`
- `stress.*`
- `generalization.*`
- `stability_sweep.*`
- `neural.*`
- `neural_training.*`
- `domain_randomization.*`
- `jax.*`
- `jax_engine.*`
- `verification.*`
- `wandb.*`
- root output/training fields
- `experiments.*`

## Current Conclusion

The old notes were directionally useful but not fully rigorous enough for a
performance-critical project because they treated raw YAML coverage as if it
were full runtime coverage.

The corrected interpretation is:

- the fixed/variable framework is still useful
- runtime config is determined by YAML plus schema/default resolution plus
  experiment overrides plus CLI overrides
- `debug` vs `small` and `default` vs `final_experiment` are not pure
  budget-only splits
- the largest hidden experiment-specific surface remains `experiments.drift.*`
- omission in YAML must never be assumed to mean "inactive" without checking
  runtime construction code

## CSV Files In This Folder

- `config_comparison_comprehensive.csv`
  - resolved root parameters plus explicit experiment override rows
- `config_comparison_summary.csv`
  - only parameters whose values differ
- `config_comparison_by_category.csv`
  - same data grouped by top-level family
- `parameter_freeze_ledger.csv`
  - parameter classification, meaning, rule, and per-profile values

Generated by:

- `generate_config_csv.py`

Verified by:

- `verify_csv_accuracy.py`

## Practical Editing Rules

1. Edit YAML source files in `configs/`.
2. Re-run CSV generation in `configs/notes/`.
3. Re-run CSV verification.
4. Re-run config validation and tests.
5. Review `parameter_freeze_ledger.csv` to confirm the intended class still
   matches the runtime behavior.

If a parameter is marked `Invariant`, treat changes as high-risk.
If a parameter is marked `Profile-Scaled`, check profile ordering.
If a parameter is marked `Workload-Defining`, document why the workload itself
must differ.
If a parameter is marked `Experiment-Specific Override`, confirm it belongs to
that experiment and is not masking a root-profile inconsistency.
