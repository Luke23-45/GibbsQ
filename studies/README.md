# studies/

This directory contains the thesis-grade experimental pipeline for the
`z2` program. It is organized into two layers.

## Directory Structure

```
studies/
├── runners/                          # Experiment execution
│   ├── run_all_z2.py                # Master runner: executes all z2 experiments
│   ├── run_verification.py          # Runs verification experiments only
│   ├── run_benchmarks.py            # Runs benchmark experiments only
│   └── run_neural_support.py        # Neural applicability studies (H7)
└── analysis/                         # Post-hoc analysis (modular)
    ├── __init__.py                  # Package root
    ├── common/                      # Shared utilities
    │   ├── visualization/           # Premium plotting & theme system
    │   ├── config.py               # YAML config loader (OmegaConf)
    │   ├── constants.py            # Hypothesis names, colors, metrics
    │   ├── io.py                   # CSV discovery, JSON/YAML loading
    │   ├── latex.py                # LaTeX table formatting
    │   ├── stats.py                # Bootstrap CIs, Cohen's d, p-values
    │   └── style.py                # Matplotlib thesis style system
    ├── configs/
    │   └── base.yaml               # Central configuration (paths, studies)
    ├── docs/
    │   └── final_thesis_asset_manifest.md
    ├── scripts/                     # CLI entry points
    │   ├── generate_all.py         # Master: tables -> figures -> stats
    │   ├── generate_figures.py     # All figures
    │   ├── generate_tables.py      # All LaTeX/MD tables
    │   └── generate_stats.py       # Statistical summary reports
    ├── regeneration/                # Artifact reproduction tools
    │   ├── ablation_regeneration.py
    │   ├── critical_regeneration.py
    │   ├── generalize_regeneration.py
    │   ├── policy_regeneration.py
    │   └── sweep_regeneration.py
    └── studies/                     # Per-hypothesis analysis modules
        ├── convergence/            # H1/H2: Reflected-ODE convergence
        ├── equilibrium/            # H2: Boundary equilibrium
        ├── drift/                  # H4: Exhaustive drift audit
        └── mismatch/               # H3: Boundary mismatch
```

## Usage

### Running Experiments

```bash
# Run all verification + benchmark experiments
python -m studies.runners.run_all_z2

# Run verification experiments only (H1-H4)
python -m studies.runners.run_verification

# Run benchmarks only (H5)
python -m studies.runners.run_benchmarks
```

### Generating Analysis Artifacts

```bash
# Generate everything (tables + figures + stats) in one shot
python -m analysis.scripts.generate_all

# Or run individually:
python -m analysis.scripts.generate_figures
python -m analysis.scripts.generate_tables
python -m analysis.scripts.generate_stats

# Legacy (still works, delegates to modular structure):
python -m studies.analysis.scripts.generate_figures
python -m studies.analysis.scripts.generate_tables
python -m studies.analysis.scripts.statistical_summary
```

## Design Principles

1. **Decoupled execution and analysis**: Experiments write CSV data only;
   figures and tables are generated in a separate step.
2. **Modular study structure**: Each hypothesis (H1-H5) has its own
   subpackage under `analysis/studies/` with dedicated `figures.py`
   and `table.py` modules - matching the BPGS/SPECTRA convention.
3. **Shared utilities**: All common functionality (I/O, styling, stats,
   LaTeX) lives in `analysis/common/` - imported by every study module.
4. **Central configuration**: A single `configs/base.yaml` defines all
   paths, study metadata, and output settings via OmegaConf.
5. **Atomic output**: Every experiment uses `ExperimentCSVWriter` with
   temp-file-and-rename for crash safety.
6. **Reproducibility**: Every CSV file has a sidecar `.meta.json` with
   timestamps, column descriptions, and experiment metadata.
7. **Idempotency**: Runners can be re-executed safely. Outputs have
   timestamped filenames to avoid overwriting.
8. **Error isolation**: Each experiment runs in its own try/except block.
   A single failure does not abort the pipeline.

## Data Flow

```
experiments/verification/*.py  ───►  outputs/data/*.csv
experiments/benchmark/*.py     ───►  outputs/data/*.csv
                                          │
                                          ▼
                           analysis/scripts/generate_all.py
                                          │
                      ┌────────────────────┴────────────────────┐
                      ▼                    ▼                    ▼
               studies/convergence  studies/equilibrium  studies/drift ...
                      │                    │                    │
                      ▼                    ▼                    ▼
               outputs/figures/     outputs/tables/      outputs/reports/
```
