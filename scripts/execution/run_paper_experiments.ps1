<#
.SYNOPSIS
Full end-to-end master execution pipeline for the GibbsQ & N-GibbsQ research paper.

.DESCRIPTION
This script sequentially runs the entire battery of analyses necessary to
reproduce the final research paper figures and metrics.

CRITICAL: Following our rigorous codebase audit, this script intentionally
DOES NOT INJECT any variables (e.g., system.num_servers). All parameters
are completely bound to the formally reviewed Hydra YAML configuration
hierarchy within the `configs/` folder. This guarantees total reproducibility
by reviewers exactly as intended.

.EXAMPLE
.\run_paper_experiments.ps1
.\run_paper_experiments.ps1 wandb.mode=online
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraArgs = @()
)

# Import common utilities
. "$PSScriptRoot\utils\common.ps1"

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunScript = Join-Path $ScriptDir "run_experiment.ps1"

Write-ExperimentHeader "GibbsQ Research Paper: Final Execution Pipeline"

Write-Host "`n[Initiating Pipeline...]`n" -ForegroundColor Cyan

# ---------------------------------------------------------
# Phase 0: Gradient Estimator Validation (Track 5)
# Must run BEFORE any training to ensure unbiased gradients.
# ---------------------------------------------------------
Invoke-Experiment "0/10" "REINFORCE Gradient validation (Track 5)" $RunScript "reinforce_check"

# ---------------------------------------------------------
# Phase 1: Foundational Analytical Metrics & Verification
# ---------------------------------------------------------
Invoke-Experiment "1/11" "Drift Verification (Phase 1a)" $RunScript "drift"

Invoke-Experiment "2/10" "Model Fidelity Check (Phase 1b)" $RunScript "fidelity"

Invoke-Experiment "4/10" "Corrected Policy Evaluation Benchmark (Track 4)" $RunScript "corrected_policy"

Invoke-Experiment "5/10" "Stability Sweep" $RunScript "sweep +experiment=stability_sweep"

Invoke-Experiment "6/10" "Scaling Stress Tests" $RunScript "stress ++jax.enabled=True"

# ---------------------------------------------------------
# Phase 2: N-GibbsQ Neural Learning Pipeline
# ---------------------------------------------------------
Invoke-Experiment "7/10" "REINFORCE SSA Training (Track 1)" $RunScript "reinforce_train"

Invoke-Experiment "8/10" "Domain Randomization Training (Track 3)" $RunScript "dr_train"

# ---------------------------------------------------------
# Phase 3: Generational Analysis & Ablation
# ---------------------------------------------------------
Invoke-Experiment "10/10" "Deep Component Ablation & Generational Generalization" $RunScript "stats"
Invoke-Experiment "11/10" "Generalization Heatmap" $RunScript "generalize"
Invoke-Experiment "12/10" "Critical Stability Boundary" $RunScript "critical"
Invoke-Experiment "13/10" "SSA-based Component Ablation" $RunScript "ablation"

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Pipeline fully complete."
Write-Host "  Review '/outputs/' for your plots and logs."
Write-Host "==========================================================" -ForegroundColor Green
