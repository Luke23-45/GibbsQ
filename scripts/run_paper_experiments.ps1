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

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunScript = Join-Path $ScriptDir "run_experiment.ps1"

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "  GibbsQ Research Paper: Final Execution Pipeline"
Write-Host "==========================================================" -ForegroundColor Magenta

Write-Host "`n[Initiating Pipeline...]`n" -ForegroundColor Cyan

# ---------------------------------------------------------
# Phase 0: Gradient Estimator Validation (Track 5)
# Must run BEFORE any training to ensure unbiased gradients.
# ---------------------------------------------------------
Write-Host "`n[0/10] Running REINFORCE Gradient validation (Track 5)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" reinforce_check $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Phase 1: Foundational Analytical Metrics & Verification
# ---------------------------------------------------------
Write-Host "`n[1/11] Running Drift Verification (Phase 1a)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" drift $($HydraArgs -join ' ')"

Write-Host "`n[2/10] Running Model Fidelity Check (Phase 1b)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" fidelity $($HydraArgs -join ' ')"

Write-Host "`n[4/10] Running Corrected Policy Evaluation Benchmark (Track 4)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" corrected_policy $($HydraArgs -join ' ')"

Write-Host "`n[5/10] Running Stability Sweep..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" sweep +experiment=stability_sweep $($HydraArgs -join ' ')"

Write-Host "`n[6/10] Running Scaling Stress Tests..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" stress ++jax.enabled=True $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Phase 2: N-GibbsQ Neural Learning Pipeline
# ---------------------------------------------------------
Write-Host "`n[7/10] Running REINFORCE SSA Training (Track 1)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" reinforce_train $($HydraArgs -join ' ')"

Write-Host "`n[8/10] Running Domain Randomization Training (Track 3)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" dr_train $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Phase 3: Generational Analysis & Ablation
# ---------------------------------------------------------
Write-Host "`n[10/10] Running Deep Component Ablation & Generational Generalization..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" stats $($HydraArgs -join ' ')"
Invoke-Expression "& `"$RunScript`" generalize $($HydraArgs -join ' ')"
Invoke-Expression "& `"$RunScript`" critical $($HydraArgs -join ' ')"
Invoke-Expression "& `"$RunScript`" ablation $($HydraArgs -join ' ')"

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Pipeline fully complete."
Write-Host "  Review '/outputs/' for your plots and logs."
Write-Host "==========================================================" -ForegroundColor Green
