<#
.SYNOPSIS
Execution script for GibbsQ experiments (Windows).

.DESCRIPTION
Wraps the invocation of the Python modules, sets up the PYTHONPATH,
and forwards arbitrary Hydra configuration overrides.

.PARAMETER Experiment
The name of the experiment to run. Valid options are: 'drift', 'sweep', 'policy'.

.PARAMETER HydraArgs
Extra arguments passed natively to the underlying Hydra application.

.EXAMPLE
.\run_experiment.ps1 drift
.\run_experiment.ps1 sweep system.num_servers=5 simulation.ssa.sim_time=5000
.\run_experiment.ps1 policy +simulation.export_trajectories=True
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false, Position=0)]
    [string]$Experiment,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraArgs = @()
)

$ErrorActionPreference = "Stop"

if (-not $Experiment) {
    Write-Host "Usage: .\run_experiment.ps1 <experiment> [hydra_args...]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available experiments:" -ForegroundColor Cyan
    Write-Host "  drift    - Run drift verification (drift_vs_norm, heatmap)"
    Write-Host "  sweep    - Run stability sweep across alpha and rho"
    Write-Host "  policy   - Run policy comparison (Softmax vs JSQ, etc.) [DEPRECATED -> corrected_policy]"
    Write-Host "  stress   - Run stress test (massive-N, critical load)"
    Write-Host "  train    - [DEPRECATED] Run DGA routing agent training"
    Write-Host "  fidelity - Run gradient survival check across horizons"
    Write-Host "  n_train  - [DEPRECATED] Run neural curriculum training"
    Write-Host "  parity   - [DEPRECATED] Run neural vs GibbsQ parity evaluation"
    Write-Host "  jacobian - [DEPRECATED] Run Jacobian AD vs finite-difference check"
    Write-Host "  stats    - Run 30-seed statistical significance benchmark"
    Write-Host "  generalize - Run generalization stress heatmap"
    Write-Host "  ablation - Run SSA-based component ablation study"
    Write-Host "  critical - Run critical stability boundary test"
    Write-Host "  reinforce_train - Run REINFORCE SSA training (Track 1)"
    Write-Host "  dr_train       - Run Domain Randomization training (Track 3)"
    Write-Host "  corrected_policy - Run corrected Tiered benchmark (Track 4)"
    Write-Host "  reinforce_check  - Run gradient validation (Track 5)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\run_experiment.ps1 drift"
    Write-Host "  .\run_experiment.ps1 sweep system.num_servers=5 simulation.ssa.sim_time=5000"
    Write-Host "  .\run_experiment.ps1 policy +simulation.export_trajectories=True"
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

$env:PYTHONPATH = $ProjectRoot

$PythonScript = ""
switch ($Experiment.ToLower()) {
    "drift"    { $PythonScript = "experiments.verification.drift_verification" }
    "sweep"    { $PythonScript = "experiments.sweeps.stability_sweep" }
    "policy"   { $PythonScript = "experiments.evaluation.policy_comparison" }
    "stress"   { $PythonScript = "experiments.testing.stress_test" }
    "train"    { $PythonScript = "experiments.training.train_dga" }
    "fidelity" { $PythonScript = "experiments.n_gibbsq.grad_check" }
    "n_train"  { $PythonScript = "experiments.n_gibbsq.train" }
    "parity"   { $PythonScript = "experiments.n_gibbsq.eval" }
    "jacobian" { $PythonScript = "experiments.n_gibbsq.jacobian_check" }
    "stats"    { $PythonScript = "experiments.n_gibbsq.stats_bench" }
    "generalize" { $PythonScript = "experiments.n_gibbsq.gen_sweep" }
    "ablation" { $PythonScript = "experiments.n_gibbsq.ablation_ssa" }
    "critical" { $PythonScript = "experiments.n_gibbsq.critical_load" }
    "bias"     { $PythonScript = "experiments.testing.verify_bias" }
    "reinforce_train" { $PythonScript = "experiments.n_gibbsq.train_reinforce" }
    "dr_train"       { $PythonScript = "experiments.n_gibbsq.train_domain_randomized" }
    "corrected_policy" { $PythonScript = "experiments.evaluation.corrected_policy_comparison" }
    "reinforce_check" { $PythonScript = "experiments.testing.reinforce_gradient_check" }
    "-h"       { Get-Help $MyInvocation.MyCommand.Definition; exit 0 }
    "help"     { Get-Help $MyInvocation.MyCommand.Definition; exit 0 }
    default    {
        Write-Host "Error: Unknown experiment '$Experiment'" -ForegroundColor Red
        Write-Host "Valid options are: drift, sweep, policy, stress, train, fidelity, n_train, parity, jacobian, stats, generalize, ablation, critical"
        exit 1
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting Experiment: $Experiment"
Write-Host " Remaining Args (Hydra Overrides): $($HydraArgs -join ' ')"
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $ProjectRoot

# Note: In PowerShell, pushing an array of strings to a native command
# correctly quotes and escapes them if necessary.
python -m $PythonScript @HydraArgs
