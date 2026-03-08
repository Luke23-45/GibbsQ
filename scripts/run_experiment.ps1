<#
.SYNOPSIS
Robust execution script for MoEQ experiments (Windows).

.DESCRIPTION
Wraps the invocation of the Python modules, sets up the PYTHONPATH,
and forwards arbitrary Hydra configuration overrides.

.PARAMETER Experiment
The name of the experiment to run. Valid options are: 'drift', 'sweep', 'policy'.

.PARAMETER HydraArgs
Extra arguments passed natively to the underlying Hydra application.

.EXAMPLE
.\run_experiment.ps1 drift
.\run_experiment.ps1 sweep system.num_servers=5 simulation.sim_time=5000
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
    Write-Host "  policy   - Run policy comparison (Softmax vs JSQ, etc.)"
    Write-Host "  stress   - Run scientific stress test (massive-N, critical load)"
    Write-Host "  train    - Run DGA routing agent training (gradient descent)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\run_experiment.ps1 drift"
    Write-Host "  .\run_experiment.ps1 sweep system.num_servers=5 simulation.sim_time=5000"
    Write-Host "  .\run_experiment.ps1 policy +simulation.export_trajectories=True"
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

$env:PYTHONPATH = $ProjectRoot

$PythonScript = ""
switch ($Experiment.ToLower()) {
    "drift"  { $PythonScript = "experiments.drift_verification" }
    "sweep"  { $PythonScript = "experiments.stability_sweep" }
    "policy" { $PythonScript = "experiments.policy_comparison" }
    "stress" { $PythonScript = "experiments.scientific_stress_test" }
    "train"  { $PythonScript = "experiments.train_routing_agent" }
    "-h"     { Get-Help $MyInvocation.MyCommand.Definition; exit 0 }
    "help"   { Get-Help $MyInvocation.MyCommand.Definition; exit 0 }
    default  {
        Write-Host "Error: Unknown experiment '$Experiment'" -ForegroundColor Red
        Write-Host "Valid options are: drift, sweep, policy, stress, train"
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
