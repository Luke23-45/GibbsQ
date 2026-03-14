<#
.SYNOPSIS
Multi-level verification script for the GibbsQ project.

.DESCRIPTION
Automates the verification steps identified in the implementation plan.
Supports 'debug' (fast) and 'trusted' (rigorous) validation levels.

.PARAMETER Level
The rigor level: 'debug' (~2 mins) or 'trusted' (~20 mins).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false, Position=0)]
    [ValidateSet("debug", "trusted", "full")]
    [string]$Level = "debug"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunScript = Join-Path $ScriptDir "run_experiment.ps1"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  GibbsQ Automated Verification ($($Level.ToUpper()))"
Write-Host "==========================================================" -ForegroundColor Cyan

# Define Parameters based on Level
$SimTime = 500
$Reps = 1
$BurnIn = 0.1
$PolicyTime = 500

if ($Level -eq "trusted") {
    $SimTime = 5000
    $Reps = 3
    $BurnIn = 0.2
    $PolicyTime = 5000
} elseif ($Level -eq "full") {
    $SimTime = 25000
    $Reps = 5
    $BurnIn = 0.2
    $PolicyTime = 10000
}

# Parameter Tuning based on Level
$StressReps = 1
$TrainEpochs = 1
if ($Level -eq "trusted") {
    $StressReps = 2
    $TrainEpochs = 5
} elseif ($Level -eq "full") {
    $StressReps = 5
    $TrainEpochs = 30
}

Write-Host "[1/5] Running Step 1: Theoretical Drift Verification..." -ForegroundColor Yellow
& $RunScript drift system.num_servers=2 system.arrival_rate=2.0

Write-Host "`n[2/5] Running Step 2: Stability Sweep..." -ForegroundColor Yellow
& $RunScript sweep simulation.ssa.sim_time=$SimTime simulation.num_replications=$Reps simulation.burn_in_fraction=$BurnIn

Write-Host "`n[3/5] Running Step 3: Policy Comparison (Heterogeneous)..." -ForegroundColor Yellow
& $RunScript policy system.num_servers=4 system.service_rates="[1.0, 2.0, 5.0, 10.0]" simulation.ssa.sim_time=$PolicyTime simulation.num_replications=$Reps

Write-Host "`n[4/5] Running Step 4: Stress Test..." -ForegroundColor Yellow
& $RunScript stress simulation.num_replications=$StressReps "debug=$($Level -ne "full")" jax.enabled=true

Write-Host "`n[5/5] Running Step 5: DGA Routing Agent Training..." -ForegroundColor Yellow
& $RunScript train train_epochs=$TrainEpochs

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Verification Complete!"
Write-Host "  Use 'python tools/analyze.py' to see the summary."
Write-Host "==========================================================" -ForegroundColor Green
