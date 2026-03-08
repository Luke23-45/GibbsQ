<#
.SYNOPSIS
Multi-level verification script for the MoEQ project.

.DESCRIPTION
Automates the verification steps identified in the project blueprint.
Supports 'smoke' (fast) and 'trusted' (rigorous) validation levels.

.PARAMETER Level
The rigor level: 'smoke' (~2 mins) or 'trusted' (~20 mins).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false, Position=0)]
    [ValidateSet("smoke", "trusted", "full")]
    [string]$Level = "smoke"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunScript = Join-Path $ScriptDir "run_experiment.ps1"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MoEQ Automated Verification ($($Level.ToUpper()))"
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

Write-Host "[1/5] Running Step 1: Theoretical Drift Verification..." -ForegroundColor Yellow
& $RunScript drift system.num_servers=2 system.arrival_rate=2.0

Write-Host "`n[2/5] Running Step 2: Stability Sweep..." -ForegroundColor Yellow
& $RunScript sweep simulation.sim_time=$SimTime simulation.num_replications=$Reps simulation.burn_in_fraction=$BurnIn

Write-Host "`n[3/5] Running Step 3: Policy Comparison (Heterogeneous)..." -ForegroundColor Yellow
& $RunScript policy system.num_servers=4 system.service_rates="[1.0, 2.0, 5.0, 10.0]" simulation.sim_time=$PolicyTime simulation.num_replications=$Reps

if ($Level -eq "full") {
    Write-Host "`n[4/5] Running Step 4: Scientific Stress Test..." -ForegroundColor Yellow
    & $RunScript stress simulation.num_replications=2
    
    Write-Host "`n[5/5] Running Step 5: DGA Routing Agent Training..." -ForegroundColor Yellow
    & $RunScript train
} else {
    Write-Host "`n[Skip] Steps 4-5 (Stress & Train) require 'full' level." -ForegroundColor Gray
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Verification Complete!"
Write-Host "  Use 'python scripts/analyze_results.py' to see the summary."
Write-Host "==========================================================" -ForegroundColor Green
