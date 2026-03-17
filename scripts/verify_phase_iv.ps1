<#
.SYNOPSIS
Surgical verification script for Phase IV Corrective Tracks.

.DESCRIPTION
This script specifically runs the new REINFORCE and SSA-based methodology 
tracks identified in the Phase IV corrective research plan (suggestions.md).
It ignores all deprecated DGA-based experiments.

TRACKS INCLUDED:
1. Track 5: Gradient Estimator Validation (reinforce_check)
2. Track 1: REINFORCE SSA Training (reinforce_train)
3. Track 3: Domain Randomization Training (dr_train)
4. Track 4: Corrected Policy Benchmark (corrected_policy)

.PARAMETER Config
The Hydra config name to use. Defaults to 'small' for fast verification.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false, Position=0)]
    [string]$Config = "small",

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$HydraArgs = @()
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunScript = Join-Path $ScriptDir "run_experiment.ps1"

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "  Phase IV Corrective Track Verification"
Write-Host "  Methodology: REINFORCE + True Gillespie SSA"
Write-Host "==========================================================" -ForegroundColor Magenta

Write-Host "`n[Configuration: $Config]`n" -ForegroundColor Cyan

# ---------------------------------------------------------
# Track 5: Gradient Estimator Validation
# ---------------------------------------------------------
Write-Host "[1/4] Track 5: Validating Gradient Estimator..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" reinforce_check --config-name $Config $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Track 1: REINFORCE SSA Training
# ---------------------------------------------------------
Write-Host "`n[2/4] Track 1: Training REINFORCE Agent (True SSA)..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" reinforce_train --config-name $Config $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Track 3: Domain Randomization
# ---------------------------------------------------------
Write-Host "`n[3/4] Track 3: Training with Domain Randomization..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" dr_train --config-name $Config $($HydraArgs -join ' ')"

# ---------------------------------------------------------
# Track 4: Corrected Policy Benchmark
# ---------------------------------------------------------
Write-Host "`n[4/4] Track 4: Running Corrected Tiered Benchmark..." -ForegroundColor Yellow
Invoke-Expression "& `"$RunScript`" corrected_policy --config-name $Config $($HydraArgs -join ' ')"

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Phase IV Verification Complete."
Write-Host "  Check '/outputs' for detailed logs and plots."
Write-Host "==========================================================" -ForegroundColor Green
