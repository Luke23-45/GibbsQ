# Common utilities for GibbsQ experiment scripts
# Reduces duplication between execution scripts

function Write-ExperimentHeader {
    param(
        [string]$Title,
        [string]$Color = "Magenta"
    )
    Write-Host "==========================================================" -ForegroundColor $Color
    Write-Host "  $Title" -ForegroundColor $Color
    Write-Host "==========================================================" -ForegroundColor $Color
}

function Invoke-Experiment {
    param(
        [string]$ExperimentName,
        [string]$Description,
        [string]$ScriptPath,
        [string[]]$HydraArgs = @()
    )
    
    Write-Host "`n[$ExperimentName] Running $Description..." -ForegroundColor Yellow
    $fullArgs = $HydraArgs -join ' '
    Invoke-Expression "& `"$ScriptPath`" $ExperimentName $fullArgs"
}

function Get-AvailableExperiments {
    return @(
        "drift    - Run drift verification (drift_vs_norm, heatmap)",
        "sweep    - Run stability sweep across alpha and rho", 
        "policy   - Run corrected policy comparison (Tiered benchmark)",
        "stress   - Run stress test (massive-N, critical load)",
        "stats    - Run 30-seed statistical significance benchmark",
        "generalize - Run generalization stress heatmap",
        "ablation - Run SSA-based component ablation study",
        "critical - Run critical stability boundary test",
        "reinforce_train - Run REINFORCE SSA training (Track 1)",
        "dr_train       - Run Domain Randomization training (Track 3)",
        "corrected_policy - Run corrected Tiered benchmark (Track 4)",
        "reinforce_check  - Run gradient validation (Track 5)"
    )
}
