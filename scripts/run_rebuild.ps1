# HaluRISC rebuild runner
# Usage:
#   pwsh -File scripts/run_rebuild.ps1                 # full run except Qwen
#   pwsh -File scripts/run_rebuild.ps1 -SkipJudge      # analysis only
#   pwsh -File scripts/run_rebuild.ps1 -RunQwen        # include local Qwen judge
#   pwsh -File scripts/run_rebuild.ps1 -Force          # rerun fixed-hp too
#
# Every step writes a log under artifacts/results/rebuild/logs/ and stops on
# the first non-zero exit code. The script never overwrites b2/b4/b6/review.

param(
    [switch]$SkipJudge,
    [switch]$RunQwen,
    [switch]$Force,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "artifacts\results\rebuild\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$Arguments
    )
    $log = Join-Path $logDir ("{0}.log" -f $Name)
    Write-Host ""
    Write-Host ("=== {0} ===" -f $Name) -ForegroundColor Cyan
    Write-Host ("{0} {1}" -f $Exe, ($Arguments -join " "))
    $start = Get-Date
    & $Exe @Arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}; see {2}" -f $Name, $LASTEXITCODE, $log)
    }
    $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    Write-Host ("=== {0} done in {1} min ===" -f $Name, $mins) -ForegroundColor Green
}

if (-not (Test-Path $py)) { throw "venv interpreter not found: $py" }

$fixedOut = Join-Path $root "artifacts\results\rebuild\fixed_hp\fixed_hp_ablation.csv"
if ($Force -or -not (Test-Path $fixedOut)) {
    Invoke-Step "wp1_fixed_hp" $py @(
        "src\models\run_b6_modified.py",
        "--fixed-params",
        "--results-dir", "artifacts\results\rebuild\fixed_hp",
        "--models-dir", "artifacts\models\rebuild\fixed_hp"
    )
} else {
    Write-Host "WP1 fixed-hp outputs exist; skipping (use -Force to rerun)." -ForegroundColor Yellow
}

if (Test-Path (Join-Path $root "src\models\rebuild_calibration.py")) {
    Invoke-Step "wp2_calibration" $py @("src\models\rebuild_calibration.py")
} else {
    Write-Warning "src\models\rebuild_calibration.py missing; skipping WP2."
}

if (Test-Path (Join-Path $root "src\models\rebuild_intervals.py")) {
    Invoke-Step "wp3_intervals" $py @("src\models\rebuild_intervals.py")
} else {
    Write-Warning "src\models\rebuild_intervals.py missing; skipping WP3."
}

if (Test-Path (Join-Path $root "src\models\rebuild_explanations.py")) {
    Invoke-Step "wp4_explanations" $py @(
        "src\models\rebuild_explanations.py",
        "--perturbation", "--n-perturb", "20"
    )
} else {
    Write-Warning "src\models\rebuild_explanations.py missing; skipping WP4."
}

if (Test-Path (Join-Path $root "src\models\rebuild_length_control.py")) {
    Invoke-Step "wp7_length_control" $py @("src\models\rebuild_length_control.py")
}
if (Test-Path (Join-Path $root "src\models\rebuild_lexical_audit.py")) {
    Invoke-Step "wp7_lexical_audit" $py @("src\models\rebuild_lexical_audit.py")
}

if (-not $SkipJudge) {
    if (Test-Path (Join-Path $root "src\models\rebuild_judge.py")) {
        Invoke-Step "wp6_judge_openai" $py @(
            "src\models\rebuild_judge.py",
            "--backend", "openai",
            "--limit", "0",
            "--repeats", "3"
        )
        Invoke-Step "wp6_judge_encoder_timing" $py @(
            "src\models\rebuild_judge.py",
            "--time-encoder"
        )
        Invoke-Step "wp6_judge_comparisons" $py @(
            "src\models\rebuild_judge.py",
            "--comparisons-only"
        )
        if ($RunQwen) {
            Invoke-Step "wp6_judge_qwen" $py @(
                "src\models\rebuild_judge.py",
                "--backend", "qwen",
                "--limit", "0",
                "--repeats", "1"
            )
        }
    } else {
        Write-Warning "src\models\rebuild_judge.py missing; skipping WP6."
    }
}

if (Test-Path (Join-Path $root "src\models\rebuild_paper_assets.py")) {
    Invoke-Step "wp8_paper_assets" $py @("src\models\rebuild_paper_assets.py")
} else {
    Write-Warning "src\models\rebuild_paper_assets.py missing; skipping asset generation."
}

if (Test-Path (Join-Path $root "src\models\rebuild_latency.py")) {
    Invoke-Step "wp8_latency" $py @("src\models\rebuild_latency.py", "--limit", "200")
}

Invoke-Step "tests_rebuild" $py @(
    "-m", "pytest",
    "tests\test_rebuild_calibration.py",
    "tests\test_b6_modified.py",
    "tests\test_api.py",
    "-q"
)

if (-not $SkipBuild) {
    $env:Path = "C:\texlive\2026\bin\windows;" + $env:Path
    Push-Location (Join-Path $root "Journal_Paper")
    try {
        Invoke-Step "latex_build" "latexmk" @(
            "-pdf", "-interaction=nonstopmode", "-outdir=out", "halurisc.tex"
        )
    } finally {
        Pop-Location
    }
}

if (Test-Path (Join-Path $root "src\models\review_number_audit.py")) {
    Invoke-Step "number_audit" $py @("src\models\review_number_audit.py")
}
if (Test-Path (Join-Path $root "src\models\rebuild_number_audit.py")) {
    Invoke-Step "rebuild_number_audit" $py @("src\models\rebuild_number_audit.py")
}

Write-Host ""
Write-Host "Rebuild run complete." -ForegroundColor Green
Write-Host ("Logs: {0}" -f $logDir)
Write-Host "Next: inspect artifacts/results/rebuild/*/REPORT.md and docs/rebuild/human_verification_checklist.md."
