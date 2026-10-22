param(
    [ValidateSet('smoke','study','prepare','test','compare','benchmark','controls')][string]$Mode='smoke',
    [string]$Run=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
. "$PSScriptRoot\project_env.ps1"
$ProjectPython="$PSScriptRoot\.venv\Scripts\python.exe"
switch ($Mode) {
    'test' { & $ProjectPython -m pytest -q --basetemp "$PSScriptRoot\.tmp\pytest" }
    'benchmark' { & $ProjectPython -m SaTP.benchmark }
    'prepare' { & $ProjectPython -m SaTP.run --config configs/study.yaml --prepare-only }
    'smoke' {
        & $ProjectPython scripts/archive_sources.py
        if ($LASTEXITCODE -ne 0) { throw 'Training source snapshot failed' }
        & $ProjectPython -m SaTP.run --config configs/smoke.yaml --no-tune
    }
    'study' {
        & $ProjectPython scripts/archive_sources.py
        if ($LASTEXITCODE -ne 0) { throw 'Training source snapshot failed' }
        & $ProjectPython -m SaTP.run --config configs/study.yaml
    }
    'controls' {
        if (-not $Run) { throw 'Pass -Run experiments/<run directory>' }
        & $ProjectPython scripts/run_ablations.py --run $Run
    }
    'compare' {
        if (-not $Run) { throw 'Pass -Run experiments/<run directory>' }
        & $ProjectPython -m Comparison.report $Run
        if ($LASTEXITCODE -ne 0) { throw 'Base comparison failed' }
        $ControlManifest=Join-Path $Run 'ablations_seed42/manifest.json'
        if ((Test-Path -LiteralPath $ControlManifest) -and ((Get-Content -LiteralPath $ControlManifest -Raw | ConvertFrom-Json).state -eq 'completed')) {
            & $ProjectPython scripts/run_ablations.py --run $Run --report-only
            if ($LASTEXITCODE -ne 0) { throw 'Controls report failed' }
            & $ProjectPython scripts/analyze_results.py --run $Run
            if ($LASTEXITCODE -ne 0) { throw 'Behavior analysis failed' }
            & $ProjectPython scripts/audit_attributes.py --run $Run
            if ($LASTEXITCODE -ne 0) { throw 'Attribute audit failed' }
            & $ProjectPython scripts/resource_report.py --run $Run
            if ($LASTEXITCODE -ne 0) { throw 'Resource report failed' }
            & $ProjectPython scripts/finalize_report.py --run $Run
            if ($LASTEXITCODE -ne 0) { throw 'Final report failed' }
            & $ProjectPython scripts/verify_completed_run.py --run $Run
        }
    }
}
if ($LASTEXITCODE -ne 0) { throw "Project command failed with exit code $LASTEXITCODE" }
