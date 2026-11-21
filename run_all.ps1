# End-to-end pipeline for Windows PowerShell.   .\run_all.ps1          (full)
#                                               .\run_all.ps1 -Fast    (quick CPU smoke run)
param([switch]$Fast)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$f = if ($Fast) { "--fast" } else { "" }
$steps = @(
    "src.data.generate_dataset", "src.data.clean_split", "src.models.train_classical $f",
    "src.models.train_embedding_lr", "src.models.train_transformer $f", "src.evaluation.evaluate",
    "src.evaluation.error_analysis", "src.evaluation.visualize"
)
foreach ($s in $steps) {
    Write-Host "`n=== python -m $s ===" -ForegroundColor Cyan
    $parts = $s.Trim().Split(" ")
    python -m @parts
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $s" }
}
python -m pytest -q
Write-Host "`nDone. Launch the app with:  streamlit run app.py" -ForegroundColor Green
