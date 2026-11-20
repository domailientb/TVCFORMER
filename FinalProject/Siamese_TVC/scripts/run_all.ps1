param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenchmarkArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$benchmark = Join-Path $repoRoot 'FinalProject\Siamese_TVC\benchmark.py'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Không tìm thấy Python của môi trường repo: $python"
}

& $python $benchmark --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather @BenchmarkArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
