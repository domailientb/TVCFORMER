param(
    [string]$Config = (Join-Path $PSScriptRoot "configs\smoke_etth1.yaml"),
    [int]$Clusters = 3,
    [double]$CrossEdgeRatio = 0.1,
    [int]$Seed = 2021,
    [int]$BatchSize = 4,
    [int]$TrainBatches = 1,
    [int]$EvalBatches = 1
)

$python = Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $command = Get-Command python -ErrorAction Stop
    $python = $command.Source
}
$script = Join-Path $PSScriptRoot "exp_clustering.py"
& $python $script --config_yaml $Config --n_clusters $Clusters `
    --cross_edge_ratio $CrossEdgeRatio --random_seed $Seed `
    --batch_size $BatchSize --max_train_batches $TrainBatches `
    --max_eval_batches $EvalBatches
exit $LASTEXITCODE
