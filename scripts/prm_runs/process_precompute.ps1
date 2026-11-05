# Precompute one roadmap per bar of an exported process folder: the Windows
# twin of process_precompute.sh. See
# that file for what the roadmap holds and what the stages are.
#
# Usage, from the repository root:
#   scripts\prm_runs\process_precompute.ps1 -Problem 261008_2ModulesDemo_purged_process -Bars B21,B22
# Launch detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\process_precompute.ps1','-Problem','261008_2ModulesDemo_purged_process','-Bars','B21'
#
# Needs the environment variables HUSKY_BENCH_ROOT (the benchmark data root)
# and HUSKY_DESIGN_STUDY_DIR (the folder holding the exported problems), the
# trimmed scenes (scripts\benchmark_cdfm.py build-cells <problem folder> --bars B21)
# and the carry starts file <root>\prm\carry\carry_v1\starts.json. Optional:
# PYTHON (the interpreter; default: the repo's venv), WORKERS, NODES, TRACK_MM
# (0 skips the tracking, for a quick smoke run), STARTS.
param(
    [Parameter(Mandatory = $true)][string]$Problem,
    [Parameter(Mandatory = $true)][string[]]$Bars,
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 8 }),
    [int]$Nodes = $(if ($env:NODES) { [int]$env:NODES } else { 6000 }),
    [int]$TrackMm = $(if ($env:TRACK_MM) { [int]$env:TRACK_MM } else { 15 }),
    [string]$Starts = $(if ($env:STARTS) { $env:STARTS } else { "carry/carry_v1/starts.json" })
)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = if ($env:PYTHON) { $env:PYTHON } else { Join-Path $repo ".venv\Scripts\python.exe" }
if (-not $env:HUSKY_DESIGN_STUDY_DIR) { throw "set HUSKY_DESIGN_STUDY_DIR to the folder holding the exported problems" }
if (-not $env:HUSKY_BENCH_ROOT) { throw "set HUSKY_BENCH_ROOT to the benchmark data root" }
$problemDir = Join-Path $env:HUSKY_DESIGN_STUDY_DIR $Problem

$commands = @()
foreach ($bar in $Bars) {
    $run = "${Problem}__$bar"
    $commands += "build --name $run --problem-dir `"$problemDir`" --bar $bar --nodes $Nodes --starts-file $Starts --workers $Workers"
    if ($TrackMm -ne 0) {
        $commands += "track --name $run --max-score-mm $TrackMm --workers $Workers"
    }
    $commands += "export --name $run"
}
$log = "process_precompute_${Problem}_$(Get-Date -Format yyyyMMdd).log"
& $python (Join-Path $PSScriptRoot "run_logged.py") --log $log @commands
exit $LASTEXITCODE
