# Experiments T2, T3 and T5 of the holding-roadmap report for one bar of a
# schema 2.1 process folder: the Windows twin of process_plans.sh. The carry
# is planned in its real scene with the roadmap planner and with the BiRRT
# (scripts\plan_process_carry.py), then every saved path is checked on its
# own (scripts\prm_check_solved.py). Results go to <root>\prm\plans\<problem>;
# nothing is written into the process folder.
#
# Usage, from the repository root:
#   scripts\prm_runs\process_plans.ps1 -Problem 261008_2ModulesDemo_purged_process -Bar B21
#   scripts\prm_runs\process_plans.ps1 -Problem 261008_2ModulesDemo_purged_process -Bar B21 -Parts PRM
# Launch detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\process_plans.ps1','-Problem','261008_2ModulesDemo_purged_process','-Bar','B21'
#
# Needs HUSKY_BENCH_ROOT and HUSKY_DESIGN_STUDY_DIR (see process_precompute.ps1)
# and the bar's roadmap. Optional: PYTHON, WORKERS, MAX_TIME (the BiRRT's budget, s).
param(
    [Parameter(Mandatory = $true)][string]$Problem,
    [Parameter(Mandatory = $true)][string]$Bar,
    # Which parts to run: T2 (both planners; or only one of them with PRM or BIRRT),
    # T5 (one worker), T3 (shifted base), NOTRACK (the roadmap planner without the
    # tracking file; not in the default list). One part: -Parts PRM
    [string[]]$Parts = @("T2", "T5", "T3"),
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 8 }),
    [int]$MaxTime = $(if ($env:MAX_TIME) { [int]$env:MAX_TIME } else { 300 })
)
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = if ($env:PYTHON) { $env:PYTHON } else { Join-Path $repo ".venv\Scripts\python.exe" }
if (-not $env:HUSKY_DESIGN_STUDY_DIR) { throw "set HUSKY_DESIGN_STUDY_DIR to the folder holding the exported problems" }
if (-not $env:HUSKY_BENCH_ROOT) { throw "set HUSKY_BENCH_ROOT to the benchmark data root" }
$problemDir = Join-Path $env:HUSKY_DESIGN_STUDY_DIR $Problem
$out = Join-Path $env:HUSKY_BENCH_ROOT "prm\plans\$Problem"

# The planner reads the process folder and saves everything into $out.
$common = "plan `"$problemDir`" --bar $Bar --save-dir `"$out`""
$check = "check `"$problemDir`" --bar $Bar --solved-dir `"$out`" --out `"$out\checks.jsonl`" --tags"
$commands = @()
if (($Parts -contains "T2") -or ($Parts -contains "PRM")) {
    # * The roadmap planner, then its check right away.
    $commands += "$common --cdfm-backend prm --prm-workers $Workers --save-tag prm"
    $commands += "$check prm"
}
if (($Parts -contains "T2") -or ($Parts -contains "BIRRT")) {
    # * The BiRRT, three runs (it is random), MaxTime seconds each at most.
    foreach ($n in 1..3) {
        $commands += "$common --cdfm-backend birrt --max-time $MaxTime --save-tag birrt$n"
    }
    $commands += "$check birrt1 birrt2 birrt3"
}
if ($Parts -contains "T5") {
    # * The roadmap planner again with one worker (everything in the caller's process).
    $commands += "$common --cdfm-backend prm --prm-workers 1 --save-tag prm_w1"
    $commands += "$check prm_w1"
}
if ($Parts -contains "NOTRACK") {
    # * T4: the roadmap planner ignoring the tracking file, to see what that tracking saves.
    $commands += "$common --cdfm-backend prm --prm-workers $Workers --prm-no-pretracked --save-tag prm_notrack"
    $commands += "$check prm_notrack"
}
if ($Parts -contains "T3") {
    # * The base parked off its planned pose: 5 cm / 3 deg (inside the error the
    # * roadmap was built for) and 10 cm / 5 deg (at its edge), two directions each.
    $shifts = @{
        "shift5a" = "0.05,0.0,3"; "shift5b" = "-0.035,-0.035,-3";
        "shift10a" = "0.0,0.10,5"; "shift10b" = "-0.07,0.07,-5"
    }
    foreach ($tag in ($shifts.Keys | Sort-Object)) {
        $commands += "$common --cdfm-backend prm --prm-workers $Workers --base-shift=$($shifts[$tag]) --save-tag $tag"
    }
    $commands += "$check " + (($shifts.Keys | Sort-Object) -join " ")
}
$log = "process_plans_${Problem}_${Bar}_$(Get-Date -Format yyyyMMdd).log"
& $python (Join-Path $PSScriptRoot "run_logged.py") --log $log @commands
exit $LASTEXITCODE
