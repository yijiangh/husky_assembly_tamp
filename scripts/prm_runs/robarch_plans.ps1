# Experiments T2, T3 and T5 for one bar of the RobArch demo export: plan the bar
# carry in its real scene with the roadmap planner and with the BiRRT, then
# check every saved path on its own (production repair at 2 mm, full collision).
# Needs the bar's roadmap (robarch_precompute.ps1). Each run writes, into
# <root>\prm\robarch_plans (nothing is written into the problem folder):
#   <bar>__J.solved_motion.<tag>.json             the path
#   <bar>__J.solved_motion.<tag>.plan_info.json   how the plan went
#   <bar>__J_planned_motion.<tag>.png             the joint curves
# and the checks go to <root>\prm\robarch_plans\checks.jsonl.
# Launch detached, from the repository root:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\robarch_plans.ps1','-Bar','B4'
param(
    [Parameter(Mandatory = $true)][string]$Bar,
    # Which parts to run: T2 (both planners; or only one of them with PRM or BIRRT),
    # T5 (one worker), T3 (shifted base), NOTRACK (the roadmap planner without the
    # night's tracking file; not in the default list). One part: -Parts PRM
    [string[]]$Parts = @("T2", "T5", "T3")
)
$study = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_design_study"
$root = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark"
$problem = "260920_RobArch_demo_revamp"
$out = "$root\prm\robarch_plans"
$common = "headless `"$study`" --problem $problem --bar-action ${Bar}__J.json --movement cdfm --no-replay --no-show-plot --save-dir `"$out`""
$check = "check `"$study\$problem`" --bar $Bar --solved-dir `"$out`" --out `"$out\checks.jsonl`" --tags"
$commands = @()
if (($Parts -contains "T2") -or ($Parts -contains "PRM")) {
    # * The roadmap planner, 8 workers, then its check right away.
    $commands += "$common --cdfm-backend prm --prm-workers 8 --save-tag prm"
    $commands += "$check prm"
}
if (($Parts -contains "T2") -or ($Parts -contains "BIRRT")) {
    # * The BiRRT, three runs (it is random), 300 s each at most.
    foreach ($n in 1..3) {
        $commands += "$common --cdfm-backend birrt --max-time 300 --save-tag birrt$n"
    }
    $commands += "$check birrt1 birrt2 birrt3"
}
if ($Parts -contains "T5") {
    # * The roadmap planner again with one worker (everything in the caller's process).
    $commands += "$common --cdfm-backend prm --prm-workers 1 --save-tag prm_w1"
    $commands += "$check prm_w1"
}
if ($Parts -contains "NOTRACK") {
    # * T4: the roadmap planner ignoring the night's tracking, to see what that tracking saves.
    $commands += "$common --cdfm-backend prm --prm-workers 8 --prm-no-pretracked --save-tag prm_notrack"
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
        $commands += "$common --cdfm-backend prm --prm-workers 8 --base-shift=$($shifts[$tag]) --save-tag $tag"
    }
    $commands += "$check " + (($shifts.Keys | Sort-Object) -join " ")
}
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "robarch_plans_${Bar}_20261002.log" @commands
