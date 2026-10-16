# Experiment T4 (first half): time per collision check of each kind, per bar of the
# RobArch demo export, on 250 configurations:
#   real scene    -> full check, robot-side check, scene-only check
#   trimmed scene -> full check (the scene the roadmap is precomputed in)
# Also re-checks that the split adds up (the verdict lists are saved).
# Writes <root>\prm\robarch_plans\split_<bar>_<mode>.json. Run it on a quiet
# machine: the numbers are times. Launch detached, from the repository root:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\robarch_split_timing.ps1'
$env:HUSKY_BENCH_ROOT = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark"
$problem = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_design_study\260920_RobArch_demo_revamp"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$out = Join-Path $env:HUSKY_BENCH_ROOT "prm\robarch_plans"
New-Item -ItemType Directory -Force $out | Out-Null
$logPath = Join-Path $env:HUSKY_BENCH_ROOT "prm\logs\robarch_split_timing_20261002.log"
foreach ($bar in @("B1", "B3", "B4")) {
    foreach ($mode in @("real", "trimmed")) {
        $file = Join-Path $out "split_${bar}_${mode}.json"
        cmd /c "echo === $bar $mode >> `"$logPath`""
        cmd /c ".venv\Scripts\python.exe -u benchmark\tests\collision_split_helper.py $mode `"$problem`" $bar 250 `"$file`" >> `"$logPath`" 2>&1"
    }
}
cmd /c "echo ALL DONE >> `"$logPath`""
