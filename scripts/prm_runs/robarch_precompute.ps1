# Experiment T1: precompute one roadmap per bar for B1, B3, B4 of the RobArch
# demo export. Each roadmap holds only the robot, its tools and the carried bar;
# it is anchored on the bar's planned goal, on 12 copies of it for a base parked
# up to 10 cm / 5 degrees off, and on the 18 carry poses. Then every roadmap edge
# within 15 mm is tracked exactly, so online planning only has to check the scene.
# Needs: scripts\benchmark_cdfm.py build-cells <problem folder> --bars B1,B3,B4
# Launch detached, from the repository root:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\robarch_precompute.ps1'
$problem = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_design_study\260920_RobArch_demo_revamp"
$commands = @()
foreach ($bar in @("B4", "B3", "B1")) {
    $run = "260920_RobArch_demo_revamp__$bar"
    $commands += "build --name $run --problem-dir `"$problem`" --bar $bar --nodes 6000 --starts-file carry/carry_v1/starts.json --workers 8"
    $commands += "track --name $run --max-score-mm 15 --workers 8"
    $commands += "export --name $run"
}
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "robarch_precompute_20261002.log" @commands
