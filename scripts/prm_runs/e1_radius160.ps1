# Experiment E1: the 4000-node B18 goal-2 roadmap rebuilt with a 160-degree edge limit
# (default 90). Same seed, so the sampled nodes are the same; only edges differ.
# Launch detached, from the repository root:
#   Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\e1_radius160.ps1' -WindowStyle Hidden
# ! Output is redirected through cmd: in Windows PowerShell 5.1 a native program
# ! writing to stderr under "*>>" becomes an error record and can stop the script.
$env:HUSKY_BENCH_ROOT = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark"
Set-Location (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$log = Join-Path $env:HUSKY_BENCH_ROOT "prm\logs\e1_radius160_20261001.log"
$name = "B18_g2_n4000_r160"
$run = ".venv\Scripts\python.exe -u scripts\prm_pilot.py"

cmd /c "$run build --name $name --library cdfm_v1 --bar B18 --goals 2 --anchor-starts 40 --nodes 4000 --radius-deg 160 --workers 6 >> `"$log`" 2>&1"
cmd /c "$run query --name $name --goal 2 --starts 0,12 --tag s0_s12 --radius-endpoint-deg 160 --workers 6 >> `"$log`" 2>&1"
cmd /c "$run query --name $name --goal 2 --starts all --tag all --radius-endpoint-deg 160 --workers 6 >> `"$log`" 2>&1"
cmd /c "$run export --name $name >> `"$log`" 2>&1"
cmd /c "echo E1 DONE >> `"$log`""
