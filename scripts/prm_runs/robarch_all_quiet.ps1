# Every changing-scene experiment on B1, B3 and B4, one run at a time, for times
# taken on a machine that is doing nothing else (the first pass ran next to the
# roadmap precompute; its summaries are kept in robarch_plans\shared_machine_run).
# Order: the quick ones first, the BiRRT runs (up to 25 minutes each) last.
# Launch detached, from the repository root:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\robarch_all_quiet.ps1'
$plans = Join-Path $PSScriptRoot "robarch_plans.ps1"
# * T4: time per collision check of each kind.
& (Join-Path $PSScriptRoot "robarch_split_timing.ps1")
# * T2 (roadmap planner) and T5 (one worker), per bar.
foreach ($bar in @("B3", "B1", "B4")) { & $plans -Bar $bar -Parts PRM, T5 }
# * T3: the base parked off its planned pose.
foreach ($bar in @("B3", "B1", "B4")) { & $plans -Bar $bar -Parts T3 }
# * T4: the roadmap planner without the night's tracking file.
foreach ($bar in @("B3", "B1", "B4")) { & $plans -Bar $bar -Parts NOTRACK }
# * T2 (BiRRT), three runs per bar.
foreach ($bar in @("B3", "B1", "B4")) { & $plans -Bar $bar -Parts BIRRT }
