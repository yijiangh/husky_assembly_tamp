# Queue B (experiment E5): score the carry candidates of carry_v1 against many goals.
# Per bar: one roadmap anchored on several goals plus every candidate's branch pairs,
# then per goal a query whose start set is the candidates, and its refinement.
# Launch detached, from the repository root, after `carry starts --tag carry_v1`:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\prm_runs\queueB_carry_stage2.ps1'
$starts = "carry/carry_v1/starts.json"
$plan = @{ "B18" = @(0, 7, 14, 21, 28, 35, 42, 49); "B3" = @(4, 11, 16, 25, 33, 42) }
$commands = @()
foreach ($bar in @("B18", "B3")) {
    $run = "${bar}_carry_v1"
    $goals = $plan[$bar] -join ","
    $commands += "build --name $run --library cdfm_v1 --bar $bar --goals $goals --anchor-starts 0 --nodes 4000 --starts-file $starts --workers 8"
    foreach ($goal in $plan[$bar]) {
        $tag = "g{0:D3}_carry" -f $goal
        $commands += "query --name $run --goal $goal --starts-file $starts --tag $tag --workers 8"
        $commands += "refine --name $run --query $tag --workers 8"
    }
}
$runs = "B18_carry_v1:" + ($plan["B18"] -join ",") + "+B3_carry_v1:" + ($plan["B3"] -join ",")
$commands += "carry stage2 --tag carry_v1 --runs $runs"
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "queueB_carry_stage2_20261001.log" @commands
