# Queue D (experiment E5, second pass): the carry candidates scored on one roadmap PER goal.
# The first pass (queueB, one 4000-node roadmap over 8 goals per bar) was too sparse to
# separate the candidates: 311 components, 12 of 36 goal anchors in the largest.
$starts = "carry/carry_v1/starts.json"
$plan = @{ "B18" = @(0, 7, 14, 21, 28, 35, 42, 49); "B3" = @(4, 11, 16, 25, 33, 42) }
$commands = @()
$runs = @()
foreach ($bar in @("B18", "B3")) {
    foreach ($goal in $plan[$bar]) {
        $run = "{0}_cv1_g{1:D3}" -f $bar, $goal
        $tag = "g{0:D3}_carry" -f $goal
        $commands += "build --name $run --library cdfm_v1 --bar $bar --goals $goal --anchor-starts 0 --nodes 3000 --starts-file $starts --workers 8"
        $commands += "query --name $run --goal $goal --starts-file $starts --tag $tag --workers 8"
        $commands += "refine --name $run --query $tag --workers 8"
        $runs += "${run}:$goal"
    }
}
$commands += "carry stage2 --tag carry_v1 --runs " + ($runs -join "+")
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "queueD_carry_stage2b_20261001.log" @commands
