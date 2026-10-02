# Queue C (experiment E6): overnight precompute, then online queries.
#   Night: one B18 roadmap anchored on 17 goals and the three carry poses, every edge
#          collision-checked, then every roadmap edge within 10 mm tracked exactly.
#   Online: goals the roadmap was NOT anchored on; query (carry poses as the start set)
#           and refine with the pretracked verdicts. Timed per phase.
$carry = "asset\carry_poses.json"
$anchorGoals = (0..16 | ForEach-Object { $_ * 3 }) -join ","
$commands = @(
    "carry pick --tag carry_v1",
    "build --name B18_overnight --library cdfm_v1 --bar B18 --goals $anchorGoals --anchor-starts 0 --nodes 6000 --starts-file $carry --workers 8",
    "track --name B18_overnight --max-score-mm 10 --workers 8"
)
foreach ($goal in @(2, 10, 20, 31, 47)) {
    $tag = "online_g{0:D3}" -f $goal
    $commands += "query --name B18_overnight --goal $goal --starts-file $carry --tag $tag --workers 8"
    $commands += "refine --name B18_overnight --query $tag --pretracked tracked_10mm.json.gz --max-gap-mm 10 --tolerances-mm 2,5,10 --workers 8"
}
# The same online question in a single process (no worker pool at all).
$commands += "query --name B18_overnight --goal 2 --starts-file $carry --tag online1_g002 --workers 1"
$commands += "refine --name B18_overnight --query online1_g002 --pretracked tracked_10mm.json.gz --max-gap-mm 10 --tolerances-mm 2,5,10 --workers 1"
$commands += "export --name B18_overnight"
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "queueC_precompute_20261001.log" @commands
