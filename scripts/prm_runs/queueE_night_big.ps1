# Queue E (experiment E6, second pass): a night-sized roadmap. The 6000-node roadmap over
# 17 goals (queueC) answered online queries in ~30 s but brought none of 5 unseen goals
# within 10 mm: coverage, not speed, is what the night has to buy.
# 30,000 nodes anchored on 45 goals (all but the 5 held out) and the three carry poses.
$carry = "asset\carry_poses.json"
$heldOut = @(2, 10, 20, 31, 47)
$anchorGoals = (0..49 | Where-Object { $heldOut -notcontains $_ }) -join ","
$commands = @(
    # First, the production check of the tube-refined paths (experiment E3).
    "bridge validate --name B18_g2_n4000 --refine all_k2.5_p10mm_mid_tube3 --workers 8",
    "build --name B18_night30k --library cdfm_v1 --bar B18 --goals $anchorGoals --anchor-starts 0 --nodes 30000 --starts-file $carry --workers 8",
    "track --name B18_night30k --max-score-mm 10 --workers 8"
)
foreach ($goal in $heldOut) {
    $tag = "online_g{0:D3}" -f $goal
    $commands += "query --name B18_night30k --goal $goal --starts-file $carry --tag $tag --workers 8"
    $commands += "refine --name B18_night30k --query $tag --pretracked tracked_10mm.json.gz --max-gap-mm 10 --tolerances-mm 2,5,10 --store-paths-mm 10 --workers 8"
}
$commands += "query --name B18_night30k --goal 2 --starts-file $carry --tag online1_g002 --workers 1"
$commands += "refine --name B18_night30k --query online1_g002 --pretracked tracked_10mm.json.gz --max-gap-mm 10 --tolerances-mm 2,5,10 --workers 1"
$commands += "export --name B18_night30k"
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "queueE_night_big_20261001.log" @commands
