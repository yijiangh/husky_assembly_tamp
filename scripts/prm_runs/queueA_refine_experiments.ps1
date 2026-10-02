# Queue A (experiments E2 and E3) on B18 goal 2, all 807 starts.
#   E2: refine (every exact path stored), production check of every stored path,
#       nodes-only knots, and the three 2000-node seeds.
#   E3: tube rounds (way 2), then the seeded BiRRT with and without the exact chains (way 1).
$commands = @(
    "refine --name B18_g2_n4000 --query all --store-paths-mm 30 --workers 8",
    "bridge validate --name B18_g2_n4000 --refine all_k2.5_p10mm_mid --workers 8",
    "refine --name B18_g2_n4000 --query all --tube-rounds 3 --workers 8",
    "bridge birrt --name B18_g2_n4000 --refine all_k2.5_p10mm_mid_tube3 --max-time 120 --workers 8",
    "bridge birrt --name B18_g2_n4000 --refine all_k2.5_p10mm_mid_tube3 --max-time 120 --workers 8 --no-chains",
    "refine --name B18_g2_n4000 --query all --knot-step-deg 0 --workers 8",
    "refine --name B18_g2_n2000 --query all --workers 8",
    "refine --name B18_g2_n2000_seed1 --query all --workers 8",
    "refine --name B18_g2_n2000_seed2 --query all --workers 8"
)
& (Join-Path $PSScriptRoot "run_logged.ps1") -Log "queueA_20261001d.log" @commands
# Then the night-sized precompute (experiment E6, second pass).
& (Join-Path $PSScriptRoot "queueE_night_big.ps1")
