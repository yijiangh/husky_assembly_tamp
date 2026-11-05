#!/usr/bin/env bash
# Experiments T2, T3 and T5 of the holding-roadmap report for one bar of a
# schema 2.1 process folder: plan the bar carry in its real scene with the
# roadmap planner and with the BiRRT (scripts/plan_process_carry.py), then
# check every saved path on its own (production repair at 2 mm, full
# collision; scripts/prm_check_solved.py). Needs the bar's roadmap
# (process_precompute.sh). Each run writes, into <root>/prm/plans/<problem>
# (nothing is written into the process folder):
#   <bar>_J.solved_motion.<tag>.json             the path
#   <bar>_J.solved_motion.<tag>.plan_info.json   how the plan went
#   <bar>_J_planned_motion.<tag>.png             the joint curves
# and the checks go to <root>/prm/plans/<problem>/checks.jsonl.
#
# Usage:
#   scripts/prm_runs/process_plans.sh <problem folder name> <bar> [part ...]
# Parts: T2 (both planners; or only one of them with PRM or BIRRT), T5 (one
# worker), T3 (shifted base), NOTRACK (the roadmap planner without the
# tracking file; not in the default list). Default: T2 T5 T3. Examples:
#   scripts/prm_runs/process_plans.sh 261008_2ModulesDemo_purged_process B21
#   scripts/prm_runs/process_plans.sh 261008_2ModulesDemo_purged_process B21 PRM
#   nohup scripts/prm_runs/process_plans.sh 261008_2ModulesDemo_purged_process B21 > /dev/null 2>&1 &
#
# Needs HUSKY_BENCH_ROOT and HUSKY_DESIGN_STUDY_DIR (see process_precompute.sh).
# Optional: PYTHON, WORKERS (8), MAX_TIME (300: the BiRRT's budget in seconds).
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
python="${PYTHON:-$repo/.venv/bin/python}"
[ -x "$python" ] || python="python3"

problem="${1:?usage: process_plans.sh <problem folder name> <bar> [T2|T5|T3|PRM|BIRRT|NOTRACK ...]}"
bar="${2:?usage: process_plans.sh <problem folder name> <bar> [T2|T5|T3|PRM|BIRRT|NOTRACK ...]}"
shift 2
parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=(T2 T5 T3)

has_part() {
    # Whether the part named $1 was asked for.
    local part
    for part in "${parts[@]}"; do [ "$part" = "$1" ] && return 0; done
    return 1
}

study="${HUSKY_DESIGN_STUDY_DIR:?set HUSKY_DESIGN_STUDY_DIR to the folder holding the exported problems}"
root="${HUSKY_BENCH_ROOT:?set HUSKY_BENCH_ROOT to the benchmark data root}"
out="$root/prm/plans/$problem"
workers="${WORKERS:-8}"
max_time="${MAX_TIME:-300}"

# The planner reads the process folder and saves everything into $out.
common="plan \"$study/$problem\" --bar $bar --save-dir \"$out\""
check="check \"$study/$problem\" --bar $bar --solved-dir \"$out\" --out \"$out/checks.jsonl\" --tags"

commands=()
if has_part T2 || has_part PRM; then
    # The roadmap planner, then its check right away.
    commands+=("$common --cdfm-backend prm --prm-workers $workers --save-tag prm")
    commands+=("$check prm")
fi
if has_part T2 || has_part BIRRT; then
    # The BiRRT, three runs (it is random), MAX_TIME seconds each at most.
    for n in 1 2 3; do
        commands+=("$common --cdfm-backend birrt --max-time $max_time --save-tag birrt$n")
    done
    commands+=("$check birrt1 birrt2 birrt3")
fi
if has_part T5; then
    # The roadmap planner again with one worker (everything in the caller's process).
    commands+=("$common --cdfm-backend prm --prm-workers 1 --save-tag prm_w1")
    commands+=("$check prm_w1")
fi
if has_part NOTRACK; then
    # T4: the roadmap planner ignoring the tracking file, to see what that tracking saves.
    commands+=("$common --cdfm-backend prm --prm-workers $workers --prm-no-pretracked --save-tag prm_notrack")
    commands+=("$check prm_notrack")
fi
if has_part T3; then
    # The base parked off its planned pose: 5 cm / 3 deg (inside the error the
    # roadmap was built for) and 10 cm / 5 deg (at its edge), two directions each.
    declare -A shifts=(
        [shift10a]="0.0,0.10,5" [shift10b]="-0.07,0.07,-5"
        [shift5a]="0.05,0.0,3" [shift5b]="-0.035,-0.035,-3"
    )
    tags=(shift10a shift10b shift5a shift5b)
    for tag in "${tags[@]}"; do
        commands+=("$common --cdfm-backend prm --prm-workers $workers --base-shift=${shifts[$tag]} --save-tag $tag")
    done
    commands+=("$check ${tags[*]}")
fi

exec "$python" "$here/run_logged.py" --log "process_plans_${problem}_${bar}_$(date +%Y%m%d).log" "${commands[@]}"
