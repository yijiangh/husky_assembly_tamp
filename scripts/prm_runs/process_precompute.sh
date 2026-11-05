#!/usr/bin/env bash
# Precompute one roadmap per bar of an exported process folder (process.json
# and actions/). Each roadmap holds only the robot, its tools and the carried
# bar (with its halves), anchored on
# the bar's planned goal, on copies of it for a base parked off, and on the
# carry poses; then every roadmap edge within TRACK_MM is tracked exactly, so
# online planning only has to check the scene.
#
# Usage:
#   scripts/prm_runs/process_precompute.sh <problem folder name> <bar> [<bar> ...]
# Example (the two-modules demo, one bar):
#   scripts/prm_runs/process_precompute.sh 261008_2ModulesDemo_purged_process B21
#
# Needs, before running:
#   HUSKY_BENCH_ROOT        the benchmark data root (holds cells/, prm/, ...)
#   HUSKY_DESIGN_STUDY_DIR  the folder holding the exported problems
#   the trimmed scenes:     python scripts/benchmark_cdfm.py build-cells "$HUSKY_DESIGN_STUDY_DIR/<problem>" --bars B21
#   the carry starts file:  <root>/prm/carry/carry_v1/starts.json (synced with the data)
# Optional: PYTHON (interpreter), WORKERS (8), NODES (6000), TRACK_MM (15; 0 skips
# the tracking, for a quick smoke run), STARTS (carry/carry_v1/starts.json).
#
# Detach it with nohup so it survives closing the terminal:
#   nohup scripts/prm_runs/process_precompute.sh 261008_2ModulesDemo_purged_process B21 > /dev/null 2>&1 &
# Progress goes to <root>/prm/logs/process_precompute_<problem>_<date>.log.
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# The interpreter: $PYTHON if set, else the repo's venv, else whatever python3 is.
python="${PYTHON:-$repo/.venv/bin/python}"
[ -x "$python" ] || python="python3"

problem="${1:?usage: process_precompute.sh <problem folder name> <bar> [<bar> ...]}"
shift
[ $# -ge 1 ] || { echo "give at least one bar id" >&2; exit 1; }
study="${HUSKY_DESIGN_STUDY_DIR:?set HUSKY_DESIGN_STUDY_DIR to the folder holding the exported problems}"
: "${HUSKY_BENCH_ROOT:?set HUSKY_BENCH_ROOT to the benchmark data root}"
workers="${WORKERS:-8}"
nodes="${NODES:-6000}"
track_mm="${TRACK_MM:-15}"
starts="${STARTS:-carry/carry_v1/starts.json}"

commands=()
for bar in "$@"; do
    run="${problem}__$bar"
    commands+=("build --name $run --problem-dir \"$study/$problem\" --bar $bar --nodes $nodes --starts-file $starts --workers $workers")
    if [ "$track_mm" != "0" ]; then
        commands+=("track --name $run --max-score-mm $track_mm --workers $workers")
    fi
    commands+=("export --name $run")
done

exec "$python" "$here/run_logged.py" --log "process_precompute_${problem}_$(date +%Y%m%d).log" "${commands[@]}"
