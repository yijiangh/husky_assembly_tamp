#!/bin/bash
seeds=(78543287 19802734 37485620 92013201 19491001 53011314 67081201 37475620 21920132 75492013)
scene=rebar_1
task=task_1

for seed in "${seeds[@]}"; do
    python scripts/test/corner_case_for_transfer.py --tampor --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --birrt --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --curobo --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl RRTConnect --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl PRM --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl LazyRRT --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl EST --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl STRIDE --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl BITstar --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl EITstar --save --scene $scene --task $task --seed $seed
    # python scripts/test/corner_case_for_transfer.py --ompl BFMT --save --scene $scene --task $task --seed $seed
done