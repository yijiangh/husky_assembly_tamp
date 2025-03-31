#!/bin/bash

file_list=("task_12" "task_13" "task_14" "task_15")

for file in "${file_list[@]}"; do
    python scripts/model/dataset_generator.py --birrt --curobo --save --random --repeat 20 --scene cuboid_1 --task "${file}" --max_attempts 600
done

