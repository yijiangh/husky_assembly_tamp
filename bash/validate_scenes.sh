#!/bin/bash

file_list=("task_18" "task_19")

for file in "${file_list[@]}"; do
    python scripts/model/dataset_generator.py --birrt --curobo --save --random --validation --scene cuboid_1 --task "${file}" --max_attempts 50 --repeat 1
done

