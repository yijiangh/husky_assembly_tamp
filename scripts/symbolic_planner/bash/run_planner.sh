#!/bin/bash

# 运行次数
NUM_RUNS=10

# 循环执行 Python 文件
for ((i = 1; i <= NUM_RUNS; i++)); do
    echo "运行第 $i 次..."
    python /home/jeong/summer_research/eth/husky_assembly/scripts/symbolic_planner/plan_generator.py
done
