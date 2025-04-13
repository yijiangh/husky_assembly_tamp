#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Read JSON file
log_path = os.path.join(os.path.dirname(__file__), 'logs/corner_case_for_transfer_250413_1740.json')
with open(log_path, 'r') as f:
    data = json.load(f)

# Parse data
algorithms = list(data.keys())
success_rates = []
success_rates_std = []
planning_times = []
planning_times_std = []

for alg in algorithms:
    # Calculate success rate
    successes = [entry[1] for entry in data[alg]]
    success_rate = np.mean(successes) * 100  # Convert to percentage
    success_rates.append(success_rate)
    
    # Calculate standard deviation of success rate
    n = len(successes)
    if n > 1:
        std = np.std(successes, ddof=1) * 100 / np.sqrt(n)  # Standard error, convert to percentage
    else:
        std = 0
    success_rates_std.append(std)
    
    # Calculate average planning time
    planning_time = [entry[2] for entry in data[alg]]
    mean_time = np.mean(planning_time)
    planning_times.append(mean_time)
    
    # Calculate standard deviation of planning time
    if n > 1:
        time_std = np.std(planning_time, ddof=1) / np.sqrt(n)  # Standard error
    else:
        time_std = 0
    planning_times_std.append(time_std)

# Define colors for each algorithm
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # 蓝色, 橙色, 绿色, 红色, 紫色, 棕色, 粉色, 灰色

# Set up figure
plt.figure(figsize=(12, 5))

# Plot success rate bar chart
plt.subplot(1, 2, 1)
x = np.arange(len(algorithms))
width = 0.5
bars = plt.bar(x, success_rates, width, yerr=success_rates_std, capsize=5, color=colors)
plt.ylabel('Success Rate (%)')
plt.title('Success Rate Comparison')
plt.xticks(x, algorithms)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend with algorithm names
plt.legend(bars, algorithms, title="Algorithms")

# Plot planning time bar chart
plt.subplot(1, 2, 2)
bars = plt.bar(x, planning_times, width, yerr=planning_times_std, capsize=5, color=colors)
plt.ylabel('Planning Time (seconds)')
plt.title('Average Planning Time Comparison')
plt.xticks(x, algorithms)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend with algorithm names
plt.legend(bars, algorithms, title="Algorithms")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'logs/algorithm_comparison.png'), dpi=300)
plt.show()
