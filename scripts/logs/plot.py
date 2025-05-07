#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))

# Set up LaTeX rendering for text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Read the log file
with open(os.path.join(HERE, 'log.json'), 'r') as f:
    data = json.load(f)

# Define scenarios and algorithms
scenarios = list(data.keys())

# Select which algorithms to include in the plots
algorithms_to_plot = [
    "TAPOM",
    "RRTConnect",
    # "PRM",
    # "LazyRRT",
    # "BIT*",
    # "EIT*",
    # "BFMT",
    # "EST",
    # "STRIDE",
    "TAMPOR_wo_Prioritization"
]

# Validate algorithm names
all_algorithms = list(data[scenarios[0]].keys())
valid_algorithms = [algo for algo in algorithms_to_plot if algo in all_algorithms]

if not valid_algorithms:
    print(f"Warning: None of the selected algorithms found. Available algorithms: {all_algorithms}")
    valid_algorithms = all_algorithms
elif len(valid_algorithms) < len(algorithms_to_plot):
    missing = set(algorithms_to_plot) - set(valid_algorithms)
    print(f"Warning: Some algorithms not found: {missing}. Available algorithms: {all_algorithms}")

algorithms = valid_algorithms

# Initialize data structures
success_rates = {scenario: {} for scenario in scenarios}
avg_times = {scenario: {} for scenario in scenarios}
avg_success_times = {scenario: {} for scenario in scenarios}

# Calculate metrics for each algorithm in each scenario
for scenario in scenarios:
    for algo in algorithms:
        trials = data[scenario][algo]
        
        # Success rate calculation
        successes = sum(1 for trial in trials if trial[1])
        total_trials = max(len(trials), 1)
        success_rate = (successes / total_trials) * 100
        success_rates[scenario][algo] = success_rate
        
        # Average time calculation (all trials)
        total_time = sum(trial[2] for trial in trials)
        avg_times[scenario][algo] = total_time / total_trials
        
        # Average time for successful planning only
        successful_times = [trial[2] for trial in trials if trial[1]]
        if successful_times:
            avg_success_times[scenario][algo] = sum(successful_times) / len(successful_times)
        else:
            avg_success_times[scenario][algo] = 0  # No successful trials

# Custom distinct color palette (more than enough for all algorithms)
distinct_colors = [
    "#E41A1C",  # Red
    "#377EB8",  # Blue
    "#4DAF4A",  # Green
    "#984EA3",  # Purple
    "#FF7F00",  # Orange
    "#FFFF33",  # Yellow
    "#A65628",  # Brown
    "#F781BF",  # Pink
    "#999999",  # Grey
    "#66C2A5",  # Mint
    "#FC8D62",  # Salmon
    "#8DA0CB",  # Light blue
    "#E78AC3",  # Light pink
    "#A6D854",  # Light green
    "#FFD92F",  # Light yellow
    "#B3B3B3"   # Light grey
]

# Create a fixed color mapping for algorithms with the custom colors
color_map = {algo: distinct_colors[i % len(distinct_colors)] for i, algo in enumerate(algorithms)}
bar_width = 0.08  # Reduced bar width
index = np.arange(len(scenarios)) * 1.5  # Increased spacing between scenario groups

# Figure 1: Success Rates
plt.figure(figsize=(16, 8))  # Wider figure
for i, algo in enumerate(algorithms):
    values = [success_rates[scenario][algo] for scenario in scenarios]
    bars = plt.bar(index + i*bar_width - (len(algorithms)-1)*bar_width/2, 
            values, bar_width, label=algo, color=color_map[algo])
    
    # Add data labels on top of bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel(r'\textbf{Scenario}', fontsize=14)
plt.ylabel(r'\textbf{Success Rate (\%)}', fontsize=14)
plt.title(r'\textbf{Algorithm Success Rates}', fontsize=16)
plt.xticks(index, [r'\textbf{' + scenario.replace(' ', '\ ') + '}' for scenario in scenarios], fontsize=12)
plt.legend(prop={'size': 10})
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'success_rates.pdf'), dpi=300, format='pdf')

# Figure 2: Average Times (all trials)
plt.figure(figsize=(16, 8))  # Wider figure
for i, algo in enumerate(algorithms):
    values = [avg_times[scenario][algo] for scenario in scenarios]
    bars = plt.bar(index + i*bar_width - (len(algorithms)-1)*bar_width/2, 
            values, bar_width, label=algo, color=color_map[algo])
    
    # Add data labels on top of bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=8, rotation=45)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel(r'\textbf{Scenario}', fontsize=14)
plt.ylabel(r'\textbf{Average Time (s)}', fontsize=14)
plt.title(r'\textbf{Average Planning Time}', fontsize=16)
plt.xticks(index, [r'\textbf{' + scenario.replace(' ', '\ ') + '}' for scenario in scenarios], fontsize=12)
plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'average_times.pdf'), dpi=300, format='pdf')

# Figure 3: Average Times for Successful Planning Only
plt.figure(figsize=(16, 8))  # Wider figure
for i, algo in enumerate(algorithms):
    values = [avg_success_times[scenario][algo] for scenario in scenarios]
    bars = plt.bar(index + i*bar_width - (len(algorithms)-1)*bar_width/2, 
            values, bar_width, label=algo, color=color_map[algo])
    
    # Add data labels on top of bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=8, rotation=45)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel(r'\textbf{Scenario}', fontsize=14)
plt.ylabel(r'\textbf{Average Time (s)}', fontsize=14)
plt.title(r'\textbf{Average Planning Time for Successful Trials Only}', fontsize=16)
plt.xticks(index, [r'\textbf{' + scenario.replace(' ', '\ ') + '}' for scenario in scenarios], fontsize=12)
plt.legend(prop={'size': 10})
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'successful_times.pdf'), dpi=300, format='pdf')

print("Plots generated: success_rates.pdf, average_times.pdf, successful_times.pdf")
