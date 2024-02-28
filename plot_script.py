from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
arg = sys.argv[1]

if arg == 'resnet':
    df = pd.read_csv('resnet18_bm.csv')
elif arg == 'vit':
    df = pd.read_csv('vit_bm.csv')
else:
    print('Error: Invalid argument')
    sys.exit()

# Convert all trial times from seconds to minutes
df.loc[:, df.columns != 'Dataset'] /= 60

df['Dataset'] = ['CIFAR10'] * 4 + ['MNIST'] * 4 + ['CelebA'] * 4

grouped_mean = df.groupby('Dataset').mean()
grouped_std = df.groupby('Dataset').std()

# List of lists where each inner list contains the features to be plotted in a separate graph
features = [
    ['None', 'Vanilla'],
    ['None', 'Vanilla', 'Multithreaded'],
    ['None', 'Vanilla', 'Cryptographic Hash'],
    ['None', 'Vanilla', 'Multithreaded Cryptographic Hash'],
    ['None', 'Vanilla', 'Multithreaded Cryptographic Hash and Plain Model'],
    ['None', 'Vanilla', 'Multithreaded', 'Cryptographic Hash', 'Multithreaded Cryptographic Hash', 'Multithreaded Cryptographic Hash and Plain Model']
]


colors = ['#2edaff', '#1a80bb', '#8cc5e3']
hatches = ['', '/', 'o']

bar_width = 0.2  # Define the width of the bars

for i, feature in enumerate(features[:-1], 1):
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, feature_name in enumerate(feature):
        ax.bar(np.arange(len(grouped_mean.index)) + j * bar_width, grouped_mean[feature_name], yerr=grouped_std[feature_name], capsize=4, color=colors[j % len(colors)], hatch=hatches[j % len(hatches)], width=bar_width, label=feature_name)
    for j, bar in enumerate(ax.patches):
        feature_index = j // len(grouped_mean.index)
        if feature[feature_index] != 'None':
            none_average_time = grouped_mean['None'].iloc[j % len(grouped_mean.index)]
            feature_average_time = grouped_mean[feature[feature_index]].iloc[j % len(grouped_mean.index)]
            overhead = ((feature_average_time / none_average_time) - 1) * 100
            ax.text(bar.get_x() + bar.get_width() * 0.9, bar.get_height() + grouped_std[feature[feature_index]].iloc[j % len(grouped_mean.index)] + 1, f'+{overhead:.2f}%', ha='center', va='bottom', fontsize=10, rotation=30)

    ax.set_xticks(np.arange(len(grouped_mean.index)) + bar_width * (len(feature) - 1) / 2)
    ax.set_xticklabels(grouped_mean.index)

    ncol = 2 if len(feature) > 2 else len(feature)
    
    ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=10, handlelength=3, handleheight=1.5)

    plt.xticks(rotation=0, fontsize=12)
    plt.tick_params(axis='x', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('Time Elapsed (minutes)', fontsize=12)
    
    if arg == 'resnet':
        ax.set_ylim(0, 160)
    else:
        ax.set_ylim(0, 3000)
        
    y_ticks = ax.get_yticks().tolist()
    y_ticks = y_ticks[:-1]
    ax.set_yticks(y_ticks)
    
    plt.tight_layout()
    
    plt.savefig(f'{arg}_{i}.svg', format='svg')
    plt.show()

combined_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']    
combined_hatches = ['', '/', 'o', '-', '\\', 'O']

bar_width = 0.1  # Define the width of the bars

fig, ax = plt.subplots(figsize=(7, 5))
feature = features[-1]
for j, feature_name in enumerate(feature):
    ax.bar(np.arange(len(grouped_mean.index)) + j * bar_width, grouped_mean[feature_name], yerr=grouped_std[feature_name], capsize=4, color=combined_colors[j], hatch=combined_hatches[j], width=bar_width, label=feature_name)
for j, bar in enumerate(ax.patches):
    feature_index = j // len(grouped_mean.index)
ax.set_xticks(np.arange(len(grouped_mean.index)) + bar_width * (len(feature) - 1) / 2)
ax.set_xticklabels(grouped_mean.index)

ncol = 2 if len(feature) > 2 else len(feature)

ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=10, handlelength=3, handleheight=1.5)

plt.xticks(rotation=0, fontsize=12)
plt.tick_params(axis='x', length=0)
ax.set_xlabel('')
ax.set_ylabel('Time Elapsed (minutes)', fontsize=12)

if arg == 'resnet':
    ax.set_ylim(0, 160)
else:
    ax.set_ylim(0, 3000)
y_ticks = ax.get_yticks().tolist()
y_ticks = y_ticks[:-1]
ax.set_yticks(y_ticks)

plt.tight_layout()

plt.savefig(f'{arg}_6.svg', format='svg')
plt.show()