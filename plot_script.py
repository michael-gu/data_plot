import matplotlib.pyplot as plt
import pandas as pd
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

# colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey']
colors = ['#b8b8b8', '#1a80bb', '#8cc5e3']

for i, feature in enumerate(features[:-1], 1):
    fig, ax = plt.subplots(figsize=(6, 5))
    grouped_mean[feature].plot.bar(ax = ax, yerr=grouped_std[feature], capsize=4, color=colors[:len(feature)])
    for j, bar in enumerate(ax.patches):
        feature_index = j // len(grouped_mean.index)
        if feature[feature_index] != 'None':
            none_average_time = grouped_mean['None'].iloc[j % len(grouped_mean.index)]
            feature_average_time = grouped_mean[feature[feature_index]].iloc[j % len(grouped_mean.index)]
            overhead = ((feature_average_time / none_average_time) - 1) * 100
            ax.text(bar.get_x() + bar.get_width() * 1.1, bar.get_height() + grouped_std[feature[feature_index]].iloc[j % len(grouped_mean.index)] + 1, f'+{overhead:.2f}%', ha='center', va='bottom', fontsize=10, rotation=30)

    ncol = 2 if len(feature) > 2 else len(feature)
    ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=10)

    plt.xticks(rotation=0, fontsize=12)
    plt.tick_params(axis='x', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('Time Elapsed (minutes)', fontsize=12)
    
    if arg == 'resnet':
        ax.set_ylim(0, 160)
    else:
        ax.set_ylim(0, 3000)
    
    plt.tight_layout()
    
    plt.savefig(f'{arg}_{i}.svg', format='svg')
    plt.show()

combined_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']    

fig, ax = plt.subplots(figsize=(6, 5))
feature = features[-1]
grouped_mean[feature].plot.bar(ax=ax, yerr=grouped_std[feature], capsize=4, color=combined_colors[:len(feature)])

ncol = 2 if len(feature) > 2 else len(feature)
ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=10)

plt.xticks(rotation=0, fontsize=12)
plt.tick_params(axis='x', length=0)
ax.set_xlabel('')
ax.set_ylabel('Time Elapsed (minutes)', fontsize=12)
plt.tight_layout()

plt.savefig(f'{arg}_6.svg', format='svg')
plt.show()