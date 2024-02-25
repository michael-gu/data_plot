import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('resnet18_bm.csv')

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
combined_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

# Define a dictionary that maps each feature to a specific hatch pattern
hatches = {
    'None': '/',
    'Vanilla': '\\',
    'Multithreaded': '|',
    'Cryptographic Hash': '-',
    'Multithreaded Cryptographic Hash': '+',
    'Multithreaded Cryptographic Hash and Plain Model': 'x'
}

for i, feature in enumerate(features[:-1], 1):
    fig, ax = plt.subplots(figsize=(10, 9))
    grouped_mean[feature].plot.bar(ax = ax, yerr=grouped_std[feature], capsize=4, color=colors[:len(feature)])
    for j, bar in enumerate(ax.patches):
        feature_index = j // len(grouped_mean.index)
        if feature[feature_index] != 'None':
            none_average_time = grouped_mean['None'].iloc[j % len(grouped_mean.index)]
            feature_average_time = grouped_mean[feature[feature_index]].iloc[j % len(grouped_mean.index)]
            overhead = ((feature_average_time / none_average_time) - 1) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + grouped_std[feature[feature_index]].iloc[j % len(grouped_mean.index)] + 1, f'+{overhead:.2f}%', ha='center', va='bottom', fontsize=8)

    ncol = 3 if len(feature) > 3 else len(feature)
    ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center')

    plt.xticks(rotation=0)
    plt.tick_params(axis='x', length=0)
    ax.set_xlabel('')
    ax.set_ylabel('Time Elapsed (minutes)')
    plt.tight_layout()
    
    plt.savefig(f'{i}.svg', format='svg')
    plt.show()
    

fig, ax = plt.subplots(figsize=(10, 9))
feature = features[-1]
grouped_mean[feature].plot.bar(ax=ax, yerr=grouped_std[feature], capsize=4, color=combined_colors[:len(feature)])

ncol = 3 if len(feature) > 3 else len(feature)
ax.legend(ncol=ncol, bbox_to_anchor=(0.5, 1), loc='lower center')

plt.xticks(rotation=0)
plt.tick_params(axis='x', length=0)
ax.set_xlabel('')
ax.set_ylabel('Time Elapsed (minutes)')
plt.tight_layout()

plt.savefig(f'{i}.svg', format='svg')
plt.show()