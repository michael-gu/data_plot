import matplotlib.pyplot as plt
import numpy as np

# Data
data = np.genfromtxt('final_bm - Vanilla.csv', delimiter=',', skip_header=1, names=True)

# Plot settings
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(data[''], data['None'], label='None', color='red')
ax.bar(data[''], data['Vanilla'], label='Vanilla', color='blue')

ax.set_xlabel('')
ax.set_ylabel('Elapsed Time (minutes)')
ax.set_xticks(np.arange(len(data)))
ax.set_xticklabels(data[''], rotation=90)

ax.legend(fontsize=10)
ax.margins(y=0.1)

plt.savefig('vanilla.pdf')
plt.show()
