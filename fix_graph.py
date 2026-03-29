import matplotlib.pyplot as plt

labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
accuracies = [97.15, 97.07, 97.02, 97.01, 96.99, 96.96, 96.94, 96.91, 96.90, 96.89]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(labels, accuracies, marker='o', color='#1a5276', linewidth=2, markersize=8, markerfacecolor='#2980b9')

ax.set_xlabel('Hyperparameter Combination', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Hyperparameter Combinations - HGBC', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(96.85, 97.20)

for label, acc in zip(labels, accuracies):
    ax.annotate(f'{acc:.2f}', (label, acc), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('hyperparameter_combination_line.png', dpi=300, bbox_inches='tight')
plt.show()
