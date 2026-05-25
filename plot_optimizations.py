import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['True Positives (Matches)', 'False Positives (Ghosts)', 'False Negatives (Misses)']
before = [1642, 24958, 8639]
after = [1424, 22618, 7753]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, before, width, label='Before Physics Filters', color='#ff6b6b')
rects2 = ax.bar(x + width/2, after, width, label='After Physics Filters', color='#4ecdc4')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count (12 Videos Aggregate)')
ax.set_title('HELLA Radar-Camera Sensor Fusion: Optimization Results')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

# Save the chart
plt.savefig('Output_Visualizations/optimization_chart.png', dpi=300)
print("Chart successfully saved to Output_Visualizations/optimization_chart.png")
