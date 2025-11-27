import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Video Search Data ---
video_methods = ['Semantic Search\n(FAISS)', 'TF-IDF\nBaseline']
video_top1 = [84.6, 76.9]
video_top5 = [100.0, 100.0]
video_latency = [64.8, 0.3]

# --- Music Recognition Data ---
music_conditions = ['Clean\n(Original)', 'Noisy\n(SNR ~40dB)']
music_accuracy = [85.7, 85.7]

# ===== VIDEO SEARCH CHARTS =====

# Chart 1: Video Search Accuracy Comparison
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(video_methods))
width = 0.35

bars1 = ax.bar(x - width/2, video_top1, width, label='Top-1 Accuracy', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, video_top5, width, label='Top-5 Accuracy', color='#2ecc71', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Video Search: Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(video_methods, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'video_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'video_accuracy_comparison.png'}")
plt.close()

# Chart 2: Video Search Latency Comparison
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(video_methods, video_latency, color=['#e74c3c', '#f39c12'], edgecolor='black', width=0.5)

ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax.set_title('Video Search: Query Latency', fontsize=14, fontweight='bold')
ax.set_yscale('log')  # Logarithmic scale due to large difference
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f} ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'video_latency_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'video_latency_comparison.png'}")
plt.close()

# ===== MUSIC RECOGNITION CHARTS =====

# Chart 3: Music Recognition Robustness
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(music_conditions, music_accuracy, color=['#9b59b6', '#e67e22'], edgecolor='black', width=0.5)

ax.set_ylabel('Identification Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Music Recognition: Robustness to Noise', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'music_robustness.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'music_robustness.png'}")
plt.close()

# ===== COMBINED SUMMARY TABLE =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Video (Semantic)', 'Video (TF-IDF)', 'Music (Clean)', 'Music (Noisy)'],
    ['Top-1 Accuracy', '84.6%', '76.9%', '85.7%', '85.7%'],
    ['Top-5 Accuracy', '100.0%', '100.0%', 'N/A', 'N/A'],
    ['Avg Latency', '64.8 ms', '0.3 ms', '118 ms', '118 ms']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.2, 0.2, 0.17, 0.18])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, 4):
    for j in range(5):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('#ffffff')
        cell.set_edgecolor('#bdc3c7')

plt.title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig(OUTPUT_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'summary_table.png'}")
plt.close()

print("\n✅ All visualizations generated successfully!")
print(f"📁 Output directory: {OUTPUT_DIR}")
