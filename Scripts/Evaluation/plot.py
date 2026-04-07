import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Setup Publication Quality Parameters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.25,
    'xtick.major.width': 1.25,
    'ytick.major.width': 1.25,
    'figure.dpi': 300
})

# 2. Input Data from your results
tau = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Domain Comparison Data
med_recall = [0.502, 0.502, 0.660, 0.660, 0.686, 0.798, 0.903]
med_k = [1.00, 1.00, 1.76, 1.76, 2.00, 2.78, 3.78]
cs_recall = [0.396, 0.396, 0.575, 0.575, 0.586, 0.734, 0.863]
cs_k = [1.00, 1.00, 1.86, 1.86, 2.00, 2.87, 3.87]

# Seniority Comparison Data
senior_recall = [0.374, 0.374, 0.536, 0.536, 0.556, 0.700, 0.844]
senior_k = [1.00, 1.00, 1.84, 1.84, 2.00, 2.85, 3.85]
junior_recall = [0.346, 0.346, 0.524, 0.524, 0.542, 0.713, 0.860]
junior_k = [1.00, 1.00, 1.86, 1.86, 2.00, 2.87, 3.87]

# 3. Create Multi-Panel Figure
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

def plot_panel(ax_main, tau, recall1, recall2, k1, k2, label1, label2, color1, color2, title):
    # Left Axis: Recall
    ax_main.plot(tau, recall1, 'o-', color=color1, label=f'{label1} Recall', linewidth=2, markersize=7)
    ax_main.plot(tau, recall2, 's-', color=color2, label=f'{label2} Recall', linewidth=2, markersize=7)
    
    ax_main.set_xlabel('Confidence Threshold (τ)', fontweight='bold')
    ax_main.set_ylabel('Recall (True Author in Set)', fontweight='bold')
    ax_main.set_ylim(0, 1.05)
    ax_main.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Chance (1/5)')
    ax_main.grid(True, which='both', linestyle=':', alpha=0.4)
    ax_main.set_title(title, loc='left', fontweight='bold', fontsize=12)

    # Right Axis: Suspect Set Size (K)
    ax_k = ax_main.twinx()
    ax_k.plot(tau, k1, '^--', color=color1, alpha=0.4, label=f'{label1} Avg K', markersize=5)
    ax_k.plot(tau, k2, 'v--', color=color2, alpha=0.4, label=f'{label2} Avg K', markersize=5)
    ax_k.set_ylabel('Avg. Suspect Set Size (K)', fontweight='bold', color='gray')
    ax_k.set_ylim(1, 5)
    
    # Legend handling
    lines_1, labels_1 = ax_main.get_legend_handles_labels()
    lines_2, labels_2 = ax_k.get_legend_handles_labels()
    ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8, frameon=True)

# Generate Panel A
plot_panel(ax1, tau, med_recall, cs_recall, med_k, cs_k, 
           'Medicine', 'CompSci', '#d7191c', '#2c7bb6', 'a Domain Sensitivity')

# Generate Panel B
plot_panel(ax3, tau, senior_recall, junior_recall, senior_k, junior_k, 
           'Senior', 'Junior', '#7b3294', '#008837', 'b Seniority Resilience')

# 4. Export for Manuscript
plt.savefig('authorship_analysis_nature.pdf', format='pdf', bbox_inches='tight')
plt.savefig('authorship_analysis_nature.png', dpi=300)
plt.show()