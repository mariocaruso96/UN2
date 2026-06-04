# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:37:36 2026

@author: WKS
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

# =========================
# Load data
# =========================
root = "C:/Users/WKS/Desktop/UNIBA/UN"

data = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")

data = data.drop(["EGDI_2008"], axis=1)
data = data.set_index("Countries")

# =========================
# Pearson correlation matrix
# =========================
corr_matrix = data.corr(method="pearson")

# =========================
# Rename columns for plotting
# =========================
plot_corr = corr_matrix.copy()

plot_corr.columns = [
    col.replace("EGDI_", "")
    for col in plot_corr.columns
]

plot_corr.index = [
    idx.replace("EGDI_", "")
    for idx in plot_corr.index
]


# =========================
# Plot
# =========================
plt.figure(figsize=(10, 8))

ax = sns.heatmap(
    plot_corr,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0.8,
    vmax=1,
    square=True,
    linewidths=0.5,
    
    # dimensione testo celle
    annot_kws={"size": 18},

    # colorbar
    cbar_kws={"shrink": 1.0}
)

# tick labels colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
ax.xaxis.tick_top()

plt.xticks(fontsize=15, rotation=25)
plt.yticks(fontsize=15, rotation=25)
# plt.title("Pearson Correlation Matrix of EGDI Years")
plt.tight_layout()
plt.savefig(f"{root}/Immagini/Paper/EGDI_correlations/egdi_correlations.png", dpi = 300)
plt.show()

# =========================
# Optional: p-values matrix
# =========================
pval_matrix = pd.DataFrame(
    np.ones((len(data.columns), len(data.columns))),
    columns=data.columns,
    index=data.columns
)

for col1 in data.columns:
    for col2 in data.columns:
        r, p = pearsonr(data[col1], data[col2])
        pval_matrix.loc[col1, col2] = p

print(corr_matrix)
print("\nP-values:")
print(pval_matrix)
