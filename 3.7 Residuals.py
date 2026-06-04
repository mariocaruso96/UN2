# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:50:51 2026

@author: WKS
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = "C:/Users/WKS/Desktop/UNIBA/UN/"

# =========================
# Settings
# =========================
data_years = np.arange(2010, 2026, 2)
model_name = "boos"

output_dir = f"{root}/Immagini/Residuals"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Build global residual dataframe
# =========================
rows = []

for data_year in data_years:

    egdi_years = np.arange(data_year, 2026, 2)

    for egdi_year in egdi_years:

        file_name = f"results_data_{data_year}_egdi_{egdi_year}.pkl"

        with open(f"{root}/Risultati/ML/{file_name}", "rb") as f:
            [
                _,  # shap_global_rf
                _,  # shap_global_boos
                _,  # test_idx_global
                _,  # train_idx_global
                _,  # explainer_rf_list
                _,  # explainer_boos_list
                _,  # c_table_rf
                _,  # c_table_boos
                _,  # rf results
                _,  # boos results
                _,  # svr results
                _,  # linr results
                egdi_true,
                y_pred_total_rf,
                y_pred_total_boos,
                y_pred_total_svr,
                y_pred_total_linr
            ] = pickle.load(f)

        # Select model
        if model_name == "rf":
            y_pred_total = y_pred_total_rf
        elif model_name == "boos":
            y_pred_total = y_pred_total_boos
        elif model_name == "svr":
            y_pred_total = y_pred_total_svr
        elif model_name == "linr":
            y_pred_total = y_pred_total_linr
        else:
            raise ValueError("model_name must be one of: rf, boos, svr, linr")

        # Align indexes
        common_idx = egdi_true.index.intersection(y_pred_total.index)

        true_vals = egdi_true.loc[common_idx].astype(float)
        pred_mean = y_pred_total.loc[common_idx].mean(axis=1).astype(float)
        pred_std = y_pred_total.loc[common_idx].std(axis=1).astype(float)

        residual = pred_mean - true_vals

        tmp = pd.DataFrame({
            "Country_ISO": common_idx,
            "data_year": data_year,
            "egdi_year": egdi_year,
            "horizon": egdi_year - data_year,
            "true_egdi": true_vals.values,
            "pred_egdi_mean": pred_mean.values,
            "pred_egdi_std": pred_std.values,
            "residual": residual.values
        })

        rows.append(tmp)

df_residuals = pd.concat(rows, axis=0).reset_index(drop=True)

# Optional save
df_residuals.to_csv(
    f"{output_dir}/global_residuals_{model_name}.csv",
    index=False
)

# =========================
# Plot 1: residuals vs true EGDI
# =========================
fig, ax = plt.subplots(figsize=(9, 6))

# =========================
# Hexbin density plot
# =========================
hb = ax.hexbin(
    df_residuals["true_egdi"],
    df_residuals["residual"],
    gridsize=45,
    cmap="Blues",
    mincnt=1,
    linewidths=0.2
)

# =========================
# Linear regression line
# =========================
sns.regplot(
    data=df_residuals,
    x="true_egdi",
    y="residual",
    scatter=False,
    ci=None,
    line_kws={
        "color": "darkred",
        "linewidth": 3
    },
    ax=ax
)

# =========================
# Horizontal zero line
# =========================
ax.axhline(
    0,
    linestyle="--",
    linewidth=2.0,
    color="black"
)

# =========================
# Labels
# =========================
ax.set_xlabel(
    "True EGDI",
    fontsize=20
)

ax.set_ylabel(
    r"Residual ($\widehat{EGDI} - EGDI$)",
    fontsize=20
)

# =========================
# Limits
# =========================
ax.set_xlim(0, 1)

# symmetric y-axis
ax.set_ylim(-0.35, 0.35)

# =========================
# Ticks
# =========================
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# =========================
# Colorbar
# =========================
cb = fig.colorbar(hb, ax=ax)

cb.set_label(
    "Counts",
    fontsize=20
)

cb.ax.tick_params(labelsize=18)

# =========================
# Title
# =========================
# ax.set_title(
#     "Global residual bias across temporal configurations",
#     fontsize=16
# )

# =========================
# Layout
# =========================
plt.tight_layout()

# =========================
# Save
# =========================
plt.savefig(
    f"{output_dir}/global_residuals_vs_true_egdi_{model_name}.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    f"{output_dir}/global_residuals_vs_true_egdi_{model_name}.pdf",
    bbox_inches="tight"
)

plt.show()

# =========================
# Plot 2: binned residuals by true EGDI
# =========================
egdi_bins = [0.0, 0.25, 0.50, 0.75, 1.0]

egdi_labels = [
    "Very low\n[0.00, 0.25)",
    "Low-medium\n[0.25, 0.50)",
    "Medium-high\n[0.50, 0.75)",
    "High\n[0.75, 1.00]"
]

df_residuals["EGDI_bin"] = pd.cut(
    df_residuals["true_egdi"],
    bins=egdi_bins,
    labels=egdi_labels,
    include_lowest=True,
    right=False
)

# Include exactly EGDI = 1.0 in last bin
df_residuals.loc[
    df_residuals["true_egdi"] == 1.0,
    "EGDI_bin"
] = egdi_labels[-1]

# =========================
# Summary statistics
# =========================
bin_summary = (
    df_residuals
    .groupby("EGDI_bin", observed=False)
    .agg(
        residual_mean=("residual", "mean"),
        residual_std=("residual", "std"),
        residual_sem=("residual", lambda x: x.std() / np.sqrt(len(x))),
        n=("residual", "size")
    )
    .reset_index()
)

# =========================
# Colors (increasing EGDI)
# =========================
# colors = [
#     "#4575b4",  # blue
#     "#74add1",  # light blue
#     "#fdae61",  # orange
#     "#d73027"   # red
# ]


colors = [
    "#dbe9f6",  # very light blue
    "#9ecae1",  # light-medium blue
    "#4292c6",  # medium-dark blue
    "#084594"   # dark blue
]

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    bin_summary["EGDI_bin"],
    bin_summary["residual_mean"],
    yerr=bin_summary["residual_sem"],
    capsize=6,
    color=colors,
    edgecolor="black",
    linewidth=1.2
)

# =========================
# Horizontal line
# =========================
ax.axhline(
    0,
    linestyle="--",
    linewidth=2,
    color="black"
)

# =========================
# Add n above bars
# =========================
for bar, n_val, mean_val in zip(
    bars,
    bin_summary["n"],
    bin_summary["residual_mean"]
):

    if mean_val >= 0:
        y_text = mean_val + 0.006
        va = "bottom"
    else:
        y_text = mean_val - 0.006
        va = "top"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_text,
        f"n = {n_val}",
        ha="center",
        va=va,
        fontsize=15,
        fontweight="bold"
    )

# =========================
# Labels
# =========================
ax.set_xlabel(
    "True EGDI class",
    fontsize=20
)

ax.set_ylabel(
    r"Mean residual ($\widehat{EGDI} - EGDI$)",
    fontsize=20
)

# =========================
# Symmetric y axis
# =========================
ax.set_ylim(-0.1, 0.1)

# =========================
# Ticks
# =========================
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# =========================
# Title
# =========================
# ax.set_title(
#     "Mean residual by EGDI range",
#     fontsize=20
# )

# =========================
# Layout
# =========================
plt.tight_layout()

# =========================
# Save
# =========================
plt.savefig(
    f"{output_dir}/binned_residuals_by_egdi_{model_name}.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    f"{output_dir}/binned_residuals_by_egdi_{model_name}.pdf",
    bbox_inches="tight"
)

plt.show()