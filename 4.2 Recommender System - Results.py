# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:30:35 2026

@author: WKS
"""

#%% 0)
# =========================
# 0) Funzioni e librerie
# =========================

import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import seaborn as sns


def classify_difference(diff_df):
    """
    Converte la heatmap numerica della differenza:
        final_egdi - target_egdi
    in una matrice categoriale.

    Classi:
        0: diff < -0.10
        1: -0.10 <= diff < 0
        2: 0 <= diff < 0.10
        3: diff >= 0.10
    """

    class_df = pd.DataFrame(
        np.nan,
        index=diff_df.index,
        columns=diff_df.columns
    )

    class_df[diff_df < -0.10] = 0
    class_df[(diff_df >= -0.10) & (diff_df < 0)] = 1
    class_df[(diff_df >= 0) & (diff_df < 0.10)] = 2
    class_df[diff_df >= 0.10] = 3

    return class_df


def add_gain_markers(ax, gain_df, threshold=0.0, fontsize=24):
    """
    Aggiunge marker nelle celle:
        ↑ se final_egdi > starting_egdi
        × altrimenti
    """

    for y_idx, rho in enumerate(gain_df.index):
        for x_idx, K in enumerate(gain_df.columns):

            val = gain_df.loc[rho, K]

            if pd.isna(val):
                continue

            marker = "↑" if val > threshold else "×"

            ax.text(
                x_idx + 0.5,
                y_idx + 0.5,
                marker,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="black"
            )


def format_heatmap_axis(
    ax,
    df,
    xlabel="K",
    ylabel=r"$\rho$",
    labelsize=38,
    ticksize=30,
    x_tick_step=2,
    y_tick_step=2
):
    """
    Formattazione assi heatmap con tick diradati.
    La granularità della heatmap resta invariata.
    """

    ax.set_xlabel(
        xlabel,
        fontsize=labelsize,
        fontweight="normal"
    )

    ax.set_ylabel(
        ylabel,
        fontsize=labelsize,
        fontweight="normal"
    )

    # Tick X diradati
    x_positions = np.arange(len(df.columns)) + 0.5
    x_labels = list(df.columns)

    x_keep = np.arange(0, len(x_labels), x_tick_step)

    ax.set_xticks(x_positions[x_keep])
    ax.set_xticklabels(
        [x_labels[i] for i in x_keep],
        fontsize=ticksize,
        rotation=0
    )

    # Tick Y diradati
    y_positions = np.arange(len(df.index)) + 0.5
    y_labels = [f"{x:.2f}" for x in df.index]

    y_keep = np.arange(0, len(y_labels), y_tick_step)

    ax.set_yticks(y_positions[y_keep])
    ax.set_yticklabels(
        [y_labels[i] for i in y_keep],
        fontsize=ticksize,
        rotation=0
    )

#%% 1)
# =========================
# 1) Upload dati
# =========================

root = "C:/Users/WKS/Desktop/UNIBA/UN/RS/Risultati"

save_plots = True
year = 2022
country = "CIV"

print(f"Selected country: {country}.")

path_country_files = f"{root}/{year}/{country}"

all_runs = {}

for num_file, run_file_name in enumerate(os.listdir(path_country_files)):

    run_path = os.path.join(path_country_files, run_file_name)

    if not os.path.isdir(run_path):
        continue

    print(f"{country} ---- File number: {num_file + 1} ---- run id: {run_file_name}")

    policy_df_total = None
    results_df_total = None
    jaccard_kendall_summary = None
    parameters = None

    for file_name in os.listdir(run_path):

        file_path = os.path.join(run_path, file_name)

        if file_name == "policy_df_total.csv":
            policy_df_total = pd.read_csv(file_path)
            if "Unnamed: 0" in policy_df_total.columns:
                policy_df_total = policy_df_total.drop(["Unnamed: 0"], axis=1)

        elif file_name == "results_df_total.csv":
            results_df_total = pd.read_csv(file_path)
            if "Unnamed: 0" in results_df_total.columns:
                results_df_total = results_df_total.drop(["Unnamed: 0"], axis=1)

        elif file_name == "jaccard_kendall_summary.json":
            with open(file_path, "r", encoding="utf-8") as f:
                jaccard_kendall_summary = json.load(f)

        elif file_name == "parameters.json":
            with open(file_path, "r", encoding="utf-8") as f:
                parameters = json.load(f)

    if policy_df_total is None or results_df_total is None or jaccard_kendall_summary is None or parameters is None:
        print(f"WARNING: incomplete run skipped: {run_file_name}")
        continue

    K = parameters["K"]
    rho = parameters["rho"]
    required_egdi_delta = parameters["required_egdi_delta"]

    key = f"K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"

    all_runs[key] = {
        "rho": rho,
        "K": K,
        "required_egdi_delta": required_egdi_delta,
        "run_file_name": run_file_name,
        "policy_df_total": policy_df_total,
        "results_df_total": results_df_total,
        "jaccard_kendall_summary": jaccard_kendall_summary,
    }

# =========================
# 2) Organizzazione risultati
# =========================

required_egdi_delta_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.5])

K_vals = np.arange(2, 21, 1)

rho_vals = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95
])

df_global = pd.DataFrame()

for required_egdi_delta in required_egdi_delta_vals:
    for K in K_vals:
        for rho in rho_vals:

            key = f"K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"

            if key not in all_runs:
                print(f"Missing run: {key}")
                continue

            dictionary = all_runs[key]

            jk_summary = dictionary["jaccard_kendall_summary"]
            results_df_total = dictionary["results_df_total"]

            jaccard_mean = jk_summary["jaccard_mean"]
            kendall_mean = jk_summary["kendall_mean"]

            success_rate = results_df_total["success"].sum() / results_df_total.shape[0]
            budget_spent_mean = results_df_total["budget_spent"].mean()

            difference_mean = results_df_total["final_egdi_minus_target_egdi"].mean()
            difference_std = results_df_total["final_egdi_minus_target_egdi"].std()

            egdi_gain_mean = (
                results_df_total["final_egdi"] -
                results_df_total["starting_egdi"]
            ).mean()

            egdi_gain_std = (
                results_df_total["final_egdi"] -
                results_df_total["starting_egdi"]
            ).std()

            row = pd.DataFrame({
                "required_egdi_delta": required_egdi_delta,
                "K": K,
                "rho": rho,
                "jaccard_mean": jaccard_mean,
                "kendall_mean": kendall_mean,
                "success_rate": success_rate,
                "budget_spent_mean": budget_spent_mean,
                "difference_mean": difference_mean,
                "difference_std": difference_std,
                "egdi_gain_mean": egdi_gain_mean,
                "egdi_gain_std": egdi_gain_std,
            }, index=[0])

            df_global = pd.concat([df_global, row], axis=0)

df_global = df_global.reset_index(drop=True)

# =========================
# 3) Pivot tables
# =========================

results_by_delta = {}

metrics = [
    "jaccard_mean",
    "kendall_mean",
    "success_rate",
    "budget_spent_mean",
    "difference_mean",
    "difference_std",
    "egdi_gain_mean",
    "egdi_gain_std",
]

for required_egdi_delta in required_egdi_delta_vals:

    df = df_global[
        df_global["required_egdi_delta"] == required_egdi_delta
    ].copy()

    results_by_delta[required_egdi_delta] = {}

    for metric in metrics:

        pivot = df.pivot_table(
            index="rho",
            columns="K",
            values=metric,
            aggfunc="mean"
        )

        results_by_delta[required_egdi_delta][metric] = pivot

# =========================
# 4) Impostazioni plot
# =========================

output_dir = f"C:/Users/WKS/Desktop/UNIBA/UN/Immagini/Recommender System/{country}"
os.makedirs(output_dir, exist_ok=True)

# Scala continua centrata in zero
global_min = -0.1
global_max = 0.1

# Classi categoriali:
# 0 = far below target
# 1 = slightly below target
# 2 = target reached
# 3 = target exceeded

class_cmap = ListedColormap([
    "#d73027",  # red
    "#fdae61",  # orange
    "#a6d96a",  # light green
    "#1a9850",  # emerald green
])

class_norm = BoundaryNorm(
    boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
    ncolors=class_cmap.N
)

class_labels = [
    "Far below target",
    "Slightly below target",
    "Target reached",
    "Target exceeded"
]


    
#%% 2)
# =========================
# 5) Figura unica: tutte le required delta
# =========================

fig = plt.figure(figsize=(48, 38))

gs = GridSpec(
    nrows=6,
    ncols=5,
    figure=fig,
    width_ratios=[0.08, 1.50, 1.50, 0.08, 0.02],
    wspace=0.25,
    hspace=0.18
)

continuous_axes = []
mask_axes = []

for i, required_egdi_delta in enumerate(required_egdi_delta_vals):

    ax_cont = fig.add_subplot(gs[i, 1])
    ax_mask = fig.add_subplot(gs[i, 2])

    continuous_axes.append(ax_cont)
    mask_axes.append(ax_mask)

cbar_ax_cont = fig.add_subplot(gs[:, 0])
cbar_ax_mask = fig.add_subplot(gs[:, 3])

for i, required_egdi_delta in enumerate(required_egdi_delta_vals):

    diff_df = results_by_delta[required_egdi_delta]["difference_mean"].copy()
    diff_df = diff_df.sort_index(ascending=False)

    gain_df = results_by_delta[required_egdi_delta]["egdi_gain_mean"].copy()
    gain_df = gain_df.sort_index(ascending=False)

    class_df = classify_difference(diff_df)

    # -------------------------
    # Heatmap continua
    # -------------------------

    sns.heatmap(
        diff_df,
        annot=False,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=global_min,
        vmax=global_max,
        cbar=(i == 0),
        cbar_ax=cbar_ax_cont if i == 0 else None,
        cbar_kws={
            "label": r"$\hat{E}_{final} - \hat{E}_{target}$"
        } if i == 0 else None,
        ax=continuous_axes[i]
    )

    # Label interna bianca, in basso a sinistra
    continuous_axes[i].text(
        0.02,
        0.08,
        rf"$\Delta_{{EGDI}}^{{req}} = {required_egdi_delta}$",
        transform=continuous_axes[i].transAxes,
        fontsize=50,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.25",
            alpha=0.90
        )
    )

    continuous_axes[i].set_title("")

    format_heatmap_axis(
    continuous_axes[i],
    diff_df,
    labelsize=38,
    ticksize=26,
    x_tick_step=2,
    y_tick_step=2
    )

    # K non bold
    continuous_axes[i].xaxis.label.set_fontweight("normal")

    # -------------------------
    # Heatmap categoriale
    # -------------------------

    sns.heatmap(
        class_df,
        annot=False,
        cmap=class_cmap,
        norm=class_norm,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=mask_axes[i]
    )

    add_gain_markers(
        mask_axes[i],
        gain_df,
        fontsize=24
    )

    mask_axes[i].set_title("")

    format_heatmap_axis(
    mask_axes[i],
    class_df,
    labelsize=38,
    ticksize=26,
    x_tick_step=2,
    y_tick_step=2)
    
    # K non bold
    mask_axes[i].xaxis.label.set_fontweight("normal")

# =========================
# Colorbar continua
# =========================

cbar_ax_cont.tick_params(
    labelsize=100
)

cbar_ax_cont.set_ylabel(
    r"$\hat{E}_{final} - \hat{E}_{target}$",
    fontsize=100,
    fontweight="bold",
    labelpad=5
)

# =========================
# Tick e label a sinistra
# =========================

cbar_ax_cont.yaxis.set_ticks_position('left')
cbar_ax_cont.yaxis.set_label_position('left')

cbar_ax_cont.tick_params(
    labelleft=True,
    labelright=False,
    pad=10
)

# =========================
# Colorbar categoriale
# =========================

sm = ScalarMappable(
    cmap=class_cmap,
    norm=class_norm
)

sm.set_array([])

cbar2 = fig.colorbar(
    sm,
    cax=cbar_ax_mask,
    ticks=[0, 1, 2, 3]
)

cbar2.ax.set_yticklabels(class_labels)

cbar2.ax.tick_params(
    labelsize=60,
    pad=40
)

for tick in cbar2.ax.get_yticklabels():
    tick.set_rotation(90)
    tick.set_horizontalalignment("center")
    tick.set_verticalalignment("center")

cbar2.set_label(
    "Target-status class",
    fontsize=100,
    fontweight="bold",
    labelpad=40
)

# =========================
# Spaziatura finale
# =========================

fig.subplots_adjust(
    left=0.045,
    right=0.955,
    top=0.985,
    bottom=0.03,
    wspace=0.14,
    hspace=0.16
)

if save_plots:

    plt.savefig(
        f"{output_dir}/{country}_all_required_egdi_delta_continuous_and_mask.png",
        dpi=500,
        bbox_inches="tight"
    )

    print(
        f"{country} ---- all required EGDI delta paired heatmaps saved."
    )

plt.show()