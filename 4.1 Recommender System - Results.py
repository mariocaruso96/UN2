# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:14:19 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import os
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

#%% 1) Upload dati

root = "C:/Users/WKS/Desktop/UNIBA/UN/RS/Risultati"

year = 2022
country = "BRA"
print(f"Selected country: {country}.")

path_country_files = f"{root}/{year}/{country}"

# Scorri tra le run
for num_file, run_file_name in enumerate(os.listdir(path_country_files)):
    print(f"{country} ---- File number: {num_file+1} ---- run id: {run_file_name}")
    
    # Scorri sui file del path della run
    for file_name in os.listdir(path_country_files + "/" + run_file_name):
        
        # Policy df
        if file_name == "policy_df_total.csv":
            policy_df_total = pd.read_csv(path_country_files + "/" + run_file_name + "/" + file_name)
            policy_df_total = policy_df_total.drop(["Unnamed: 0"], axis = 1)
        
        # Results df
        if file_name == "results_df_total.csv":
            results_df_total = pd.read_csv(path_country_files + "/" + run_file_name + "/" + file_name)
            results_df_total = results_df_total.drop(["Unnamed: 0"], axis = 1)
        
        # Jaccard-Kendall summary
        if file_name == "jaccard_kendall_summary.json":
            with open(path_country_files + "/" + run_file_name + "/" + file_name, "r", encoding="utf-8") as f:
                jaccard_kendall_summary = json.load(f)
                del f
        
        # Parametri del json
        if file_name == "parameters.json":
            with open(path_country_files + "/" + run_file_name + "/" + file_name, "r", encoding="utf-8") as f:
                parameters = json.load(f)
                del f
            
            # Salvo i parametri
            K = parameters["K"]
            required_egdi_delta = parameters["required_egdi_delta"]
            rho = parameters["rho"]
                
            del parameters
    
    # Creo un dizionario con tutto salvato
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"] = {}
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["rho"] = rho
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["K"] = K
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["required_egdi_delta"] = required_egdi_delta
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["run_file_name"] = run_file_name
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["policy_df_total"] = policy_df_total
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["results_df_total"] = results_df_total
    globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]["jaccard_kendall_summary"] = jaccard_kendall_summary
    
    del file_name, rho, K, required_egdi_delta, policy_df_total, results_df_total, jaccard_kendall_summary, num_file
del run_file_name, path_country_files

########## 2) Data results organization

# budget = 1.0 sempre

required_egdi_delta_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.5])
K_vals = np.arange(2, 21, 1)
rho_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

globals()["df_global"] = pd.DataFrame()

for required_egdi_delta in required_egdi_delta_vals:
    for K in K_vals:
        for rho in rho_vals:
        
            # Dizionario particolare
            dictionary = globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"].copy()
            
            # Jaccard-Kendall summary results
            jk_summary = dictionary["jaccard_kendall_summary"]
            jaccard_mean = jk_summary["jaccard_mean"]
            kendall_mean = jk_summary["kendall_mean"]
            
            # Results summary
            results_df_total = dictionary["results_df_total"]
            success_rate = results_df_total["success"].sum()/results_df_total.shape[0]
            budget_spent_mean =  results_df_total["budget_spent"].mean()
            difference_mean = results_df_total["final_egdi_minus_target_egdi"].mean()
            difference_std = results_df_total["final_egdi_minus_target_egdi"].std()
            
            # Aggiorno df
            row = pd.DataFrame({"required_egdi_delta": required_egdi_delta,
                                "K": K,
                                "rho": rho,
                                "jaccard_mean": jaccard_mean,
                                "kendall_mean": kendall_mean,
                                "success_rate": success_rate,
                                "budget_spent_mean": budget_spent_mean,
                                "difference_mean": difference_mean,
                                "difference_std": difference_std}, index = [0])
            
            globals()["df_global"] = pd.concat([globals()["df_global"], row], axis = 0)
            
            del row
            del dictionary
            del jk_summary, jaccard_mean, kendall_mean
            del results_df_total, success_rate, budget_spent_mean, difference_mean, difference_std
            del globals()[f"dict_K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"]
            del rho
        del K
    del required_egdi_delta
del K_vals, rho_vals

########## 3) Long table to pivot tables

required_egdi_delta_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.5])

for required_egdi_delta in required_egdi_delta_vals:
    
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"] = {}
    
    df = globals()["df_global"][globals()["df_global"]["required_egdi_delta"] == required_egdi_delta].copy()
    
    # Jaccard
    pivot_jaccard = df.pivot_table(
        index="rho",
        columns="K",
        values="jaccard_mean",
        aggfunc="mean" 
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["jaccard_mean"] = pivot_jaccard
    del pivot_jaccard
    
    # Kendall
    pivot_kendall = df.pivot_table(
        index="rho",
        columns="K",
        values="kendall_mean",
        aggfunc="mean" 
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["kendall_mean"] = pivot_kendall
    del pivot_kendall
    
    # Success rate
    pivot_success = df.pivot_table(
        index="rho",
        columns="K",
        values="success_rate",
        aggfunc="mean"   
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["success_rate"] = pivot_success
    del pivot_success
    
    # Budget spent
    pivot_budget = df.pivot_table(
        index="rho",
        columns="K",
        values="budget_spent_mean",
        aggfunc="mean"   
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["budget_spent_mean"] = pivot_budget
    del pivot_budget
    
    # Difference_mean
    pivot_diff_mean = df.pivot_table(
        index="rho",
        columns="K",
        values="difference_mean",
        aggfunc="mean"   
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["difference_mean"] = pivot_diff_mean
    del pivot_diff_mean
    
    # Difference_std
    pivot_diff_std = df.pivot_table(
        index="rho",
        columns="K",
        values="difference_std",
        aggfunc="mean"   
    )
    globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["difference_std"] = pivot_diff_std
    del pivot_diff_std
    
    del df, required_egdi_delta
del required_egdi_delta_vals

########## 4) Plots

save_plots = True
required_egdi_delta_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.5])

# =========================
# STEP 1: trova min/max globali
# =========================
# global_min = np.inf
# global_max = -np.inf

# for required_egdi_delta in required_egdi_delta_vals:
#     df = globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["difference_mean"]
#     global_min = min(global_min, df.min().min())
#     global_max = max(global_max, df.max().max())

# global_min = round(global_min, 1)
# global_max = -global_min

global_min = -0.5
global_max = -global_min

print("Global min:", global_min)
print("Global max:", global_max)

# =========================
# STEP 2: plot con stessa scala
# =========================

dfs = {}

for required_egdi_delta in required_egdi_delta_vals:

    df = globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["difference_mean"].copy()
    df = df.sort_index(ascending = False)
    
    # Distribuzione valori
    df_values = df.values.flatten()
    median = np.median(df_values)
    
    # Normalizzazione coerente con heatmap
    norm = colors.TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)
    
    # Colormap
    cmap = plt.get_cmap("viridis")
    
    # Colore associato alla mediana
    median_color = cmap(norm(median))
    
    dfs[required_egdi_delta] = {"df_values": df_values,
                                "df_median": median,
                                "df_color": median_color}
    
    del df_values, median


    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df,
        annot=False,
        fmt=".3f",
        cmap="viridis",
        center=0,
        vmin=global_min,
        vmax=global_max,
        cbar_kws={'label': 'Difference'},
        ax=ax
    )

    ax.set_xlabel("K", fontsize=20)
    ax.set_ylabel("rho", fontsize=20)
    ax.set_title(f"{country} --- Required EGDI delta: {required_egdi_delta}\nHeatmap (rho vs K)", fontsize=20)

    # Tick asse x
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticklabels(df.columns, fontsize=15)

    # Tick asse y centrati e con 2 decimali
    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_yticklabels([f"{x:.2f}" for x in df.index], rotation=0, fontsize=15)

    plt.tight_layout()
    if save_plots:
        output_dir = f"C:/Users/WKS/Desktop/UNIBA/UN/Immagini/Recommender System/{country}"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{country}_required_egdi_delta_{required_egdi_delta}.png", dpi = 300)
        
        print(f"{country} ---- Required EGDI delta = {required_egdi_delta} ---- File saved.")
    plt.show()

    del df


for required_egdi_delta in required_egdi_delta_vals:
    plt.hist(dfs[required_egdi_delta]["df_values"], bins = 10, color = dfs[required_egdi_delta]["df_color"], alpha = 0.8)
    # plt.axvline(dfs[required_egdi_delta]["df_median"], color = dfs[required_egdi_delta]["df_color"])
plt.axvline(0, color = "red", label = "Predicted EGDI = Target EGDI")
plt.title(f"{country} ---- Difference values: Predicted EGDI - Target EGDI")
plt.xlabel("Difference values: Predicted EGDI - Target EGDI")
plt.xlim((global_min, global_max))
plt.ylabel("#")
plt.ylim((0, 100))
plt.legend(fontsize = "small")
plt.show()


# =========================
# STEP 3: unica figura con 6 sottofigure
# =========================

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(30, 17))

gs = GridSpec(
    2, 4,
    figure=fig,
    width_ratios=[1, 1, 1, 0.1],
    wspace=0.22,
    hspace=0.22
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
]

cbar_ax = fig.add_subplot(gs[:, 3])

for i, required_egdi_delta in enumerate(required_egdi_delta_vals):

    df = globals()[f"results_required_egdi_delta_{required_egdi_delta}"]["difference_mean"].copy()
    df = df.sort_index(ascending=False)

    sns.heatmap(
        df,
        annot=False,
        fmt=".3f",
        cmap="viridis",
        center=0,
        vmin=global_min,
        vmax=global_max,
        cbar=(i == 0),
        cbar_ax=cbar_ax if i == 0 else None,
        cbar_kws={"label": "Difference"} if i == 0 else None,
        ax=axes[i]
    )

    axes[i].set_title(
        rf"Required $\Delta_{{EGDI}}$ = {required_egdi_delta}",
        fontsize=28
    )

    axes[i].set_xlabel("K", fontsize=22)
    axes[i].set_ylabel(r"$\rho$", fontsize=22)

    axes[i].set_xticks(np.arange(len(df.columns)) + 0.5)
    axes[i].set_xticklabels(df.columns, fontsize=18)

    axes[i].set_yticks(np.arange(len(df.index)) + 0.5)
    axes[i].set_yticklabels(
        [f"{x:.2f}" for x in df.index],
        rotation=0,
        fontsize=18
    )

# Colorbar più leggibile
cbar_ax.tick_params(labelsize=28)
cbar_ax.set_ylabel("Difference", fontsize=28)

fig.suptitle(
    f"{country} — Difference Mean Heatmaps (Predicted EGDI - Target EGDI)",
    fontsize=40,
    y=0.95
)

if save_plots:
    output_dir = f"C:/Users/WKS/Desktop/UNIBA/UN/Immagini/Recommender System/{country}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/{country}_all_required_egdi_delta_heatmaps.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()



    
            


