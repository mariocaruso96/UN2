# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:05:34 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def results_matrix(medians_matr, iqr_matr, model, metric, path, cbar=False, regional_results=False, save=False):
    
    data_years = [str(year)[5:] for year in list(medians_matr.index)]
    egdi_years = [str(year)[5:] for year in list(medians_matr.columns)]
    
    if metric == "R^2":
        cmap_name = "Blues"
        vmin, vmax = 0.5, 0.9
        if model == "linr":
            vmin, vmax = -0.5, 0.5
    
    elif metric == "RMSE":
        cmap_name = "Oranges"
        vmin, vmax = 0.0, 0.1
    
    elif metric == "MAPE":
        cmap_name = "Purples"
        vmin, vmax = 0.1, 1.0

    else:
        # fallback sensato
        cmap_name = "viridis"
        vmin, vmax = np.nanmin(medians_matr.values), np.nanmax(medians_matr.values)

    # --- colormap & normalization (serve per leggere il colore reale di ogni cella) ---
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(20, 6))

    sns.heatmap(
        medians_matr,
        annot=False,
        cmap=cmap_name,
        cbar=cbar,
        linewidths=1,
        square=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )

    # Helper: luminanza percettiva di un colore RGB (sRGB)
    def luminance(rgb):
        r, g, b = rgb[:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Annotazioni con colore dinamico
    for i, sdg_year in enumerate(list(medians_matr.index)):
        for j, egdi_year in enumerate(list(medians_matr.columns)):

            if j < i:
                text = " "
                # (opzionale) salta proprio il disegno del testo
                # continue
            else:
                median_val = round(float(medians_matr.loc[sdg_year, egdi_year]), 2)
                iqr_val = round(float(iqr_matr.loc[sdg_year, egdi_year]), 2)
                text = f"{median_val} ± {iqr_val}"

            # Colore di sfondo della cella basato sul valore median
            val = medians_matr.loc[sdg_year, egdi_year]
            rgba = cmap(norm(val))
            lum = luminance(rgba)

            # Soglia: se sfondo chiaro -> testo nero, altrimenti bianco
            txt_color = "black" if lum > 0.55 else "white"

            ax.text(
                j + 0.5, i + 0.5, text,
                ha="center", va="center",
                fontsize=15,
                color=txt_color,
            )
    if regional_results:
        title = f"All {model.upper()} results for {metric} metric. 100 Repetitions Leave One Area Out."
    else:
        title = f"All {model.upper()} results for {metric} metric. 100 Repetitions 10 folds CV."

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("EGDI years", fontsize=18)
    ax.set_ylabel("Data years", fontsize=18)
    
    xticks_pos = [x + 0.5 for x in range(medians_matr.shape[1])]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(egdi_years, fontsize=15)
    
    yticks_pos = [y + 0.5 for y in range(medians_matr.shape[0])]
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(data_years, fontsize=15)

    if save:
        if regional_results:
            path_to_save = f"{path}/Immagini/Risultati Globali/results_{model}_{metric}_regional.png"
        else:
            path_to_save = f"{path}/Immagini/Risultati Globali/results_{model}_{metric}.png"

        plt.savefig(
            path_to_save,
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()
    
root = "C:/Users/WKS/Desktop/UNIBA/UN/"
    
#%% 1) Caricamento dei risultati

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)
regional = False

if regional:
    models = ["rf", "boos", "svr"]
else:
    models = ["rf", "boos", "svr", "linr"]
metrics = ["R^2", "RMSE", "MAPE"] 

# Generazione matrici
# Sono matrici che hanno in riga l'anno del dataset e in colonna l'anno dell'EGDI da predire
# Si plottano solo i risultati medi e le deviazioni standard
for model in models:
    for metric in metrics:
        globals()[f"{metric}_medians_{model}"] = pd.DataFrame(np.zeros((len(data_years), len(egdi_years))),
                                                              index = [f"Data_{data_year}" for data_year in data_years],
                                                              columns = [f"EGDI_{egdi_year}" for egdi_year in egdi_years])
        globals()[f"{metric}_iqr_{model}"] = pd.DataFrame(np.zeros((len(data_years), len(egdi_years))),
                                                          index = [f"Data_{data_year}" for data_year in data_years],
                                                          columns = [f"EGDI_{egdi_year}" for egdi_year in egdi_years])

del egdi_years

for data_year in data_years:
    egdi_years = list(np.arange(data_year, 2026, 2))
    for egdi_year in egdi_years:
        
        if regional:
            file_name = f"results_data_{data_year}_egdi_{egdi_year}_regional.pkl"
        else:
            file_name = f"results_data_{data_year}_egdi_{egdi_year}.pkl"

        with open(f"{root}/Risultati/ML/{file_name}", "rb") as f:
            [_, # shap_global_rf,
             _, # shap_global_boos,
             _, # test_idx_global,
             _, # train_idx_global,
             _, # explainer_rf_list,
             _, # explainer_boos_list,
             _, # c_table_rf,
             _, # c_table_boos,
             globals()[f"rf_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"boos_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"svr_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"linr_data_{data_year}_egdi_{egdi_year}"],
             _, # egdi,
             _, # y_pred_total_rf,
             _, # y_pred_total_boos,
             _, # y_pred_total_svr,
             y_pred_total_linr] = pickle.load(f)
                    
        del f, y_pred_total_linr
        
        for model in models:
            
            for metric in metrics:
        
                median_result = globals()[f"{model}_data_{data_year}_egdi_{egdi_year}"][metric].median()
                iqr_result = globals()[f"{model}_data_{data_year}_egdi_{egdi_year}"][metric].quantile(0.75) - globals()[f"{model}_data_{data_year}_egdi_{egdi_year}"][metric].quantile(0.25)
                
                globals()[f"{metric}_medians_{model}"].loc[f"Data_{data_year}"][f"EGDI_{egdi_year}"] = median_result 
                globals()[f"{metric}_iqr_{model}"].loc[f"Data_{data_year}"][f"EGDI_{egdi_year}"] = iqr_result
        
        del median_result, iqr_result
        del globals()[f"rf_data_{data_year}_egdi_{egdi_year}"]
        del globals()[f"boos_data_{data_year}_egdi_{egdi_year}"]
        del globals()[f"svr_data_{data_year}_egdi_{egdi_year}"]
        del globals()[f"linr_data_{data_year}_egdi_{egdi_year}"]

del model, metric
del data_year
del egdi_year, egdi_years

#%% Plot dei risultati

# Plottiamo
for metric in metrics:
    for model in models:
        
        results_matrix(globals()[f"{metric}_medians_{model}"],
                       globals()[f"{metric}_iqr_{model}"],
                       model,
                       metric,
                       root,
                       cbar = True,
                       regional_results = regional,
                       save = True)
    
        del model
    del metric