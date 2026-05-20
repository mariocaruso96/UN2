# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:49:04 2026

@author: WKS
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:09:12 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error


def compute_r2_distributions_by_region(region, cv_df, loao_df, series_egdi):
    """
    Calcola le distribuzioni di R² sulle 100 ripetizioni
    per una specifica regione, confrontando CV e LOAO.
    """

    r2_cv = []
    r2_loao = []

    region_cv = cv_df[cv_df["Region"] == region].drop(["Region"], axis=1)
    region_loao = loao_df[loao_df["Region"] == region].drop(["Region"], axis=1)

    egdi_region = series_egdi.loc[region_cv.index]

    for i in np.arange(1, 101, 1):
        r2_cv_rep = r2_score(egdi_region, region_cv[f"Rep_{i}"])
        r2_loao_rep = r2_score(egdi_region, region_loao[f"Rep_{i}"])

        r2_cv.append(r2_cv_rep)
        r2_loao.append(r2_loao_rep)

    return np.array(r2_cv), np.array(r2_loao)



def regional_error_computation(error_func, region, cv_df, loao_df, series_egdi, p_threshold = 0.05):

    error_region_cv = []
    error_region_loao = []
    
    region_cv = cv_df[cv_df["Region"] == region].drop(["Region"], axis = 1)            
    region_loao = loao_df[loao_df["Region"] == region].drop(["Region"], axis = 1)
    
    egdi_region = series_egdi.loc[region_cv.index]
    
    for i in np.arange(1, 101, 1):
        error_region_cv_rep = error_func(egdi_region, region_cv[f"Rep_{i}"])
        error_region_loao_rep = error_func(egdi_region, region_loao[f"Rep_{i}"])
        
        error_region_cv.append(error_region_cv_rep)
        error_region_loao.append(error_region_loao_rep)
        
    median_region_cv = np.median(error_region_cv)
    # iqr_region_cv = np.quantile(error_region_cv, 0.75) - np.quantile(error_region_cv, 0.25)
    
    median_region_loao = np.median(error_region_loao)
    # iqr_region_loao = np.quantile(error_region_loao, 0.75) - np.quantile(error_region_loao, 0.25)
    
    # NB: la sottrazione è CV - LOAO
    delta_region = np.array(error_region_cv) - np.array(error_region_loao)
    median_delta = np.median(delta_region)
    iqr_delta = np.quantile(delta_region, 0.75) - np.quantile(delta_region, 0.25)
    
    # Le due distribuzioni sono statisticamente differenti?
    coeff, p = wilcoxon(error_region_cv, error_region_loao)
    
    p_bool = True if p < p_threshold else False
    
    # return median_delta, iqr_delta, median_region_cv, iqr_region_cv, median_region_loao, iqr_region_loao
    return median_delta, iqr_delta, median_region_cv, median_region_loao, p_bool



def plot_delta_r2_heatmap_from_globals(
    data_years,
    egdi_years,
    globals_dict,
    model: str = "rf",                       # "rf" | "boos" | "svr"
    mode: str = "global",                    # "global" | "regional"
    region: str | None = None,               # richiesto se mode="regional"
    areas=("Asia-Pacific", "Africa", "Europe", "Americas"),
    agg: str = "mean",                       # "mean" | "weighted"
    area_weights: dict | None = None,        # es. {"Africa": 54, ...} se agg="weighted"
    value_col: str = "delta_R^2_median",           # colonna che contiene ΔR²
    area_col: str = "Areas",
    cmap: str = "coolwarm",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    annot: bool = True,
    fmt: str = ".3f",
    figsize=(14, 6),
    title: str | None = None,
    save_path: str = "C:/Users/WKS/Desktop/UNIBA/UN/Immagini/Heatmaps Regionali/",
    save: bool = False
):
    """
    Legge i DataFrame:
        globals_dict[f"metric_data_{data}_egdi_{egdi}"]

    Ogni DF contiene 12 righe = 4 aree x 3 modelli, index ripetuto ("rf","boos","svr"),
    con colonne: "Areas" e "R^2_median" (che qui è già ΔR² per area e modello).

    Ritorna anche la matrice (DataFrame) usata per la heatmap.
    """

    data_years = list(data_years)
    egdi_years = list(egdi_years)

    if mode not in ("global", "regional"):
        raise ValueError("mode deve essere 'global' o 'regional'.")

    if mode == "regional" and region is None:
        raise ValueError("mode='regional' richiede region='Africa' (etc.).")

    if agg not in ("mean", "weighted"):
        raise ValueError("agg deve essere 'mean' o 'weighted'.")

    if agg == "weighted" and area_weights is None:
        raise ValueError("agg='weighted' richiede area_weights (dict).")

    # matrice SDG x EGDI
    M = np.full((len(data_years), len(egdi_years)), np.nan, dtype=float)

    for i, data_year in enumerate(data_years):
        for j, egdi in enumerate(egdi_years):
            if egdi < data_year:
                continue

            key = f"metric_data_{data_year}_egdi_{egdi}"
            if key not in globals_dict:
                continue

            df = globals_dict[key]
            if not isinstance(df, pd.DataFrame):
                continue

            # filtro modello (index = modello)
            if model not in df.index:
                continue

            dfm = df.loc[model].copy()
            # dfm può essere Series se una riga sola (non nel tuo caso), forziamo DataFrame
            if isinstance(dfm, pd.Series):
                dfm = dfm.to_frame().T

            # controlli colonne
            if area_col not in dfm.columns or value_col not in dfm.columns:
                continue

            dfm[area_col] = dfm[area_col].astype(str).str.strip()

            # selezione aree valide
            dfm = dfm[dfm[area_col].isin(areas)].copy()
            if dfm.empty:
                continue

            if mode == "regional":
                sel = dfm[dfm[area_col] == region]
                if sel.empty:
                    continue
                M[i, j] = float(sel[value_col].iloc[0])

            else:  # mode == "global"
                if agg == "mean":
                    M[i, j] = float(dfm[value_col].mean())
                else:  # weighted
                    w = dfm[area_col].map(area_weights).astype(float)
                    if w.isna().any():
                        missing = dfm.loc[w.isna(), area_col].unique().tolist()
                        raise ValueError(f"area_weights manca per: {missing}")
                    M[i, j] = float(np.average(dfm[value_col].astype(float), weights=w.values))

    mat = pd.DataFrame(M, index=data_years, columns=egdi_years)

    # titolo
    if title is None:
        if mode == "global":
            title = f"ΔR² heatmap ({model.upper()}), CV-LOAO loss aggregated results ({agg})"
        else:
            title = f"ΔR² heatmap ({model.upper()}), CV-LOAO loss results for region = {region}"

    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(
        mat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "ΔR²"},
        annot_kws={
            "size": 18,          # <-- dimensione numeri (aumenta se vuoi)
       #     "weight": "bold",    # <-- grassetto
            "color": "black"
        }
    )
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("EGDI year", fontsize=18)
    ax.set_ylabel("Data year", fontsize=18)
    ax.set_title(title, fontsize=20)
    
    # tick più leggibili
    ax.tick_params(axis='both', labelsize=15)
    
    # colorbar più leggibile
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("ΔR²", fontsize=20)
    
    plt.tight_layout()
    if save:
        if mode == "global":
            plt.savefig(save_path + f"{model}_{mode}_{agg}_aggregation_r2_heatmap.png", dpi = 100)
        else:
            plt.savefig(save_path + f"{model}_{region}_r2_heatmap.png", dpi = 100)
    plt.show()

    return mat

#%% 1) Caricamento dei risultati

root = "C:/Users/WKS/Desktop/UNIBA/UN/"

countries_df = pd.read_csv(f"{root}/Data/countries_region.csv").drop(["Unnamed: 0", "Countries"], axis = 1).set_index(["ISO3"]).sort_index()
unique_areas = ["Asia-Pacific", "Africa", "Europe", "Americas"]

data_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
egdi_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

models = ["linr", "rf", "boos", "svr"]
metrics = ["R^2", "RMSE", "MAPE"] 
metrics_func = [r2_score, root_mean_squared_error, mean_absolute_percentage_error]
p_threshold = 0.05

for data_year in data_years:
    egdi_years = list(np.arange(data_year, 2026, 2))
    for egdi_year in egdi_years:
        
        print(f"Data year = {data_year}; EGDI year = {egdi_year}.")
                
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
             _, # globals()[f"rf_sdg_{sdg_year}_egdi_{egdi_year}"],
             _, # globals()[f"boos_sdg_{sdg_year}_egdi_{egdi_year}"],
             _, # globals()[f"svr_sdg_{sdg_year}_egdi_{egdi_year}"],
             _, # globals()[f"linr_sdg_{sdg_year}_egdi_{egdi_year}"],
             globals()[f"egdi_{egdi_year}"],
             globals()[f"rf_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"boos_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"svr_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"linr_y_pred_df_data_{data_year}_egdi_{egdi_year}"]] = pickle.load(f)
            
        del file_name
        del f
            
        file_name = f"results_data_{data_year}_egdi_{egdi_year}_regional.pkl"
        with open(f"{root}/Risultati/ML/{file_name}", "rb") as f:
            [# _, # shap_global_rf,
             # _, # shap_global_boos,
             # _, # test_idx_global,
             # _, # train_idx_global,
             # _, # explainer_rf_list,
             # _, # explainer_boos_list,
             _, # c_table_rf,
             _, # c_table_boos,
             _, # globals()[f"rf_sdg_{sdg_year}_egdi_{egdi_year}_regional"],
             _, # globals()[f"boos_sdg_{sdg_year}_egdi_{egdi_year}_regional"],
             _, # globals()[f"svr_sdg_{sdg_year}_egdi_{egdi_year}_regional"],
             _, # globals()[f"linr_sdg_{sdg_year}_egdi_{egdi_year}_regional"],
             _, # egdi,
             globals()[f"rf_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"boos_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"svr_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"],
             globals()[f"linr_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"]] = pickle.load(f)
            
        del file_name
        del f
        
        # Mettiamo assieme le predizioni
        
        # Calcoliamo i delta per Regione
        # Calcolo la predizione media
        globals()[f"metric_data_{data_year}_egdi_{egdi_year}"] = pd.DataFrame()
        
        for model in models:
            df_cv = globals()[f"{model}_y_pred_df_data_{data_year}_egdi_{egdi_year}"].copy()
            df_cv = pd.concat([countries_df["Region"], df_cv], axis = 1)
            df_loao = globals()[f"{model}_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"].copy()
            df_loao = pd.concat([countries_df["Region"], df_loao], axis = 1)
            
            df_model = pd.DataFrame()
            
            for region in unique_areas:
                metric_dict = {}
                
                for metric, metric_func in zip(metrics, metrics_func):
                    metric_median_delta_region, metric_iqr_delta_region, _, _, p_bool = regional_error_computation(metric_func,
                                                                                                                   region,
                                                                                                                   df_cv,
                                                                                                                   df_loao,
                                                                                                                   globals()[f"egdi_{egdi_year}"],
                                                                                                                   p_threshold)
                        
                    # Aggiorno
                    metric_dict[f"delta_{metric}_median"] = metric_median_delta_region
                    metric_dict[f"delta_{metric}_iqr"] = metric_iqr_delta_region
                    metric_dict[f"{metric}_p < {p_threshold}"] = p_bool
                    del metric, metric_func, metric_iqr_delta_region, metric_median_delta_region, p_bool
                    
                row = pd.DataFrame(metric_dict, index = [region])
                df_model = pd.concat([df_model, row], axis = 0)
                del row, region, metric_dict
                
            df_model.index.name = "Areas"
            df_model = df_model.reset_index(drop = False)
            df_model.index = [model]*4
            
            # Aggiorno
            globals()[f"metric_data_{data_year}_egdi_{egdi_year}"] = pd.concat([globals()[f"metric_data_{data_year}_egdi_{egdi_year}"], df_model], axis = 0)
            
            del df_model, model, df_cv, df_loao

del metrics_func

#%% 2) Heatmaps

for model_of_interest in models:

    # Percentuale di volte in cui le distribuzioni dell'R^2 calcolato da CV differiscono da quelle dell'R^2 calcolato da LOAO
    total_per_area = {a: 0 for a in unique_areas}
    r2_p = {a: 0 for a in unique_areas}
    
    for data_year in data_years:
        egdi_years_valid = list(np.arange(data_year, 2026, 2))
        for egdi_year in egdi_years_valid:
    
            df_all = globals()[f"metric_data_{data_year}_egdi_{egdi_year}"].copy().loc[model_of_interest]
            df_all["Areas"] = df_all["Areas"].astype(str).str.strip()
    
            col = f"R^2_p < {p_threshold}"
    
            for area in unique_areas:
                total_per_area[area] += 1
    
                row = df_all[df_all["Areas"] == area]
                if row.shape[0] != 1:
                    continue  # oppure raise ValueError(...)
    
                r2_bool = bool(row[col].iloc[0])
                if r2_bool:
                    r2_p[area] += 1
                    
                    del r2_bool
                del row, area
            del col, df_all
            del egdi_year
        del egdi_years_valid, data_year
    
    # percentuali
    r2_p_percent = {a: (r2_p[a] / total_per_area[a]) * 100 for a in unique_areas}
    print("% of statistical differneces between R^2 distributions computed respectively from CV and LOAO cases.")
    for area, val in r2_p_percent.items():
        print(f"{area}: {round(val, 2)} %")
    
        del area, val
    print("\n")
    
    del r2_p, r2_p_percent
    
    # Calcolo la media temporale dei delta di R^2 intra cluster
    delta_r2_sum = {a: 0 for a in unique_areas}
    
    for data_year in data_years:
        egdi_years_valid = list(np.arange(data_year, 2026, 2))
        for egdi_year in egdi_years_valid:
            for area in unique_areas:
                
                # Delta R^2
                df_all = globals()[f"metric_data_{data_year}_egdi_{egdi_year}"].copy().loc[model_of_interest]
                df = df_all[df_all["Areas"] == area]
                delta_r2 = df["delta_R^2_median"].values[0]
                delta_r2_sum[area] += delta_r2
                
                del df_all, df, delta_r2
            del area, egdi_year
        del egdi_years_valid, data_year
                
    
    # Perdita media di performance cross-region lungo l’intero orizzonte temporale
    delta_r2_mean = {a: round(delta_r2_sum[a] / total_per_area[a], 2) for a in unique_areas}
    
    # del total_per_area, delta_r2_sum
    
    vmin=0
    vmax=1
    
    # Delta globale
    delta_rf_global = plot_delta_r2_heatmap_from_globals(
        data_years=np.arange(2010, 2026, 2),
        egdi_years=np.arange(2010, 2026, 2),
        globals_dict=globals(),
        model=model_of_interest,
        mode="global",
        agg="mean",
        vmin=vmin,
        vmax=vmax,
        save = True
    )
    
    # Delta globale pesato
    area_weights = dict(countries_df["Region"].value_counts())
    
    delta_rf_global_w = plot_delta_r2_heatmap_from_globals(
        data_years=np.arange(2010, 2026, 2),
        egdi_years=np.arange(2010, 2026, 2),
        globals_dict=globals(),
        model=model_of_interest,
        mode="global",
        agg="weighted",
        area_weights=area_weights,
        vmin=vmin,
        vmax=vmax,
        save = True
    )


    # Delta regionale
    for area in unique_areas:
        delta_rf_africa = plot_delta_r2_heatmap_from_globals(
            data_years=np.arange(2010, 2026, 2),
            egdi_years=np.arange(2010, 2026, 2),
            globals_dict=globals(),
            model=model_of_interest,
            mode="regional",
            region=area,
            vmin=vmin,
            vmax=vmax,
            save = True
        )
        
#%% 3) R² CV vs LOAO - griglia heatmap con mediana ± IQR e scala comune leggibile

output_r2_grid_path = f"{root}/Immagini/R2 CV vs LOAO/"
os.makedirs(output_r2_grid_path, exist_ok=True)

all_data_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
all_egdi_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

for model_of_interest in models:

    for region in unique_areas:

        # =========================
        # 1) Precalcolo distribuzioni
        # =========================
        dist_dict = {}
        all_values = []

        for data_year in all_data_years:
            for egdi_year in all_egdi_years:

                if egdi_year < data_year:
                    continue

                key_cv = f"{model_of_interest}_y_pred_df_data_{data_year}_egdi_{egdi_year}"
                key_loao = f"{model_of_interest}_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"
                key_egdi = f"egdi_{egdi_year}"

                if key_cv not in globals() or key_loao not in globals() or key_egdi not in globals():
                    continue

                df_cv = globals()[key_cv].copy()
                df_cv = pd.concat([countries_df["Region"], df_cv], axis=1)

                df_loao = globals()[key_loao].copy()
                df_loao = pd.concat([countries_df["Region"], df_loao], axis=1)

                r2_cv, r2_loao = compute_r2_distributions_by_region(
                    region=region,
                    cv_df=df_cv,
                    loao_df=df_loao,
                    series_egdi=globals()[key_egdi]
                )

                dist_dict[(data_year, egdi_year)] = {
                    "cv": r2_cv,
                    "loao": r2_loao
                }

                all_values.extend(r2_cv)
                all_values.extend(r2_loao)

                del df_cv, df_loao, r2_cv, r2_loao

        if len(all_values) == 0:
            continue

        # =========================
        # 2) Scala Y comune
        # =========================
        y_min = np.nanquantile(all_values, 0.01)
        y_max = np.nanquantile(all_values, 0.99)

        y_pad = 0.08 * (y_max - y_min)

        y_min -= y_pad
        y_max += y_pad

        y_min = min(y_min, 0)
        y_max = max(y_max, 0)

        yticks = np.linspace(y_min, y_max, 5)

        # =========================
        # 3) Plot griglia
        # =========================
        fig, axes = plt.subplots(
            len(all_data_years),
            len(all_egdi_years),
            figsize=(36, 18),
            sharey=False,
            gridspec_kw={
                "wspace": 0.08,
                "hspace": 0.10
            }
        )

        for i, data_year in enumerate(all_data_years):

            for j, egdi_year in enumerate(all_egdi_years):

                ax = axes[i, j]

                # Celle sotto diagonale
                if egdi_year < data_year or (data_year, egdi_year) not in dist_dict:
                    ax.axis("off")
                    continue

                r2_cv = dist_dict[(data_year, egdi_year)]["cv"]
                r2_loao = dist_dict[(data_year, egdi_year)]["loao"]

                cv_med = np.nanmedian(r2_cv)
                cv_q25 = np.nanquantile(r2_cv, 0.25)
                cv_q75 = np.nanquantile(r2_cv, 0.75)

                loao_med = np.nanmedian(r2_loao)
                loao_q25 = np.nanquantile(r2_loao, 0.25)
                loao_q75 = np.nanquantile(r2_loao, 0.75)

                # CV
                ax.errorbar(
                    x=0,
                    y=cv_med,
                    yerr=[[cv_med - cv_q25], [cv_q75 - cv_med]],
                    fmt="o",
                    color="royalblue",
                    markersize=9,
                    capsize=6,
                    linewidth=2.0
                )

                # LOAO
                ax.errorbar(
                    x=1,
                    y=loao_med,
                    yerr=[[loao_med - loao_q25], [loao_q75 - loao_med]],
                    fmt="o",
                    color="firebrick",
                    markersize=9,
                    capsize=6,
                    linewidth=2.0
                )

                # linea
                ax.plot(
                    [0, 1],
                    [cv_med, loao_med],
                    color="black",
                    linewidth=1.0,
                    alpha=0.45
                )

                # linea y=0
                ax.axhline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55
                )

                ax.set_ylim(y_min, y_max)
                ax.set_yticks(yticks)

                # Tick Y solo sulla diagonale
                if j == i:
                    ax.set_yticklabels(
                        [f"{v:.2f}" for v in yticks],
                        fontsize=14
                    )
                    ax.tick_params(axis="y", labelsize=14)

                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)

                ax.set_xlim(-0.45, 1.45)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(
                    ["CV", "LOAO"],
                    fontsize=13,
                    rotation=45
                )

                ax.tick_params(axis="x", labelsize=13)

                # EGDI titles
                if i == 0:
                    ax.set_title(
                        f"EGDI\n{egdi_year}",
                        fontsize=20,
                        fontweight="bold",
                        pad=10
                    )

                for spine in ax.spines.values():
                    spine.set_linewidth(0.7)

        # =========================
        # 4) Etichette Data ALLINEATE
        # =========================

        fig.canvas.draw()

        # posizione X fissa per tutte
        left_x = min(
            [
                axes[i, i].get_position().x0
                for i in range(len(all_data_years))
            ]
        ) - 0.045

        for i, data_year in enumerate(all_data_years):

            diag_ax = axes[i, i]
            pos = diag_ax.get_position()

            center_y = pos.y0 + pos.height / 2

            fig.text(
                left_x,
                center_y,
                f"Data\n{data_year}",
                ha="right",
                va="center",
                fontsize=18,
                fontweight="bold"
            )

        # =========================
        # 5) Titoli finali
        # =========================

        fig.suptitle(
            f"R² median ± IQR: CV vs LOAO | "
            f"Model = {model_of_interest.upper()} | Region = {region}",
            fontsize=24,
            y=0.995
        )

        fig.text(
            0.5,
            0.012,
            "Columns: EGDI prediction year | Rows: SDG/data year | Common Y-axis within each figure",
            ha="center",
            fontsize=15
        )

        plt.tight_layout(
            rect=[0.07, 0.04, 1, 0.955]
        )

        save_name = (
            f"{model_of_interest}_{region}_r2_cv_vs_loao_median_iqr_grid_common_scale.png"
        ).replace("/", "-")

        plt.savefig(
            output_r2_grid_path + save_name,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        del fig, axes, dist_dict, all_values, save_name
        
#%% 4) Heatmap testuale CV vs LOAO: mediana ± IQR, colore = differenza mediane CV - LOAO

output_r2_text_heatmap_path = f"{root}/Immagini/R2 CV vs LOAO Text Heatmap/"
os.makedirs(output_r2_text_heatmap_path, exist_ok=True)

all_data_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
all_egdi_years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

# =========================
# Scala globale condivisa
# =========================


for model_of_interest in models:

    for region in unique_areas:

        delta_mat = pd.DataFrame(
            np.nan,
            index=all_data_years,
            columns=all_egdi_years
        )

        annot_mat = pd.DataFrame(
            "",
            index=all_data_years,
            columns=all_egdi_years
        )

        for data_year in all_data_years:

            for egdi_year in all_egdi_years:

                if egdi_year < data_year:
                    continue

                key_cv = f"{model_of_interest}_y_pred_df_data_{data_year}_egdi_{egdi_year}"
                key_loao = f"{model_of_interest}_regional_y_pred_df_data_{data_year}_egdi_{egdi_year}"
                key_egdi = f"egdi_{egdi_year}"

                if key_cv not in globals() or key_loao not in globals() or key_egdi not in globals():
                    continue

                df_cv = globals()[key_cv].copy()
                df_cv = pd.concat([countries_df["Region"], df_cv], axis=1)

                df_loao = globals()[key_loao].copy()
                df_loao = pd.concat([countries_df["Region"], df_loao], axis=1)

                r2_cv, r2_loao = compute_r2_distributions_by_region(
                    region=region,
                    cv_df=df_cv,
                    loao_df=df_loao,
                    series_egdi=globals()[key_egdi]
                )

                cv_med = np.nanmedian(r2_cv)
                cv_iqr = np.nanquantile(r2_cv, 0.75) - np.nanquantile(r2_cv, 0.25)

                loao_med = np.nanmedian(r2_loao)
                loao_iqr = np.nanquantile(r2_loao, 0.75) - np.nanquantile(r2_loao, 0.25)

                delta_mat.loc[data_year, egdi_year] = cv_med - loao_med

                annot_mat.loc[data_year, egdi_year] = (
                    "CV:\n"
                    rf"$\bf{{{cv_med:.2f} \pm {cv_iqr:.2f}}}$" "\n"
                    "LOAO:\n"
                    rf"$\bf{{{loao_med:.2f} \pm {loao_iqr:.2f}}}$"
                )
                

                del df_cv, df_loao, r2_cv, r2_loao

        if delta_mat.isna().all().all():
            continue

        max_abs_delta = 2
        
        fig, ax = plt.subplots(figsize=(20, 10))

        sns.heatmap(
            delta_mat,
            cmap="coolwarm",
            center=0,
            vmin=-max_abs_delta,
            vmax=max_abs_delta,
            annot=annot_mat,
            fmt="",
            linewidths=0.8,
            linecolor="white",
            cbar_kws={
                "label": "Median difference: CV - LOAO"
            },
            annot_kws={
                "size": 14.5,
                "color": "black"
            },
            ax=ax
        )

        ax.set_xlabel("EGDI year", fontsize=20)
        ax.set_ylabel("Data year", fontsize=20)

        ax.set_title(
            f"R² CV vs LOAO median ± IQR | "
            rf"Model = $\bf{{{model_of_interest.upper()}}}$ | Region = $\bf{{{region}}}$",
            fontsize=25,
            pad=16
        )

        ax.tick_params(axis="x", labelsize=20, rotation=0)
        ax.tick_params(axis="y", labelsize=20, rotation=0)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Median difference: CV - LOAO", fontsize=17)

        plt.tight_layout()

        save_name = (
            f"{model_of_interest}_{region}_r2_cv_loao_text_heatmap_delta_color.png"
        ).replace("/", "-")

        plt.savefig(
            output_r2_text_heatmap_path + save_name,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        del delta_mat, annot_mat, fig, ax, save_name