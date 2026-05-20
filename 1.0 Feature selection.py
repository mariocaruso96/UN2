# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:07:39 2026

@author: WKS
"""

#%% 0) Utilities

import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def find_redundant_pairs_by_pearson(
    df: pd.DataFrame,
    threshold: float = 0.90,
    min_common_obs: int = 30
) -> pd.DataFrame:
    """
    Calcola le coppie di feature ridondanti usando Pearson
    sui valori pairwise non-null.

    Parametri
    ---------
    df : pd.DataFrame
        DataFrame con index = paesi e colonne = indicatori.
    threshold : float
        Soglia su |corr| oltre la quale una coppia è considerata ridondante.
    min_common_obs : int
        Numero minimo di osservazioni comuni non-NaN per calcolare Pearson.

    Ritorna
    -------
    redundant_pairs : pd.DataFrame
        Tabella con una riga per ogni coppia ridondante.
    """

    X = df.copy()
    X = X.select_dtypes(include=[np.number])

    indicators = list(X.columns)
    rows = []

    warnings.filterwarnings("ignore")

    for i, ind_1 in enumerate(indicators):
        print(f"  feature {i+1}/{len(indicators)}: {ind_1}")

        s1 = X[ind_1]
        nan_mask_1 = s1.isna()
        num_nans_1 = int(nan_mask_1.sum())
        states_nan_1 = s1.index[nan_mask_1].tolist()

        for j in range(i):
            ind_2 = indicators[j]
            s2 = X[ind_2]

            nan_mask_2 = s2.isna()
            num_nans_2 = int(nan_mask_2.sum())
            states_nan_2 = s2.index[nan_mask_2].tolist()

            valid_mask = s1.notna() & s2.notna()
            n_common = int(valid_mask.sum())

            if n_common < min_common_obs:
                continue

            x1 = s1[valid_mask].values
            x2 = s2[valid_mask].values

            # Se una delle due feature è costante sui valori comuni, salto
            if np.nanstd(x1) == 0 or np.nanstd(x2) == 0:
                continue

            try:
                cor_coef, p_value = pearsonr(x1, x2)
            except Exception:
                continue

            if np.isfinite(cor_coef) and abs(cor_coef) > threshold:
                rows.append({
                    "Correlation": cor_coef,
                    "Abs_correlation": abs(cor_coef),
                    "P_value": p_value,
                    "N_common": n_common,
                    "Ind_1": ind_1,
                    "Ind_2": ind_2,
                    "Num_nans_ind_1": num_nans_1,
                    "Num_nans_ind_2": num_nans_2,
                    "States_nan_ind_1": states_nan_1,
                    "States_nan_ind_2": states_nan_2,
                })

    redundant_pairs = pd.DataFrame(rows)

    if not redundant_pairs.empty:
        redundant_pairs = redundant_pairs.sort_values(
            by=["Abs_correlation", "N_common"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return redundant_pairs


def select_non_redundant_features(
    df: pd.DataFrame,
    redundant_pairs: pd.DataFrame
) -> list:
    """
    Dato un DataFrame e la tabella delle coppie ridondanti,
    restituisce la lista delle feature da mantenere.

    Criterio gerarchico:
    1. meno NaN
    2. a parità di NaN, maggiore varianza
    3. a parità completa, tiene ind_1
    """

    X = df.copy()
    X = X.select_dtypes(include=[np.number])

    selected = set(X.columns)

    if redundant_pairs.empty:
        return sorted(selected)

    nan_count = X.isna().sum().to_dict()
    variances = X.var(skipna=True).to_dict()

    for _, row in redundant_pairs.iterrows():
        ind_1 = row["Ind_1"]
        ind_2 = row["Ind_2"]

        # se una delle due è già stata rimossa, non faccio nulla
        if ind_1 not in selected or ind_2 not in selected:
            continue

        n1 = nan_count[ind_1]
        n2 = nan_count[ind_2]

        v1 = variances[ind_1]
        v2 = variances[ind_2]

        # 1) tengo quella con meno NaN
        if n1 < n2:
            selected.remove(ind_2)
        elif n2 < n1:
            selected.remove(ind_1)
        else:
            # 2) a parità di NaN, tengo quella con maggiore varianza
            if v1 > v2:
                selected.remove(ind_2)
            elif v2 > v1:
                selected.remove(ind_1)
            else:
                # 3) parità completa -> tengo ind_1
                selected.remove(ind_2)

    return sorted(selected)

root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Uploading data

data = pd.read_csv(f"{root}/Data/ALL/df_all.csv")
data = data.set_index(["Country_ISO"])

# Removal of problematic columns
ordinal_cols_master = [col for col in data.columns if "0 = No status; 1 = Status B, partially compliant; 2 = Status A, fully compliant" in col]
categorical_cols_master = [col for col in data.columns if "1 = YES; 0 = NO" in col]
others_cols_master = ['SG_INF_ACCSS - Countries that adopt and implement constitutional, statutory and/or policy guarantees for public access to information - Goal 16 - Reporting Type: G - Units: NUMBER - Nature: C - SDG',
                      'Secure Internet servers (per 1 million people) - WDI']
manufacturing_cols_master = [col for col in data.columns if "Manufacturing" in col]

cols_to_delete = ordinal_cols_master + categorical_cols_master + others_cols_master + manufacturing_cols_master
data = data.drop(cols_to_delete, axis = 1)

del ordinal_cols_master, categorical_cols_master, others_cols_master, manufacturing_cols_master, cols_to_delete

# Year-based division
years = np.arange(2010, 2026, 2)

for year in years:
    globals()[f"df_{year}"] = data[data["Year"] == year]
    
    del year
del data

#%% 2) Removal indicators with missing ratio >= nan_th

nan_th = 0.5  # soglia
removed_indicators_high_nans = {}

print("\n")
print(f"Removing indicators with missing ratio >= {nan_th:.0%}\n")

for year in years:

    print(f"High-missing indicators removal ---- Year: {year}.")
    df = globals()[f"df_{year}"].copy()
    df = df.drop(["Year"], axis = 1)
    print(f"Starting number: {len(df.columns)}")

    # Duplicates removal
    df = df.loc[:, ~df.columns.duplicated()]

    # Share of NaN for each column
    nan_ratio = df.isna().mean()

    # Indicators with nan_ratio >= nan_th
    high_nan_indicators = nan_ratio[nan_ratio >= nan_th].index.tolist()
    df = df.drop(high_nan_indicators, axis = 1)
    
    removed_indicators_high_nans[year] = len(high_nan_indicators)
    print(f"Removed indicators: {len(high_nan_indicators)}")
    print(f"Final number: {len(df.columns)}\n")

    # New df
    globals()[f"new_df_{year}"] = df
    
    del globals()[f"df_{year}"]
    del df, nan_ratio, high_nan_indicators, year
del nan_th

#%% 3) Removal of constant indicators

removed_indicators_constant = {}

print("\n")
print("Removing constant indicators.")

for year in years:
    
    df = globals()[f"new_df_{year}"].copy()
    starting_number = len(df.columns)
    print(f"Starting number: {starting_number}")
    
    df = df.loc[:, df.nunique() > 1]
    final_number = len(df.columns)
    print(f"Final number: {final_number}\n")
    globals()[f"new_df_{year}"] = df
    
    removed_indicators_constant[year] = starting_number - final_number
    
    del df, year, starting_number, final_number
    
#%% 4) Removal of redundant indicators: Pearson correlation

cor_coef_threshold = 0.95
min_common_obs = 50

redundant_results = {}
selected_features_by_year = {}
summary = []

for year in years:
    print(f"\n{'='*70}")
    print(f"Pearson correlation filter ---- Year: {year}")
    print(f"{'='*70}")

    df_year = globals()[f"new_df_{year}"].copy()

    print(f"Initial dataframe shape: {df_year.shape}")

    redundant_pairs = find_redundant_pairs_by_pearson(
        df=df_year,
        threshold=cor_coef_threshold,
        min_common_obs=min_common_obs
    )

    redundant_results[year] = redundant_pairs.copy()
    globals()[f"redundant_indicators_{year}"] = redundant_pairs.copy()

    selected_indicators = select_non_redundant_features(
        df=df_year,
        redundant_pairs=redundant_pairs
    )

    df_filtered = df_year[selected_indicators].copy()

    selected_features_by_year[year] = selected_indicators
    globals()[f"new_df_{year}"] = df_filtered.copy()

    n_numeric_before = df_year.select_dtypes(include=[np.number]).shape[1]
    n_numeric_after = len(selected_indicators)

    print(f"Redundant pairs found: {len(redundant_pairs)}")
    print(f"Numeric indicators before: {n_numeric_before}")
    print(f"Numeric indicators after:  {n_numeric_after}")
    print(f"Final dataframe shape: {df_filtered.shape}")

    summary.append({
        "Year": year,
        "Redundant_pairs": len(redundant_pairs),
        "Numeric_before": n_numeric_before,
        "Numeric_after": n_numeric_after,
        "Final_rows": df_filtered.shape[0],
        "Final_cols": df_filtered.shape[1]
    })
    
    del df_filtered, df_year, n_numeric_before, n_numeric_after
    del redundant_pairs, globals()[f"redundant_indicators_{year}"], year
    del selected_indicators
    
del selected_features_by_year

summary_df = pd.DataFrame(summary)

removed_indicators_pearson = summary_df["Numeric_before"] - summary_df["Numeric_after"]
removed_indicators_pearson = dict(zip(years, removed_indicators_pearson.values))

del summary, summary_df
del min_common_obs, cor_coef_threshold
    
#%% 5) Studying NaN presence

countries = list(globals()["new_df_2022"].index)
nan_df = pd.DataFrame(index = countries)

for year in years:
    countries_num_nan = []
    countries_perc_nan = []
    
    for country in countries:
        country_df = globals()[f"new_df_{year}"].loc[country]
        country_num_nan = country_df.isna().sum()
        country_perc_nan = round(np.divide(country_num_nan, globals()[f"new_df_{year}"].shape[1])*100, 2)
        
        countries_num_nan.append(country_num_nan)
        countries_perc_nan.append(country_perc_nan)
        
        del country, country_df, country_num_nan, country_perc_nan
    
    nan_df_year = pd.concat([pd.Series(countries_num_nan), pd.Series(countries_perc_nan)], axis = 1)
    nan_df_year.index = countries
    nan_df_year.columns = [f"# nans {year}", f"% nans {year}"]
    
    nan_df = pd.concat([nan_df, nan_df_year], axis = 1)

    del year
    del countries_num_nan, countries_perc_nan
    del nan_df_year

nan_df.to_csv("C:/Users/WKS/Desktop/UNIBA/UN/Data/nan_df_global.csv")

#%% 6) Imputation with IterativeImputer

imputers = {}
imputers_details = pd.DataFrame()

for year in years:
    print(f"\nYear = {year}")
    
    df = globals()[f"new_df_{year}"].copy()

    # assicura numericità
    df = df.apply(pd.to_numeric, errors="coerce")
    starting_nans_percentage = np.round(df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100, 2)

    # scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    # imputer
    imputer = IterativeImputer(
        random_state=year,
        max_iter=50,
        tol=1e-4,
        initial_strategy="median",
        imputation_order="ascending",
        skip_complete=True
    )

    arr_imputed = imputer.fit_transform(df_scaled)

    df_imputed_scaled = pd.DataFrame(
        arr_imputed,
        index=df.index,
        columns=df.columns
    )

    df_imputed = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled),
        index=df.index,
        columns=df.columns
    )

    # clipping opzionale ai min/max osservati
    col_mins = df.min(skipna=True)
    col_maxs = df.max(skipna=True)
    df_imputed = df_imputed.clip(lower=col_mins, upper=col_maxs, axis=1)
    imputers[year] = imputer

    globals()[f"new_df_{year}"] = df_imputed
    print(f"  Shape: {df_imputed.shape}")
    print(f"  Residual NaNs: {df_imputed.isna().sum().sum()}")
    print(f"  Iterations: {imputer.n_iter_}")
    
    residual_nans_percentage = np.round(df_imputed.isna().sum().sum()/(df_imputed.shape[0]*df_imputed.shape[1])*100, 2)
    
    row_details = pd.DataFrame({"Shape": df_imputed.shape[1],
                                "Startin Nans %": starting_nans_percentage,
                                "Residual Nans %": residual_nans_percentage,
                                "Iterations": imputer.n_iter_}, index = [year])
    
    imputers_details = pd.concat([imputers_details, row_details], axis = 0)
    
    del row_details
    del year
    del df, df_scaled, arr_imputed, df_imputed_scaled, df_imputed, scaler, imputer
    del col_mins, col_maxs
    

# Plot NaNs before and after imputation
plot_df = imputers_details.copy()
plot_df.index.name = "Year"

plot_df["Startin Nans %"].loc[2010] = 0.1
  
plot_df = plot_df.reset_index()
plot_df["Residual Nans %"] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

x = np.arange(len(plot_df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

# =========================
# Bars
# =========================
bars1 = ax.bar(
    x - width/2,
    plot_df["Startin Nans %"],
    width,
    color="purple",
    label="Starting NaNs %"
)

bars2 = ax.bar(
    x + width/2,
    plot_df["Residual Nans %"],
    width,
    color="cyan",
    label="Residual NaNs %"
)

# =========================
# Labels above each pair of bars
# =========================
for i, row in plot_df.iterrows():

    max_height = max(
        row["Startin Nans %"],
        row["Residual Nans %"]
    )

    text = (
        f"Shape: {int(row['Shape'])}\n"
        f"Iterations: {int(row['Iterations'])}"
    )

    ax.text(
        x[i],
        max_height + 0.8,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

# =========================
# Axis formatting
# =========================
ax.set_xticks(x)
ax.set_xticklabels(plot_df["Year"], fontsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.set_ylabel("NaNs Percentage (%)", fontsize=20)
ax.set_xlabel("Year", fontsize=20)

# ax.set_title(
#     "Starting vs Residual NaNs Percentage Across Years",
#     fontsize=15,
#     fontweight="bold"
# )

ax.legend(loc="lower right", framealpha = 1, fontsize=20)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.ylim((0, 20))

plt.tight_layout()
plt.savefig(f"{root}/Immagini/Paper/nans_tackling.png", dpi = 300)
plt.show()
    
#%% 7) Variance filter

print("\n")
print("Remove indicators with low variance...\n")

th_var = 0.015
removed_indicators_variance = {}
    
for year in years:
    
    print(f"Low variance indicators removal ---- Year: {year}.")
    df = globals()[f"new_df_{year}"].copy()
    
    scaler = MinMaxScaler()
    
    df_scaled = pd.DataFrame(scaler.fit_transform(df),
                             columns = list(df.columns),
                             index = list(df.index))
    
    variance = df_scaled.var(ddof = 0).to_frame()
    variance.columns = ['Variance']
    variance = variance.sort_values(by = ['Variance'], ascending = False)
    
    starting_number = variance.shape[0]
    
    globals()[f"reduced_variance_{year}"] = variance[variance['Variance'] >= th_var]
    
    final_number = globals()[f'reduced_variance_{year}'].shape[0]
    
    plt.bar(np.arange(0, variance.shape[0], 1), height = variance['Variance'].values, color = 'darkorange')
    plt.axhline(th_var, color = 'darkred', label = f"Threshold = {th_var}")
    plt.xlim((-0.5, 45))
    plt.ylim((0, 0.25))
    plt.xlabel("# Indicators")
    plt.ylabel("Variance")
    plt.title(f"Variance of {year} SDG dataset.\nStarting indicators number = {starting_number}. Final number = {final_number}.\n")
    plt.legend()
    plt.show()
    
    to_keep = list(globals()[f"reduced_variance_{year}"].index)
    to_keep.sort()
    
    removed_indicators_variance[year] = starting_number - final_number
    
    globals()[f"new_df_{year}"] = globals()[f"new_df_{year}"][to_keep]
    print(f"Remaining indicators: {len(to_keep)}")
    
    del df, variance, globals()[f"reduced_variance_{year}"], df_scaled, to_keep, scaler, year

del th_var

#%% 8) Savings

# for year in years:
    
#     globals()[f"new_df_{year}"].to_csv(f"{root}/Data/ALL/cleaned_df_{year}.csv")
    
#%% 9) Plots

removed_indicators_df = pd.DataFrame()

starting_vals = np.array([270]*8)

removed_indicators_df = pd.concat([pd.Series(starting_vals),
                                   pd.Series(list(removed_indicators_high_nans.values())),
                                   pd.Series(list(removed_indicators_constant.values())),
                                   pd.Series(list(removed_indicators_pearson.values())),
                                   pd.Series(list(removed_indicators_variance.values()))], axis = 1)

removed_indicators_df.index = years
removed_indicators_df.index.name = "Years"

del starting_vals
del removed_indicators_high_nans, removed_indicators_constant, removed_indicators_pearson, removed_indicators_variance

removed_indicators_df.columns = ["Starting", "High Nans", "Constant", "Pearson", "Variance"]
removed_indicators_df["Remaining"] = removed_indicators_df["Starting"] - removed_indicators_df["High Nans"] - removed_indicators_df["Constant"] - removed_indicators_df["Pearson"] - removed_indicators_df["Variance"]

#%%
# =========================
# Data
# =========================
plot_df = removed_indicators_df.copy()

x = np.arange(len(plot_df))

# =========================
# Figure with broken axis
# =========================
fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(12, 7),
    gridspec_kw={'height_ratios': [1, 3]}
)

# =========================
# Function to plot bars
# =========================
def plot_stacked(ax):

    # =========================
    # Remaining (BOTTOM)
    # =========================
    ax.bar(
        x,
        plot_df["Remaining"],
        color="green",
        label="Remaining"
    )

    # =========================
    # Variance
    # =========================
    ax.bar(
        x,
        plot_df["Variance"],
        bottom=plot_df["Remaining"],
        color="cyan",
        label="Variance"
    )

    # =========================
    # Pearson
    # =========================
    ax.bar(
        x,
        plot_df["Pearson"],
        bottom=(
            plot_df["Remaining"] +
            plot_df["Variance"]
        ),
        color="blue",
        label="Pearson"
    )

    # =========================
    # High NaNs (TOP)
    # =========================
    ax.bar(
        x,
        plot_df["High Nans"],
        bottom=(
            plot_df["Remaining"] +
            plot_df["Variance"] +
            plot_df["Pearson"]
        ),
        color="red",
        label="High NaNs"
    )

    # =========================
    # Reference line
    # =========================
    ax.axhline(
        270,
        color="darkred",
        ls="--",
        lw=3
    )

# Plot on both axes
plot_stacked(ax_top)
plot_stacked(ax_bottom)

# =========================
# Broken axis limits
# =========================
ax_top.set_ylim(255, 275)
ax_bottom.set_ylim(0, 65)

# =========================
# Hide spines
# =========================
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)

ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

# =========================
# Diagonal break marks
# =========================
d = .015

kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)

ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)

ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# =========================
# Labels and formatting
# =========================
ax_bottom.set_xticks(x)
ax_bottom.set_xticklabels(
    plot_df.index.astype(str),
    fontsize=20
)

ax_top.tick_params(axis='y', labelsize=20)
ax_bottom.tick_params(axis='y', labelsize=20)

ax_bottom.set_xlabel("Year", fontsize=20)
ax_bottom.set_ylabel("Number of Features", fontsize=20)

ax_bottom.legend(
    loc="lower left",
    fontsize=18,
    framealpha=1
)

ax_top.grid(axis="y", linestyle="--", alpha=0.4)
ax_bottom.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{root}/Immagini/Paper/indicators_feature_selection.png", dpi = 300)
plt.show()























