# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:44:31 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

# =============================================================================
# 0) IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
import warnings
import re
from collections import Counter
from typing import Tuple, List, Dict, Optional, Callable, Any

import shap
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score

country_to_iso3 = {
'Afghanistan': 'AFG',
'Albania': 'ALB',
'Algeria': 'DZA',
'Andorra': 'AND',
'Angola': 'AGO',
'Antigua and Barbuda': 'ATG',
'Argentina': 'ARG',
'Armenia': 'ARM',
'Australia': 'AUS',
'Austria': 'AUT',
'Azerbaijan': 'AZE',
'Bahamas': 'BHS',
'Bahrain': 'BHR',
'Bangladesh': 'BGD',
'Barbados': 'BRB',
'Belarus': 'BLR',
'Belgium': 'BEL',
'Belize': 'BLZ',
'Benin': 'BEN',
'Bhutan': 'BTN',
'Bolivia': 'BOL',
'Bosnia and Herzegovina': 'BIH',
'Botswana': 'BWA',
'Brazil': 'BRA',
'Brunei Darussalam': 'BRN',
'Bulgaria': 'BGR',
'Burkina Faso': 'BFA',
'Burundi': 'BDI',
'Cabo Verde': 'CPV',
'Cambodia': 'KHM',
'Cameroon': 'CMR',
'Canada': 'CAN',
'Central African Republic': 'CAF',
'Chad': 'TCD',
'Chile': 'CHL',
'China': 'CHN',
'Colombia': 'COL',
'Comoros': 'COM',
'Congo': 'COG',
'Costa Rica': 'CRI',
"Cote d'Ivoire": 'CIV',
'Croatia': 'HRV',
'Cuba': 'CUB',
'Cyprus': 'CYP',
'Czech Republic': 'CZE',
"Democratic People's Republic of Korea": 'PRK',
'Democratic Republic of the Congo': 'COD',
'Denmark': 'DNK',
'Djibouti': 'DJI',
'Dominica': 'DMA',
'Dominican Republic': 'DOM',
'Ecuador': 'ECU',
'Egypt': 'EGY',
'El Salvador': 'SLV',
'Equatorial Guinea': 'GNQ',
'Eritrea': 'ERI',
'Estonia': 'EST',
'Eswatini': 'SWZ',
'Ethiopia': 'ETH',
'Fiji': 'FJI',
'Finland': 'FIN',
'France': 'FRA',
'Gabon': 'GAB',
'Gambia': 'GMB',
'Georgia': 'GEO',
'Germany': 'DEU',
'Ghana': 'GHA',
'Greece': 'GRC',
'Grenada': 'GRD',
'Guatemala': 'GTM',
'Guinea': 'GIN',
'Guinea-Bissau': 'GNB',
'Guyana': 'GUY',
'Haiti': 'HTI',
'Honduras': 'HND',
'Hungary': 'HUN',
'Iceland': 'ISL',
'India': 'IND',
'Indonesia': 'IDN',
'Iran': 'IRN',
'Iraq': 'IRQ',
'Ireland': 'IRL',
'Israel': 'ISR',
'Italy': 'ITA',
'Jamaica': 'JAM',
'Japan': 'JPN',
'Jordan': 'JOR',
'Kazakhstan': 'KAZ',
'Kenya': 'KEN',
'Kiribati': 'KIR',
'Kuwait': 'KWT',
'Kyrgyzstan': 'KGZ',
'Laos': 'LAO',
'Latvia': 'LVA',
'Lebanon': 'LBN',
'Lesotho': 'LSO',
'Liberia': 'LBR',
'Libya': 'LBY',
'Liechtenstein': 'LIE',
'Lithuania': 'LTU',
'Luxembourg': 'LUX',
'Madagascar': 'MDG',
'Malawi': 'MWI',
'Malaysia': 'MYS',
'Maldives': 'MDV',
'Mali': 'MLI',
'Malta': 'MLT',
'Marshall Islands': 'MHL',
'Mauritania': 'MRT',
'Mauritius': 'MUS',
'Mexico': 'MEX',
'Micronesia': 'FSM',
'Moldova': 'MDA',
'Monaco': 'MCO',
'Mongolia': 'MNG',
'Montenegro': 'MNE',
'Morocco': 'MAR',
'Mozambique': 'MOZ',
'Myanmar': 'MMR',
'Namibia': 'NAM',
'Nauru': 'NRU',
'Nepal': 'NPL',
'Netherlands': 'NLD',
'New Zealand': 'NZL',
'Nicaragua': 'NIC',
'Niger': 'NER',
'Nigeria': 'NGA',
'North Macedonia': 'MKD',
'Norway': 'NOR',
'Oman': 'OMN',
'Pakistan': 'PAK',
'Palau': 'PLW',
'Panama': 'PAN',
'Papua New Guinea': 'PNG',
'Paraguay': 'PRY',
'Peru': 'PER',
'Philippines': 'PHL',
'Poland': 'POL',
'Portugal': 'PRT',
'Qatar': 'QAT',
'Republic of Korea': 'KOR',
'Romania': 'ROU',
'Russian Federation': 'RUS',
'Rwanda': 'RWA',
'Saint Kitts and Nevis': 'KNA',
'Saint Lucia': 'LCA',
'Saint Vincent and the Grenadines': 'VCT',
'Samoa': 'WSM',
'San Marino': 'SMR',
'Sao Tome and Principe': 'STP',
'Saudi Arabia': 'SAU',
'Senegal': 'SEN',
'Serbia': 'SRB',
'Seychelles': 'SYC',
'Sierra Leone': 'SLE',
'Singapore': 'SGP',
'Slovakia': 'SVK',
'Slovenia': 'SVN',
'Solomon Islands': 'SLB',
'Somalia': 'SOM',
'South Africa': 'ZAF',
'South Sudan': 'SSD',
'Spain': 'ESP',
'Sri Lanka': 'LKA',
'Sudan': 'SDN',
'Suriname': 'SUR',
'Sweden': 'SWE',
'Switzerland': 'CHE',
'Syria': 'SYR',
'Tajikistan': 'TJK',
'Tanzania': 'TZA',
'Thailand': 'THA',
'Timor-Leste': 'TLS',
'Togo': 'TGO',
'Tonga': 'TON',
'Trinidad and Tobago': 'TTO',
'Tunisia': 'TUN',
'Turkey': 'TUR',
'Turkmenistan': 'TKM',
'Tuvalu': 'TUV',
'Uganda': 'UGA',
'Ukraine': 'UKR',
'United Arab Emirates': 'ARE',
'United Kingdom': 'GBR',
'United States of America': 'USA',
'Uruguay': 'URY',
'Uzbekistan': 'UZB',
'Vanuatu': 'VUT',
'Venezuela': 'VEN',
'Vietnam': 'VNM',
'Yemen': 'YEM',
'Zambia': 'ZMB',
'Zimbabwe': 'ZWE'
}


# =============================================================================
# 1) UTILITIES
# =============================================================================
def iqr_series(x) -> float:
    """Interquartile range (Q3 - Q1) robust to NaNs."""
    x = pd.Series(x).dropna()
    return float(x.quantile(0.75) - x.quantile(0.25))


def apply_plot_style(ax: plt.Axes, plot_cfg: dict) -> None:
    """
    Central place to modify plot aesthetics quickly.
    Edit plot_cfg in CONFIG section.
    """
    ax.grid(plot_cfg.get("grid", True), alpha=plot_cfg.get("grid_alpha", 0.3))
    ax.tick_params(axis="both", labelsize=plot_cfg.get("tick_fontsize", 10))
    if plot_cfg.get("invert_xaxis", True):
        ax.invert_xaxis()


# =============================================================================
# 2) SHAP / TOP FEATURES
# =============================================================================
def shap_summary_and_top_features(
    dataset: pd.DataFrame,
    global_shap: List[np.ndarray],
    year_egdi: int,
    year_sdg: int,
    global_idx_test: List[np.ndarray],
    how_many: int = 20,
    save: bool = True,
    save_path: Optional[str] = None,
    plot: bool = True,
):
    """
    Replica la tua shap_summary_plot:
      - concat shap values e test idx
      - summary_plot
      - calcolo mean(|shap|) e restituisce top_features (DataFrame ordinato)

    NOTE:
    - global_shap: lista (fold-wise) di array (n_test_fold x p)
    - global_idx_test: lista (fold-wise) di indici test (n_test_fold,)
    """
    warnings.filterwarnings("ignore")

    shap_values_concatenated = np.concatenate(global_shap, axis=0)
    index_list_concatenated = np.concatenate(global_idx_test, axis=0)

    data_shap = dataset.iloc[index_list_concatenated]

    if plot:
        plt.figure()
        plt.title(
            f"RF SHAP Summary Plot - Global case - EGDI: {year_egdi} - SDG: {year_sdg}",
            fontsize=14,
            loc="center",
        )
        shap.summary_plot(
            shap_values_concatenated,
            np.array(data_shap),
            feature_names=list(dataset.columns),
            show=False,
        )
        if save and save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.show()

    mean_abs_shap = np.mean(np.abs(shap_values_concatenated), axis=0)

    feature_importance = pd.DataFrame(
        {"MeanAbsSHAP": mean_abs_shap},
        index=dataset.columns,
    )
    feature_importance["idx"] = np.arange(0, dataset.shape[1], 1)
    feature_importance = feature_importance.sort_values("MeanAbsSHAP", ascending=False)

    return feature_importance.head(how_many)

# =============================================================================
# 3) FEATURE REMOVAL (TOP-FIRST / RANDOM)
# =============================================================================
def cumulative_drop_top_features(
    data: pd.DataFrame,
    top_features: pd.DataFrame,
    block_size: int = 3,
    max_steps: Optional[int] = None,
    ignore_missing: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Cumulativo TOP-FIRST: rimuove le feature in base all'ordine in top_features.index.

    step s rimuove: top_list[: s*block_size]
    """
    if not isinstance(top_features, pd.DataFrame):
        raise TypeError("top_features deve essere un pd.DataFrame con index = feature names.")

    # dedup preservando ordine
    top_idx = pd.Index(dict.fromkeys(top_features.index).keys()) if top_features.index.has_duplicates else top_features.index
    top_list = [str(f) for f in top_idx.tolist()]

    total_blocks = (len(top_list) + block_size - 1) // block_size
    if max_steps is None:
        max_steps = total_blocks
    else:
        max_steps = max(0, min(max_steps, total_blocks))

    dfs: Dict[str, pd.DataFrame] = {}
    removed_dict: Dict[str, List[str]] = {}

    for step in range(1, max_steps + 1):
        k = step * block_size
        to_remove = top_list[:k]

        if ignore_missing:
            to_remove_in_data = [c for c in to_remove if c in data.columns]
        else:
            missing = [c for c in to_remove if c not in data.columns]
            if missing:
                raise KeyError(f"Feature non presenti in data.columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
            to_remove_in_data = to_remove

        degraded = data.drop(columns=to_remove_in_data, errors="ignore" if ignore_missing else "raise")
        key = f"top_step_{step:02d}_removed_{len(to_remove_in_data):03d}"

        dfs[key] = degraded
        removed_dict[key] = to_remove_in_data

        if verbose:
            missing_n = len(to_remove) - len(to_remove_in_data)
            msg = f"[{key}] remaining={degraded.shape[1]} | removed_cum={len(to_remove_in_data)}"
            if missing_n > 0:
                msg += f" | missing_in_data={missing_n}"
            print(msg)

        if degraded.shape[1] == 0:
            if verbose:
                print(f"[STOP] {key}: data.shape[1]==0.")
            break

    return dfs, removed_dict


def cumulative_drop_random_features(
    data: pd.DataFrame,
    top_features: pd.DataFrame,
    block_size: int = 3,
    seed: int = 0,
    max_steps: Optional[int] = None,
    ignore_missing: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Cumulativo RANDOM: come top-first ma con ordine random di top_features.index.
    """
    if not isinstance(top_features, pd.DataFrame):
        raise TypeError("top_features deve essere un pd.DataFrame con index = feature names.")

    top_idx = pd.Index(dict.fromkeys(top_features.index).keys()) if top_features.index.has_duplicates else top_features.index
    top_list = [str(f) for f in top_idx.tolist()]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(top_list).tolist()

    total_blocks = (len(perm) + block_size - 1) // block_size
    if max_steps is None:
        max_steps = total_blocks
    else:
        max_steps = max(0, min(max_steps, total_blocks))

    dfs: Dict[str, pd.DataFrame] = {}
    removed_dict: Dict[str, List[str]] = {}

    for step in range(1, max_steps + 1):
        k = step * block_size
        to_remove = perm[:k]

        if ignore_missing:
            to_remove_in_data = [c for c in to_remove if c in data.columns]
        else:
            missing = [c for c in to_remove if c not in data.columns]
            if missing:
                raise KeyError(f"Feature non presenti in data.columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
            to_remove_in_data = to_remove

        degraded = data.drop(columns=to_remove_in_data, errors="ignore" if ignore_missing else "raise")
        key = f"rand_seed_{seed:03d}_step_{step:02d}_removed_{len(to_remove_in_data):03d}"

        dfs[key] = degraded
        removed_dict[key] = to_remove_in_data

        if verbose:
            missing_n = len(to_remove) - len(to_remove_in_data)
            msg = f"[{key}] remaining={degraded.shape[1]} | removed_cum={len(to_remove_in_data)}"
            if missing_n > 0:
                msg += f" | missing_in_data={missing_n}"
            print(msg)

        if degraded.shape[1] == 0:
            if verbose:
                print(f"[STOP] {key}: data.shape[1]==0.")
            break

    return dfs, removed_dict


# =============================================================================
# 4) CV CORE: ONE REPETITION
# =============================================================================
def run_one_repetition(
    i: int,
    data: pd.DataFrame,
    y: pd.Series,
    k: int,
    boxcox: bool,
    compute_shap: bool = False,
) -> Dict[str, Any]:
    """
    Esegue UNA ripetizione della KFold CV (shuffle=True, random_state=i).
    Restituisce predizioni, metriche, feature importance fold-wise e (opzionale) shap per fold.
    """
    data_i = data.copy()

    rf = RandomForestRegressor(n_jobs=1, random_state=i)
    boos = xgb.XGBRegressor(n_jobs=1, random_state=i)
    svr = SVR()
    linr = LinearRegression()

    truth_labels = []
    pred_labels_rf, pred_labels_boos, pred_labels_svr, pred_labels_linr = [], [], [], []

    c_table_rf, c_table_boos = [], []
    shap_list_rf, shap_list_boos = [], []
    explainer_rf_expected, explainer_boos_expected = [], []
    train_idx_rep, test_idx_rep = [], []

    kf = KFold(n_splits=k, shuffle=True, random_state=i)

    for fold, (train_index, test_index) in enumerate(kf.split(data_i, y)):
        train_idx_rep.append(train_index)
        test_idx_rep.append(test_index)

        train = data_i.iloc[train_index, :]
        test = data_i.iloc[test_index, :]

        y_train = y.iloc[train_index].copy()
        y_test = y.iloc[test_index].copy()

        # riallineo indici con i paesi
        y_train.index = train.index
        y_test.index = test.index

        # BoxCox sul training (e riuso lambda sul test)
        if boxcox:
            y_train_bc, lambda_val = scipy.stats.boxcox(y_train.values)
            y_train = pd.Series(y_train_bc, index=train.index)

            y_test_bc = scipy.stats.boxcox(y_test.values, lmbda=lambda_val)
            y_test = pd.Series(y_test_bc, index=test.index)

        # scaling X (fit SOLO su train)
        scaler_X = MinMaxScaler()
        train_scaled = pd.DataFrame(
            scaler_X.fit_transform(train),
            columns=train.columns,
            index=train.index,
        )
        test_scaled = pd.DataFrame(
            scaler_X.transform(test),
            columns=test.columns,
            index=test.index,
        )

        # scaling y (fit SOLO su train)
        scaler_y = MinMaxScaler()
        y_train_scaled = pd.Series(
            scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
            index=y_train.index,
        )

        # train
        rf.fit(train_scaled, y_train_scaled)
        boos.fit(train_scaled.values, y_train_scaled.values)
        svr.fit(train_scaled, y_train_scaled)
        linr.fit(train_scaled, y_train_scaled)

        # feature importance fold-wise
        c_table_rf.append(pd.Series(rf.feature_importances_, index=train_scaled.columns, name=f"fold_{fold+1}"))
        c_table_boos.append(pd.Series(boos.feature_importances_, index=train_scaled.columns, name=f"fold_{fold+1}"))

        # prediction (inverse-transform su scala originale)
        y_pred_rf = pd.Series(
            scaler_y.inverse_transform(rf.predict(test_scaled).reshape(-1, 1)).ravel(),
            index=test.index,
        )
        y_pred_boos = pd.Series(
            scaler_y.inverse_transform(boos.predict(test_scaled.values).reshape(-1, 1)).ravel(),
            index=test.index,
        )
        y_pred_svr = pd.Series(
            scaler_y.inverse_transform(svr.predict(test_scaled).reshape(-1, 1)).ravel(),
            index=test.index,
        )
        y_pred_linr = pd.Series(
            scaler_y.inverse_transform(linr.predict(test_scaled).reshape(-1, 1)).ravel(),
            index=test.index,
        )

        truth_labels.append(y_test)
        pred_labels_rf.append(y_pred_rf)
        pred_labels_boos.append(y_pred_boos)
        pred_labels_svr.append(y_pred_svr)
        pred_labels_linr.append(y_pred_linr)

        # SHAP per fold
        if compute_shap:
            explainer_rf = shap.TreeExplainer(rf, train_scaled.values)
            shap_vals_rf = explainer_rf.shap_values(test_scaled.values, check_additivity=False)
            shap_list_rf.append(pd.DataFrame(shap_vals_rf, columns=test_scaled.columns, index=test_scaled.index))
            explainer_rf_expected.append(float(np.array(explainer_rf.expected_value).ravel()[0]))

            explainer_boos = shap.TreeExplainer(boos, train_scaled.values)
            shap_vals_boos = explainer_boos.shap_values(test_scaled.values, check_additivity=False)
            shap_list_boos.append(pd.DataFrame(shap_vals_boos, columns=test_scaled.columns, index=test_scaled.index))
            explainer_boos_expected.append(float(np.array(explainer_boos.expected_value).ravel()[0]))

    # concat fold -> singola serie per ripetizione
    y_true_rep = pd.concat(truth_labels, axis=0).sort_index()
    pred_rf_rep = pd.concat(pred_labels_rf, axis=0).sort_index()
    pred_boos_rep = pd.concat(pred_labels_boos, axis=0).sort_index()
    pred_svr_rep = pd.concat(pred_labels_svr, axis=0).sort_index()
    pred_linr_rep = pd.concat(pred_labels_linr, axis=0).sort_index()

    metrics = {
        "RF": {
            "RMSE": root_mean_squared_error(y_true_rep.values, pred_rf_rep.values),
            "MAPE": mean_absolute_percentage_error(y_true_rep.values, pred_rf_rep.values),
            "R2": r2_score(y_true_rep.values, pred_rf_rep.values),
        },
        "XgB": {
            "RMSE": root_mean_squared_error(y_true_rep.values, pred_boos_rep.values),
            "MAPE": mean_absolute_percentage_error(y_true_rep.values, pred_boos_rep.values),
            "R2": r2_score(y_true_rep.values, pred_boos_rep.values),
        },
        "SVR": {
            "RMSE": root_mean_squared_error(y_true_rep.values, pred_svr_rep.values),
            "MAPE": mean_absolute_percentage_error(y_true_rep.values, pred_svr_rep.values),
            "R2": r2_score(y_true_rep.values, pred_svr_rep.values),
        },
        "LinR": {
            "RMSE": root_mean_squared_error(y_true_rep.values, pred_linr_rep.values),
            "MAPE": mean_absolute_percentage_error(y_true_rep.values, pred_linr_rep.values),
            "R2": r2_score(y_true_rep.values, pred_linr_rep.values),
        },
    }

    out = {
        "i": i,
        "y_true": y_true_rep,
        "pred_rf": pred_rf_rep,
        "pred_boos": pred_boos_rep,
        "pred_svr": pred_svr_rep,
        "pred_linr": pred_linr_rep,
        "metrics": metrics,
        "c_table_rf": pd.concat(c_table_rf, axis=1),
        "c_table_boos": pd.concat(c_table_boos, axis=1),
    }

    if compute_shap:
        out.update({
            "train_idx": train_idx_rep,
            "test_idx": test_idx_rep,
            "shap_rf": shap_list_rf,
            "shap_boos": shap_list_boos,
            "exp_rf": explainer_rf_expected,
            "exp_boos": explainer_boos_expected,
        })

    return out


# =============================================================================
# 5) ABLATION RUNNERS
# =============================================================================
def run_model_ablation_from_dfs(
    dfs: Dict[str, pd.DataFrame],
    egdi: pd.Series,
    model,
    run_one_repetition_fn: Callable[..., Dict[str, Any]],
    k: int,
    n: int,
    boxcox: bool = False,
    compute_shap: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose_parallel: int = 10,
    do_plot: bool = False,
    plot_cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Per ogni dataset degradato:
      - esegue Parallel su n ripetizioni di run_one_repetition
      - salva median(R2) e IQR(R2)
    """
    shap_ablation_results = {}
    summary_rows = []

    sizes, r2s, err_r2s = [], [], []

    for key, data in dfs.items():

        if data.shape[1] == 0:
            print(f"\n[STOP] {key}: data.shape[1]==0")
            break

        y = egdi.copy()
        print(f"\n[ABLATION] {key} | remaining features = {data.shape[1]}")

        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose_parallel)(
            delayed(run_one_repetition_fn)(
                i=i,
                data=data,
                y=y,
                k=k,
                boxcox=boxcox,
                compute_shap=compute_shap,
            )
            for i in range(n)
        )

        r2_global_model = [r["metrics"][model]["R2"] for r in results]

        results_model = pd.DataFrame({"R^2": r2_global_model}, index=np.arange(1, n + 1, 1))
        shap_ablation_results[key] = results_model.copy()

        size = int(data.shape[1])
        r2_med = float(np.round(results_model["R^2"].median(), 2))
        r2_iqr = float(np.round(iqr_series(results_model["R^2"]), 2))

        sizes.append(size)
        r2s.append(r2_med)
        err_r2s.append(r2_iqr)

        summary_rows.append({
            "key": key,
            "remaining_features": size,
            "r2_median": r2_med,
            "r2_iqr": r2_iqr,
        })

        print(f"R2 {model} --- median {r2_med}, iqr = {r2_iqr}")

    summary_df = pd.DataFrame(summary_rows).sort_values("remaining_features", ascending=False)

    if do_plot and len(sizes) > 0:
        if plot_cfg is None:
            plot_cfg = {}
        fig, ax = plt.subplots()
        ax.plot(sizes, r2s, marker="o")
        ax.set_xlabel("# remaining features")
        ax.set_ylabel("R^2 (median across repetitions)")
        apply_plot_style(ax, {"invert_xaxis": True, **plot_cfg})
        plt.show()

    return {
        "shap_ablation_results": shap_ablation_results,
        "summary_df": summary_df,
        "sizes": sizes,
        "r2s": r2s,
        "err_r2s": err_r2s,
    }


def aggregate_random_baseline(
    data: pd.DataFrame,
    top_features: pd.DataFrame,
    egdi: pd.Series,
    run_one_repetition_fn: Callable[..., Dict[str, Any]],
    k: int,
    n: int,
    block_size: int = 3,
    B: int = 20,
    seeds: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    ignore_missing: bool = True,
    boxcox: bool = False,
    compute_shap: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose_parallel: int = 0,
) -> pd.DataFrame:
    """
    Costruisce B curve random e aggrega per #remaining_features:
      - median delle median (over seeds)
      - q25/q75 (over seeds)
    """
    if seeds is None:
        seeds = list(range(B))
    else:
        B = len(seeds)

    curves = []

    for s in seeds:
        dfs_rand, _ = cumulative_drop_random_features(
            data=data,
            top_features=top_features,
            block_size=block_size,
            seed=s,
            max_steps=max_steps,
            ignore_missing=ignore_missing,
            verbose=False,
        )

        out_rand = run_model_ablation_from_dfs(
            dfs=dfs_rand,
            egdi=egdi,
            model="XgB",
            run_one_repetition_fn=run_one_repetition_fn,
            k=k,
            n=n,
            boxcox=boxcox,
            compute_shap=compute_shap,
            n_jobs=n_jobs,
            backend=backend,
            verbose_parallel=verbose_parallel,
            do_plot=False,
        )

        df_seed = out_rand["summary_df"][["remaining_features", "r2_median"]].copy()
        df_seed["seed"] = s
        curves.append(df_seed)

    all_curves = pd.concat(curves, axis=0, ignore_index=True)

    agg = (
        all_curves
        .groupby("remaining_features")["r2_median"]
        .agg(
            r2_med_over_seeds="median",
            r2_q25_over_seeds=lambda x: x.quantile(0.25),
            r2_q75_over_seeds=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .sort_values("remaining_features", ascending=False)
    )
    return agg


# =============================================================================
# 6) PLOTTING (easy to modify)
# =============================================================================
def plot_top_vs_random_baseline(
    out_top: Dict[str, Any],
    agg_random: pd.DataFrame,
    *,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    plot_cfg: Optional[dict] = None,
    save: bool = False,
    root: Optional[str] = None,
    data_year: Optional[int] = None,
    egdi_year: Optional[int] = None,
):
    """
    Plot:
      - Top-first curve: median + IQR band (across repetitions)
      - Random baseline: median + IQR band (across seeds)

    If save=True, saves figure to:
      {root}/Immagini/Ablation/ablation_data_{data_year}_egdi_{egdi_year}.png
      with dpi=100
    """
    if plot_cfg is None:
        plot_cfg = {}

    top_df = out_top["summary_df"].copy().sort_values("remaining_features", ascending=False)
    x_top = top_df["remaining_features"].values
    y_top = top_df["r2_median"].values
    yerr_top = top_df["r2_iqr"].values
    y_top_lo = y_top - yerr_top
    y_top_hi = y_top + yerr_top

    x_r = agg_random["remaining_features"].values
    y_r = agg_random["r2_med_over_seeds"].values
    y_r_lo = agg_random["r2_q25_over_seeds"].values
    y_r_hi = agg_random["r2_q75_over_seeds"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=plot_cfg.get("figsize", (7, 5)))
    else:
        fig = ax.figure

    # --- Random baseline: line + band ---
    ax.fill_between(
        x_r, y_r_lo, y_r_hi,
        alpha=plot_cfg.get("random_band_alpha", 0.2),
        label=plot_cfg.get("label_random_band", "Random removal (IQR across seeds)")
    )
    ax.plot(
        x_r, y_r,
        marker=plot_cfg.get("random_marker", "o"),
        linestyle=plot_cfg.get("random_linestyle", "--"),
        label=plot_cfg.get("label_random_line", "Random removal (median across seeds)")
    )

    # --- Top-first: line + band ---
    ax.fill_between(
        x_top, y_top_lo, y_top_hi,
        alpha=plot_cfg.get("top_band_alpha", 0.2),
        label=plot_cfg.get("label_top_band", "Top-first removal (IQR across repetitions)")
    )
    ax.plot(
        x_top, y_top,
        marker=plot_cfg.get("top_marker", "o"),
        linestyle=plot_cfg.get("top_linestyle", "-"),
        label=plot_cfg.get("label_top_line", "Top-first removal (median across repetitions)")
    )

    ax.set_xlabel(plot_cfg.get("xlabel", "# remaining features"))
    ax.set_ylabel(plot_cfg.get("ylabel", "R^2"))

    if title:
        ax.set_title(title, fontsize=plot_cfg.get("title_fontsize", 12))

    apply_plot_style(ax, plot_cfg)
    ax.legend(fontsize=plot_cfg.get("legend_fontsize", 10), loc="lower left")

    if save:
        if root is None or data_year is None or egdi_year is None:
            raise ValueError("Per salvare l'immagine devi fornire root, data_year ed egdi_year.")

        save_path = f"{root}/Immagini/Ablation/ablation_data_{data_year}_egdi_{egdi_year}.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.show()


#%% 1) Analisi

# =============================================================================
# 7) MAIN
# =============================================================================
if __name__ == "__main__":

    # -------------------------
    # CONFIG
    # -------------------------
    root = "C:/Users/WKS/Desktop/UNIBA/UN/"

    data_year = 2022
    egdi_year = 2024

    n_reps = 10
    k_folds = 10

    block_size = 1
    B_random = 3  # increase for stability

    plot_cfg = {
        "figsize": (7, 5),
        "grid": True,
        "grid_alpha": 0.25,
        "tick_fontsize": 10,
        "legend_fontsize": 10,
        "title_fontsize": 12,
        "invert_xaxis": True,
    
        "random_band_alpha": 0.2,
        "random_marker": "o",
        "random_linestyle": "--",
    
        "top_band_alpha": 0.2,
        "top_marker": "o",
        "top_linestyle": "-",
    
        "xlabel": "# remaining features",
        "ylabel": "R^2",
    
        "label_random_band": "Random removal (IQR across seeds)",
        "label_random_line": "Random removal (median across seeds)",
        "label_top_band": "Top-first removal (IQR across repetitions)",
        "label_top_line": "Top-first removal (median across repetitions)",
    }
    

    # -------------------------
    # LOAD EGDI
    # -------------------------
    all_egdis = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")
    countries = list(all_egdis["Countries"])

    egdi = all_egdis[f"EGDI_{egdi_year}"].copy()
    egdi.index = list(country_to_iso3.values())
    egdi = egdi.replace([0], [0.01])

    del all_egdis, countries

    # -------------------------
    # LOAD SDG DATASET
    # -------------------------
    ###### CARICO DATASET ######
    globals()[f"df_{data_year}"] = pd.read_csv(f"{root}/Data/ALL/cleaned_df_{data_year}.csv")
    globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].set_index(["Country_ISO"])
    
    # Sort
    egdi = egdi.sort_index()
    globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].sort_index()

    # -------------------------
    # LOAD PKL (SHAP + test idx)
    # -------------------------
    pkl_path = f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}.pkl"
    with open(pkl_path, "rb") as f:
        [
            _, # shap_global_rf,
            shap_global_boos,
            test_idx_global,
            _,  # train_idx_global
            _,  # explainer_rf_list
            _,  # explainer_boos_list
            _,  # c_table_rf
            _,  # c_table_boos
            _,  # results_rf
            _,  # results_boos
            _,  # results_svr
            _,  # results_linr
            _,  # egdi
            _,  # y_pred_total_rf
            _,  # y_pred_total_boos
            _,  # y_pred_total_svr
            _   # y_pred_total_linr
        ] = pickle.load(f)

    # -------------------------
    # TOP FEATURES (all columns)
    # -------------------------
    top_features = shap_summary_and_top_features(
        dataset=globals()[f"df_{data_year}"].copy(),
        global_shap=shap_global_boos,
        year_egdi=egdi_year,
        year_sdg=data_year,
        global_idx_test=test_idx_global,
        how_many=globals()[f"df_{data_year}"].shape[1],
        save=False,
        save_path=None,
        plot=False,  # metti True se vuoi il summary plot qui
    )

    # -------------------------
    # TOP-FIRST ABLATION
    # -------------------------
    dfs_top, _ = cumulative_drop_top_features(
        data=globals()[f"df_{data_year}"].copy(),
        top_features=top_features,
        block_size=block_size,
        ignore_missing=True,
        verbose=True,
    )

    out_top = run_model_ablation_from_dfs(
        dfs=dfs_top,
        egdi=egdi,
        model="XgB",
        run_one_repetition_fn=run_one_repetition,
        k=k_folds,
        n=n_reps,
        boxcox=False,
        compute_shap=False,
        n_jobs=-1,
        backend="loky",
        verbose_parallel=10,
        do_plot=False,
        plot_cfg=plot_cfg,
    )

    # -------------------------
    # RANDOM BASELINE (B seeds)
    # -------------------------
    agg_random = aggregate_random_baseline(
        data=globals()[f"df_{data_year}"].copy(),
        top_features=top_features,
        egdi=egdi,
        run_one_repetition_fn=run_one_repetition,
        k=k_folds,
        n=n_reps,
        block_size=block_size,
        B=B_random,
        max_steps=None,
        ignore_missing=True,
        n_jobs=-1,
        backend="loky",
        verbose_parallel=0,
    )

    # -------------------------
    # PLOT COMPARISON
    # -------------------------
    
    plot_top_vs_random_baseline(
    out_top=out_top,
    agg_random=agg_random,
    title="Top-first vs Random feature removal",
    plot_cfg=plot_cfg,
    save=True,
    root=root,
    data_year=data_year,
    egdi_year=egdi_year,
    )
#%% 2) Ablation per macrofamiglie

from matplotlib.lines import Line2D

if __name__ == "__main__":

    # =========================
    # 1) Dizionario macrofamiglie
    # =========================

    macrofamily_dict = {
        # Digital / technological / ICT
        "High-technology exports (% of manufactured exports) - WDI": "Digital and technological infrastructure",
        "ICT service exports (% of service exports, BoP) - WDI": "Digital and technological infrastructure",
        "ICT goods exports (% of total goods exports) - WDI": "Digital and technological infrastructure",
        "ICT goods imports (% total goods imports) - WDI": "Digital and technological infrastructure",
        "Medium and high-tech exports (% manufactured exports) - WDI": "Digital and technological infrastructure",
        "Medium and high-tech manufacturing value added (% manufacturing value added) - WDI": "Digital and technological infrastructure",

        # Education / human capital
        "Lower secondary school starting age (years) - WDI": "Education and human capital",
        "Primary education, duration (years) - WDI": "Education and human capital",
        "Primary school starting age (years) - WDI": "Education and human capital",
        "Labor force with advanced education (% of total working-age population with advanced education) - WDI": "Education and human capital",
        "Educational attainment, at least Bachelor's or equivalent, population 25+, total (%) (cumulative) - WDI": "Education and human capital",

        # Economy / services / trade
        "Services, value added (% of GDP) - WDI": "Economic structure and services",
        "Services, value added per worker (constant 2015 US$) - WDI": "Economic structure and services",
        "Services, value added (annual % growth) - WDI": "Economic structure and services",
        "Merchandise exports to high-income economies (% of total merchandise exports) - WDI": "Economic structure and services",
        "Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports) - WDI": "Economic structure and services",
        "Merchandise imports from high-income economies (% of total merchandise imports) - WDI": "Economic structure and services",

        # Governance / institutions
        "Control of Corruption: Estimate - WDI": "Governance and institutions",
        "Government Effectiveness: Estimate - WDI": "Governance and institutions",
        "Regulatory Quality: Estimate - WDI": "Governance and institutions",
        "Rule of Law: Estimate - WDI": "Governance and institutions",
        "Voice and Accountability: Estimate - WDI": "Governance and institutions",
        "SG_NHR_IMPL - Proportion of countries with independent National Human Rights Institutions in compliance with the Paris Principles (%) - Goal 16 - Reporting Type: G - Units: PERCENT - Nature: G - SDG": "Governance and institutions",

        # Statistical capacity
        "Statistical performance indicators (SPI): Pillar 1 data use score (scale 0-100) - WDI": "Statistical capacity and data infrastructure",
        "Statistical performance indicators (SPI): Pillar 2 data services score (scale 0-100) - WDI": "Statistical capacity and data infrastructure",
        "Statistical performance indicators (SPI): Pillar 3 data products score  (scale 0-100) - WDI": "Statistical capacity and data infrastructure",
        "Statistical performance indicators (SPI): Pillar 4 data sources score (scale 0-100) - WDI": "Statistical capacity and data infrastructure",
        "Statistical performance indicators (SPI): Pillar 5 data infrastructure score (scale 0-100) - WDI": "Statistical capacity and data infrastructure",
        "IQ_SPI_PIL4 - Data Sources performance index (Statistical Performance Indicators Pillar 4) (Index) - Goal 17 - Reporting Type: G - Units: INDEX - Nature: G - SDG": "Statistical capacity and data infrastructure",
        "SG_STT_ODIN - Open Data Inventory (ODIN) Coverage Index - Goal 17 - Reporting Type: G - Units: INDEX - Nature: E - SDG": "Statistical capacity and data infrastructure",

        # Financial system
        "FI_FSI_FSANL - Non-performing loans to total gross loans (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",
        "FI_FSI_FSKA - Regulatory capital to assets (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",
        "FI_FSI_FSKRTC - Regulatory Tier 1 capital to risk-weighted assets (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",
        "FI_FSI_FSERA - Return on assets (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",
        "FI_FSI_FSLS - Liquid assets to short term liabilities (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",
        "FB_BNK_ACCSS - Proportion of adults (15 years and older) with an account at a financial institution or mobile-money-service provider, by sex (% of adults aged 15 years and older) - Goal 8 - Sex: BOTHSEX - Education level: AGG_0_1 - Location: ALLAREA - Reporting Type: G - Units: PERCENT - Age: 15+ - Nature: G - Quantile: _T - SDG": "Financial system and inclusion",
        "FB_BNK_ACCSS - Proportion of adults (15 years and older) with an account at a financial institution or mobile-money-service provider, by sex (% of adults aged 15 years and older) - Goal 8 - Sex: FEMALE - Education level: _T - Location: ALLAREA - Reporting Type: G - Units: PERCENT - Age: 15+ - Nature: G - Quantile: _T - SDG": "Financial system and inclusion",
        "FI_FSI_FSKNL - Non-performing loans net of provisions to capital (%) - Goal 10 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Financial system and inclusion",

        # Public finance
        "GC_GOB_TAXD - Proportion of domestic budget funded by domestic taxes (%) - Goal 17 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Public finance and fiscal capacity",
        "GR_G14_GDP - Total government revenue (budgetary central government) as a proportion of GDP (%) - Goal 17 - Reporting Type: G - Observation Status: A - Units: PERCENT - Nature: C - SDG": "Public finance and fiscal capacity",
        "Primary government expenditures as a proportion of original approved budget (%) - WDI": "Public finance and fiscal capacity",

        # Logistics
        "Logistics performance index: Ability to track and trace consignments (1=low to 5=high) - WDI": "Logistics and trade facilitation",
        "Logistics performance index: Efficiency of customs clearance process (1=low to 5=high) - WDI": "Logistics and trade facilitation",
    }

    macrofamily_order = [
        ("Governance and institutions", "GI"),
        ("Statistical capacity and data infrastructure", "SD"),
        ("Economic structure and services", "ES"),
        ("Logistics and trade facilitation", "LT"),
        ("Digital and technological infrastructure", "DT"),
        ("Education and human capital", "EH"),
        ("Financial system and inclusion", "FI"),
        ("Public finance and fiscal capacity", "PF"),
    ]

    data_macro = globals()[f"df_{data_year}"].copy()

    unmapped_features = [
        c for c in data_macro.columns
        if c not in macrofamily_dict
    ]

    if len(unmapped_features) > 0:
        raise ValueError(
            f"Ci sono {len(unmapped_features)} feature NON mappate.\n"
            f"Prime feature non mappate:\n{unmapped_features[:30]}"
        )

    dfs_macro = {}
    removed_macro_dict = {}
    macro_rows = []
    removed_so_far = []

    for step, (macro_name, macro_abbr) in enumerate(macrofamily_order, start=1):

        features_this_macro = [
            feature
            for feature, family in macrofamily_dict.items()
            if family == macro_name and feature in data_macro.columns
        ]

        removed_so_far.extend(features_this_macro)
        removed_so_far = list(dict.fromkeys(removed_so_far))

        degraded = data_macro.drop(columns=removed_so_far)

        removed_label = "+".join([
            abbr for _, abbr in macrofamily_order[:step]
        ])

        key = f"macro_step_{step:02d}_removed_{removed_label}"

        dfs_macro[key] = degraded
        removed_macro_dict[key] = removed_so_far.copy()

        macro_rows.append({
            "key": key,
            "step": step,
            "removed_macrofamilies": removed_label,
            "last_removed_macrofamily": macro_name,
            "last_removed_abbr": macro_abbr,
            "removed_features": len(removed_so_far),
            "remaining_features": degraded.shape[1],
        })

        print(
            f"[{key}] removed={removed_label} | "
            f"removed_features={len(removed_so_far)} | "
            f"remaining_features={degraded.shape[1]}"
        )

    macro_summary_df = pd.DataFrame(macro_rows)

    dfs_macro_for_cv = {
        key: df
        for key, df in dfs_macro.items()
        if df.shape[1] > 0
    }

    if len(dfs_macro_for_cv) < len(dfs_macro):
        print("[INFO] Lo step finale con zero feature è stato escluso dalla CV.")

    out_macro = run_model_ablation_from_dfs(
        dfs=dfs_macro_for_cv,
        egdi=egdi,
        model="XgB",
        run_one_repetition_fn=run_one_repetition,
        k=k_folds,
        n=n_reps,
        boxcox=False,
        compute_shap=False,
        n_jobs=-1,
        backend="loky",
        verbose_parallel=10,
        do_plot=False,
        plot_cfg=plot_cfg,
    )

    macro_perf_df = out_macro["summary_df"].copy()

    macro_plot_df = macro_perf_df.merge(
    macro_summary_df[
        [
            "key",
            "step",
            "removed_macrofamilies",
            "last_removed_abbr",
            "removed_features",
            "remaining_features",
        ]
    ],
    on=["key", "remaining_features"],
    how="left")

    macro_plot_df = macro_plot_df.sort_values("step")
    
    
    for macro_name, macro_abbr in macrofamily_order:

        n_features = sum(
            1
            for feature, family in macrofamily_dict.items()
            if family == macro_name and feature in data_macro.columns
        )
    
        print(f"{macro_abbr}: {n_features}")

    # =========================
    # Plot paper-style
    # =========================
    
    x_labels = [
    (
        row['last_removed_abbr'],
        f"{row['remaining_features']} features"
    )
    for _, row in macro_plot_df.iterrows()]
    x = np.arange(len(x_labels))
    
    y = macro_plot_df["r2_median"].values
    yerr = macro_plot_df["r2_iqr"].values
    
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    
    ax.fill_between(
        x,
        y - yerr,
        y + yerr,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
        label="IQR"
    )
    
    ax.plot(
        x,
        y,
        color="tab:red",
        marker="o",
        markersize=5,
        linewidth=2.2,
        label="Median $R^2$"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(
    [f"{abbr}\n{nfeat}" for abbr, nfeat in x_labels],
    fontsize=10
    )
    
    macro_colors = {
    "GI": "darkviolet",
    "SD": "mediumblue",
    "ES": "royalblue",
    "LT": "cornflowerblue",
    "DT": "teal",
    "EH": "green",
    "FI": "darkorange",
    "PF": "orange",
    }
    
    for tick, (abbr, nfeat) in zip(ax.get_xticklabels(), x_labels):

        tick.set_text(f"{abbr}\n{nfeat}")
        tick.set_color(macro_colors[abbr])
        tick.set_fontweight("bold")
    
    # ax.set_xlabel("Removed macrofamily at each step (remaining features)", fontsize=14)
    ax.set_ylabel("$R^2$", fontsize=18)
    plt.yticks(fontsize=13)
    
    # ax.set_title(
    #     f"Macrofamily-based ablation: {data_year} $\\rightarrow$ EGDI {egdi_year}",
    #     fontsize=12,
    #     pad=10
    # )
    
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.legend(
        frameon=False,
        fontsize=12,
        loc="lower left"
    )
    
    ax.set_ylim(
        max(0, np.nanmin(y - yerr) - 0.05),
        min(1, np.nanmax(y + yerr) + 0.05)
    )
    
    fig.tight_layout()
    
    save_dir = f"{root}/Immagini/Ablation"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_png = (
        f"{save_dir}/macrofamily_ablation_data_"
        f"{data_year}_egdi_{egdi_year}.png"
    )
    
    save_path_pdf = (
        f"{save_dir}/macrofamily_ablation_data_"
        f"{data_year}_egdi_{egdi_year}.pdf"
    )
    
    fig.savefig(save_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(save_path_pdf, bbox_inches="tight")
    
    plt.show()