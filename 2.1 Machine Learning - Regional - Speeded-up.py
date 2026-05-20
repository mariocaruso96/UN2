# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:24:24 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import iqr
from joblib import Parallel, delayed
import pickle
import winsound

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

def run_one_repetition(
    i: int,
    areas: list,
    data: pd.DataFrame,
    y: pd.Series,
    compute_shap: bool = False,
):
    
    # 2) modelli: n_jobs=1 per evitare oversubscription
    # 1) shuffle indicatori (null model) - indipendente per ripetizione
    data_i = data.copy()

    # 2) modelli: n_jobs=1 per evitare oversubscription
    rf = RandomForestRegressor(n_estimators=1500,
                               max_depth=3,
                               max_features=0.5,
                               n_jobs=1, random_state=i)
    
    boos = xgb.XGBRegressor(n_estimators=1500,
                            max_depth=3,
                            learning_rate=0.01,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            min_child_weight=3,
                            # gamma=0.2,
                            n_jobs=1, random_state=i)
    
    svr = SVR()
    
    linr = LinearRegression()

    # contenitori
    truth_labels = []
    pred_labels_rf = []
    # pred_labels_lgbm = []
    # pred_labels_cat = []
    pred_labels_boos = []
    pred_labels_svr = []
    pred_labels_linr = []

    # opzionali
    c_table_rf = []
    # c_table_lgbm = []
    # c_table_cat = []
    c_table_boos = []
    shap_list_rf = []
    # shap_list_lgbm = []
    # shap_list_cat = []
    shap_list_boos = []
    explainer_rf_expected = []
    # explainer_lgbm_expected = []
    # explainer_cat_expected = []
    explainer_boos_expected = []

    train_idx_rep, test_idx_rep = [], []

    # LOAO
    # mapping Paese -> Regione
    region_map = countries_df["Region"]

    for holdout_region in areas:
        
        print(f"Holdout region: {holdout_region}")
        
        # maschere
        test_mask = (region_map == holdout_region)
        train_mask = (region_map.isin([r for r in areas if r != holdout_region]))
        
        test_countries = list(test_mask == True)
        train_countries = list(train_mask == True)
        
        test_idx_rep.append(test_countries)
        train_idx_rep.append(train_countries)
        
        # Split
        train = data.loc[train_mask]
        test  = data.loc[test_mask]
        
        y_train = y.loc[train_mask]
        y_test  = y.loc[test_mask]
        
        del train_mask, test_mask
        
        # scaling X e y (fit SOLO su train)
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

        scaler_y = MinMaxScaler()
        y_train_scaled = pd.Series(
            scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
            index=y_train.index,
        )
        y_test_scaled = pd.Series(
            scaler_y.transform(y_test.values.reshape(-1, 1)).ravel(),
            index=y_test.index,
        )

        # training
        rf.fit(train_scaled, y_train_scaled)
        boos.fit(train_scaled.values, y_train_scaled.values)
        svr.fit(train_scaled, y_train_scaled)
        linr.fit(train_scaled, y_train_scaled)
        
        # feature importance
        c_table_rf.append(pd.Series(rf.feature_importances_, index=train_scaled.columns, name=f"HO_{holdout_region}"))
        c_table_boos.append(pd.Series(boos.feature_importances_, index=train_scaled.columns, name=f"HO_{holdout_region}"))

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

        truth_labels.append(y_test)  # y_test già in scala originale (o boxcox)
        pred_labels_rf.append(y_pred_rf)
        pred_labels_boos.append(y_pred_boos)
        pred_labels_svr.append(y_pred_svr)
        pred_labels_linr.append(y_pred_linr)

        # SHAP (come vuoi tu: per ogni fold)
        if compute_shap:
            # RF
            explainer_rf = shap.TreeExplainer(rf, train_scaled.values)
            shap_vals_rf = explainer_rf.shap_values(test_scaled.values, check_additivity=False)
            shap_list_rf.append(pd.DataFrame(shap_vals_rf, columns=test_scaled.columns, index=test_scaled.index))
            explainer_rf_expected.append(float(np.array(explainer_rf.expected_value).ravel()[0]))

            # XGB
            explainer_boos = shap.TreeExplainer(boos, train_scaled.values)
            shap_vals_boos = explainer_boos.shap_values(test_scaled.values, check_additivity=False)
            shap_list_boos.append(pd.DataFrame(shap_vals_boos, columns=test_scaled.columns, index=test_scaled.index))
            explainer_boos_expected.append(float(np.array(explainer_boos.expected_value).ravel()[0]))

    # concat fold -> singola serie per ripetizione, poi sort_index
    y_true_rep = pd.concat(truth_labels, axis=0).sort_index()

    pred_rf_rep = pd.concat(pred_labels_rf, axis=0).sort_index()
    pred_boos_rep = pd.concat(pred_labels_boos, axis=0).sort_index()
    pred_svr_rep = pd.concat(pred_labels_svr, axis=0).sort_index()
    pred_linr_rep = pd.concat(pred_labels_linr, axis=0).sort_index()

    # metriche (per ripetizione)
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
        }
    }

    # feature importance per ripetizione (fold-wise)
    c_table_rf_rep = pd.concat(c_table_rf, axis=1)
    c_table_boos_rep = pd.concat(c_table_boos, axis=1)

    out = {
        "i": i,
        "y_true": y_true_rep,
        "pred_rf": pred_rf_rep,
        "pred_boos": pred_boos_rep,
        "pred_svr": pred_svr_rep,
        "pred_linr": pred_linr_rep,
        "metrics": metrics,
        "c_table_rf": c_table_rf_rep,
        "c_table_boos": c_table_boos_rep,
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

#%% 1) Esecuzione ML con CV Stratificata e Ripetuta

root = "C:/Users/WKS/Desktop/UNIBA/UN/"

n = 100
compute_shap = False  # mettilo True solo se hai RAM e tempo
save = True

# Carico l'EGDI
all_egdis = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")
countries_df = pd.read_csv(f"{root}/Data/ALL/countries_region.csv").drop(["Unnamed: 0"], axis = 1).set_index(["Countries"])
countries_df.index = countries_df.index.map(country_to_iso3)

unique_areas = ["Asia-Pacific", "Africa", "Europe", "Americas"]

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)

# data_years = np.array([2022])
# egdi_years = np.array([2022])

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:
        
        ###### CARICO EGDI ######
        egdi = all_egdis[f"EGDI_{egdi_year}"]
        egdi.index = list(country_to_iso3.values())
        egdi = egdi.replace([0], [0.01])

        ###### CARICO DATASET ######
        globals()[f"df_{data_year}"] = pd.read_csv(f"{root}/Data/ALL/cleaned_df_{data_year}.csv")
        globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].set_index(["Country_ISO"])
        
        # Sort
        egdi = egdi.sort_index()
        globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].sort_index()
        
        if (egdi.index == globals()[f"df_{data_year}"].index).sum() == globals()[f"df_{data_year}"].shape[0]:
            print("Indici ok.")
            
        # ATTENZIONE: data e y devono essere "puliti" e pronti PRIMA della Parallel
        data = globals()[f"df_{data_year}"].copy()
        y = egdi.copy()
        
        print(f"Analysis: Data year = {data_year}; EGDI year = {egdi_year}.\n")
        
        results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
            delayed(run_one_repetition)(
                i=i,
                areas=unique_areas,
                data=data,   # oppure data
                y=egdi,             # oppure y
                compute_shap = compute_shap, # cambia se serve
            )
            for i in range(n)
        )
        
        # ricostruzione DataFrame predizioni (colonne Rep_1..Rep_n) e y_true unico
        y_pred_total_rf = pd.concat(
            [r["pred_rf"].rename(f"Rep_{r['i']+1}") for r in results],
            axis=1
        )
        y_pred_total_boos = pd.concat(
            [r["pred_boos"].rename(f"Rep_{r['i']+1}") for r in results],
            axis=1
        )
        y_pred_total_svr = pd.concat(
            [r["pred_svr"].rename(f"Rep_{r['i']+1}") for r in results],
            axis=1
        )
        y_pred_total_linr = pd.concat(
            [r["pred_linr"].rename(f"Rep_{r['i']+1}") for r in results],
            axis=1
        )
        
        # y_true: è lo stesso “set completo” (193) per ogni ripetizione, quindi prendo quello della prima
        y_truth_total = results[0]["y_true"]
        
        # metriche globali come nel tuo loop finale
        rmse_global_rf  = [r["metrics"]["RF"]["RMSE"] for r in results]
        mape_global_rf  = [r["metrics"]["RF"]["MAPE"] for r in results]
        r2_global_rf    = [r["metrics"]["RF"]["R2"]   for r in results]
        
        rmse_global_boos = [r["metrics"]["XgB"]["RMSE"] for r in results]
        mape_global_boos = [r["metrics"]["XgB"]["MAPE"] for r in results]
        r2_global_boos   = [r["metrics"]["XgB"]["R2"]   for r in results]
        
        rmse_global_svr = [r["metrics"]["SVR"]["RMSE"] for r in results]
        mape_global_svr = [r["metrics"]["SVR"]["MAPE"] for r in results]
        r2_global_svr   = [r["metrics"]["SVR"]["R2"]   for r in results]
        
        rmse_global_linr = [r["metrics"]["LinR"]["RMSE"] for r in results]
        mape_global_linr = [r["metrics"]["LinR"]["MAPE"] for r in results]
        r2_global_linr   = [r["metrics"]["LinR"]["R2"]   for r in results]
        
        # feature importance globali (concat su tutte rep)
        c_table_rf = pd.concat([r["c_table_rf"] for r in results], axis=1)
        c_table_boos = pd.concat([r["c_table_boos"] for r in results], axis=1)
        
        # Risultati metriche globali di RF, SVR e LinR
        print("\n")
        print(f"Global Error Metrics for {egdi_year} EGDI and {data_year} Data.")
        print("\n")
        
        # RF
        print("RMSE RF --- median {}, iqr = {}".format(np.round(np.median(rmse_global_rf), 3), np.round(iqr(rmse_global_rf), 3)))
        print("MAPE RF --- median {}, iqr = {}".format(np.round(np.median(mape_global_rf), 3), np.round(iqr(mape_global_rf), 3)))
        print("R2 RF --- median {}, iqr = {}".format(np.round(np.median(r2_global_rf), 3), np.round(iqr(r2_global_rf), 3)))
        print("\n")
        
        # XgB
        print("RMSE Boos --- median {}, iqr = {}".format(np.round(np.median(rmse_global_boos), 3), np.round(iqr(rmse_global_boos), 3)))
        print("MAPE Boos --- median {}, iqr = {}".format(np.round(np.median(mape_global_boos), 3), np.round(iqr(mape_global_boos), 3)))
        print("R2 Boos --- median {}, iqr = {}".format(np.round(np.median(r2_global_boos), 3), np.round(iqr(r2_global_boos), 3)))
        print("\n")
        
        # SVR
        print("RMSE SVR --- median {}, iqr = {}".format(np.round(np.median(rmse_global_svr), 3), np.round(iqr(rmse_global_svr), 3)))
        print("MAPE SVR --- median {}, iqr = {}".format(np.round(np.median(mape_global_svr), 3), np.round(iqr(mape_global_svr), 3)))
        print("R2 SVR --- median {}, iqr = {}".format(np.round(np.median(r2_global_svr), 3), np.round(iqr(r2_global_svr), 3)))
        print("\n")
        
        # LinR
        print("RMSE LinR --- median {}, iqr = {}".format(np.round(np.median(rmse_global_linr), 3), np.round(iqr(rmse_global_linr), 3)))
        print("MAPE LinR --- median {}, iqr = {}".format(np.round(np.median(mape_global_linr), 3), np.round(iqr(mape_global_linr), 3)))
        print("R2 LinR --- median {}, iqr = {}".format(np.round(np.median(r2_global_linr), 3), np.round(iqr(r2_global_linr), 3)))
        print("\n")
        
        # Assicuro ordine deterministico: per i (ripetizione) crescente
        results = sorted(results, key=lambda d: d["i"])
        
        if compute_shap:
            # Liste globali come nel codice originale
            train_idx_global = []
            test_idx_global = []
            
            for r in results:
                train_idx_global.extend(r["train_idx"])
                test_idx_global.extend(r["test_idx"])
            
            # Flatten: lista di DataFrame lunga n*k (stesso concetto del tuo shap_global_rf iniziale)
            shap_global_rf = [df for r in results for df in r["shap_rf"]]
            shap_global_boos = [df for r in results for df in r["shap_boos"]]
            
            # Se ti servono anche gli expected_value come nel vecchio codice
            explainer_rf_list = [v for r in results for v in r["exp_rf"]]
            explainer_boos_list = [v for r in results for v in r["exp_boos"]]
        
        print("Done.")
        
        ##########################    Salvo i risultati    ##########################
        
        
        # RF
        results_rf = pd.concat([pd.Series(rmse_global_rf), pd.Series(mape_global_rf), pd.Series(r2_global_rf)], axis = 1)
        results_rf.columns = ["RMSE", "MAPE", "R^2"]
        results_rf.index = np.arange(1, n + 1, 1)
        
        
        # Boos 
        results_boos = pd.concat([pd.Series(rmse_global_boos), pd.Series(mape_global_boos), pd.Series(r2_global_boos)], axis = 1)
        results_boos.columns = ["RMSE", "MAPE", "R^2"]
        results_boos.index = np.arange(1, n + 1, 1)
        
        
        # SVR
        results_svr = pd.concat([pd.Series(rmse_global_svr), pd.Series(mape_global_svr), pd.Series(r2_global_svr)], axis = 1)
        results_svr.columns = ["RMSE", "MAPE", "R^2"]
        results_svr.index = np.arange(1, n + 1, 1)
        
        
        # LinR
        results_linr = pd.concat([pd.Series(rmse_global_linr), pd.Series(mape_global_linr), pd.Series(r2_global_linr)], axis = 1)
        results_linr.columns = ["RMSE", "MAPE", "R^2"]
        results_linr.index = np.arange(1, n + 1, 1)
            
        
        del rmse_global_rf, mape_global_rf, r2_global_rf
        del rmse_global_boos, mape_global_boos, r2_global_boos
        del rmse_global_svr, mape_global_svr, r2_global_svr
        del rmse_global_linr, mape_global_linr, r2_global_linr
        
        if save:
            if compute_shap:
                with open(f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}_regional.pkl", "wb") as f:
                    pickle.dump([shap_global_rf,
                                 shap_global_boos,
                                 test_idx_global,
                                 train_idx_global,
                                 explainer_rf_list,
                                 explainer_boos_list,
                                 c_table_rf,
                                 c_table_boos,
                                 results_rf,
                                 results_boos,
                                 results_svr,
                                 results_linr,
                                 egdi,
                                 y_pred_total_rf,
                                 y_pred_total_boos,
                                 y_pred_total_svr,
                                 y_pred_total_linr], f)
                    
                del f
            else:
                with open(f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}_regional.pkl", "wb") as f:
                    pickle.dump([c_table_rf,
                                 c_table_boos,
                                 results_rf,
                                 results_boos,
                                 results_svr,
                                 results_linr,
                                 egdi,
                                 y_pred_total_rf,
                                 y_pred_total_boos,
                                 y_pred_total_svr,
                                 y_pred_total_linr], f)
        
            print("Valutazioni finite, dati salvati.\n")
        else:
            print("Valutazioni finite, dati NON salvati.\n")
        
        # winsound.Beep(3000, 1500)
