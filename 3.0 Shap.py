# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:18:07 2026

@author: WKS
"""

#%% 0) Caricamento librerie, funzioni e path

import os
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

root = "C:/Users/WKS/Desktop/UNIBA/UN/"

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


def shap_summary_global(
    dataset,
    global_shap,
    year_egdi,
    year_data,
    global_idx_test,
    model_name,
    how_many=20,
    do_plot=False,
    save=True,
    show=True
):

    warnings.filterwarnings("ignore")

    shap_values_concatenated = np.concatenate(global_shap)
    index_list_concatenated = np.concatenate(global_idx_test)

    mean_abs_shap = np.mean(np.abs(shap_values_concatenated), axis=0)

    feature_importance = pd.DataFrame(
        {"MeanAbsSHAP": mean_abs_shap},
        index=dataset.columns
    )

    feature_importance["idx"] = np.arange(dataset.shape[1])
    feature_importance = feature_importance.sort_values(
        by="MeanAbsSHAP",
        ascending=False
    )

    data_shap = dataset.iloc[index_list_concatenated].copy()

    # mapping: nome feature originale -> feat. idx
    feature_name_mapping = {
        feature_name: f"feat. {idx}"
        for feature_name, idx in feature_importance["idx"].items()
    }

    # nomi ordinati secondo l'ordine effettivo delle colonne di data_shap
    mapped_feature_names = [
        feature_name_mapping[col]
        for col in data_shap.columns
    ]

    if do_plot:
        plt.title(
            f"{model_name} Shap Summary Plot - Global case - Data: {year_data} - EGDI: {year_egdi}",
            fontsize=15,
            loc="center"
        )

        shap.summary_plot(
            shap_values_concatenated,
            np.array(data_shap),
            feature_names=mapped_feature_names,
            show=False,
            max_display=how_many
        )

        if save:
            plt.savefig(
                f"{root}/Immagini/Shap/Summary/shap_globale_data_{year_data}_egdi_{year_egdi}.png",
                bbox_inches="tight",
                dpi=100
            )

        if show:
            plt.show()

        plt.close()

    top_features = feature_importance.head(how_many)

    return top_features


def shap_force(dataset, global_shap, egdi_values, year_egdi, year_data, global_explainers,
                      country_name, how_many, do_plot=False, save=True, show=False):

    shap_df_country_total = pd.DataFrame()
    feature_values_country_total = pd.DataFrame()
    explainer_base_values = []

    for i in range(len(global_shap)):
        shap_df = global_shap[i]

        if country_name in shap_df.index:
            # shap values del paese nella i-esima fold/ripetizione
            shap_row = pd.DataFrame(shap_df.loc[country_name]).T

            # valori originali delle feature dello stesso paese
            x_row = pd.DataFrame(dataset.loc[country_name]).T

            explainer_exp_value = global_explainers[i]

            explainer_base_values.append(explainer_exp_value)
            shap_df_country_total = pd.concat([shap_df_country_total, shap_row], axis=0)
            feature_values_country_total = pd.concat([feature_values_country_total, x_row], axis=0)

    if shap_df_country_total.empty:
        print(f"{country_name} non trovato in global_shap.")
        return None

    base_value_mean = np.mean(explainer_base_values)

    # media sugli shap values del paese
    shap_mean = shap_df_country_total.mean(axis=0)

    if do_plot:
        
        feature_labels = [f"feat. {i}" for i in range(len(shap_mean))]
        
        shap.force_plot(
            base_value=base_value_mean,
            shap_values=shap_mean.values,
            features=None,
            feature_names=feature_labels,
            matplotlib=True,
            show=False
        )
        
        plt.title(f"{country_name} ---- Actual EGDI = {np.round(egdi_values[country_name], 2)}\nData = {year_data}; EGDI = {year_egdi}.",
                  fontsize=30, y=-0.001)

        if save:
            folder = Path(root) / "Immagini" / "Shap" / "Forceplots" / country_name
            folder.mkdir(parents=True, exist_ok=True)
        
            plt.savefig(
                folder / f"force_plot_{country_name}_data_{year_data}_egdi_{year_egdi}.png",
                bbox_inches='tight',
                dpi=100
            )
        if show:
            plt.show()

    # feature importance per paese
    feature_importance = pd.DataFrame(shap_mean, columns=[f"SHAP_{country_name}"])
    feature_importance["sign"] = np.where(feature_importance[f"SHAP_{country_name}"] >= 0,
                                          "Positive", "Negative")
    feature_importance[f"SHAP_{country_name}"] = feature_importance[f"SHAP_{country_name}"].abs()
    feature_importance["idx"] = np.arange(feature_importance.shape[0])
    feature_importance = feature_importance.sort_values(
        by=[f"SHAP_{country_name}"], ascending=False
    )

    top_features = feature_importance.head(how_many)
    top_features.index.name = "Features"

    return top_features



def shap_barplot(global_shap, year_egdi, year_data, how_many, do_plot = False, save = True, show = False):
    
    # Dataframe dei valori assoluti degli shap values ottenuti in CV
    shap_abs_tot = pd.DataFrame()
    
    for i in range(0, len(global_shap)):
        shap_df = global_shap[i].abs()
        shap_abs_tot = pd.concat([shap_abs_tot, shap_df], axis = 0)
    
    del i, shap_df
    
    shap_abs_tot = shap_abs_tot.astype(float)
    
    shap_abs_tot_mean = shap_abs_tot.mean().to_frame()
    shap_abs_tot_mean.columns = ['ShapMeanAbs']
    shap_abs_tot_mean['idx'] = np.arange(0, shap_abs_tot_mean.shape[0], 1)
    shap_abs_tot_mean = shap_abs_tot_mean.sort_values(by = ['ShapMeanAbs'], ascending = False)
    
    shap_abs_tot_std = shap_abs_tot.std()
    shap_abs_tot_std.name = "ShapMeanAbsStd"
    shap_abs_tot_std = shap_abs_tot_std.to_frame().reindex(shap_abs_tot_mean.index)
    shap_abs = pd.concat([shap_abs_tot_mean, shap_abs_tot_std], axis = 1)
    shap_abs = shap_abs[['ShapMeanAbs', 'ShapMeanAbsStd', 'idx']]
    shap_abs_partial = shap_abs.iloc[:how_many, :]
    
    del shap_abs_tot, shap_abs_tot_mean, shap_abs_tot_std
    
    if do_plot:
        plt.bar(x = np.arange(0, shap_abs_partial.shape[0], 1),
                height = shap_abs_partial['ShapMeanAbs'],
                yerr = shap_abs_partial['ShapMeanAbsStd'],
                width = 0.9,
                edgecolor = 'black',
                color = 'crimson')
        plt.xticks(np.arange(0, shap_abs_partial.shape[0], 1), [f"feat. {num}" for num in list(shap_abs_partial['idx'])], rotation = 90)
        plt.yscale('log')
        plt.ylim((0, 0.5))
        plt.ylabel('Mean absolute SHAP values', fontsize = 15)
        plt.title(f"Mean SHAP absolute values barplot.\nData = {year_data}; EGDI = {year_egdi}.")
        if save:
            plt.savefig(f"{root}/Immagini/Shap/Barplots/barplot_data_{year_data}_egdi_{year_egdi}.png", dpi = 100, bbox_inches = 'tight')
        if show:
            plt.show()
    
    return shap_abs_partial

#%% 1) Caricamento dati ed analisi SHAP

# Carico l'EGDI
all_egdis = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")
countries = list(country_to_iso3.values())

# Casi singoli
interesting_countries = ["ITA",
                         "DZA",
                         "AFG"]

data_years = np.arange(2022, 2026, 2)
egdi_years = np.arange(2022, 2026, 2)

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:
        
        print(f"Data = {data_year}; EGDI = {egdi_year};")
        
        egdi = all_egdis[f"EGDI_{egdi_year}"]
        egdi.index = list(country_to_iso3.values())
        egdi = egdi.replace([0], [0.01])

        with open(f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}.pkl", "rb") as f:
            [shap_global_rf,
             shap_global_boos,
             test_idx_global,
             _, # train_idx_global,
             explainer_rf_list,
             explainer_boos_list,
             _, # c_table_rf,
             _, # c_table_boos,
             _, # results_rf,
             _, # results_boos,
             _, # results_svr,
             _, # results_linr,
             _, # egdi,
             _, # y_pred_total_rf,
             _, # y_pred_total_boos,
             _, # y_pred_total_svr,
             y_pred_total_linr] = pickle.load(f)
            
        del f, y_pred_total_linr

        # Carico il dataset
        globals()[f"df_{data_year}"] = pd.read_csv(f"{root}/Data/ALL/cleaned_df_{data_year}.csv")
        globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].set_index(["Country_ISO"])        
        
        # Sort
        egdi = egdi.sort_index()
        globals()[f"df_{data_year}"] = globals()[f"df_{data_year}"].sort_index()
    
        data = globals()[f"df_{data_year}"].copy()
        
        # Feature importance: numero di feature da considerare
        num_cols = globals()[f"df_{data_year}"].shape[1]
        # how_many = num_cols if num_cols < 20 else 20
        how_many = num_cols
        print(f"Features to consider: {how_many}.")

        # Caso globale
        top_features_global = shap_summary_global(data,
                                                  shap_global_boos, # shap_global_rf,
                                                  egdi_year,
                                                  data_year,
                                                  test_idx_global,
                                                  how_many = how_many,
                                                  model_name = "XgB",
                                                  do_plot = True, # if (data_year >= 2020 and egdi_year >= 2020) else False,
                                                  save = True, # True if (data_year >= 2020 and egdi_year >= 2020) else False,
                                                  show = True) #if (data_year >= 2020 and egdi_year >= 2020) else False)
        
        top_features_global.to_excel(f"{root}/Risultati/ML/Shap/Top_features/Global/top_features_data_{data_year}_egdi_{egdi_year}.xlsx")
        
        # Barplot
        _ = shap_barplot(shap_global_boos, # shap_global_rf,
                         egdi_year,
                         data_year,
                         how_many = how_many,
                         do_plot = True, # if (data_year >= 2020 and egdi_year >= 2020) else False,
                         save = True, # True if (data_year >= 2020 and egdi_year >= 2020) else False,
                         show = True) # if (data_year >= 2020 and egdi_year >= 2020) else False)

        for country in interesting_countries: # countries
            
            globals()[f"top_features_{country}"] = shap_force(data,
                                                              shap_global_boos,
                                                              egdi,
                                                              egdi_year,
                                                              data_year,
                                                              explainer_boos_list,
                                                              country,
                                                              how_many = how_many,
                                                              do_plot = True, # if (data_year >= 2020 and egdi_year >= 2020) else False,
                                                              save = True, # if (data_year >= 2020 and egdi_year >= 2020) else False,
                                                              show = True) # if (data_year >= 2020 and egdi_year >= 2020) else False)
            folder = Path(root) / "Risultati" / "ML" / "Shap" / "Top features" / "Countries" / country
            folder.mkdir(parents=True, exist_ok=True)
            globals()[f"top_features_{country}"].to_csv(f"{folder}/top_features_{country}_data_{data_year}_egdi_{egdi_year}.csv")
            
            del country
        del how_many
        print("\n")
    print("\n")

del explainer_rf_list
del data_year, egdi_year
# del top_features_global

