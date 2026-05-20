# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:04:45 2026

@author: WKS
"""

#%% 0) Utilities

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from adjustText import adjust_text

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
#%% 1) Temporal stability analysis: aggregated forecast horizon

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)

metric = "R^2"   # oppure "RMSE" o "MAPE"

models_dict = {
    "RF": [],
    "XGB": [],
    "SVR": [],
    "LinR": []
}

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:

        forecast_horizon = egdi_year - data_year

        with open(f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}.pkl", "rb") as f:
            [_,  # shap_global_rf
             _,  # shap_global_boos
             _,  # test_idx_global
             _,  # train_idx_global
             _,  # explainer_rf_list
             _,  # explainer_boos_list
             _,  # c_table_rf
             _,  # c_table_boos
             results_rf,
             results_boos,
             results_svr,
             results_linr,
             egdi,
             _,  # y_pred_total_rf
             _,  # y_pred_total_boos
             _,  # y_pred_total_svr
             _] = pickle.load(f)

        for model_name, results_df in zip(
            ["RF", "XGB", "SVR", "LinR"],
            [results_rf, results_boos, results_svr, results_linr]
        ):
            temp = pd.DataFrame({
                "model": model_name,
                "data_year": data_year,
                "egdi_year": egdi_year,
                "forecast_horizon": forecast_horizon,
                metric: results_df[metric].values
            })

            models_dict[model_name].append(temp)


# Concateno tutto in un unico dataframe lungo
temporal_stability_df = pd.concat(
    [pd.concat(v, ignore_index=True) for v in models_dict.values()],
    ignore_index=True
)


#%% 2) Aggregazione per forecast horizon

summary_df = (
    temporal_stability_df
    .groupby(["model", "forecast_horizon"])[metric]
    .agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
        mean="mean",
        std="std",
        n="count"
    )
    .reset_index()
)

summary_df["iqr"] = summary_df["q75"] - summary_df["q25"]


#%% 3) Plot: temporal degradation curve aggregata

plt.figure(figsize=(10, 6))

models_dict_names = dict(zip(["RF", "XGB", "SVR", "LinR"], ["Random Forest Regressor", "XGBoost Regressor", "Support Vector Regressor", "Linear Regression"]))

for model_name, model_ext_name in models_dict_names.items():

    df_model = summary_df[summary_df["model"] == model_name].copy()
    df_model = df_model.sort_values("forecast_horizon")

    x = df_model["forecast_horizon"].values
    y = df_model["median"].values
    y_low = df_model["q25"].values
    y_high = df_model["q75"].values

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label=model_ext_name
    )

    plt.fill_between(
        x,
        y_low,
        y_high,
        alpha=0.18
    )

plt.xlabel("Forecast horizon (years)", fontsize=18)
plt.ylabel(f"Cross-validation {metric}", fontsize=18)
# plt.title("Temporal degradation analysis aggregated by forecast horizon", fontsize=14)

xticks = np.arange(0, 16, 2)
xticklabels = ["Nowcasting"] + [f"+{i}y" for i in xticks[1:]]

plt.xticks(xticks, xticklabels, fontsize=15)
plt.yticks(fontsize = 15)
plt.grid(alpha=0.3)
plt.legend(frameon=True, loc="lower left", fontsize="x-large")
plt.tight_layout()
save_dir = f"{root}/Immagini/Temporal_decay/"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(
    f"{save_dir}/temporal_degradation_curve_{metric.replace('^', '')}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()