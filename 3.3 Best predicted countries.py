# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:07:33 2026

@author: WKS
"""

#%% 0) Caricamento librerie, path e dati EGDI 2024 (EGDI e Ranking) e dati SDG 2022.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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

root = "C:/Users/WKS/Desktop/UNIBA/UN/"

#%% 1) Upload dati

save = True
data_year = 2022
egdi_year = 2022

with open(f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}.pkl", "rb") as f:
    [_, # shap_global_rf,
     _, # shap_global_boos,
     _, # test_idx_global,
     _, # train_idx_global,
     _, # explainer_rf_list,
     _, # explainer_boos_list,
     _, # c_table_rf,
     _, # c_table_boos,
     results_rf,
     results_boos,
     results_svr,
     results_linr,
     egdi,
     y_pred_total_rf,
     y_pred_total_boos,
     y_pred_total_svr,
     y_pred_total_linr] = pickle.load(f)
    
del f

egdi_df = y_pred_total_boos.copy()

# Sottrazione tra egdi predetto ed egdi vero
egdi_df = egdi_df.T - egdi

# Ordine dei Paesi in base all'EGDI reale
# ascending=False -> dal più alto EGDI al più basso
country_order = egdi.sort_values(ascending=False).index

# Riordino delle colonne del dataframe delle differenze
egdi_df = egdi_df[country_order]

plt.figure(figsize=(32, 8))

for idx, country in enumerate(egdi_df.columns):
    data = egdi_df[country].dropna()
    plt.boxplot(
        data,
        positions=[idx + 1],
        widths=0.7,
        patch_artist=True,
        boxprops=dict(facecolor='cornflowerblue', color='black'),
        medianprops=dict(color='purple'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
        showfliers=False
    )

plt.axhline(0, color='darkred', linewidth=2, label='Perfect agreement')

xticks_labels = egdi_df.columns
xticks_positions = np.arange(1, len(egdi_df.columns) + 1)

for tick, label in zip(xticks_positions, xticks_labels):
    plt.text(
        tick,
        -0.45,
        label,
        fontsize=11,
        rotation=90,
        ha='center'
    )

plt.xticks(ticks=xticks_positions, labels=xticks_labels, fontsize=0)
plt.yticks(fontsize=15)
plt.xlabel(f"Countries ordered by actual {egdi_year} EGDI", labelpad=40, fontsize=40)
plt.ylabel("Difference\nPredicted - Actual EGDI", fontsize=40)
plt.grid(False)

plt.xlim(0, len(egdi_df.columns) + 1)
plt.ylim(-0.4, 0.4)
plt.tight_layout()
plt.legend(fontsize=28)
if save:
    plt.savefig(f"{root}/Immagini/Boxplots differenze/bp_data_{data_year}_egdi_{egdi_year}.png", dpi = 100)
plt.show()

#%% 2) Best predicted countries

threshold = 0.007

median_deviations = egdi_df.median().abs()
print(f"Best predicted countries with median deviation <= {threshold}.")
print("\n")

# EGDI <= 1st EGDI quartile
first_group = list(egdi[egdi <= egdi.quantile(0.25)].index)
first_group_deviations = median_deviations.loc[first_group]
best_predicted_first_group_countries = first_group_deviations[first_group_deviations <= threshold]
best_predicted_first_group_countries = best_predicted_first_group_countries.sort_values(ascending = False)
print("EGDI <= 1st EGDI quartile are:")
for country in best_predicted_first_group_countries.index:
    print(country + f" ---- EGDI: {egdi.loc[country]}")
    del country
print("\n")
del first_group_deviations, best_predicted_first_group_countries
# AFG
    
# 1st EGDI quartile < EGDI <= 2nd EGDI quartile
second_group = list(egdi[(egdi.quantile(0.25) < egdi) & (egdi <= egdi.quantile(0.50))].index)
second_group_deviations = median_deviations.loc[second_group]
best_predicted_second_group_countries = second_group_deviations[second_group_deviations <= threshold]
best_predicted_second_group_countries = best_predicted_second_group_countries.sort_values(ascending = False)
print("1st EGDI quartile < EGDI <= 2nd EGDI quartile are:")
for country in best_predicted_second_group_countries.index:
    print(country + f" ---- EGDI: {egdi.loc[country]}")
    del country
print("\n")
del second_group_deviations, best_predicted_second_group_countries
# CIV

# 2nd EGDI quartile < EGDI <= 3rd EGDI quartile
third_group = list(egdi[(egdi.quantile(0.50) < egdi) & (egdi <= egdi.quantile(0.75))].index)
third_group_deviations = median_deviations.loc[third_group]
best_predicted_third_group_countries = third_group_deviations[third_group_deviations <= threshold]
best_predicted_third_group_countries = best_predicted_third_group_countries.sort_values(ascending = False)
print("2nd EGDI quartile < EGDI <= 3rd EGDI quartile are:")
for country in best_predicted_third_group_countries.index:
    print(country + f" ---- EGDI: {egdi.loc[country]}")
    del country
print("\n")
del third_group_deviations, best_predicted_third_group_countries
# MEX

# EGDI > 4th EGDI quartile
fourth_group = list(egdi[egdi > egdi.quantile(0.75)].index)
fourth_group_deviations = median_deviations.loc[fourth_group]
best_predicted_fourth_group_countries = fourth_group_deviations[fourth_group_deviations <= threshold]
best_predicted_fourth_group_countries = best_predicted_fourth_group_countries.sort_values(ascending = False)
print("EGDI > 4th EGDI quartile are:")
for country in best_predicted_fourth_group_countries.index:
    print(country + f" ---- EGDI: {egdi.loc[country]}")
    del country
print("\n")
del fourth_group_deviations, best_predicted_fourth_group_countries
# JPN

countries = first_group + second_group + third_group + fourth_group
countries.sort()
if countries == list(egdi.index):
    print("Counts are ok.")


