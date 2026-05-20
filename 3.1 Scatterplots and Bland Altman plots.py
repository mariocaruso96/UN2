# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:05:03 2026

@author: WKS
"""
#%% 0) Utilities

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

#%% 1) Uploading

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:

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
             y_pred_total_rf,
             y_pred_total_boos,
             y_pred_total_svr,
             y_pred_total_linr] = pickle.load(f)
        
        models_names = ["rf", "boos", "svr", "linr"]
        egdi.index = list(country_to_iso3.values())
        
        for model in models_names:
        
            if model == "rf":
                model_name = "Random Forest Regressor"
                color = "green"
            elif model == "boos":
                model_name = "XGBoost Regressor"
                color = "orange"
            elif model == "svr":
                model_name = "Support Vector Regressor"
                color = "blue"
            elif model == "linr":
                model_name = "Linear Regression"
                color = "purple"
        
            predictions_df = globals()[f"y_pred_total_{model}"].copy()
            results_model = globals()[f"results_{model}"].copy()
        
            mean_predictions = predictions_df.mean(axis=1)
        
            median_result = round(results_model["R^2"].median(), 3)
            iqr_result = round(
                results_model["R^2"].quantile(0.75) - results_model["R^2"].quantile(0.25),
                3
            )
        
            # --- Data for plots ---
            true = egdi.astype(float).values
            pred = mean_predictions.astype(float).values
        
            # Scatter: y vs x
            bisector_x = np.arange(0, 1.001, 0.001)
            bisector_y = bisector_x
        
            # Bland–Altman: difference vs mean
            ba_mean = (pred + true) / 2.0
            ba_diff = pred - true
            ba_bias = ba_diff.mean()
            ba_sd = ba_diff.std(ddof=1)
            loa_upper = ba_bias + 1.96 * ba_sd
            loa_lower = ba_bias - 1.96 * ba_sd
        
            # --- Figure with two rows: Scatter + Bland–Altman ---
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), sharex=False)
        
            # ============================================================
            # (1) Scatter plot: Actual vs Predicted EGDI
            # ============================================================
            ax = axes[0]
        
            ax.scatter(
                true,
                pred,
                color=color,
                marker="^",
                s=40,
                label=f"{model_name}\n$R^2$ = {median_result} ± {iqr_result}"
            )
        
            ax.plot(
                bisector_x,
                bisector_y,
                color="red",
                label="Bisector"
            )
        
            texts_scatter = []
        
            for i, country in enumerate(mean_predictions.index):
                if (ba_diff[i] > loa_upper) or (ba_diff[i] < loa_lower):
                    texts_scatter.append(
                        ax.text(
                            true[i],
                            pred[i],
                            country,
                            color="black",
                            fontsize=12
                        )
                    )
        
            adjust_text(
                texts_scatter,
                ax=ax,
                expand_points=(1.5, 1.8),
                expand_text=(1.4, 1.7),
                force_text=(0.8, 1.0),
                force_points=(0.5, 0.7),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.6
                )
            )
        
            ax.set_xlabel(f"Actual {egdi_year} EGDI", fontsize=18)
            ax.set_ylabel("Predicted EGDI", fontsize=18)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=13)
            ax.legend(fontsize=14, loc="upper left")
        
            # ============================================================
            # (2) Bland–Altman plot
            # ============================================================
            ax2 = axes[1]
        
            ax2.scatter(
                ba_mean,
                ba_diff,
                s=35,
                marker="o",
                color=color,
                alpha=0.85
            )
        
            ax2.axhline(
                ba_bias,
                color="black",
                linewidth=1.5,
                label=f"Bias = {ba_bias:.3f}"
            )
        
            ax2.axhline(
                loa_upper,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"LoA upper = {loa_upper:.3f}"
            )
        
            ax2.axhline(
                loa_lower,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"LoA lower = {loa_lower:.3f}"
            )
        
            ax2.axhline(
                0.0,
                color="gray",
                linestyle=":",
                linewidth=1.2
            )
        
            texts_ba = []
        
            for i, country in enumerate(mean_predictions.index):
                if (ba_diff[i] > loa_upper) or (ba_diff[i] < loa_lower):
                    texts_ba.append(
                        ax2.text(
                            ba_mean[i],
                            ba_diff[i],
                            country,
                            color="black",
                            fontsize=12
                        )
                    )
        
            adjust_text(
                texts_ba,
                ax=ax2,
                expand_points=(1.5, 1.8),
                expand_text=(1.4, 1.7),
                force_text=(1.0, 1.2),
                force_points=(0.6, 0.8),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.6
                )
            )
        
            ax2.set_xlim(0, 1)
            ax2.set_ylim(-0.5, 0.5)
            ax2.set_xlabel("Mean of (Predicted, Actual)", fontsize=18)
            ax2.set_ylabel("Difference (Predicted − Actual)", fontsize=18)
            ax2.tick_params(labelsize=13)
            ax2.legend(fontsize=13, loc="upper left")
        
            # ============================================================
            # Save + show
            # ============================================================
            outpath = (
                f"{root}/Immagini/Scatterplots/"
                f"scatterplot_and_bland_altman_{model}_data_{data_year}_egdi_{egdi_year}.png"
            )
        
            fig.suptitle(f"Data {data_year}; EGDI {egdi_year}", fontsize=20)
            fig.tight_layout()
            fig.savefig(outpath, dpi=100, bbox_inches="tight")
            plt.show()









