# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:52:08 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Elaborazione

# =========================
# 0) Parametri
# =========================

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)

output_dir = f"{root}/Immagini/Shap/Macrofamily"
os.makedirs(output_dir, exist_ok=True)
all_egdis = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")


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

    # Governance / institutions / rule of law
    "Control of Corruption: Estimate - WDI": "Governance and institutions",
    "Government Effectiveness: Estimate - WDI": "Governance and institutions",
    "Regulatory Quality: Estimate - WDI": "Governance and institutions",
    "Rule of Law: Estimate - WDI": "Governance and institutions",
    "Voice and Accountability: Estimate - WDI": "Governance and institutions",
    "SG_NHR_IMPL - Proportion of countries with independent National Human Rights Institutions in compliance with the Paris Principles (%) - Goal 16 - Reporting Type: G - Units: PERCENT - Nature: G - SDG": "Governance and institutions",

    # Statistical capacity / data governance
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


# =========================
# 2) Tabella feature uniche + frequenza + macrofamiglia
# =========================

feature_counter = Counter()

for data_year in data_years:

    df = pd.read_csv(f"{root}/Data/ALL/cleaned_df_{data_year}.csv")
    df = df.set_index("Country_ISO")

    globals()[f"df_{data_year}"] = df

    feature_counter.update(df.columns)

features_frequency_df = pd.DataFrame({
    "Feature": list(feature_counter.keys()),
    "Frequency": list(feature_counter.values())
})

features_frequency_df["Presence_Ratio"] = (
    features_frequency_df["Frequency"] / len(data_years)
)

features_frequency_df["Presence_Percentage"] = (
    100 * features_frequency_df["Presence_Ratio"]
)

features_frequency_df["Macrofamily"] = (
    features_frequency_df["Feature"]
    .map(macrofamily_dict)
    .fillna("Other / not classified")
)

features_frequency_df = (
    features_frequency_df
    .sort_values(
        by=["Frequency", "Macrofamily", "Feature"],
        ascending=[False, True, True]
    )
    .reset_index(drop=True)
)

# features_frequency_df.to_csv(
#     f"{root}/features_frequency_with_macrofamily.csv",
#     index=False
# )

print(features_frequency_df)


# =========================
# 3) SHAP macrofamily importance
# =========================

macrofamily_records = []
feature_records = []

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:

        print(f"Data = {data_year}; EGDI = {egdi_year};")

        egdi = all_egdis[f"EGDI_{egdi_year}"]
        egdi.index = list(country_to_iso3.values())
        egdi = egdi.replace([0], [0.01])

        with open(
            f"{root}/Risultati/ML/results_data_{data_year}_egdi_{egdi_year}.pkl",
            "rb"
        ) as f:

            [
                shap_global_rf,
                shap_global_boos,
                test_idx_global,
                _,  # train_idx_global
                explainer_rf_list,
                explainer_boos_list,
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
                y_pred_total_linr
            ] = pickle.load(f)

        del y_pred_total_linr

        df = pd.read_csv(f"{root}/Data/ALL/cleaned_df_{data_year}.csv")
        df = df.set_index("Country_ISO")

        globals()[f"df_{data_year}"] = df

        # =========================
        # Concatenazione SHAP values
        # =========================

        shap_dfs = []

        for shap_fold in shap_global_boos:

            if isinstance(shap_fold, pd.DataFrame):
                tmp = shap_fold.copy()
            else:
                tmp = pd.DataFrame(
                    shap_fold,
                    columns=df.columns
                )

            shap_dfs.append(tmp)

        shap_concat = pd.concat(shap_dfs, axis=0, ignore_index=True)
        shap_concat = shap_concat[[c for c in shap_concat.columns if c in df.columns]]

        # =========================
        # Feature-level importance
        # =========================

        mean_abs_shap = shap_concat.abs().mean(axis=0)

        tmp_feature_importance = pd.DataFrame({
            "Data_Year": data_year,
            "EGDI_Year": egdi_year,
            "Configuration": f"{data_year}→{egdi_year}",
            "Forecast_Horizon": egdi_year - data_year,
            "Feature": mean_abs_shap.index,
            "MeanAbsSHAP": mean_abs_shap.values
        })

        tmp_feature_importance["Macrofamily"] = (
            tmp_feature_importance["Feature"]
            .map(macrofamily_dict)
            .fillna("Other / not classified")
        )

        feature_records.append(tmp_feature_importance)

        # =========================
        # Macrofamily-level importance
        # =========================
        # MeanAbsSHAP_Mean = importanza media della singola feature nella famiglia
        # MeanAbsSHAP_Sum  = contributo cumulativo della famiglia
        # Feature_Count    = numero di feature della famiglia in quella configurazione

        tmp_macro = (
            tmp_feature_importance
            .groupby(
                [
                    "Data_Year",
                    "EGDI_Year",
                    "Configuration",
                    "Forecast_Horizon",
                    "Macrofamily"
                ],
                as_index=False
            )
            .agg(
                MeanAbsSHAP_Mean=("MeanAbsSHAP", "mean"),
                MeanAbsSHAP_Sum=("MeanAbsSHAP", "sum"),
                Feature_Count=("Feature", "count")
            )
        )

        # Normalizzazione sulla media:
        # confronta le famiglie correggendo per il diverso numero di feature.
        total_mean_importance = tmp_macro["MeanAbsSHAP_Mean"].sum()

        tmp_macro["RelativeMeanImportance"] = (
            tmp_macro["MeanAbsSHAP_Mean"] / total_mean_importance
        )

        tmp_macro["RelativeMeanImportance_Percentage"] = (
            100 * tmp_macro["RelativeMeanImportance"]
        )

        # Normalizzazione sulla somma:
        # utile come informazione supplementare sul contributo cumulativo.
        total_sum_importance = tmp_macro["MeanAbsSHAP_Sum"].sum()

        tmp_macro["RelativeSumImportance"] = (
            tmp_macro["MeanAbsSHAP_Sum"] / total_sum_importance
        )

        tmp_macro["RelativeSumImportance_Percentage"] = (
            100 * tmp_macro["RelativeSumImportance"]
        )

        macrofamily_records.append(tmp_macro)


# =========================
# 4) DataFrame finali
# =========================

feature_shap_importance_df = pd.concat(
    feature_records,
    axis=0,
    ignore_index=True
)

macrofamily_shap_importance_df = pd.concat(
    macrofamily_records,
    axis=0,
    ignore_index=True
)

# feature_shap_importance_df.to_csv(
#     f"{root}/feature_level_shap_importance_all_configurations.csv",
#     index=False
# )

# macrofamily_shap_importance_df.to_csv(
#     f"{root}/macrofamily_level_shap_importance_all_configurations.csv",
#     index=False
# )


# =========================
# 5) Numerosità globale delle macrofamiglie
# =========================

macrofamily_global_count = (
    features_frequency_df
    .groupby("Macrofamily")["Feature"]
    .nunique()
    .to_dict()
)


# =========================
# 6) Heatmap macrofamiglie - MEDIA
# =========================

heatmap_df = macrofamily_shap_importance_df.pivot_table(
    index="Macrofamily",
    columns="Configuration",
    values="RelativeMeanImportance_Percentage",
    aggfunc="mean"
)

# Ordino le colonne temporalmente
ordered_columns = []

for data_year in data_years:
    for egdi_year in egdi_years[egdi_years >= data_year]:
        col = f"{data_year}→{egdi_year}"
        if col in heatmap_df.columns:
            ordered_columns.append(col)

heatmap_df = heatmap_df[ordered_columns]

# Ordino le righe per importanza media, ignorando i NaN
heatmap_df["Mean_Importance"] = heatmap_df.mean(axis=1, skipna=True)
heatmap_df = heatmap_df.sort_values("Mean_Importance", ascending=False)
heatmap_df = heatmap_df.drop(columns="Mean_Importance")

# Versione con NaN per availability matrix
heatmap_df_with_nan = heatmap_df.copy()

# Versione con zero per plot principale
heatmap_df = heatmap_df.fillna(0)

# heatmap_df.to_csv(
#     f"{root}/macrofamily_mean_shap_heatmap_matrix_filled_zero.csv"
# )

# heatmap_df_with_nan.to_csv(
#     f"{root}/macrofamily_mean_shap_heatmap_matrix_with_nan.csv"
# )


# =========================
# 7) Availability matrix
# =========================

availability_df = heatmap_df_with_nan.notna().astype(int)

# availability_df.to_csv(
#     f"{root}/macrofamily_availability_matrix.csv"
# )


# =========================
# 8) Etichette y con numerosità famiglia
# =========================

ytick_labels = [
    f"{family} (n={macrofamily_global_count.get(family, 0)})"
    for family in heatmap_df.index
]


# =========================
# 9) Plot heatmap SHAP importance - MEDIA
# =========================

plt.figure(figsize=(23, 8))

ax = sns.heatmap(
    heatmap_df,
    cmap="viridis",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={
        "label": "Relative mean SHAP importance (%)"
    }
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.label.set_size(16)

ax.set_xlabel("Temporal configuration", fontsize = 18)
ax.set_ylabel("Macrofamily", fontsize = 18)
ax.set_yticklabels(ytick_labels, rotation=0, fontsize = 16)

# ax.set_title(
#     "Macrofamily-level mean SHAP importance across temporal EGDI prediction configurations"
# )

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/macrofamily_mean_shap_importance_heatmap_filled_zero.png",
    dpi=600,
    bbox_inches="tight"
)

# plt.savefig(
#     f"{output_dir}/macrofamily_mean_shap_importance_heatmap_filled_zero.pdf",
#     bbox_inches="tight"
# )

plt.show()


# =========================
# 10) Plot availability matrix
# =========================

plt.figure(figsize=(23, 8))

ax = sns.heatmap(
    availability_df.loc[heatmap_df.index],
    cmap="Blues",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={
        "label": "Feature-family availability"
    }
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.label.set_size(16)

ax.set_xlabel("Temporal configuration", fontsize = 18)
ax.set_ylabel("Macrofamily", fontsize = 18)
ax.set_yticklabels(ytick_labels, rotation=0, fontsize = 16)

# ax.set_title(
#     "Macrofamily availability across temporal EGDI prediction configurations"
# )

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    f"{output_dir}/macrofamily_availability_matrix.png",
    dpi=600,
    bbox_inches="tight"
)

# plt.savefig(
#     f"{output_dir}/macrofamily_availability_matrix.pdf",
#     bbox_inches="tight"
# )

plt.show()
