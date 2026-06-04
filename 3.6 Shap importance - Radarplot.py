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

#%%

# =========================
# 6) Radar plot macrofamiglie - MEDIA SHAP
# =========================

macrofamily_abbrev = {
    "Governance and institutions": "GI",
    "Statistical capacity and data infrastructure": "SD",
    "Digital and technological infrastructure": "DT",
    "Economic structure and services": "ES",
    "Education and human capital": "EH",
    "Financial system and inclusion": "FI",
    "Public finance and fiscal capacity": "PF",
    "Logistics and trade facilitation": "LT",
    "Other / not classified": "OT"
}

label_colors = {
    "GI": "darkviolet",
    "SD": "mediumblue",
    "ES": "royalblue",
    "LT": "cornflowerblue",
    "DT": "teal",
    "EH": "green",
    "FI": "darkorange",
    "PF": "orange",
    "OT": "gray"
}

# =========================
# Aggregazione globale per macrofamiglia
# =========================

radar_df = (
    macrofamily_shap_importance_df
    .groupby("Macrofamily", as_index=False)
    .agg(
        Mean_Relative_Importance=("RelativeMeanImportance_Percentage", "mean")
    )
)

radar_df["Abbreviation"] = radar_df["Macrofamily"].map(macrofamily_abbrev)

radar_df = radar_df.dropna(subset=["Abbreviation"]).copy()

# ordine decrescente per importanza media
radar_df = (
    radar_df
    .sort_values("Mean_Relative_Importance", ascending=False)
    .reset_index(drop=True)
)

radar_order = radar_df["Abbreviation"].tolist()

labels = radar_df["Abbreviation"].values
values = radar_df["Mean_Relative_Importance"].values

values_closed = np.concatenate([values, [values[0]]])

angles = np.linspace(
    0,
    2 * np.pi,
    len(labels),
    endpoint=False
)

angles_closed = np.concatenate([angles, [angles[0]]])

fig, ax = plt.subplots(
    figsize=(14, 14),
    subplot_kw=dict(polar=True)
)

# primo valore in alto, poi senso orario
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.plot(
    angles_closed,
    values_closed,
    color="blue",
    linewidth=3,
    marker=".",
    markersize=10
)

ax.fill(
    angles_closed,
    values_closed,
    color="dodgerblue",
    alpha=0.35
)

ax.set_xticks(angles)

xtick_labels = ax.set_xticklabels(
    labels,
    fontsize=25,
    fontweight="bold"
)

for label in xtick_labels:
    txt = label.get_text()
    label.set_color(label_colors[txt])

ax.grid(
    True,
    linewidth=1.8,
    alpha=0.8
)

ax.tick_params(
    axis="y",
    labelsize=16
)

ax.set_rlabel_position(315)

from matplotlib.lines import Line2D

legend_elements = []

for abbr in radar_order:

    name = radar_df.loc[
        radar_df["Abbreviation"] == abbr,
        "Macrofamily"
    ].iloc[0]

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"{abbr} = {name}",
            markerfacecolor=label_colors[abbr],
            markersize=22
        )
    )

legend = ax.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.25, 0.5),
    fontsize=14,
    frameon=True,
    edgecolor="white",
    borderpad=1.5,
    labelspacing=2.3,
    handletextpad=1
)

legend.get_frame().set_linewidth(2)

for text in legend.get_texts():

    abbr = text.get_text().split(" = ")[0]

    text.set_color(label_colors[abbr])
    text.set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{root}/Immagini/Shap/Macrofamily/shap_macrofamilies_radarplot.png", bbox_inches = 'tight', dpi = 300)
plt.show()