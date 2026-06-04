# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:32:29 2026

@author: WKS
"""

import numpy as np
import pandas as pd
from collections import Counter

root = "C:/Users/WKS/Desktop/UNIBA/UN"

# =========================
# Parametri
# =========================

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)


# =========================
# Dizionario macrofamiglie
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
# Dizionario abbreviazioni macrofamiglie
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


# =========================
# Counter per frequenza feature
# =========================

feature_counter = Counter()


# =========================
# Loop dataset
# =========================

for data_year in data_years:

    # =========================
    # Caricamento dataset
    # =========================

    df = pd.read_csv(
        f"{root}/Data/ALL/cleaned_df_{data_year}.csv"
    )

    df = df.set_index("Country_ISO")

    # Salvo eventualmente in globals
    globals()[f"df_{data_year}"] = df

    # =========================
    # Estrazione feature
    # =========================

    features = list(df.columns)

    # Aggiorna conteggio frequenze
    feature_counter.update(features)


# =========================
# Creazione dataframe finale
# =========================

features_frequency_df = pd.DataFrame({
    "Feature": list(feature_counter.keys()),
    "Frequency": list(feature_counter.values())
})


# =========================
# Associazione macrofamiglia
# =========================

features_frequency_df["Macrofamily"] = (
    features_frequency_df["Feature"]
    .map(macrofamily_dict)
    .fillna("Other / not classified")
)

features_frequency_df["Macrofamily_Abbrev"] = (
    features_frequency_df["Macrofamily"]
    .map(macrofamily_abbrev)
)


# =========================
# Ordinamento
# =========================

features_frequency_df = (
    features_frequency_df
    .sort_values(
        by=["Frequency", "Macrofamily", "Feature"],
        ascending=[False, True, True]
    )
    .reset_index(drop=True)
)


# =========================
# Statistiche aggiuntive
# =========================

features_frequency_df["Presence_Ratio"] = (
    features_frequency_df["Frequency"] / len(data_years)
)

features_frequency_df["Presence_Percentage"] = (
    100 * features_frequency_df["Presence_Ratio"]
)


# =========================
# Output
# =========================

print(features_frequency_df)

features_frequency_df = features_frequency_df.drop(["Macrofamily", "Presence_Ratio"], axis = 1)
features_frequency_df = features_frequency_df[["Feature", "Macrofamily_Abbrev", "Frequency", "Presence_Percentage"]]

# =========================
# Salvataggio
# =========================

features_frequency_df.to_csv(
    f"{root}/Data/features_frequency_analysis.csv",
    index=False
)