# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:32:29 2026

@author: WKS
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import Counter

root = "C:/Users/WKS/Desktop/UNIBA/UN"

# =========================
# Parametri
# =========================

data_years = np.arange(2010, 2026, 2)
egdi_years = np.arange(2010, 2026, 2)

# Counter per frequenza feature
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
# Ordinamento
# =========================

features_frequency_df = (
    features_frequency_df
    .sort_values(
        by=["Frequency", "Feature"],
        ascending=[False, True]
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

# =========================
# Salvataggio
# =========================

features_frequency_df.to_csv(
    f"{root}/features_frequency_analysis.csv",
    index=False
)