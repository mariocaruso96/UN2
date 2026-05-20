# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:14:29 2026

@author: WKS
"""

#%% 0) Utilities

import numpy as np
import pandas as pd
from pathlib import Path

root = Path("C:/Users/WKS/Desktop/UNIBA/UN")

#%% 1) Data loading

data = pd.read_csv(root / "Data" / "ITU" / "ITU_data.csv")

# 2) Basic cleaning

cols_to_drop = [
    "seriesID",
    "seriesCode",
    "seriesParent",
    "seriesUnits",
    "entityID",
    "entityName",
    "dataNote",
    "dataSource",
    "seriesDescription"
]

data = data.drop(columns=cols_to_drop, errors="ignore")

other_cols = [f"dataValue {i}" for i in np.arange(1, 13, 1)]
data = data.drop(columns=other_cols, errors="ignore")

data = data.rename(columns={
    "dataYear": "Year",
    "entityIso": "Country_ISO",
    "seriesName": "Feature",
    "dataValue": "Value"
})

data = data[["Year", "Country_ISO", "Feature", "Value"]].copy()

# opzionale ma consigliato
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

# 3) Basic checks

data = data.sort_values(["Country_ISO", "Year", "Feature"]).reset_index(drop=True)

print("Shape raw reduced data:", data.shape)
print("Unique countries:", data["Country_ISO"].nunique())
print("Unique years:", data["Year"].nunique())
print("Unique features:", data["Feature"].nunique())

# 4) Check duplicates on (Country_ISO, Year, Feature)

dup_mask = data.duplicated(subset=["Country_ISO", "Year", "Feature"], keep=False)

if dup_mask.any():
    print("\nATTENZIONE: trovati duplicati per (Country_ISO, Year, Feature).")
    dup_df = data.loc[dup_mask].sort_values(["Country_ISO", "Year", "Feature"])
    print(dup_df.head(20))
    print(f"Numero righe duplicate: {dup_df.shape[0]}")
else:
    print("\nNessun duplicato trovato per (Country_ISO, Year, Feature).")

# 5) Handle duplicates

def first_non_null(series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan

data_clean = (
    data
    .groupby(["Country_ISO", "Year", "Feature"], as_index=False)["Value"]
    .agg(first_non_null)
)

print("\nShape after duplicate handling:", data_clean.shape)

# 6) Wide table creation

data_country_cleaned = data_clean.pivot(
    index=["Country_ISO", "Year"],
    columns="Feature",
    values="Value"
)

data_country_cleaned.columns.name = None

# 7) Add missing Country-Year rows explicitly

all_countries = sorted(data["Country_ISO"].dropna().unique().tolist())
all_years = sorted(data["Year"].dropna().unique().tolist())

full_index = pd.MultiIndex.from_product(
    [all_countries, all_years],
    names=["Country_ISO", "Year"]
)

data_country_cleaned = data_country_cleaned.reindex(full_index)

# torno a dataframe classico
data_country_cleaned = data_country_cleaned.reset_index()

# ordine finale
data_country_cleaned = data_country_cleaned.sort_values(
    ["Country_ISO", "Year"]
).reset_index(drop=True)

print("\nFinal shape with missing Country-Year rows added:", data_country_cleaned.shape)
print(data_country_cleaned.head(20))

# 8) Optional checks

# righe completamente NaN nelle feature
feature_cols = [col for col in data_country_cleaned.columns if col not in ["Country_ISO", "Year"]]

all_nan_rows = data_country_cleaned[feature_cols].isna().all(axis=1)
print("\nNumero righe Country-Year con tutte le feature NaN:", all_nan_rows.sum())

all_nan_columns = [col for col in data_country_cleaned.columns if data_country_cleaned.isna().sum()[col] == data_country_cleaned.shape[0]]

data_country_cleaned = data_country_cleaned.drop(all_nan_columns, axis = 1)
data_country_cleaned = data_country_cleaned.set_index(["Country_ISO"])

data_country_cleaned = data_country_cleaned.rename(
    columns=lambda c: f"{c} - ITU" if c != "Year" else c
)

data_country_cleaned.to_csv(f"{root}/Data/ITU/itu_wide.csv")
