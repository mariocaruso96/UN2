# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:57:27 2026

@author: WKS
"""

#%% 0) Utilities

import os
import numpy as np
import pandas as pd

def build_indicator(row):

    parts = []

    # parte principale
    base = str(row["Indicator"])

    if pd.notna(row["Breakdown"]):
        base += f" - {row['Breakdown']}"

    if pd.notna(row["Unit of measure"]):
        base += f" ({row['Unit of measure']})"

    parts.append(base)

    # campi opzionali
    optional_cols = [
        "Industry",
        "Field",
        "Level of education",
        "Flow",
        "Scope"
    ]

    for col in optional_cols:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))

    return " - ".join(parts)

root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Data uploading

data_list = []

for folder_name in os.listdir(f"{root}/Data/OECD/"):
    folder_path = f"{root}/Data/OECD/{folder_name}"

    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        file_path = f"{folder_path}/{file_name}"

        if not file_name.lower().endswith(".csv"):
            continue

        df = pd.read_csv(file_path)
        data_list.append(df)

data = pd.concat(data_list, axis=0, ignore_index=True)

# 2) Basic cleaning

# metto ISO come colonna standard Country_ISO
data = data.rename(columns={"ISO": "Country_ISO"})

# tolgo colonne inutili solo se esistono
data = data.drop(columns=["Country", "Gender"], errors="ignore")

# costruzione colonna indicator testuale
data["Indicator"] = data.apply(build_indicator, axis=1)

# tengo solo le colonne utili
data = data[["Country_ISO", "Year", "Indicator", "Value"]].copy()

# conversioni
data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

# tolgo righe senza paese o anno
data = data.dropna(subset=["Country_ISO", "Year"])

# Year intero
data["Year"] = data["Year"].astype(int)

# 3) Pivot wide

df_wide = data.pivot_table(
    index=["Country_ISO", "Year"],
    columns="Indicator",
    values="Value",
    aggfunc="first"
)

df_wide.columns.name = None

# 4) Add missing years 2000-2025 for each country

all_countries = sorted(data["Country_ISO"].dropna().unique().tolist())
all_years = list(range(2000, 2026))

full_index = pd.MultiIndex.from_product(
    [all_countries, all_years],
    names=["Country_ISO", "Year"]
)

df_wide = df_wide.reindex(full_index)

# 5) Final format:
# Country_ISO = index
# Year = normal column

df_wide = df_wide.reset_index(level="Year")
df_wide = df_wide.sort_index()

df_wide = df_wide.rename(
    columns=lambda c: f"{c} - OECD" if c != "Year" else c
)

df_wide.to_csv(f"{root}/Data/OECD/oecd_wide.csv")
