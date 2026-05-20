# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:43:53 2026

@author: WKS
"""

#%% 0) Utilities

import os
import numpy as np
import pandas as pd

def align_dataset(df, countries, index_full, prefix=None):
    """
    Porta il dataset su indice MultiIndex (Country_ISO, Year),
    lo reindicizza sulla griglia completa e opzionalmente
    aggiunge un prefisso alle colonne.
    """
    tmp = df.copy()

    # Country_ISO è nell'indice, Year è colonna
    tmp = tmp.reset_index()

    # tengo solo i 193 paesi target
    tmp = tmp[tmp["Country_ISO"].isin(countries)]

    # imposto indice doppio
    tmp = tmp.set_index(["Country_ISO", "Year"])

    # reindex sulla griglia completa
    tmp = tmp.reindex(index_full)

    # eventuale prefisso alle colonne per evitare collisioni
    if prefix is not None:
        tmp = tmp.add_prefix(f"{prefix}_")

    return tmp


root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Uploading data

first_year = 2008
last_year = 2024

wdi_data = pd.read_csv(f"{root}/Data/WDI/wdi_wide.csv")
wdi_data = wdi_data.set_index(["Country_ISO"])
wdi_data = wdi_data[(wdi_data["Year"] >= first_year) & (wdi_data["Year"] <= last_year)]

oecd_data = pd.read_csv(f"{root}/Data/OECD/oecd_wide.csv")
oecd_data = oecd_data.set_index(["Country_ISO"])
oecd_data = oecd_data[(oecd_data["Year"] >= first_year) & (oecd_data["Year"] <= last_year)]

itu_data = pd.read_csv(f"{root}/Data/ITU/itu_wide.csv")
itu_data = itu_data.set_index(["Country_ISO"])
itu_data = itu_data[(itu_data["Year"] >= first_year) & (itu_data["Year"] <= last_year)]

sdg_data = pd.read_csv(f"{root}/Data/SDG/sdg_wide.csv")
sdg_data = sdg_data.set_index(["Country_ISO"])
sdg_data = sdg_data[(sdg_data["Year"] >= first_year) & (sdg_data["Year"] <= last_year)]

del first_year, last_year

# Countries X years grid
valid_countries = sorted(sdg_data.index.unique())

all_years = sorted(set(oecd_data["Year"].dropna().unique())
                   | set(sdg_data["Year"].dropna().unique())
                   | set(wdi_data["Year"].dropna().unique())
                   | set(itu_data["Year"].dropna().unique()))

full_index = pd.MultiIndex.from_product(
    [valid_countries, all_years],
    names=["Country_ISO", "Year"]
)

# Datasets alignement
oecd_aligned = align_dataset(oecd_data, countries = valid_countries, index_full = full_index, prefix = None)
sdg_aligned  = align_dataset(sdg_data, countries = valid_countries, index_full = full_index, prefix = None)
wdi_aligned  = align_dataset(wdi_data, countries = valid_countries, index_full = full_index, prefix = None)
itu_aligned  = align_dataset(itu_data, countries = valid_countries, index_full = full_index, prefix = None)

del full_index, all_years, valid_countries, wdi_data, oecd_data, itu_data, sdg_data

# Merge
df_merged = pd.concat(
    [oecd_aligned, sdg_aligned, wdi_aligned, itu_aligned],
    axis=1
).reset_index()

del wdi_aligned, oecd_aligned, itu_aligned, sdg_aligned

df_merged = df_merged.set_index(["Country_ISO"])

# Save
df_merged.to_csv(f"{root}/Data/ALL/df_all.csv")

