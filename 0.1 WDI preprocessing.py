# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:34:19 2026

@author: WKS
"""

#%% 0) Utilities

import numpy as np
import pandas as pd

root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Data uploading

# Upload
data = pd.read_csv(f"{root}/Data/WDI/WDI_data.csv")

# Starting cleaning
data = data.drop(["Country Name", "Series Code"], axis = 1)
data = data.set_index(["Country Code"])
data = data.drop(np.nan, axis = 0)

# Unique features
unique_features = data["Series Name"].unique().tolist()
del unique_features

# Unique country codes
unique_country_codes = data.index.unique().tolist()

data_cleaned = pd.DataFrame()

# Data cleaning
for country_code in unique_country_codes:
    data_country = data.loc[country_code]
    data_country = data_country.T
    features_country = data_country.loc["Series Name"].tolist()
    features_country = ["Year"] + features_country
    data_country = data_country.reset_index(drop = False)
    data_country = data_country.drop(0, axis = 0)
    data_country.columns = features_country
    data_country.index = [country_code] * data_country.shape[0]
    data_country["Year"] = data_country["Year"].map(lambda x: str(x)[:4]).astype(int)
    data_country = data_country.replace({"..": np.nan})
    data_country = data_country.sort_values(by = "Year", ascending = False)
    data_cleaned = pd.concat([data_cleaned, data_country], axis = 0)
    
    del country_code, data_country, features_country

data_cleaned = data_cleaned.astype(float)
col_names = list(data_cleaned.columns)[1:]
col_names = [f"{name} - WDI" for name in col_names]
col_names = ["Year"] + col_names
data_cleaned.columns = col_names

del col_names, unique_country_codes

data_cleaned.index.name = "Country_ISO"
data_cleaned.to_csv(f"{root}/Data/WDI/wdi_wide.csv")
