# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:23:23 2026

@author: WKS
"""

#%% 0) Utilities

import os
import numpy as np
import pandas as pd

country_to_iso3 = {
'Afghanistan':'AFG','Albania':'ALB','Algeria':'DZA','Andorra':'AND','Angola':'AGO',
'Antigua and Barbuda':'ATG','Argentina':'ARG','Armenia':'ARM','Australia':'AUS',
'Austria':'AUT','Azerbaijan':'AZE','Bahamas':'BHS','Bahrain':'BHR','Bangladesh':'BGD',
'Barbados':'BRB','Belarus':'BLR','Belgium':'BEL','Belize':'BLZ','Benin':'BEN',
'Bhutan':'BTN','Bolivia (Plurinational State of)':'BOL','Bosnia and Herzegovina':'BIH',
'Botswana':'BWA','Brazil':'BRA','Brunei Darussalam':'BRN','Bulgaria':'BGR',
'Burkina Faso':'BFA','Burundi':'BDI','Cabo Verde':'CPV','Cambodia':'KHM',
'Cameroon':'CMR','Canada':'CAN','Central African Republic':'CAF','Chad':'TCD',
'Chile':'CHL','China':'CHN','Colombia':'COL','Comoros':'COM','Congo':'COG',
'Costa Rica':'CRI','Croatia':'HRV','Cuba':'CUB','Cyprus':'CYP','Czechia':'CZE',
"Côte d'Ivoire":'CIV',"Democratic People's Republic of Korea":'PRK',
'Democratic Republic of the Congo':'COD','Denmark':'DNK','Djibouti':'DJI',
'Dominica':'DMA','Dominican Republic':'DOM','Ecuador':'ECU','Egypt':'EGY',
'El Salvador':'SLV','Equatorial Guinea':'GNQ','Eritrea':'ERI','Estonia':'EST',
'Eswatini':'SWZ','Ethiopia':'ETH','Fiji':'FJI','Finland':'FIN','France':'FRA',
'Gabon':'GAB','Gambia':'GMB','Georgia':'GEO','Germany':'DEU','Ghana':'GHA',
'Greece':'GRC','Grenada':'GRD','Guatemala':'GTM','Guinea':'GIN','Guinea-Bissau':'GNB',
'Guyana':'GUY','Haiti':'HTI','Honduras':'HND','Hungary':'HUN','Iceland':'ISL',
'India':'IND','Indonesia':'IDN','Iran (Islamic Republic of)':'IRN','Iraq':'IRQ',
'Ireland':'IRL','Israel':'ISR','Italy':'ITA','Jamaica':'JAM','Japan':'JPN',
'Jordan':'JOR','Kazakhstan':'KAZ','Kenya':'KEN','Kiribati':'KIR','Kuwait':'KWT',
'Kyrgyzstan':'KGZ',"Lao People's Democratic Republic":'LAO','Latvia':'LVA',
'Lebanon':'LBN','Lesotho':'LSO','Liberia':'LBR','Libya':'LBY','Liechtenstein':'LIE',
'Lithuania':'LTU','Luxembourg':'LUX','Madagascar':'MDG','Malawi':'MWI',
'Malaysia':'MYS','Maldives':'MDV','Mali':'MLI','Malta':'MLT','Marshall Islands':'MHL',
'Mauritania':'MRT','Mauritius':'MUS','Mexico':'MEX',
'Micronesia (Federated States of)':'FSM','Monaco':'MCO','Mongolia':'MNG',
'Montenegro':'MNE','Morocco':'MAR','Mozambique':'MOZ','Myanmar':'MMR',
'Namibia':'NAM','Nauru':'NRU','Nepal':'NPL','Netherlands (Kingdom of the)':'NLD',
'New Zealand':'NZL','Nicaragua':'NIC','Niger':'NER','Nigeria':'NGA',
'North Macedonia':'MKD','Norway':'NOR','Oman':'OMN','Pakistan':'PAK',
'Palau':'PLW','Panama':'PAN','Papua New Guinea':'PNG','Paraguay':'PRY',
'Peru':'PER','Philippines':'PHL','Poland':'POL','Portugal':'PRT','Qatar':'QAT',
'Republic of Korea':'KOR','Republic of Moldova':'MDA','Romania':'ROU',
'Russian Federation':'RUS','Rwanda':'RWA','Saint Kitts and Nevis':'KNA',
'Saint Lucia':'LCA','Saint Vincent and the Grenadines':'VCT','Samoa':'WSM',
'San Marino':'SMR','Sao Tome and Principe':'STP','Saudi Arabia':'SAU',
'Senegal':'SEN','Serbia':'SRB','Seychelles':'SYC','Sierra Leone':'SLE',
'Singapore':'SGP','Slovakia':'SVK','Slovenia':'SVN','Solomon Islands':'SLB',
'Somalia':'SOM','South Africa':'ZAF','South Sudan':'SSD','Spain':'ESP',
'Sri Lanka':'LKA','Sudan':'SDN','Suriname':'SUR','Sweden':'SWE',
'Switzerland':'CHE','Syrian Arab Republic':'SYR','Tajikistan':'TJK',
'Thailand':'THA','Timor-Leste':'TLS','Togo':'TGO','Tonga':'TON',
'Trinidad and Tobago':'TTO','Tunisia':'TUN','Turkmenistan':'TKM',
'Tuvalu':'TUV','Türkiye':'TUR','Uganda':'UGA','Ukraine':'UKR',
'United Arab Emirates':'ARE',
'United Kingdom of Great Britain and Northern Ireland':'GBR',
'United Republic of Tanzania':'TZA','United States of America':'USA',
'Uruguay':'URY','Uzbekistan':'UZB','Vanuatu':'VUT',
'Venezuela (Bolivarian Republic of)':'VEN','Viet Nam':'VNM',
'Yemen':'YEM','Zambia':'ZMB','Zimbabwe':'ZWE'
}

optional_columns = [
    'Sex', 'Policy instruments',
    'Education level', 'Location', 'Reporting Type',
    'Observation Status', 'Type of occupation', 'Units',
    'Age', 'Disability status', 'Activity', 'Nature', 'Quantile',
    'Population Group'
]

def build_indicator_name(row, cols_optional = optional_columns):

    # parte principale
    parts = [
        f"{row['SeriesCode']} - {row['SeriesDescription']} - Goal {row['Goal']}"
    ]

    for col in cols_optional:
        val = row[col]
        if pd.notna(val) and str(val).strip() != "":
            parts.append(f"{col}: {val}")

    return " - ".join(parts)


root = "C:/Users/WKS/Desktop/UNIBA/UN"

#%% 1) Data uploading and first cleaning

indicators = pd.read_csv(f"{root}/Data/SDG/UN_indicators.txt")
indicators = indicators["Indicators"].tolist()

df_all = pd.DataFrame()

for sdg_file in os.listdir(f"{root}/Data/SDG/"):
    
    print(f"Inspecting folder: {sdg_file} ...")
    
    if sdg_file == "UN_indicators.txt":
        print(f"{sdg_file} not useful. Skip.")
        continue
    
    for goal_file in os.listdir(f"{root}/Data/SDG/{sdg_file}"):
        
        print(f"Inspecting file: {goal_file} ...")
        df = pd.read_excel(f"{root}/Data/SDG/{sdg_file}/{goal_file}")
        df_filtered = df[df["SeriesCode"].isin(indicators)]
        df_all = pd.concat([df_all, df_filtered], axis = 0)
        
        del df, df_filtered
        del goal_file
    
    print("\n")
    
    del sdg_file

del indicators

#%%

# All nan columns drop
all_nans_columns = df_all.isna().sum() == df_all.shape[0]
all_nans_columns = all_nans_columns[all_nans_columns == True]
all_nans_columns = all_nans_columns.index.tolist()

df_all = df_all.drop(all_nans_columns, axis = 1)

del all_nans_columns

# Other non useful columns
other_columns = ["Target",
                 "Indicator",
                 "GeoAreaCode",
                 "Time_Detail",
                 "BasePeriod",
                 "Source",
                 "FootNote"]

df_all = df_all.drop(other_columns, axis = 1)

del other_columns


# Creating full indicator name
df_all["Indicator_full"] = df_all.apply(build_indicator_name, axis = 1)

# Deleting optional columns
optional_columns = optional_columns + ["Goal", "SeriesCode", "SeriesDescription"]
df_all = df_all.drop(optional_columns, axis = 1)

del optional_columns

df_all.columns = ["Country", "Year", "Value", "Indicator"]
df_all = df_all[["Year", "Country", "Indicator", "Value"]]

# Long format to wide format

# pivot base
df_wide = df_all.pivot_table(
    index=["Year", "Country"],
    columns="Indicator",
    values="Value",
    aggfunc="first"
)

# all possible combinations
full_index = pd.MultiIndex.from_product(
    [sorted(df_all["Year"].dropna().unique()),
     sorted(df_all["Country"].dropna().unique())],
    names=["Year", "Country"]
)

# Reindex to obtain all possible couples
df_wide = df_wide.reindex(full_index).reset_index()
df_wide.columns.name = None

del full_index, df_all

# Last adjustements
df_wide["ISO3"] = df_wide["Country"].map(country_to_iso3)
df_wide = df_wide.set_index("ISO3")
df_wide.index.name = "Country_ISO"
df_wide = df_wide.drop(["Country"], axis = 1)
df_wide = df_wide.astype(float)

del country_to_iso3

df_wide = df_wide.rename(
    columns=lambda c: f"{c} - SDG" if c != "Year" else c
)


# Save
df_wide.to_csv(f"{root}/Data/SDG/sdg_wide.csv")
