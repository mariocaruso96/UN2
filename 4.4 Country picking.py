# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:23:12 2026

@author: WKS
"""

#%% 0) Librerie

import pandas as pd
import numpy as np

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


def sample_countries_by_egdi_quartile(
    df,
    country_col="ISO3",
    egdi_col="EGDI",
    n_per_quartile=5,
    preselected_countries=None,
    random_state=42
):
    """
    Divide i paesi in quartili sulla base dell'EGDI ordinato e seleziona
    n_per_quartile paesi per ciascun quartile.

    I paesi già disponibili vengono mantenuti se cadono nel rispettivo quartile.
    Se in un quartile ce ne sono meno di n_per_quartile, il codice campiona
    casualmente altri paesi dallo stesso quartile per completare il gruppo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente almeno codice paese ed EGDI.
    country_col : str
        Nome della colonna con il codice paese, ad esempio 'ISO3'.
    egdi_col : str
        Nome della colonna con il valore EGDI.
    n_per_quartile : int
        Numero finale di paesi da selezionare per ogni quartile.
    preselected_countries : list
        Lista dei paesi già disponibili per altre analisi.
    random_state : int
        Seed per rendere il campionamento casuale riproducibile.

    Returns
    -------
    selected_df : pd.DataFrame
        DataFrame con i paesi selezionati e il relativo quartile.
    full_df : pd.DataFrame
        DataFrame originale con colonna aggiuntiva 'EGDI_quartile'.
    """

    if preselected_countries is None:
        preselected_countries = [
            "AFG", "ALB", "BRA", "CHN", "CIV", "COG", "DZA",
            "ESP", "GBR", "ITA", "JPN", "MEX", "TUR"
        ]

    rng = np.random.default_rng(random_state)

    data = df.copy()

    # Rimuove valori mancanti
    data = data.dropna(subset=[egdi_col])

    # Ordina per EGDI crescente
    data = data.sort_values(egdi_col).reset_index(drop=True)

    # Crea quartili sulla distribuzione ordinata
    data["EGDI_quartile"] = pd.qcut(
        data[egdi_col],
        q=4,
        labels=["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]
    )

    selected_rows = []

    for quartile in data["EGDI_quartile"].cat.categories:

        subset_q = data[data["EGDI_quartile"] == quartile].copy()

        # Paesi già pronti presenti nel quartile
        already_available = subset_q[
            subset_q[country_col].isin(preselected_countries)
        ].copy()

        # Se ce ne sono più di 5, ne prende 5 casualmente
        if len(already_available) >= n_per_quartile:
            chosen = already_available.sample(
                n=n_per_quartile,
                random_state=random_state
            )

        else:
            n_missing = n_per_quartile - len(already_available)

            candidates = subset_q[
                ~subset_q[country_col].isin(already_available[country_col])
            ].copy()

            additional = candidates.sample(
                n=n_missing,
                random_state=random_state
            )

            chosen = pd.concat([already_available, additional], axis=0)

        selected_rows.append(chosen)

    selected_df = pd.concat(selected_rows, axis=0)

    selected_df = selected_df.sort_values(
        ["EGDI_quartile", egdi_col]
    ).reset_index(drop=True)

    return selected_df, data

#%% 1) Estrazione

# =========================
# Load data
# =========================
root = "C:/Users/WKS/Desktop/UNIBA/UN"

data = pd.read_csv(f"{root}/Data/EGDIs/egdi_vals.csv")

data = data.drop(["EGDI_2008"], axis=1)
data = data.set_index("Countries")

data.index = data.index.map(country_to_iso3)
data = data.reset_index()

year = 2022

selected_countries, df_with_quartiles = sample_countries_by_egdi_quartile(
    df=data,
    country_col="Countries",
    egdi_col=f"EGDI_{year}",
    n_per_quartile=5,
    random_state=42
)

print(selected_countries[["Countries", f"EGDI_{year}", "EGDI_quartile"]])

# Da runnare ancora
# 
