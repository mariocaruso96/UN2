# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:16:19 2026

@author: WKS
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:48:53 2026

@author: WKS
"""

#%% 0) Librerie e funzioni

import numpy as np
import pandas as pd
import json
import re
import pickle
import hashlib

from dataclasses import dataclass
from itertools import combinations
from typing import List, Any, Tuple, Dict

# from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau

from pathlib import Path

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

@dataclass
class DataPaths:
    """
    Tutti i path in un posto solo.
    L'utente modifica qui e basta.
    """
    root: str

    # EGDI
    egdi_vals_csv: str
    egdi_rank_csv: str

    # Data post-processed
    data_post_processed_csv_template: str

    # ML results (country RMSE and SHAP)
    data_pkl_template: str

    # Top-features per country
    top_features_csv_template: str
    
def load_egdi_values(paths: DataPaths, isocodes_dict: dict, year: int) -> pd.Series:
    """
    Carica egdi_vals.csv e ritorna Series EGDI_{year} indicizzata per nome paese.
    Mantiene il replace 0 -> 0.01 come nel tuo script.
    """
    df = pd.read_csv(paths.egdi_vals_csv)
    df = df.set_index(["Countries"])
    df.index = df.index.map(isocodes_dict)
    df.index.name = "Country_ISO"
    
    y = df[f"EGDI_{year}"].copy()
    y = y.replace([0], [0.01])
    y.name = f"EGDI_{year}"
    y = y.sort_index()
    return y


def load_true_ranking(paths: DataPaths, isocodes_dict: dict, year: int) -> pd.Series:
    """
    Carica egdi_rank.csv e ritorna ranking "True" per EGDI_Rank_{year}.
    """
    df = pd.read_csv(paths.egdi_rank_csv)
    df = df.set_index(["Countries"])
    df.index = df.index.map(isocodes_dict)
    df.index.name = "Country_ISO"

    rank = df[f"EGDI_Rank_{year}"].copy()
    rank = rank.sort_values()
    rank.name = "True"
    rank = rank.sort_index()
    return rank


def load_data_post_processed(paths: DataPaths, year: int) -> pd.DataFrame:
    """
    Carica data post-processed. Index = Countries (nome paese).
    """
    fp = paths.data_post_processed_csv_template.format(year=year)
    data = pd.read_csv(fp)
    data = data.set_index(["Country_ISO"])
    data = data.sort_index()
    return data


def load_top_features(paths: DataPaths, year: int, country: str) -> pd.DataFrame:
    """
    Carica top-features per country, mette index=Features e droppa colonna 'Features'.
    """
    fp = paths.top_features_csv_template.format(year=year, country=country)
    df = pd.read_csv(fp)
    feats = df["Features"]
    df.index = feats
    df = df.drop(["Features"], axis=1)
    return df


def load_rmse_error(paths: DataPaths, year: int, country: str) -> float:
    """
    Carica pkl e ritorna RMSE del country
    """
    fp = paths.data_pkl_template.format(year=year)
    with open(fp, "rb") as f:
        obj = pickle.load(f)
    egdi_true_values = obj[12]
    model_predictions = obj[14]
    
    egdi_country = egdi_true_values.loc[country]
    model_predictions_country = model_predictions.loc[country]
    errors_country = model_predictions_country - egdi_country
    errors_country = errors_country.abs()
    return errors_country.median()
    

def load_shap_pickle(paths: DataPaths, year: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Carica pkl e ritorna (shap_global_model, test_idx_global) come nel tuo script.
    """
    fp = paths.data_pkl_template.format(year=year)
    with open(fp, "rb") as f:
        obj = pickle.load(f)
    shap_global_model = obj[1] # 0 per RF, # 1 per XGB
    test_idx_global = obj[2]
    return shap_global_model, test_idx_global


def load_json_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def build_paths_from_config(cfg: Dict[str, Any]) -> DataPaths:
    root = Path(cfg["root"])

    return DataPaths(
        root=str(root),
        egdi_vals_csv=str(root / cfg["egdi_vals_csv"]),
        egdi_rank_csv=str(root / cfg["egdi_rank_csv"]),
        data_post_processed_csv_template=str(root / cfg["data_post_processed_csv_template"]),
        data_pkl_template=str(root / cfg["data_pkl_template"]),
        top_features_csv_template=str(root / cfg["top_features_csv_template"]),
    )


def build_shap_dataframe(
    data_df: pd.DataFrame,
    shap_global_rf: np.ndarray,
    test_idx_global: List[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replica identica:
      index_list_concatenated = np.concatenate(test_idx_global)
      data_shap = data_df.iloc[index_list_concatenated]
      shap_df = DataFrame(np.concatenate(shap_global_rf)) con stesso index/columns di data_shap
    """
    idx = np.concatenate(test_idx_global)
    data_shap = data_df.iloc[idx]
    shap_values = pd.DataFrame(np.concatenate(shap_global_rf), index=data_shap.index, columns=data_shap.columns)
    return data_shap, shap_values



def infer_improvement_signs_for_country(
    country: str,
    top_features_df: pd.DataFrame,
    data_shap: pd.DataFrame,
    shap_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.Series:
    """
    Determina il segno di miglioramento per ciascuna feature di un Paese.

    Logica:
      - Applica MinMax scaling ai valori della feature su data_shap
      - Identifica i valori >= threshold
      - Calcola la media dei valori SHAP nei punti selezionati
      - Se media SHAP > 0 => '+', altrimenti '-'
    """

    signs = []
    scaler = MinMaxScaler()

    for feat in top_features_df.index:
        # Valori della feature
        feature_values = data_shap[feat].to_frame()

        # MinMax scaling
        feature_values_norm = pd.DataFrame(
            scaler.fit_transform(feature_values),
            index=feature_values.index,
            columns=feature_values.columns
        )

        # SHAP associati
        shap_values_feat = shap_df[feat]

        # Maschera valori "alti"
        high_mask = feature_values_norm[feat] >= threshold

        # Media SHAP sui valori alti
        mean_high_shap = shap_values_feat[high_mask].mean()

        # Segno
        signs.append("+" if mean_high_shap > 0 else "-")

    return pd.Series(signs, index=top_features_df.index, name=f"{country}_improve_sign")


def divide_and_scale_data(
        df_data: pd.DataFrame,
        y_egdi: pd.Series,
        country: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
    """
    Funzione che prende in input i dati e li divide tra TUTTI
    i Paesi meno quello che vogliamo considerare. Dopodichè
    scala con MinMaxScaler.
    """
    
    X = df_data.copy()
    y = y_egdi.copy()
    
    country_X = X.loc[country].to_frame().T
    country_y = y.loc[country]
    
    # Rimuovo il Paese dal dataframe
    X = X.drop(country, axis = 0)
    X = X.sort_index()
    y = y.drop(country, axis = 0)
    y = y.sort_index()
    
    # Scaling
    scaler_X = MinMaxScaler()
    
    # Dataset di Training
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), index = X.index, columns = X.columns)
    
    # Paese
    country_X_scaled = pd.DataFrame(scaler_X.transform(country_X), index = [country], columns = country_X.columns)
    
    return X_scaled, y, country_X_scaled, country_y


def compute_feature_caps_from_signs(
    X_scaled_df: pd.DataFrame,
    signs_series: pd.Series,
    q_hi: float = 0.95,
    q_lo: float = 0.05,
    bool_tol: float = 1e-12,
) -> pd.DataFrame:
    """
    Calcola i cap "migliorativi" per ciascuna feature in `signs`, coerentemente col segno:
      - segno '+'  => cap = quantile alto (q_hi) per continue; cap = 1 per booleane
      - segno '-'  => cap = quantile basso (q_lo) per continue; cap = 0 per booleane

    Parametri
    ---------
    X_scaled : pd.DataFrame
        Dataset Data già scalato (es. MinMax), con colonne = features.
    signs : pd.Series
        Serie con index = nomi feature, values in {'+','-'} (spazi ignorati).
    q_hi, q_lo : float
        Quantili da usare per le continue.
    bool_tol : float
        Tolleranza numerica per riconoscere booleane (0/1) in presenza di floating.

    Ritorna
    -------
    pd.DataFrame con index=feature e colonne:
        - sign: '+' o '-'
        - feature_type: 'boolean' o 'continuous'
        - cap: valore cap nello spazio scalato
        - q_used: quantile usato (NaN per booleane)
        - unique_vals: numero di valori unici osservati (utile debug)
    """
    if not isinstance(X_scaled_df, pd.DataFrame):
        raise TypeError("X_scaled deve essere un pd.DataFrame.")
    if not isinstance(signs_series, pd.Series):
        raise TypeError("signs deve essere un pd.Series (index=features, values='+'/'-').")
    if signs_series.index.duplicated().any():
        raise ValueError("signs contiene indici duplicati (feature ripetute).")
    if not (0.0 < q_lo < q_hi < 1.0):
        raise ValueError("Richiesto 0 < q_lo < q_hi < 1.")

    # Normalizza signs
    signs_norm = signs_series.astype(str).str.strip()
    invalid = ~signs_norm.isin(["+", "-"])
    if invalid.any():
        bad = signs_norm[invalid]
        raise ValueError(f"signs contiene valori non validi (attesi '+' o '-'): {bad.to_dict()}")

    # Controllo features presenti
    missing = [f for f in signs_norm.index if f not in X_scaled_df.columns]
    if missing:
        raise KeyError(f"Le seguenti feature in signs non sono presenti in X_scaled: {missing}")

    rows = []
    for feat, sgn in signs_norm.items():
        col = X_scaled_df[feat].dropna()
        if col.empty:
            raise ValueError(f"La colonna '{feat}' in X_scaled è vuota (tutti NaN).")

        # Riconoscimento booleano: valori ~0 o ~1 (o un sottoinsieme)
        vals = col.to_numpy(dtype=float)
        is_01 = np.isclose(vals, 0.0, atol=bool_tol, rtol=0.0) | np.isclose(vals, 1.0, atol=bool_tol, rtol=0.0)
        is_boolean = bool(is_01.all())

        uniq = int(pd.unique(np.round(vals, 12)).size)  # solo diagnostica

        if is_boolean:
            cap = 1.0 if sgn == "+" else 0.0
            q_used = np.nan
            ftype = "boolean"
        else:
            q_used = q_hi if sgn == "+" else q_lo
            cap = float(col.quantile(q_used))
            # safety: se per quantili strani esce fuori [0,1], clippa
            cap = float(np.clip(cap, 0.0, 1.0))
            ftype = "continuous"

        rows.append(
            {
                "feature": feat,
                "sign": sgn,
                "feature_type": ftype,
                "cap": cap,
                "q_used": q_used,
                "unique_vals": uniq,
            }
        )

    out = pd.DataFrame(rows).set_index("feature")
    return out


def modify_feature(
        country_scaled_df: pd.DataFrame,
        country_name: str,
        caps_df: pd.DataFrame,
        feature_name: str,
        change: float
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Funzione che prende in input il df di un Paese la feature e i caps e modifica il df
    del Paese modificando la sua feature in maniera incrementale (se continua) o flippandola
    a seconda del "segno" che controlla il concetto di "miglioramento".
    """

    # Creo una copia del Paese di cui modifico una feature per volta
    country_copy = country_scaled_df.copy()
    
    feature_type = caps_df["feature_type"][feature_name]
    sign = caps_df["sign"][feature_name]
    z_cap = caps_df["cap"].loc[feature_name]
    z_old = country_copy.loc[country_name][feature_name]
    
    # Se la feature è continua, modifico con la formula incrementale
    if feature_type == "continuous":
        delta_z = change * (z_cap - z_old)
        z_new = z_old + delta_z
        z_new = np.clip(z_new, 0, 1)
        country_copy[feature_name] = z_new
        cost = abs(delta_z)
                    
    # Se la feature è booleana, flippo a seconda di sign
    else:
        z_new = 1.0 if sign == "+" else 0.0
        cost = float(1)
        delta_z = z_new - z_old
        country_copy[feature_name] = z_new
    
    return country_copy, cost, float(delta_z), z_old, z_new


def find_candidates_features(
        how_many: int,
        country_name: str,
        country_scaled_df: pd.DataFrame,
        country_shap_series: pd.Series,
        caps_df: pd.DataFrame,
        starting_egdi_pred: float,
        change: float,
        epsilon_scoring_value: float,
        trained_model_base,
        random_state: float,
        verbose: bool = True,
        mode: str = "Probabilities", # "Casual" in alternativa se non vogliamo considerare le probabilità calcolate
) -> Tuple[list, pd.DataFrame]:
    """
    Seleziona un sottoinsieme di feature candidate (max = how_many) da utilizzare
    nella costruzione della policy greedy del Recommender System.

    Procedura:
    -----------
    1) Per ciascuna feature candidata (tipicamente top-SHAP per il Paese):
        - Calcola il "cap" coerente con il segno di miglioramento (+/-),
          usando quantili del training set scalato (X_scaled_df).
        - Applica una modifica adattiva (change = rho) verso il cap
          oppure effettua il flip se la feature è booleana.
        - Valuta il guadagno marginale predetto dal modello:
              ΔEGDI = f(x_modificato) - f(x_iniziale)
        - Calcola uno score proporzionale a:
              |SHAP| * (ΔEGDI / costo)

    2) Filtra le feature con score nullo (o sotto epsilon_scoring_value).

    3) Se il numero di feature con score positivo:
        - è 0 → la run fallisce (nessuna azione efficace individuata);
        - è ≤ how_many → vengono selezionate tutte;
        - è > how_many → viene effettuato un sampling senza replacement,
          con probabilità proporzionali allo score normalizzato.

    Output:
    -------
    - features_to_consider : lista delle feature candidate per la greedy.
    - scores_df : DataFrame con score e probabilità associate.

    Note metodologiche:
    -------------------
    - Il modello deve essere già addestrato (idealmente in setting LOCO).
    - epsilon_scoring_value è una soglia tecnica per evitare micro-mosse
      sotto il rumore del modello nella fase di scoring.
    - La selezione è stocastica ma controllata da random_state,
      permettendo analisi di stabilità (Jaccard, Kendall, ecc.).
    """
    
    # Sto selezionando quali features considerare per poterne selezionare K
    scores = []
    
    for feature in caps_df.index:
        
        # Modifico una feature per volta in maniera incrementale (se continua) o flippangola (se booleana)
        # in accordo al segno di signs_series.
        new_country_scaled, cost, _, _, _ = modify_feature(country_scaled_df=country_scaled_df,
                                                           country_name=country_name,
                                                           caps_df=caps_df,
                                                           feature_name=feature,
                                                           change=change)
        
            
        # Volta per volta, ricalcolo l'EGDI predetto
        egdi_pred = trained_model_base.predict(new_country_scaled)[0]
    
        # Calcolo il guadagno marginale cambiando una singola feature
        egdi_delta_marginal = egdi_pred - starting_egdi_pred
        shap_value = float(abs(country_shap_series[feature]))
        
        # Se il deltaE è positivo e maggiore di una soglia di tolleranza, calcola gli score; altrimenti aggiungi zero.
        if egdi_delta_marginal >= 0 and egdi_delta_marginal > epsilon_scoring_value:
            # Se il costo è troppo piccolo rischiamo una divisione per zero
            if cost <= 1e-12:
                scores.append(float(0))
            # Altrimenti calcola lo score come shap * deltaE/costo
            else:
                score = shap_value * np.divide(egdi_delta_marginal, cost)
                scores.append(score)
        else:
            scores.append(float(0))
            
    scores = pd.Series(scores, index = caps_df.index)
    scores.name = "Scores"     
    scores = scores.sort_values(ascending = False)
    
    # Features da considerare per creare l'ensamble di policy
    # Se gli score sono tutti nulli, la run fallisce
    if scores.sum() <= 1e-12:
        if verbose:
            print("FEATURES SAMPLING ---- FAILED.")
            print("All scores are zero. Run could not be performed.")
            features_to_consider = []
    
    # Se alcuni score NON sono nulli:
    else:
        # Trasformo gli scores in probabilità: sj/sum(sj)
        scores = scores[scores > 0].to_frame()
        scores["Probability"] = scores["Scores"] / scores["Scores"].sum()
        scores = scores.sort_values(by = "Probability", ascending = False)
        
        # Se il numero di score NON nulli è minore o uguale a K, prendi tutte le features
        if len(scores) <= how_many:
            features_to_consider = list(scores.index)
            if verbose:
                print("FEATURE SAMPLING ---- SUCCESS.")
                print(f"# Non-zero scores: {len(scores)} <= {how_many}.")
                print(f"Features considered: {len(scores)}.")
        
        # Altrimenti, fai sampling o sulla base della probabilità (mode = "Probabilities") o a caso (mode = "Casual")
        # SENZA possibilità di reinserimento: replace = False
        else:
            if mode == "Probabilities":
                features_to_consider = scores.sample(n = how_many, weights = scores["Probability"], random_state = random_state, replace = False).index.tolist()
            if mode == "Casual":
                features_to_consider = scores.sample(n = how_many, random_state = random_state, replace = False).index.tolist()
            if verbose:
                print("FEATURE SAMPLING ---- SUCCESS.")
                print(f"# Non-zero scores: {len(scores)} > {how_many}.")
                print(f"Features considered: {how_many}.")
    
    return features_to_consider, scores


def run_greedy_policy(
    country_scaled_df: pd.DataFrame,
    candidates_list: List[str],
    caps_df: pd.DataFrame,
    trained_base_model: Any,
    country_name: str,
    starting_egdi_pred: float,
    target_egdi: float,
    how_many: int,
    change: float,
    budget: float = 1.0,
    epsilon_scoring_value: float = 0.0,
) -> Tuple[List[str], pd.DataFrame, bool, float, float, pd.DataFrame]:
    """
    Esegue la strategia greedy per costruire una policy di miglioramento EGDI.

    Restituisce:
    ------------
    - policy_list: lista ordinata delle feature selezionate
    - policy_df: DataFrame con log dettagliato degli step
    - success: True se target_egdi raggiunto
    - budget_spent: budget consumato
    - final_egdi_pred: EGDI finale predetto
    - final_country_scaled_df: DataFrame 1×N finale dopo le modifiche
    """

    # Stato iniziale
    country_copy = country_scaled_df.copy()
    features_remained = list(candidates_list)

    policy_list: List[str] = []
    policy_steps = []

    E_current = float(starting_egdi_pred)
    budget_left = float(budget)

    for step in range(1, how_many + 1):

        best_feat = None
        best_roi = -np.inf
        best_cost = None
        best_country = None
        best_E = None

        # Logging
        best_delta_z = None
        best_z_old = None
        best_z_new = None
        best_delta_E = None

        # Valutazione mosse possibili
        for feature_name in list(features_remained):

            new_country_scaled_df, cost, delta_z, z_old, z_new = modify_feature(
                country_scaled_df=country_copy,
                country_name=country_name,
                caps_df=caps_df,
                feature_name=feature_name,
                change=change
            )

            cost = float(cost)

            if cost <= 1e-12:
                continue

            if cost > budget_left:
                continue

            E_new = float(trained_base_model.predict(new_country_scaled_df)[0])
            delta_E = E_new - E_current

            if delta_E <= float(epsilon_scoring_value):
                continue

            roi = delta_E / cost

            if roi > best_roi:
                best_roi = roi
                best_feat = feature_name
                best_cost = cost
                best_country = new_country_scaled_df
                best_E = E_new

                best_delta_z = float(delta_z)
                best_z_old = float(z_old)
                best_z_new = float(z_new)
                best_delta_E = float(delta_E)

        # Nessuna mossa valida
        if best_feat is None:
            break

        # Logging step
        policy_steps.append(
            {
                "step": step,
                "feature": best_feat,
                "z_old": best_z_old,
                "z_new": best_z_new,
                "delta_z": best_delta_z,
                "cost": best_cost,
                "E_before": E_current,
                "E_after": best_E,
                "delta_E": best_delta_E,
                "roi": best_roi,
                "budget_left_before": budget_left,
                "budget_left_after": budget_left - best_cost,
            }
        )

        # Aggiornamento stato
        policy_list.append(best_feat)
        country_copy = best_country
        E_current = float(best_E)
        budget_left -= float(best_cost)
        features_remained.remove(best_feat)

        # Stop se raggiunto target
        if E_current >= float(target_egdi):
            break

    # Costruzione DataFrame policy
    policy_df = pd.DataFrame(policy_steps)
    if not policy_df.empty:
        policy_df = policy_df.set_index("step")

    success = bool(E_current >= float(target_egdi))
    budget_spent = float(budget - budget_left)
    final_egdi_pred = float(E_current)
    final_country_scaled_df = country_copy

    return (
        policy_list,
        policy_df,
        success,
        budget_spent,
        final_egdi_pred,
        final_country_scaled_df,
    )





def jaccard_similarity(l1: List[Any], l2: List[Any]) -> float:
    s1, s2 = set(l1), set(l2)
    if len(s1 | s2) == 0:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def kendall_on_common(l1: List[Any], l2: List[Any]) -> float:
    """
    Kendall tau calcolato SOLO sugli elementi in comune, preservando l'ordine
    della prima lista come riferimento.
    Ritorna:
      - 1.0 se ordine identico sugli elementi comuni
      - np.nan se <2 elementi in comune
    """
    pos1 = {v: i for i, v in enumerate(l1)}
    pos2 = {v: i for i, v in enumerate(l2)}

    common = [v for v in l1 if v in pos2]  # ordine di l1
    if len(common) < 2:
        return np.nan

    r1 = [pos1[v] for v in common]
    r2 = [pos2[v] for v in common]

    tau = kendalltau(r1, r2).statistic
    return float(tau)


@dataclass
class StabilityResult:
    pair: Tuple[str, str]
    jaccard: float
    kendall_tau: float  # can be nan if not enough common items
    exact_match: int
    overlap_size: int
    union_size: int


def compute_stability(d: Dict[str, List[Any]]) -> List[StabilityResult]:
    keys = list(d.keys())
    results: List[StabilityResult] = []

    for k1, k2 in combinations(keys, 2):
        l1, l2 = d[k1], d[k2]
        s1, s2 = set(l1), set(l2)

        jac = jaccard_similarity(l1, l2)
        tau = kendall_on_common(l1, l2)
        exact = int(l1 == l2)

        results.append(
            StabilityResult(
                pair=(k1, k2),
                jaccard=jac,
                kendall_tau=tau,
                exact_match=exact,
                overlap_size=len(s1 & s2),
                union_size=len(s1 | s2),
            )
        )
    return results


def summarize(results: List[StabilityResult]) -> dict:
    j = np.array([r.jaccard for r in results], dtype=float)
    t = np.array([r.kendall_tau for r in results], dtype=float)  # contains nan
    e = np.array([r.exact_match for r in results], dtype=float)

    summary = {
        "n_pairs": len(results),

        # set stability
        "jaccard_mean": float(np.mean(j)),
        "jaccard_median": float(np.median(j)),
        "jaccard_min": float(np.min(j)),
        "jaccard_max": float(np.max(j)),

        # order stability (ignoring nan)
        "kendall_mean": float(np.nanmean(t)) if np.any(~np.isnan(t)) else np.nan,
        "kendall_median": float(np.nanmedian(t)) if np.any(~np.isnan(t)) else np.nan,
        "kendall_min": float(np.nanmin(t)) if np.any(~np.isnan(t)) else np.nan,
        "kendall_max": float(np.nanmax(t)) if np.any(~np.isnan(t)) else np.nan,

        # strict identity
        "exact_match_rate": float(np.mean(e)),
    }
    return summary


# -------- Salvataggio --------


def _safe_folder_name(x: Any) -> str:
    s = str(x).strip().replace(" ", "")
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "empty"


def _jsonify(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(x) for x in obj]
    return str(obj)


def _params_hash(parameters: Dict[str, Any], n_chars: int = 10) -> str:
    s = json.dumps(_jsonify(parameters), sort_keys=True, ensure_ascii=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n_chars]


def _save_df(df: pd.DataFrame, path_no_ext: Path) -> None:
    try:
        df.to_parquet(path_no_ext.with_suffix(".parquet"), index=True)
    except Exception:
        df.to_csv(path_no_ext.with_suffix(".csv"), index=True)


def save_rs_outputs(
    run_base_path: str,
    parameters: Dict[str, Any],
    jaccard_kendall_summary: Dict[str, Any],
    results_df_total: pd.DataFrame,
    policy_df_total: pd.DataFrame,
    scores_total: Dict[str, Any],
    country_scaled_final_df: pd.DataFrame,
) -> Path:
    """
    Saves under:
    {run_base_path}/{year}/{country}/run__{hash}/

    If run__{hash} already exists → it is reused and overwritten.
    """

    base_dir = Path(run_base_path)

    year = _safe_folder_name(parameters["year"])
    country = _safe_folder_name(parameters["country"])

    target_dir = base_dir / year / country
    target_dir.mkdir(parents=True, exist_ok=True)

    run_hash = _params_hash(parameters, n_chars=10)
    run_dir = target_dir / f"run__{run_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)  # <-- riusa cartella

    # save parameters
    with open(run_dir / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(parameters), f, indent=2, ensure_ascii=False)

    # save dict outputs
    with open(run_dir / "jaccard_kendall_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(jaccard_kendall_summary), f, indent=2, ensure_ascii=False)

    with open(run_dir / "scores_total.json", "w", encoding="utf-8") as f:
        json.dump(_jsonify(scores_total), f, indent=2, ensure_ascii=False)

    # save dataframes
    _save_df(results_df_total, run_dir / "results_df_total")
    _save_df(policy_df_total, run_dir / "policy_df_total")
    _save_df(country_scaled_final_df, run_dir / "country_scaled_final_df")

    return run_dir

#%% 1) Uploading dei dati
                        
# --- Load configurations ---
rs_config = load_json_config(
    r"C:/Users/WKS/Desktop/UNIBA/UN/configs/rs_config.json"
)

# --- Parameters --- (mantengo le variabili come prima)
parameters = rs_config["arguments"].copy()

year = parameters["year"]
country = parameters["country"]
base_model_seed = parameters["base_model_seed"]
required_egdi_delta = parameters["required_egdi_delta"]
epsilon_scoring = parameters["epsilon_scoring"]
budget = parameters["budget"]
K = parameters["K"]
rho = parameters["rho"]
caps_low_perc = parameters["caps_low_perc"]
caps_high_perc = parameters["caps_high_perc"]
n_repetitions = parameters["n_repetitions"]

print("Parameters set.\n")

# --- Paths ---
paths_config = rs_config["paths"].copy()   
# Prossima riga è da rimuovere su recas:
paths_config["root"] = "C:/Users/WKS/Desktop/UNIBA/UN"
paths = build_paths_from_config(paths_config)
run_base_path = str(Path(paths.root) / paths_config["run_base_path"])

print("Configs set.\n")

# Dati
egdi_y = load_egdi_values(paths, isocodes_dict=country_to_iso3, year=year)
data_df = load_data_post_processed(paths, year=year)
top_features_df = load_top_features(paths, year=year, country=country)
shap_global_model, test_idx_global = load_shap_pickle(paths, year=year)
epsilon_target = load_rmse_error(paths, year=year, country=country)
data_shap, shap_df = build_shap_dataframe(data_df, shap_global_model, test_idx_global)
country_shap = shap_df.loc[country].abs().median()
country_shap.name = "Median Abs Shap"

del shap_global_model, test_idx_global, paths

# Segno feature
signs = infer_improvement_signs_for_country(country, top_features_df, data_shap, shap_df)

del data_shap, top_features_df, shap_df

# Training di un RF su 192 Paesi e predizione iniziale circa il Paese di interesse
X_scaled, y, country_scaled, country_y = divide_and_scale_data(data_df, egdi_y, country)

# Modello base del RS:
base_model = xgb.XGBRegressor(n_estimators=1500,
                        max_depth=3,
                        learning_rate=0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=3,
                        # gamma=0.2,
                        n_jobs=1, random_state=base_model_seed)

del base_model_seed

# Addestramento del RF
base_model.fit(X_scaled, y)

# Baseline EGDI: predizione iniziale del modello
egdi_pred_start = base_model.predict(country_scaled)[0]


# Calcoliamo l'EGDI target: ovvero la correzione che tiene conto della predizione del modello, dell'errore che il modello commette e del
# delta che voglio.
# Ovviamente, siccome l'EGDI NON può essere superiore ad 1, c'è un clipping dato dalla funzione di minimo
egdi_target = min(1, egdi_pred_start + required_egdi_delta + epsilon_target)

# Calcoliamo i CAPS
caps_df = compute_feature_caps_from_signs(X_scaled_df=X_scaled,
                                          signs_series=signs,
                                          q_hi=caps_high_perc,
                                          q_lo=caps_low_perc)

del signs, X_scaled

results_df_total = pd.DataFrame()
policy_df_total = pd.DataFrame()
country_scaled_final_df = pd.DataFrame()
scores_total = {}
candidates_total = {}

print("Policies evaluations starting...\n")
for random_seed in range(n_repetitions):
    
    print(f"Repetition: {random_seed + 1}")
    
    # Trova le candidates
    candidates, scores = find_candidates_features(how_many=K,
                                                  country_name=country,
                                                  country_scaled_df=country_scaled,
                                                  country_shap_series=country_shap,
                                                  caps_df=caps_df,
                                                  starting_egdi_pred=egdi_pred_start,
                                                  change=rho,
                                                  epsilon_scoring_value=epsilon_scoring,
                                                  trained_model_base=base_model,
                                                  random_state=random_seed,
                                                  verbose=False,
                                                  mode="Probabilities")
    
    # Trova la policy
    policy, policy_df, success, budget_spent, egdi_pred_final, country_scaled_final = run_greedy_policy(country_scaled_df=country_scaled,
                                                                                                        candidates_list=candidates,
                                                                                                        caps_df=caps_df,
                                                                                                        trained_base_model=base_model,
                                                                                                        country_name=country,
                                                                                                        starting_egdi_pred=egdi_pred_start,
                                                                                                        target_egdi=egdi_target,
                                                                                                        how_many=K,
                                                                                                        change=rho,
                                                                                                        budget=budget,
                                                                                                        epsilon_scoring_value=epsilon_scoring)
    
    # Salvo
    scores_total[f"seed_{random_seed}"] = scores
    candidates_total[f"seed_{random_seed}"] = candidates
    features_names = list(country_scaled_final.columns)
    country_scaled_final["policy"] = f"seed_{random_seed}"
    features_names = ["policy"] + features_names
    country_scaled_final = country_scaled_final[features_names]
    country_scaled_final_df = pd.concat([country_scaled_final_df, country_scaled_final], axis = 0)
    results_df = pd.DataFrame({"budget": budget,
                               "budget_spent": budget_spent,
                               "success": success,
                               "starting_egdi": egdi_pred_start,
                               "final_egdi": egdi_pred_final,
                               "final_egdi_minus_target_egdi": egdi_pred_final - egdi_target}, index = [f"seed_{random_seed}"])
    results_df_total = pd.concat([results_df_total, results_df], axis = 0)
    
    
    policy_df = policy_df.reset_index(drop = False)
    policy_df["step"] = policy_df["step"].apply(lambda x: f"seed_{random_seed}_step_{x}")
    policy_df_total = pd.concat([policy_df_total, policy_df], axis = 0)
    
    del candidates, policy, policy_df, success, budget_spent, egdi_pred_final, country_scaled_final, results_df, features_names, scores
del base_model, caps_df, country_scaled, country_shap, epsilon_scoring, epsilon_target, data_df, y


print("\nPolicies evaluations finished.\n")

# Sommario del Jaccard score e del Kendall score:
jaccard_kendall = compute_stability(candidates_total)
jaccard_kendall_summary = summarize(jaccard_kendall)

del jaccard_kendall, candidates_total

#%%

# Salvo
run_dir = save_rs_outputs(
    run_base_path=r"C:/Users/WKS/Desktop/UNIBA/UN/RS/Risultati/",
    parameters=parameters,
    jaccard_kendall_summary=jaccard_kendall_summary,
    results_df_total=results_df_total,
    policy_df_total=policy_df_total,
    scores_total=scores_total,
    country_scaled_final_df=country_scaled_final_df,
)
print(f"Data saved in {run_dir}")

