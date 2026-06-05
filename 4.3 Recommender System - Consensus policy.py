# -*- coding: utf-8 -*-

#%% 0) Librerie e funzioni
"""
Recommender System - Best minimal configuration and final recommended policy

Obiettivo:
1) Leggere gli output già salvati dal Recommender System.
2) Ricostruire la tabella globale delle performance per rho, K, required_egdi_delta.
3) Selezionare, per ogni required_egdi_delta, la configurazione minimale:
   - success_rate >= SUCCESS_THRESHOLD
   - difference_mean >= 0
   - rho minimo
   - K minimo
   - difference_mean minimo
   - budget_spent_mean minimo
4) Estrarre la consensus policy dalle ripetizioni.
5) Salvare un unico file Excel per paese:
   - Executive_summary
   - Final_policy
   - Consensus_policies
   - Policy_stability
   - All_configurations
   - All_policy_steps
"""

import os
import re
import json
import numpy as np
import pandas as pd
import xlsxwriter

from pathlib import Path
from itertools import combinations
from scipy.stats import kendalltau


# =============================================================================
# 0) Utility functions
# =============================================================================

def read_dataframe(path_no_ext):
    """
    Read dataframe saved either as parquet or csv.
    """

    path_no_ext = Path(path_no_ext)

    parquet_path = path_no_ext.with_suffix(".parquet")
    csv_path = path_no_ext.with_suffix(".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        return df

    raise FileNotFoundError(f"No dataframe found for: {path_no_ext}")


def parse_policy_seed_and_step(policy_df):
    """
    Convert step labels like:
        seed_0_step_1

    into:
        seed = seed_0
        step_number = 1
    """

    df = policy_df.copy()

    if df.empty:
        return df

    if "step" not in df.columns:
        raise KeyError("policy_df_total must contain a 'step' column.")

    seeds = []
    steps = []

    for x in df["step"].astype(str):

        match = re.match(r"(seed_\d+)_step_(\d+)", x)

        if match is None:
            seeds.append(np.nan)
            steps.append(np.nan)

        else:
            seeds.append(match.group(1))
            steps.append(int(match.group(2)))

    df["seed"] = seeds
    df["step_number"] = steps

    return df


def extract_policy_lists(policy_df):
    """
    Extract ordered final policy lists from policy_df_total.
    One list for each repetition/seed.
    """

    df = parse_policy_seed_and_step(policy_df)

    if df.empty:
        return {}

    df = df.dropna(subset=["seed", "step_number"])
    df = df.sort_values(["seed", "step_number"])

    policies = {}

    for seed, g in df.groupby("seed"):
        policies[seed] = g["feature"].tolist()

    return policies


def jaccard_similarity(l1, l2):
    """
    Jaccard similarity between two feature sets.
    """

    s1, s2 = set(l1), set(l2)

    if len(s1 | s2) == 0:
        return 1.0

    return len(s1 & s2) / len(s1 | s2)


def kendall_on_common(l1, l2):
    """
    Kendall tau computed only on common elements.
    """

    pos1 = {v: i for i, v in enumerate(l1)}
    pos2 = {v: i for i, v in enumerate(l2)}

    common = [v for v in l1 if v in pos2]

    if len(common) < 2:
        return np.nan

    r1 = [pos1[v] for v in common]
    r2 = [pos2[v] for v in common]

    return float(kendalltau(r1, r2).statistic)


def compute_policy_stability(policy_lists):
    """
    Compute Jaccard and Kendall stability directly on final policies.

    This is different from the jaccard/kendall already saved by the RS,
    which were computed on sampled candidate lists.
    """

    keys = list(policy_lists.keys())

    if len(keys) < 2:
        return {
            "policy_jaccard_mean": np.nan,
            "policy_jaccard_median": np.nan,
            "policy_kendall_mean": np.nan,
            "policy_kendall_median": np.nan,
            "policy_exact_match_rate": np.nan,
            "policy_n_pairs": 0,
        }

    jaccards = []
    kendalls = []
    exacts = []

    for k1, k2 in combinations(keys, 2):

        l1 = policy_lists[k1]
        l2 = policy_lists[k2]

        jaccards.append(jaccard_similarity(l1, l2))
        kendalls.append(kendall_on_common(l1, l2))
        exacts.append(int(l1 == l2))

    jaccards = np.array(jaccards, dtype=float)
    kendalls = np.array(kendalls, dtype=float)
    exacts = np.array(exacts, dtype=float)

    return {
        "policy_jaccard_mean": float(np.nanmean(jaccards)),
        "policy_jaccard_median": float(np.nanmedian(jaccards)),
        "policy_kendall_mean": float(np.nanmean(kendalls)),
        "policy_kendall_median": float(np.nanmedian(kendalls)),
        "policy_exact_match_rate": float(np.nanmean(exacts)),
        "policy_n_pairs": len(jaccards),
    }


def build_consensus_policy(policy_df, n_repetitions=None):
    """
    Build consensus policy from all repeated RS policies.

    Ranking criterion:
    1) higher frequency across repetitions
    2) lower mean rank inside the policies
    """

    df = parse_policy_seed_and_step(policy_df)

    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["seed", "step_number", "feature"])

    if df.empty:
        return pd.DataFrame()

    if n_repetitions is None:
        n_repetitions = df["seed"].nunique()

    agg_dict = {
        "seed": "nunique",
        "step_number": ["mean", "median", "min", "max"],
    }

    optional_cols = ["delta_E", "cost", "roi"]

    for col in optional_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    out = df.groupby("feature").agg(agg_dict)

    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in out.columns
    ]

    out = out.rename(columns={
        "seed_nunique": "count",
        "step_number_mean": "mean_rank",
        "step_number_median": "median_rank",
        "step_number_min": "min_rank",
        "step_number_max": "max_rank",
        "delta_E_mean": "mean_delta_E",
        "cost_mean": "mean_cost",
        "roi_mean": "mean_roi",
    })

    out["frequency"] = out["count"] / n_repetitions

    out = out.reset_index()

    out = out.sort_values(
        by=["frequency", "mean_rank"],
        ascending=[False, True]
    )

    out["consensus_rank"] = np.arange(1, len(out) + 1)

    cols_first = [
        "consensus_rank",
        "feature",
        "count",
        "frequency",
        "mean_rank",
        "median_rank",
        "min_rank",
        "max_rank",
    ]

    other_cols = [c for c in out.columns if c not in cols_first]

    return out[cols_first + other_cols]


# =============================================================================
# Excel utility functions
# =============================================================================

def _write_sheet(writer, df, sheet_name, freeze_panes=(1, 0)):
    """
    Write a dataframe to Excel with readable formatting.
    """

    if df is None or df.empty:
        return

    df.to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#D9EAF7",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })

    percent_format = workbook.add_format({"num_format": "0.0%"})
    float_format = workbook.add_format({"num_format": "0.0000"})
    integer_format = workbook.add_format({"num_format": "0"})
    text_format = workbook.add_format({"text_wrap": False})

    for col_num, value in enumerate(df.columns):
        worksheet.write(0, col_num, value, header_format)

    worksheet.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))
    worksheet.freeze_panes(*freeze_panes)

    for col_idx, col_name in enumerate(df.columns):

        series = df[col_name].astype(str)

        max_len = max(
            [len(str(col_name))] +
            [len(x) for x in series.head(1000)]
        )

        width = min(max(max_len + 2, 10), 45)

        if "frequency" in col_name or "rate" in col_name:
            worksheet.set_column(col_idx, col_idx, width, percent_format)

        elif (
            "mean" in col_name
            or "std" in col_name
            or "delta" in col_name
            or "gain" in col_name
            or "difference" in col_name
            or "budget" in col_name
            or "rho" in col_name
            or "roi" in col_name
            or "kendall" in col_name
            or "jaccard" in col_name
        ):
            worksheet.set_column(col_idx, col_idx, width, float_format)

        elif (
            col_name in ["K", "best_K", "year", "count", "consensus_rank"]
            or "rank" in col_name
            or "step_number" in col_name
            or "n_pairs" in col_name
        ):
            worksheet.set_column(col_idx, col_idx, width, integer_format)

        else:
            worksheet.set_column(col_idx, col_idx, width, text_format)


def _build_executive_summary(best_configurations_df):
    """
    Compact sheet: one row per required EGDI delta.
    """

    if best_configurations_df.empty:
        return pd.DataFrame()

    preferred_cols = [
        "country",
        "year",
        "required_egdi_delta",
        "rho",
        "K",
        "selection_status",
        "success_rate",
        "difference_mean",
        "budget_spent_mean",
        "egdi_gain_mean",
        "policy_jaccard_mean",
        "policy_kendall_mean",
        "policy_exact_match_rate",
        "consensus_policy_topK",
        "run_file_name",
    ]

    preferred_cols = [
        c for c in preferred_cols
        if c in best_configurations_df.columns
    ]

    return best_configurations_df[preferred_cols].copy()


def _build_policy_stability(best_configurations_df):
    """
    Dedicated sheet for stability metrics on final policies and candidates.
    """

    if best_configurations_df.empty:
        return pd.DataFrame()

    preferred_cols = [
        "country",
        "year",
        "required_egdi_delta",
        "rho",
        "K",
        "success_rate",
        "policy_jaccard_mean",
        "policy_jaccard_median",
        "policy_kendall_mean",
        "policy_kendall_median",
        "policy_exact_match_rate",
        "policy_n_pairs",
        "jaccard_mean_candidates",
        "jaccard_median_candidates",
        "kendall_mean_candidates",
        "kendall_median_candidates",
        "exact_match_rate_candidates",
    ]

    preferred_cols = [
        c for c in preferred_cols
        if c in best_configurations_df.columns
    ]

    return best_configurations_df[preferred_cols].copy()


def save_country_excel(
    output_dir,
    country,
    year,
    df_global,
    best_configurations_df,
    consensus_policy_df,
    final_recommended_policy_df,
    policy_steps_df,
):
    """
    Save one readable Excel workbook per country.
    """

    output_path = output_dir / f"{country}_RS_analysis.xlsx"

    executive_summary_df = _build_executive_summary(best_configurations_df)
    policy_stability_df = _build_policy_stability(best_configurations_df)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

        _write_sheet(
            writer,
            executive_summary_df,
            "Executive_summary"
        )

        _write_sheet(
            writer,
            final_recommended_policy_df,
            "Final_policy"
        )

        _write_sheet(
            writer,
            consensus_policy_df,
            "Consensus_policies"
        )

        _write_sheet(
            writer,
            policy_stability_df,
            "Policy_stability"
        )

        _write_sheet(
            writer,
            df_global,
            "All_configurations"
        )

        _write_sheet(
            writer,
            policy_steps_df,
            "All_policy_steps"
        )

    print(f"Excel saved in: {output_path}")


#%% 1) User settings

year = 2022

countries = ["CIV", "AFG", "ITA"]
# countries = ["AFG", "DZA", "ITA", "CIV"]

required_egdi_delta_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50]

SUCCESS_THRESHOLD = 1.00
REQUIRE_POSITIVE_DIFFERENCE = True

root_results = Path(r"C:/Users/WKS/Desktop/UNIBA/UN/RS/Risultati")
base_output_dir = Path(rf"C:/Users/WKS/Desktop/UNIBA/UN/RS/Best_policy_outputs/{year}")

base_output_dir.mkdir(parents=True, exist_ok=True)


#%% 2) Main analysis

for country in countries:

    print(f"\nProcessing country: {country}")

    output_dir = base_output_dir / country
    output_dir.mkdir(parents=True, exist_ok=True)

    path_country_files = root_results / str(year) / country

    if not path_country_files.exists():
        print(f"WARNING: country folder not found: {path_country_files}")
        continue

    all_best_rows = []
    all_consensus_rows = []
    all_final_policy_rows = []
    all_policy_step_rows = []

    all_runs = {}

    # =========================================================================
    # 2.1) Read all runs for the selected country
    # =========================================================================

    for run_file_name in os.listdir(path_country_files):

        run_path = path_country_files / run_file_name

        if not run_path.is_dir():
            continue

        try:
            parameters_path = run_path / "parameters.json"
            jk_path = run_path / "jaccard_kendall_summary.json"

            if not parameters_path.exists() or not jk_path.exists():
                print(f"WARNING: incomplete run skipped: {run_file_name}")
                continue

            with open(parameters_path, "r", encoding="utf-8") as f:
                parameters = json.load(f)

            with open(jk_path, "r", encoding="utf-8") as f:
                jaccard_kendall_summary = json.load(f)

            results_df_total = read_dataframe(run_path / "results_df_total")
            policy_df_total = read_dataframe(run_path / "policy_df_total")

            K = parameters["K"]
            rho = parameters["rho"]
            required_egdi_delta = parameters["required_egdi_delta"]

            key = f"K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"

            all_runs[key] = {
                "country": country,
                "year": year,
                "rho": rho,
                "K": K,
                "required_egdi_delta": required_egdi_delta,
                "run_file_name": run_file_name,
                "run_path": str(run_path),
                "parameters": parameters,
                "jaccard_kendall_summary": jaccard_kendall_summary,
                "results_df_total": results_df_total,
                "policy_df_total": policy_df_total,
            }

        except Exception as e:
            print(f"WARNING: run skipped due to error: {run_file_name}")
            print(e)

    if len(all_runs) == 0:
        print(f"WARNING: no valid runs found for {country}.")
        continue

    # =========================================================================
    # 2.2) Build global configuration summary for the country
    # =========================================================================

    df_global = pd.DataFrame()

    for key, dictionary in all_runs.items():

        results_df_total = dictionary["results_df_total"]
        jk_summary = dictionary["jaccard_kendall_summary"]

        success_rate = results_df_total["success"].sum() / results_df_total.shape[0]

        budget_spent_mean = results_df_total["budget_spent"].mean()
        budget_spent_std = results_df_total["budget_spent"].std()

        difference_mean = results_df_total["final_egdi_minus_target_egdi"].mean()
        difference_std = results_df_total["final_egdi_minus_target_egdi"].std()

        egdi_gain = (
            results_df_total["final_egdi"] -
            results_df_total["starting_egdi"]
        )

        egdi_gain_mean = egdi_gain.mean()
        egdi_gain_std = egdi_gain.std()

        row = pd.DataFrame({
            "country": dictionary["country"],
            "year": dictionary["year"],
            "required_egdi_delta": dictionary["required_egdi_delta"],
            "K": dictionary["K"],
            "rho": dictionary["rho"],
            "success_rate": success_rate,
            "budget_spent_mean": budget_spent_mean,
            "budget_spent_std": budget_spent_std,
            "difference_mean": difference_mean,
            "difference_std": difference_std,
            "egdi_gain_mean": egdi_gain_mean,
            "egdi_gain_std": egdi_gain_std,

            "jaccard_mean_candidates": jk_summary.get("jaccard_mean", np.nan),
            "jaccard_median_candidates": jk_summary.get("jaccard_median", np.nan),
            "kendall_mean_candidates": jk_summary.get("kendall_mean", np.nan),
            "kendall_median_candidates": jk_summary.get("kendall_median", np.nan),
            "exact_match_rate_candidates": jk_summary.get("exact_match_rate", np.nan),

            "run_file_name": dictionary["run_file_name"],
            "run_path": dictionary["run_path"],
            "run_key": key,
        }, index=[0])

        df_global = pd.concat([df_global, row], axis=0)

    df_global = df_global.reset_index(drop=True)

    # =========================================================================
    # 2.3) Select best minimal configuration for each required_egdi_delta
    # =========================================================================

    for required_egdi_delta in required_egdi_delta_vals:

        df_delta = df_global[
            np.isclose(df_global["required_egdi_delta"], required_egdi_delta)
        ].copy()

        if df_delta.empty:
            print(f"WARNING: no runs for {country}, delta={required_egdi_delta}")
            continue

        df_valid = df_delta[
            df_delta["success_rate"] >= SUCCESS_THRESHOLD
        ].copy()

        if REQUIRE_POSITIVE_DIFFERENCE:
            df_valid = df_valid[df_valid["difference_mean"] >= 0].copy()

        if df_valid.empty:

            print(
                f"WARNING: no valid configuration for "
                f"{country}, delta={required_egdi_delta}. "
                f"Fallback: best available by success_rate."
            )

            best_row = (
                df_delta
                .sort_values(
                    by=[
                        "success_rate",
                        "rho",
                        "K",
                        "difference_mean",
                        "budget_spent_mean",
                    ],
                    ascending=[False, True, True, True, True]
                )
                .iloc[0]
            )

            selection_status = "fallback_best_available"

        else:

            best_row = (
                df_valid
                .sort_values(
                    by=[
                        "rho",
                        "K",
                        "difference_mean",
                        "budget_spent_mean",
                    ],
                    ascending=[True, True, True, True]
                )
                .iloc[0]
            )

            selection_status = "valid_minimal_successful"

        run_key = best_row["run_key"]
        best_run = all_runs[run_key]

        policy_df_total = best_run["policy_df_total"]
        results_df_total = best_run["results_df_total"]

        # ---------------------------------------------------------------------
        # Stability directly on final policies
        # ---------------------------------------------------------------------

        policy_lists = extract_policy_lists(policy_df_total)
        policy_stability = compute_policy_stability(policy_lists)

        # ---------------------------------------------------------------------
        # Consensus policy
        # ---------------------------------------------------------------------

        consensus_policy = build_consensus_policy(
            policy_df_total,
            n_repetitions=results_df_total.shape[0]
        )

        if not consensus_policy.empty:

            consensus_policy.insert(0, "country", country)
            consensus_policy.insert(1, "year", year)
            consensus_policy.insert(2, "required_egdi_delta", required_egdi_delta)
            consensus_policy.insert(3, "best_rho", best_row["rho"])
            consensus_policy.insert(4, "best_K", best_row["K"])
            consensus_policy.insert(5, "selection_status", selection_status)
            consensus_policy.insert(6, "run_file_name", best_row["run_file_name"])

            all_consensus_rows.append(consensus_policy)

        # ---------------------------------------------------------------------
        # Final recommended policy: top-K features from consensus policy
        # ---------------------------------------------------------------------

        if not consensus_policy.empty:

            final_policy = consensus_policy.head(int(best_row["K"])).copy()

            final_policy["success_rate"] = best_row["success_rate"]
            final_policy["difference_mean"] = best_row["difference_mean"]
            final_policy["budget_spent_mean"] = best_row["budget_spent_mean"]
            final_policy["egdi_gain_mean"] = best_row["egdi_gain_mean"]

            final_policy["policy_jaccard_mean"] = policy_stability["policy_jaccard_mean"]
            final_policy["policy_jaccard_median"] = policy_stability["policy_jaccard_median"]
            final_policy["policy_kendall_mean"] = policy_stability["policy_kendall_mean"]
            final_policy["policy_kendall_median"] = policy_stability["policy_kendall_median"]
            final_policy["policy_exact_match_rate"] = policy_stability["policy_exact_match_rate"]
            final_policy["policy_n_pairs"] = policy_stability["policy_n_pairs"]

            preferred_cols = [
                "country",
                "year",
                "required_egdi_delta",
                "best_rho",
                "best_K",
                "selection_status",
                "success_rate",
                "difference_mean",
                "budget_spent_mean",
                "egdi_gain_mean",
                "policy_jaccard_mean",
                "policy_kendall_mean",
                "policy_jaccard_median",
                "policy_kendall_median",
                "policy_exact_match_rate",
                "consensus_rank",
                "feature",
                "count",
                "frequency",
                "mean_rank",
                "median_rank",
                "min_rank",
                "max_rank",
                "mean_delta_E",
                "mean_cost",
                "mean_roi",
                "run_file_name",
            ]

            preferred_cols = [
                c for c in preferred_cols
                if c in final_policy.columns
            ]

            other_cols = [
                c for c in final_policy.columns
                if c not in preferred_cols
            ]

            final_policy = final_policy[preferred_cols + other_cols]

            all_final_policy_rows.append(final_policy)

        # ---------------------------------------------------------------------
        # Detailed policy steps
        # ---------------------------------------------------------------------

        policy_steps = parse_policy_seed_and_step(policy_df_total)

        if not policy_steps.empty:

            policy_steps.insert(0, "country", country)
            policy_steps.insert(1, "year", year)
            policy_steps.insert(2, "required_egdi_delta", required_egdi_delta)
            policy_steps.insert(3, "best_rho", best_row["rho"])
            policy_steps.insert(4, "best_K", best_row["K"])
            policy_steps.insert(5, "selection_status", selection_status)
            policy_steps.insert(6, "run_file_name", best_row["run_file_name"])

            all_policy_step_rows.append(policy_steps)

        # ---------------------------------------------------------------------
        # Best configuration summary row
        # ---------------------------------------------------------------------

        best_dict = best_row.to_dict()
        best_dict.update(policy_stability)

        best_dict["selection_status"] = selection_status
        best_dict["n_policy_repetitions"] = len(policy_lists)

        if not consensus_policy.empty:
            top_features = consensus_policy.head(int(best_row["K"]))["feature"].tolist()
            best_dict["consensus_policy_topK"] = " | ".join(top_features)
        else:
            best_dict["consensus_policy_topK"] = ""

        all_best_rows.append(best_dict)

        print(
            f"{country} | delta={required_egdi_delta} | "
            f"best rho={best_row['rho']} | K={best_row['K']} | "
            f"success={best_row['success_rate']:.2f} | "
            f"diff={best_row['difference_mean']:.4f} | "
            f"status={selection_status}"
        )

    # =========================================================================
    # 2.4) Build final dataframes and save one Excel file for the country
    # =========================================================================

    best_configurations_df = pd.DataFrame(all_best_rows)

    if len(all_consensus_rows) > 0:
        consensus_policy_df = pd.concat(all_consensus_rows, axis=0)
    else:
        consensus_policy_df = pd.DataFrame()

    if len(all_final_policy_rows) > 0:
        final_recommended_policy_df = pd.concat(all_final_policy_rows, axis=0)
    else:
        final_recommended_policy_df = pd.DataFrame()

    if len(all_policy_step_rows) > 0:
        policy_steps_df = pd.concat(all_policy_step_rows, axis=0)
    else:
        policy_steps_df = pd.DataFrame()

    # Add percentage-readable columns.
    # They are formatted as percentages in Excel.
    if not consensus_policy_df.empty and "frequency" in consensus_policy_df.columns:
        consensus_policy_df["frequency_percent"] = consensus_policy_df["frequency"]

    if not final_recommended_policy_df.empty and "frequency" in final_recommended_policy_df.columns:
        final_recommended_policy_df["frequency_percent"] = final_recommended_policy_df["frequency"]

    save_country_excel(
        output_dir=output_dir,
        country=country,
        year=year,
        df_global=df_global,
        best_configurations_df=best_configurations_df,
        consensus_policy_df=consensus_policy_df,
        final_recommended_policy_df=final_recommended_policy_df,
        policy_steps_df=policy_steps_df,
    )


print("\nDone.")
print(f"Outputs saved in: {base_output_dir}")