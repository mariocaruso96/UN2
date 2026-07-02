# -*- coding: utf-8 -*-

"""
Scenario-based final policy extraction for EGDI Recommender System.

This script:
1) Reads all RS runs for each country.
2) Builds one configuration-level summary table.
3) Keeps only robust configurations:
   - success_rate >= SUCCESS_THRESHOLD
   - difference_mean >= 0
4) Selects three final scenarios:
   - Conservative
   - Efficient
   - Ambitious
5) Extracts the final policy as the top-K consensus features.
   IMPORTANT:
   No frequency threshold is applied to select policy features.
   Frequency is retained only as diagnostic information.
6) Saves clean Excel and CSV outputs.
"""

import os
import re
import json
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations
from scipy.stats import kendalltau


# =============================================================================
# 1) USER SETTINGS
# =============================================================================

YEAR = 2022

COUNTRIES = ["AFG_random", "DZA_random", "ITA_random"]

ROOT_RESULTS = Path(r"C:/Users/WKS/Desktop/UNIBA/UN/RS/Risultati")
OUTPUT_DIR = Path(rf"C:/Users/WKS/Desktop/UNIBA/UN/RS/Final_policy_outputs/{YEAR}")

SUCCESS_THRESHOLD = 0.95
REQUIRE_POSITIVE_DIFFERENCE = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2) I/O UTILITIES
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


def load_country_runs(country, year, root_results):
    """
    Load all RS runs for a given country and year.
    """

    path_country = root_results / str(year) / country

    if not path_country.exists():
        print(f"WARNING: folder not found: {path_country}")
        return {}

    all_runs = {}

    for run_file_name in os.listdir(path_country):

        run_path = path_country / run_file_name

        if not run_path.is_dir():
            continue

        try:
            parameters_path = run_path / "parameters.json"
            jk_path = run_path / "jaccard_kendall_summary.json"

            if not parameters_path.exists():
                print(f"WARNING: missing parameters.json, skipped: {run_file_name}")
                continue

            if not jk_path.exists():
                print(f"WARNING: missing jaccard_kendall_summary.json, skipped: {run_file_name}")
                continue

            with open(parameters_path, "r", encoding="utf-8") as f:
                parameters = json.load(f)

            with open(jk_path, "r", encoding="utf-8") as f:
                jk_summary = json.load(f)

            results_df_total = read_dataframe(run_path / "results_df_total")
            policy_df_total = read_dataframe(run_path / "policy_df_total")

            K = parameters["K"]
            rho = parameters["rho"]
            required_egdi_delta = parameters["required_egdi_delta"]

            run_key = f"K_{K}_rho_{rho}_required_egdi_delta_{required_egdi_delta}"

            all_runs[run_key] = {
                "country": country,
                "year": year,
                "K": K,
                "rho": rho,
                "required_egdi_delta": required_egdi_delta,
                "run_file_name": run_file_name,
                "run_path": str(run_path),
                "parameters": parameters,
                "jk_summary": jk_summary,
                "results_df_total": results_df_total,
                "policy_df_total": policy_df_total,
                "run_key": run_key,
            }

        except Exception as e:
            print(f"WARNING: run skipped due to error: {run_file_name}")
            print(e)

    return all_runs


# =============================================================================
# 3) POLICY UTILITIES
# =============================================================================

def parse_policy_seed_and_step(policy_df):
    """
    Parse seed and step number from the step column.

    Expected format:
    seed_0_step_1
    seed_1_step_2
    ...
    """

    df = policy_df.copy()

    if df.empty:
        return df

    if "step" not in df.columns:
        raise KeyError("policy_df_total must contain a 'step' column.")

    seeds = []
    step_numbers = []

    for value in df["step"].astype(str):

        match = re.match(r"(seed_\d+)_step_(\d+)", value)

        if match is None:
            seeds.append(np.nan)
            step_numbers.append(np.nan)
        else:
            seeds.append(match.group(1))
            step_numbers.append(int(match.group(2)))

    df["seed"] = seeds
    df["step_number"] = step_numbers

    return df


def extract_policy_lists(policy_df):
    """
    Extract one ordered policy list for each repetition/seed.
    """

    df = parse_policy_seed_and_step(policy_df)

    if df.empty:
        return {}

    df = df.dropna(subset=["seed", "step_number", "feature"])
    df = df.sort_values(["seed", "step_number"])

    policy_lists = {}

    for seed, g in df.groupby("seed"):
        policy_lists[seed] = g["feature"].tolist()

    return policy_lists


def jaccard_similarity(list_1, list_2):
    """
    Jaccard similarity between two feature sets.
    """

    set_1 = set(list_1)
    set_2 = set(list_2)

    if len(set_1 | set_2) == 0:
        return 1.0

    return len(set_1 & set_2) / len(set_1 | set_2)


def kendall_on_common(list_1, list_2):
    """
    Kendall tau computed only on common features.
    """

    pos_1 = {v: i for i, v in enumerate(list_1)}
    pos_2 = {v: i for i, v in enumerate(list_2)}

    common = [v for v in list_1 if v in pos_2]

    if len(common) < 2:
        return np.nan

    rank_1 = [pos_1[v] for v in common]
    rank_2 = [pos_2[v] for v in common]

    return float(kendalltau(rank_1, rank_2).statistic)


def compute_policy_stability(policy_lists):
    """
    Compute pairwise policy stability across repeated runs.
    """

    seeds = list(policy_lists.keys())

    if len(seeds) < 2:
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
    exact_matches = []

    for seed_1, seed_2 in combinations(seeds, 2):

        policy_1 = policy_lists[seed_1]
        policy_2 = policy_lists[seed_2]

        jaccards.append(jaccard_similarity(policy_1, policy_2))
        kendalls.append(kendall_on_common(policy_1, policy_2))
        exact_matches.append(int(policy_1 == policy_2))

    jaccards = np.array(jaccards, dtype=float)
    kendalls = np.array(kendalls, dtype=float)
    exact_matches = np.array(exact_matches, dtype=float)

    return {
        "policy_jaccard_mean": float(np.nanmean(jaccards)),
        "policy_jaccard_median": float(np.nanmedian(jaccards)),
        "policy_kendall_mean": float(np.nanmean(kendalls)),
        "policy_kendall_median": float(np.nanmedian(kendalls)),
        "policy_exact_match_rate": float(np.nanmean(exact_matches)),
        "policy_n_pairs": len(jaccards),
    }


def build_consensus_policy(policy_df, n_repetitions):
    """
    Build consensus policy from repeated runs.

    Features are ranked by:
    1) frequency descending
    2) mean_rank ascending

    No frequency threshold is applied.
    """

    df = parse_policy_seed_and_step(policy_df)

    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["seed", "step_number", "feature"])

    if df.empty:
        return pd.DataFrame()

    agg_dict = {
        "seed": "nunique",
        "step_number": ["mean", "median", "min", "max"],
    }

    optional_cols = ["delta_E", "cost", "roi"]

    for col in optional_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    consensus = df.groupby("feature").agg(agg_dict)

    consensus.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in consensus.columns
    ]

    consensus = consensus.rename(columns={
        "seed_nunique": "count",
        "step_number_mean": "mean_rank",
        "step_number_median": "median_rank",
        "step_number_min": "min_rank",
        "step_number_max": "max_rank",
        "delta_E_mean": "mean_delta_E",
        "cost_mean": "mean_cost",
        "roi_mean": "mean_roi",
    })

    consensus["frequency"] = consensus["count"] / n_repetitions

    consensus = consensus.reset_index()

    consensus = consensus.sort_values(
        by=["frequency", "mean_rank"],
        ascending=[False, True]
    )

    consensus["consensus_rank"] = np.arange(1, len(consensus) + 1)

    first_cols = [
        "consensus_rank",
        "feature",
        "count",
        "frequency",
        "mean_rank",
        "median_rank",
        "min_rank",
        "max_rank",
    ]

    first_cols = [c for c in first_cols if c in consensus.columns]
    other_cols = [c for c in consensus.columns if c not in first_cols]

    return consensus[first_cols + other_cols]


# =============================================================================
# 4) CONFIGURATION SUMMARY
# =============================================================================

def add_efficiency_column(df):
    """
    Efficiency = required_egdi_delta / (rho * K)
    """

    df = df.copy()

    denominator = df["rho"] * df["K"]
    denominator = denominator.replace(0, np.nan)

    df["efficiency"] = df["required_egdi_delta"] / denominator

    return df


def build_country_configuration_table(all_runs):
    """
    Build one row per RS configuration.
    """

    rows = []

    for run_key, run in all_runs.items():

        results_df = run["results_df_total"]
        jk_summary = run["jk_summary"]

        success_rate = results_df["success"].mean()

        difference = results_df["final_egdi_minus_target_egdi"]
        difference_mean = difference.mean()
        difference_std = difference.std()

        budget_spent_mean = results_df["budget_spent"].mean()
        budget_spent_std = results_df["budget_spent"].std()

        egdi_gain = results_df["final_egdi"] - results_df["starting_egdi"]
        egdi_gain_mean = egdi_gain.mean()
        egdi_gain_std = egdi_gain.std()

        row = {
            "country": run["country"],
            "year": run["year"],
            "required_egdi_delta": run["required_egdi_delta"],
            "rho": run["rho"],
            "K": run["K"],
            "success_rate": success_rate,
            "difference_mean": difference_mean,
            "difference_std": difference_std,
            "budget_spent_mean": budget_spent_mean,
            "budget_spent_std": budget_spent_std,
            "egdi_gain_mean": egdi_gain_mean,
            "egdi_gain_std": egdi_gain_std,
            "jaccard_mean_candidates": jk_summary.get("jaccard_mean", np.nan),
            "jaccard_median_candidates": jk_summary.get("jaccard_median", np.nan),
            "kendall_mean_candidates": jk_summary.get("kendall_mean", np.nan),
            "kendall_median_candidates": jk_summary.get("kendall_median", np.nan),
            "exact_match_rate_candidates": jk_summary.get("exact_match_rate", np.nan),
            "run_key": run_key,
            "run_file_name": run["run_file_name"],
            "run_path": run["run_path"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = add_efficiency_column(df)

    df = df.sort_values(
        by=["country", "required_egdi_delta", "rho", "K"],
        ascending=[True, True, True, True]
    )

    return df.reset_index(drop=True)


def filter_valid_configurations(df_global):
    """
    Robust configuration filter.

    A configuration is valid if:
    - success_rate >= SUCCESS_THRESHOLD
    - difference_mean >= 0, if required.
    """

    df_valid = df_global.copy()

    df_valid = df_valid[df_valid["success_rate"] >= SUCCESS_THRESHOLD]

    if REQUIRE_POSITIVE_DIFFERENCE:
        df_valid = df_valid[df_valid["difference_mean"] >= 0]

    df_valid = df_valid.sort_values(
        by=["required_egdi_delta", "rho", "K"],
        ascending=[True, True, True]
    )

    return df_valid.reset_index(drop=True)


# =============================================================================
# 5) SCENARIO SELECTION
# =============================================================================

def select_scenarios(df_valid):
    """
    Select the three final scenarios from valid configurations only.

    Conservative:
        minimize rho, K, budget consumption;
        then prefer larger target if still tied.

    Efficient:
        maximize required_egdi_delta / (rho * K);
        then prefer higher success_rate and larger target.

    Ambitious:
        maximize required_egdi_delta;
        then prefer lower rho, lower K, lower budget.
    """

    if df_valid.empty:
        return {}

    conservative = (
        df_valid
        .sort_values(
            by=[
                "rho",
                "K",
                "budget_spent_mean",
                "required_egdi_delta",
                "difference_mean",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
            ]
        )
        .iloc[0]
    )

    efficient = (
        df_valid
        .sort_values(
            by=[
                "efficiency",
                "success_rate",
                "required_egdi_delta",
                "budget_spent_mean",
                "rho",
                "K",
                "difference_mean",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ]
        )
        .iloc[0]
    )

    ambitious = (
        df_valid
        .sort_values(
            by=[
                "required_egdi_delta",
                "success_rate",
                "rho",
                "K",
                "budget_spent_mean",
                "difference_mean",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
                True,
            ]
        )
        .iloc[0]
    )

    scenarios = {
        "Conservative": conservative,
        "Efficient": efficient,
        "Ambitious": ambitious,
    }

    return scenarios


# =============================================================================
# 6) FINAL POLICY EXTRACTION
# =============================================================================

def extract_final_policy_for_scenario(
    scenario_name,
    scenario_row,
    selected_run
):
    """
    Extract final policy for one selected scenario.

    The final policy is defined as:
        top-K features of the consensus policy.

    No minimum feature frequency threshold is applied.
    """

    policy_df = selected_run["policy_df_total"]
    results_df = selected_run["results_df_total"]

    n_repetitions = results_df.shape[0]
    K = int(scenario_row["K"])

    consensus_policy = build_consensus_policy(
        policy_df=policy_df,
        n_repetitions=n_repetitions
    )

    policy_lists = extract_policy_lists(policy_df)
    stability = compute_policy_stability(policy_lists)

    if consensus_policy.empty:
        return pd.DataFrame(), "", 0, stability

    final_policy = (
        consensus_policy
        .sort_values(
            by=["frequency", "mean_rank"],
            ascending=[False, True]
        )
        .head(K)
        .copy()
    )

    final_policy = final_policy.rename(columns={
        "consensus_rank": "rank"
    })

    final_policy["rank"] = np.arange(1, final_policy.shape[0] + 1)

    final_policy.insert(0, "scenario", scenario_name)
    final_policy.insert(1, "country", scenario_row["country"])
    final_policy.insert(2, "year", scenario_row["year"])
    final_policy.insert(3, "required_egdi_delta", scenario_row["required_egdi_delta"])
    final_policy.insert(4, "rho", scenario_row["rho"])
    final_policy.insert(5, "K", K)
    final_policy.insert(6, "success_rate", scenario_row["success_rate"])
    final_policy.insert(7, "difference_mean", scenario_row["difference_mean"])
    final_policy.insert(8, "budget_spent_mean", scenario_row["budget_spent_mean"])
    final_policy.insert(9, "egdi_gain_mean", scenario_row["egdi_gain_mean"])
    final_policy.insert(10, "efficiency", scenario_row["efficiency"])

    final_policy["policy_jaccard_mean"] = stability["policy_jaccard_mean"]
    final_policy["policy_jaccard_median"] = stability["policy_jaccard_median"]
    final_policy["policy_kendall_mean"] = stability["policy_kendall_mean"]
    final_policy["policy_kendall_median"] = stability["policy_kendall_median"]
    final_policy["policy_exact_match_rate"] = stability["policy_exact_match_rate"]
    final_policy["policy_n_pairs"] = stability["policy_n_pairs"]
    final_policy["run_file_name"] = scenario_row["run_file_name"]

    preferred_cols = [
        "country",
        "year",
        "scenario",
        "required_egdi_delta",
        "rho",
        "K",
        "rank",
        "feature",
        "frequency",
        "count",
        "mean_rank",
        "median_rank",
        "min_rank",
        "max_rank",
        "mean_delta_E",
        "mean_cost",
        "mean_roi",
        "success_rate",
        "difference_mean",
        "budget_spent_mean",
        "egdi_gain_mean",
        "efficiency",
        "policy_jaccard_mean",
        "policy_jaccard_median",
        "policy_kendall_mean",
        "policy_kendall_median",
        "policy_exact_match_rate",
        "policy_n_pairs",
        "run_file_name",
    ]

    preferred_cols = [c for c in preferred_cols if c in final_policy.columns]
    other_cols = [c for c in final_policy.columns if c not in preferred_cols]

    final_policy = final_policy[preferred_cols + other_cols]

    final_policy_string = " ; ".join(final_policy["feature"].tolist())
    n_selected_features = final_policy.shape[0]

    return final_policy, final_policy_string, n_selected_features, stability


def build_outputs_for_country(country, year, all_runs):
    """
    Full country-level pipeline.
    """

    df_global = build_country_configuration_table(all_runs)
    df_valid = filter_valid_configurations(df_global)

    scenarios = select_scenarios(df_valid)

    if len(scenarios) == 0:
        print(f"WARNING: no valid scenario for {country}")

        empty_summary = pd.DataFrame([{
            "country": country,
            "year": year,
            "scenario": "None",
            "selection_status": "no_valid_configuration",
        }])

        return empty_summary, pd.DataFrame(), pd.DataFrame(), df_valid, df_global

    summary_rows = []
    policy_steps_all = []
    stability_rows = []

    for scenario_name, scenario_row in scenarios.items():

        run_key = scenario_row["run_key"]
        selected_run = all_runs[run_key]

        final_policy_df, final_policy_string, n_selected_features, stability = (
            extract_final_policy_for_scenario(
                scenario_name=scenario_name,
                scenario_row=scenario_row,
                selected_run=selected_run
            )
        )

        summary_row = {
            "country": scenario_row["country"],
            "year": scenario_row["year"],
            "scenario": scenario_name,
            "selection_status": "valid_configuration_selected",
            "required_egdi_delta": scenario_row["required_egdi_delta"],
            "rho": scenario_row["rho"],
            "K": scenario_row["K"],
            "n_selected_features": n_selected_features,
            "success_rate": scenario_row["success_rate"],
            "difference_mean": scenario_row["difference_mean"],
            "budget_spent_mean": scenario_row["budget_spent_mean"],
            "egdi_gain_mean": scenario_row["egdi_gain_mean"],
            "efficiency": scenario_row["efficiency"],
            "final_policy": final_policy_string,
            "run_file_name": scenario_row["run_file_name"],
            "run_path": scenario_row["run_path"],
        }

        stability_row = {
            "country": scenario_row["country"],
            "year": scenario_row["year"],
            "scenario": scenario_name,
            "required_egdi_delta": scenario_row["required_egdi_delta"],
            "rho": scenario_row["rho"],
            "K": scenario_row["K"],
            "policy_jaccard_mean": stability["policy_jaccard_mean"],
            "policy_jaccard_median": stability["policy_jaccard_median"],
            "policy_kendall_mean": stability["policy_kendall_mean"],
            "policy_kendall_median": stability["policy_kendall_median"],
            "policy_exact_match_rate": stability["policy_exact_match_rate"],
            "policy_n_pairs": stability["policy_n_pairs"],
            "run_file_name": scenario_row["run_file_name"],
        }

        summary_rows.append(summary_row)
        stability_rows.append(stability_row)

        if not final_policy_df.empty:
            policy_steps_all.append(final_policy_df)

        print(
            f"{country} | {scenario_name} | "
            f"delta={scenario_row['required_egdi_delta']} | "
            f"rho={scenario_row['rho']} | "
            f"K={scenario_row['K']} | "
            f"features={n_selected_features} | "
            f"success={scenario_row['success_rate']:.2f} | "
            f"diff={scenario_row['difference_mean']:.4f}"
        )

    final_policy_summary = pd.DataFrame(summary_rows)
    policy_stability = pd.DataFrame(stability_rows)

    summary_cols = [
        "country",
        "year",
        "scenario",
        "selection_status",
        "required_egdi_delta",
        "rho",
        "K",
        "n_selected_features",
        "success_rate",
        "difference_mean",
        "budget_spent_mean",
        "egdi_gain_mean",
        "efficiency",
        "final_policy",
        "run_file_name",
        "run_path",
    ]

    summary_cols = [c for c in summary_cols if c in final_policy_summary.columns]
    other_cols = [c for c in final_policy_summary.columns if c not in summary_cols]

    final_policy_summary = final_policy_summary[summary_cols + other_cols]

    if len(policy_steps_all) > 0:
        final_policy_steps = pd.concat(policy_steps_all, axis=0).reset_index(drop=True)
    else:
        final_policy_steps = pd.DataFrame()

    return (
        final_policy_summary,
        final_policy_steps,
        policy_stability,
        df_valid,
        df_global
    )


# =============================================================================
# 7) EXCEL EXPORT
# =============================================================================

def write_sheet(writer, df, sheet_name):
    """
    Save formatted Excel sheet.
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

    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name, header_format)

    worksheet.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))
    worksheet.freeze_panes(1, 0)

    for col_idx, col_name in enumerate(df.columns):

        values = df[col_name].astype(str).head(1000)
        max_len = max([len(str(col_name))] + [len(v) for v in values])
        width = min(max(max_len + 2, 10), 70)

        if "frequency" in col_name or "rate" in col_name:
            worksheet.set_column(col_idx, col_idx, width, percent_format)

        elif (
            "rho" in col_name
            or "mean" in col_name
            or "std" in col_name
            or "difference" in col_name
            or "budget" in col_name
            or "gain" in col_name
            or "efficiency" in col_name
            or "jaccard" in col_name
            or "kendall" in col_name
            or "roi" in col_name
            or "delta" in col_name
        ):
            worksheet.set_column(col_idx, col_idx, width, float_format)

        elif (
            col_name in ["year", "K", "rank", "count", "n_selected_features"]
            or "rank" in col_name
            or "n_pairs" in col_name
        ):
            worksheet.set_column(col_idx, col_idx, width, integer_format)

        else:
            worksheet.set_column(col_idx, col_idx, width, text_format)


def save_country_outputs(
    country,
    output_dir,
    final_policy_summary,
    final_policy_steps,
    policy_stability,
    all_valid_configurations,
    all_configurations
):
    """
    Save one clean Excel file per country.
    """

    country_dir = output_dir / country
    country_dir.mkdir(parents=True, exist_ok=True)

    output_path = country_dir / f"{country}_RS_final_policy.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

        write_sheet(
            writer=writer,
            df=final_policy_summary,
            sheet_name="Final_policy_summary"
        )

        write_sheet(
            writer=writer,
            df=final_policy_steps,
            sheet_name="Final_policy_steps"
        )

        write_sheet(
            writer=writer,
            df=policy_stability,
            sheet_name="Policy_stability"
        )

        write_sheet(
            writer=writer,
            df=all_valid_configurations,
            sheet_name="All_valid_configurations"
        )

        write_sheet(
            writer=writer,
            df=all_configurations,
            sheet_name="All_configurations"
        )

    print(f"Saved: {output_path}")


# =============================================================================
# 8) MAIN
# =============================================================================

all_countries_summary = []
all_countries_policy_steps = []
all_countries_policy_stability = []
all_countries_valid_configurations = []

for country in COUNTRIES:

    print("\n" + "=" * 80)
    print(f"Processing country: {country}")
    print("=" * 80)

    all_runs = load_country_runs(
        country=country,
        year=YEAR,
        root_results=ROOT_RESULTS
    )

    if len(all_runs) == 0:
        print(f"WARNING: no runs found for {country}")
        continue

    (
        final_policy_summary,
        final_policy_steps,
        policy_stability,
        all_valid_configurations,
        all_configurations
    ) = build_outputs_for_country(
        country=country,
        year=YEAR,
        all_runs=all_runs
    )

    save_country_outputs(
        country=country,
        output_dir=OUTPUT_DIR,
        final_policy_summary=final_policy_summary,
        final_policy_steps=final_policy_steps,
        policy_stability=policy_stability,
        all_valid_configurations=all_valid_configurations,
        all_configurations=all_configurations
    )

    if not final_policy_summary.empty:
        all_countries_summary.append(final_policy_summary)

    if not final_policy_steps.empty:
        all_countries_policy_steps.append(final_policy_steps)

    if not policy_stability.empty:
        all_countries_policy_stability.append(policy_stability)

    if not all_valid_configurations.empty:
        all_countries_valid_configurations.append(all_valid_configurations)


# =============================================================================
# 9) GLOBAL CSV OUTPUTS
# =============================================================================

if len(all_countries_summary) > 0:

    df_summary_all = pd.concat(all_countries_summary, axis=0).reset_index(drop=True)

    path_summary_all = OUTPUT_DIR / "RS_final_policy_summary_all_countries.csv"
    df_summary_all.to_csv(path_summary_all, index=False)

    print(f"\nSaved global summary: {path_summary_all}")


if len(all_countries_policy_steps) > 0:

    df_policy_steps_all = pd.concat(all_countries_policy_steps, axis=0).reset_index(drop=True)

    path_policy_steps_all = OUTPUT_DIR / "RS_final_policy_steps_all_countries.csv"
    df_policy_steps_all.to_csv(path_policy_steps_all, index=False)

    print(f"Saved global policy steps: {path_policy_steps_all}")


if len(all_countries_policy_stability) > 0:

    df_stability_all = pd.concat(all_countries_policy_stability, axis=0).reset_index(drop=True)

    path_stability_all = OUTPUT_DIR / "RS_policy_stability_all_countries.csv"
    df_stability_all.to_csv(path_stability_all, index=False)

    print(f"Saved global policy stability: {path_stability_all}")


if len(all_countries_valid_configurations) > 0:

    df_valid_all = pd.concat(all_countries_valid_configurations, axis=0).reset_index(drop=True)

    path_valid_all = OUTPUT_DIR / "RS_valid_configurations_all_countries.csv"
    df_valid_all.to_csv(path_valid_all, index=False)

    print(f"Saved global valid configurations: {path_valid_all}")


print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")