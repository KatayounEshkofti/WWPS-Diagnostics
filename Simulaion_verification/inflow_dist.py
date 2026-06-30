#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow-rate distribution analysis for sump level data.

This script prepares an empirical inflow distribution from sump-level
measurements. It reads a set of Excel workbooks, extracts the time and level
signals, estimates the signed flow rate from level changes, and saves the
cleaned flow-rate series together with summary statistics and diagnostic plots.

The second part of the script builds an empirical-CDF sampler from the positive
flow rates. This sampler is used later by the pump-station simulator to draw
realistic inflow values.

Expected input
--------------
The input folder should contain files named like:

    output_nov_dec_part_1.xlsx
    output_nov_dec_part_2.xlsx
    ...

Each workbook should have a first sheet with at least two columns:

    time
    level

Column names are treated case-insensitively.

Outputs
-------
The script writes the following files to ``results/flow_rate_analysis``:

    flow_rates.csv
    flow_rate_summary.txt
    flow_rate_histogram.png
    ECDF.pdf
    ECDF.eps
    inflow_distribution_fit.png
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

area_m2 = 8.0

data_dir = Path("data/hogdalen_2024_11_20_to_2024_12_16")
file_glob = "output_nov_dec_part_*.xlsx"

output_dir = Path("results/flow_rate_analysis")
timezone_name: Optional[str] = "Europe/Stockholm"

flow_csv = output_dir / "flow_rates.csv"

output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data preparation helpers
# =============================================================================

def numeric_sort_key(path: str | Path) -> int:
    """
    Extract the numeric suffix from names such as
    ``output_nov_dec_part_17.xlsx`` so the files are processed in order.
    """
    match = re.search(r"part_(\d+)\.xlsx$", Path(path).name)
    return int(match.group(1)) if match else 0


def load_single_file(path: str | Path) -> pd.DataFrame:
    """
    Read one workbook and return a cleaned time-level dataframe.

    The function keeps only the columns needed for the flow calculation:
    ``time`` and ``level``. Duplicate timestamps are resolved by keeping the
    last observation at each timestamp.
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(column).lower() for column in df.columns]

    required_columns = {"time", "level"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{path} does not contain the required columns: {required_columns}."
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "level"]).copy()

    if timezone_name:
        if df["time"].dt.tz is None:
            df["time"] = (
                df["time"]
                .dt.tz_localize(
                    "UTC",
                    ambiguous="infer",
                    nonexistent="shift_forward",
                )
                .dt.tz_convert(timezone_name)
            )
        else:
            df["time"] = df["time"].dt.tz_convert(timezone_name)

    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")
    return df[["time", "level"]]


def compute_flow_rate(df_level: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the signed flow rate from the level time series.

    Positive values correspond to rising water level, interpreted as inflow.
    The calculation follows:

        Q = A * dL / dt

    where ``A`` is the sump cross-sectional area.
    """
    df = df_level.copy().set_index("time")

    delta_t = df.index.to_series().diff().dt.total_seconds()
    delta_level = df["level"].diff()

    df["flow_rate_m3_s"] = area_m2 * delta_level / delta_t
    df["flow_rate_L_s"] = df["flow_rate_m3_s"] * 1000.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["flow_rate_m3_s"])

    # Very small time steps usually come from duplicate or nearly duplicate
    # measurements and can create unrealistic spikes.
    df = df[delta_t >= 0.1]

    return df.reset_index()


def build_flow_rate_table() -> pd.DataFrame:
    """
    Load all matching workbooks, compute flow rates, and return one table.
    """
    pattern = str(data_dir / file_glob)
    paths = sorted(glob.glob(pattern), key=numeric_sort_key)

    if not paths:
        raise FileNotFoundError(
            f"No files matched {pattern!r}. Check data_dir and file_glob."
        )

    logger.info("Found %d workbook(s). Starting preprocessing.", len(paths))

    flow_parts: list[pd.DataFrame] = []

    for path in tqdm(paths, desc="Processing files", unit="file"):
        try:
            df_level = load_single_file(path)
            df_flow = compute_flow_rate(df_level)
            flow_parts.append(df_flow)
        except Exception as exc:
            logger.exception("Skipping %s because of: %s", path, exc)

    if not flow_parts:
        raise RuntimeError("No valid flow-rate data could be extracted.")

    df_all = pd.concat(flow_parts, ignore_index=True)
    logger.info("Total valid flow-rate samples: %s", f"{len(df_all):,}")

    return df_all


def save_flow_outputs(df_flow: pd.DataFrame) -> None:
    """
    Save the cleaned flow-rate table, descriptive statistics, and histogram.
    """
    df_flow.to_csv(flow_csv, index=False)
    logger.info("Cleaned flow-rate table saved to %s", flow_csv)

    summary_path = output_dir / "flow_rate_summary.txt"
    stats = df_flow["flow_rate_m3_s"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# Flow-rate descriptive statistics (m³ s⁻¹)\n\n")
        file.write(stats.to_string())

    logger.info("Summary statistics saved to %s", summary_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        df_flow["flow_rate_m3_s"],
        bins=200,
        density=True,
        alpha=0.6,
        label="Histogram",
    )
    df_flow["flow_rate_m3_s"].plot(kind="kde", linewidth=2, label="KDE", ax=ax)

    ax.set_xlabel("Flow rate [m³ s⁻¹]")
    ax.set_ylabel("Probability density")
    ax.set_title("Sump flow-rate distribution")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "flow_rate_histogram.png", dpi=300)
    plt.close(fig)

    logger.info("Histogram saved to %s", output_dir / "flow_rate_histogram.png")


# =============================================================================
# Empirical-CDF sampler
# =============================================================================

def load_positive_flows(csv_path: str | Path = flow_csv) -> np.ndarray:
    """
    Load positive flow rates from the cleaned CSV file.

    Positive signed flow rates are treated as inflow.
    """
    df_flow = pd.read_csv(csv_path)

    positive = df_flow.loc[df_flow["flow_rate_m3_s"] > 0, "flow_rate_m3_s"].to_numpy()
    positive.sort()

    if positive.size == 0:
        raise ValueError("No positive flow rates were found in the cleaned CSV file.")

    return positive


def ecdf_sampler(
    positive_flows: np.ndarray,
    n: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Draw random samples from the empirical inflow distribution.

    Samples are returned in m³ s⁻¹.
    """
    rng = rng or np.random.default_rng()
    u = rng.random(n)
    return np.quantile(positive_flows, u)


# =============================================================================
# Diagnostic plots and distribution fit
# =============================================================================

def plot_ecdf_check(positive_flows: np.ndarray) -> None:
    """
    Compare the original positive-flow distribution with ECDF samples.
    """
    sns.set(style="whitegrid", font_scale=1.2)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(
        positive_flows,
        bins="fd",
        stat="density",
        element="step",
        fill=False,
        lw=3,
        ax=ax,
        label="Empirical",
        color="#0077b6",
    )
    sns.histplot(
        ecdf_sampler(positive_flows, 2_000_000),
        bins="fd",
        stat="density",
        element="step",
        fill=False,
        lw=3,
        ax=ax,
        label="ECDF draws",
        color="#ef476f",
    )

    ax.set_xlabel(r"Flow rate\,$(\mathrm{m^{3}\,s^{-1}})$")
    ax.set_xlim(0.0, 0.06)
    ax.set_ylabel("Density")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.grid(axis="x", visible=False)

    sns.despine(ax=ax, top=True, right=True)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("#444444")

    ax.legend(frameon=True, handlelength=3, borderpad=0.2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "ECDF.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_dir / "ECDF.eps", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("ECDF comparison plot saved to %s", output_dir)


def fit_candidate_distributions(csv_path: str | Path = flow_csv) -> pd.DataFrame:
    """
    Fit a small set of candidate distributions to positive inflow values.

    The fitting step follows the original script: it uses a random 50,000-sample
    subset of positive inflow values in L/s and compares candidates using the
    KS statistic and AIC.
    """
    df_flow = pd.read_csv(csv_path)

    inflow = df_flow.loc[df_flow["flow_rate_L_s"] > 0, "flow_rate_L_s"]

    if inflow.empty:
        raise ValueError("No positive inflow values were found for distribution fitting.")

    sample_size = min(50_000, len(inflow))
    sample = inflow.sample(sample_size, random_state=1)

    dist_names = ["gamma", "lognorm", "norm", "expon"]
    fit_summary = []

    for name in dist_names:
        dist = getattr(st, name)
        params = dist.fit(sample)
        ks_stat, _ = st.kstest(sample, name, args=params)
        log_likelihood = np.sum(dist.logpdf(sample, *params))
        n_params = len(params)
        aic = 2 * n_params - 2 * log_likelihood

        fit_summary.append(
            {
                "distribution": name,
                "params": [round(float(param), 4) for param in params],
                "KS statistic": round(float(ks_stat), 4),
                "AIC": round(float(aic), 1),
            }
        )

    fit_df = pd.DataFrame(fit_summary).sort_values("AIC").reset_index(drop=True)
    fit_df.to_csv(output_dir / "distribution_fit_summary.csv", index=False)

    best_name = fit_df.iloc[0]["distribution"]
    best_dist = getattr(st, best_name)
    best_params = tuple(fit_df.iloc[0]["params"])

    x = np.linspace(inflow.min(), inflow.max(), 300)
    pdf = best_dist.pdf(x, *best_params)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(inflow, bins=80, density=True, alpha=0.5, label="Empirical")
    ax.plot(x, pdf, linewidth=2, label=f"{best_name.title()} PDF")
    ax.set_xlabel("Flow rate (L/s)")
    ax.set_ylabel("Density")
    ax.set_title("Inflow rate distribution fit")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "inflow_distribution_fit.png", dpi=300)
    plt.close(fig)

    logger.info("Distribution-fit summary saved to %s", output_dir)

    return fit_df


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    """
    Run the full flow-rate analysis pipeline.
    """
    df_flow = build_flow_rate_table()
    save_flow_outputs(df_flow)

    positive_flows = load_positive_flows(flow_csv)

    preview = ecdf_sampler(positive_flows, 100)
    print("Example ECDF draws [m³/s]:", preview)

    plot_ecdf_check(positive_flows)

    fit_df = fit_candidate_distributions(flow_csv)
    print("\nDistribution-fit summary:")
    print(fit_df)

    logger.info("Flow-rate analysis complete.")


if __name__ == "__main__":
    main()
