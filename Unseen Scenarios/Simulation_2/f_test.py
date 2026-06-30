#!/usr/bin/env python3
# %%
"""
Three-state per-cycle classifier based on nested F-tests.

The script classifies each operating cycle as one of three states:
Normal, Pump Fault, or System Fault.

The pipeline uses two complementary tests:

1. Pump-fault test, applied cycle by cycle
   - H0: H = a0 + a1 Q + a2 Q²
   - H1: H = (a0 + α0 t̃) + (a1 + α1 t̃) Q + (a2 + α2 t̃) Q²

2. System-fault test, applied over sliding super-cycles
   - H0: H = Hs + K Q²
   - H1: H = (Hs + β0 t̃) + (K + β2 t̃) Q²

A few practical safeguards are included because the simulation data can be
nearly noise-free during healthy operation:

- Cycle labels are mapped from the names used in the cycle-time file.
- The pump F-test is only considered when the static pump-curve residual is
  above a small noise floor. This avoids flagging numerical noise as a fault.
- System-fault decisions are based on voting across all windows that contain a
  cycle, which reduces boundary artifacts around fault transitions.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =============================================================================
# User settings
# =============================================================================

data_xlsx = Path("data/pump_system_faultsimualtion.xlsx")
cycles_xlsx = Path("data/Cycle-time.xlsx")
output_dir = Path("results/ftest_classifier")
output_dir.mkdir(parents=True, exist_ok=True)

nominal_freq = 50.0
freq_tol = 0.5
min_points_per_cycle = 30

# Pump-fault nested F-test thresholds.
alpha_pump = 0.01
delta_min_pump = 0.03
ssr0_floor = 0.01

# System-fault super-cycle F-test thresholds.
alpha_sys = 0.01
delta_min_sys = 0.02

# Super-cycle windowing.
window_size = 5
window_stride = 1
sys_vote_frac = 0.90

# Input column names.
time_col = "Timestamp"
freq_col = "Pump1_Freq_Hz"
flow_col = "Pump1_Flow_m3h"
head_col = "Pump1_Head_m"

# Labels used by the cycle-time file and their classifier names.
label_map = {
    "normal": "Normal",
    "Normal": "Normal",
    "pump_blockage_window": "Pump Fault",
    "pump_blockage": "Pump Fault",
    "pump_fault": "Pump Fault",
    "pump fault": "Pump Fault",
    "Pump Fault": "Pump Fault",
    "system_fault_window": "System Fault",
    "system_fault": "System Fault",
    "system fault": "System Fault",
    "System Fault": "System Fault",
}


# =============================================================================
# Utilities
# =============================================================================

def to_seconds(series: pd.Series) -> pd.Series:
    """Convert a datetime-like series to seconds from its first sample."""
    if np.issubdtype(series.dtype, np.datetime64):
        return (series - series.iloc[0]).dt.total_seconds()
    return series.astype(float)


def f_test(ssr0: float, ssr1: float, n_samples: int, p0: int, p1: int) -> tuple[float, float]:
    """Run a nested-model F-test and return the statistic and p-value."""
    if n_samples <= p1 + 1:
        return np.nan, np.nan

    numerator = (ssr0 - ssr1) / (p1 - p0)
    denominator = ssr1 / (n_samples - p1)

    if denominator <= 0:
        return np.nan, np.nan

    f_value = numerator / denominator
    p_value = 1 - stats.f.cdf(f_value, p1 - p0, n_samples - p1)
    return float(f_value), float(p_value)


# =============================================================================
# Model fitting helpers
# =============================================================================

def fit_pump_h0(flow: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the static pump curve: H = a0 + a1 Q + a2 Q²."""
    design = np.column_stack([np.ones_like(flow), flow, flow ** 2])
    beta, *_ = np.linalg.lstsq(design, head, rcond=None)
    ssr = float(np.sum((head - design @ beta) ** 2))
    return beta, ssr


def fit_pump_h1(flow: np.ndarray, head: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the time-varying pump curve with drifting quadratic coefficients."""
    time_centered = time_s - time_s.mean()
    time_scale = float(time_centered.std()) if time_centered.std() > 0 else 1.0
    time_norm = time_centered / time_scale

    design = np.column_stack([
        np.ones_like(flow),
        flow,
        flow ** 2,
        time_norm,
        time_norm * flow,
        time_norm * (flow ** 2),
    ])
    beta, *_ = np.linalg.lstsq(design, head, rcond=None)
    ssr = float(np.sum((head - design @ beta) ** 2))
    return beta, ssr


def fit_system_h0(flow: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the static system curve: H = Hs + K Q²."""
    design = np.column_stack([np.ones_like(flow), flow ** 2])
    beta, *_ = np.linalg.lstsq(design, head, rcond=None)
    ssr = float(np.sum((head - design @ beta) ** 2))
    return beta, ssr


def fit_system_h1(flow: np.ndarray, head: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the time-varying system curve with drifting Hs and K."""
    time_centered = time_s - time_s.mean()
    time_scale = float(time_centered.std()) if time_centered.std() > 0 else 1.0
    time_norm = time_centered / time_scale

    design = np.column_stack([
        np.ones_like(flow),
        flow ** 2,
        time_norm,
        time_norm * (flow ** 2),
    ])
    beta, *_ = np.linalg.lstsq(design, head, rcond=None)
    ssr = float(np.sum((head - design @ beta) ** 2))
    return beta, ssr


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full classification pipeline and save the decision tables."""

    # Load simulation data and the cycle table.
    df = pd.read_excel(data_xlsx)
    cycles_raw = pd.read_excel(cycles_xlsx)

    print(f"Data shape: {df.shape}")
    print(f"Cycle columns: {list(cycles_raw.columns)}")

    # Keep only nominal-frequency samples and normalize the pump measurements
    # to the 50 Hz operating point.
    df["t_s"] = to_seconds(df[time_col])
    nominal_mask = (np.abs(df[freq_col] - nominal_freq) <= freq_tol) & (df[freq_col] > 0)
    df_nominal = df[nominal_mask].copy()

    scale = nominal_freq / df_nominal[freq_col]
    df_nominal["Q50"] = df_nominal[flow_col] * scale
    df_nominal["H50"] = df_nominal[head_col] * (scale ** 2)
    df_nominal["t_rel"] = df_nominal["t_s"]

    # Prepare the cycle metadata and map labels to the three classifier states.
    cycles = cycles_raw.rename(columns={
        "start_t_sec": "start_s",
        "stop_t_sec": "end_s",
        "label": "truth",
    })

    if "cycle_id" not in cycles.columns:
        cycles["cycle_id"] = np.arange(1, len(cycles) + 1)

    cycles["truth_clean"] = cycles["truth"].astype(str).str.strip().map(label_map)
    cycles = cycles.sort_values("cycle_id").reset_index(drop=True)

    unmapped = cycles["truth_clean"].isna().sum()
    if unmapped:
        print(f"[Warning] {unmapped} cycles have labels that were not mapped:")
        print(cycles.loc[cycles["truth_clean"].isna(), "truth"].unique())

    print(f"\nLabel distribution:\n{cycles['truth_clean'].value_counts().to_string()}")

    # -------------------------------------------------------------------------
    # 1. Per-cycle pump nested F-test
    # -------------------------------------------------------------------------
    per_cycle_records = []

    for _, row in cycles.iterrows():
        block = df_nominal[
            (df_nominal["t_s"] >= row["start_s"])
            & (df_nominal["t_s"] <= row["end_s"])
        ]
        n_samples = int(len(block))
        record = {
            "cycle_id": int(row["cycle_id"]),
            "m": n_samples,
            "true_label": row["truth_clean"],
        }

        if n_samples < min_points_per_cycle:
            record.update(
                F_pump=np.nan,
                p_pump=np.nan,
                dSSR_pump=np.nan,
                SSR0_pump=np.nan,
            )
            per_cycle_records.append(record)
            continue

        flow = block["Q50"].to_numpy(float)
        head = block["H50"].to_numpy(float)
        time_s = block["t_rel"].to_numpy(float)

        _, ssr0 = fit_pump_h0(flow, head)
        _, ssr1 = fit_pump_h1(flow, head, time_s)
        f_pump, p_pump = f_test(ssr0, ssr1, n_samples, p0=3, p1=6)
        dssr_pump = (ssr0 - ssr1) / max(ssr0, 1e-12)

        record.update(
            F_pump=f_pump,
            p_pump=p_pump,
            dSSR_pump=dssr_pump,
            SSR0_pump=ssr0,
        )
        per_cycle_records.append(record)

    per_cycle_df = pd.DataFrame(per_cycle_records)

    # -------------------------------------------------------------------------
    # 2. Super-cycle system nested F-test
    # -------------------------------------------------------------------------
    n_cycles = len(cycles)
    window_records = []

    for start_idx in range(0, n_cycles - window_size + 1, window_stride):
        end_idx = start_idx + window_size - 1
        cycle_ids = cycles.loc[start_idx:end_idx, "cycle_id"].tolist()
        start_s = float(cycles.loc[start_idx, "start_s"])
        end_s = float(cycles.loc[end_idx, "end_s"])

        block = df_nominal[(df_nominal["t_s"] >= start_s) & (df_nominal["t_s"] <= end_s)]
        n_samples = int(len(block))

        if n_samples < min_points_per_cycle * 2:
            window_records.append(
                dict(
                    cycles=cycle_ids,
                    m=n_samples,
                    F_sys=np.nan,
                    p_sys=np.nan,
                    dSSR_sys=np.nan,
                )
            )
            continue

        flow = block["Q50"].to_numpy(float)
        head = block["H50"].to_numpy(float)
        time_s = block["t_s"].to_numpy(float)

        _, ssr0 = fit_system_h0(flow, head)
        _, ssr1 = fit_system_h1(flow, head, time_s)
        f_sys, p_sys = f_test(ssr0, ssr1, n_samples, p0=2, p1=4)
        dssr_sys = (ssr0 - ssr1) / max(ssr0, 1e-12)

        window_records.append(
            dict(
                cycles=cycle_ids,
                m=n_samples,
                F_sys=f_sys,
                p_sys=p_sys,
                dSSR_sys=dssr_sys,
            )
        )

    window_df = pd.DataFrame(window_records)

    # -------------------------------------------------------------------------
    # 3. Aggregate system-fault evidence by window voting
    # -------------------------------------------------------------------------
    system_vote_records = []

    for cycle_id in cycles["cycle_id"]:
        selected_windows = window_df[window_df["cycles"].apply(lambda ids: cycle_id in ids)]

        if selected_windows.empty:
            system_vote_records.append(
                dict(
                    cycle_id=cycle_id,
                    sys_vote_frac=0.0,
                    n_windows=0,
                    n_flagged=0,
                    sys_p_min=np.nan,
                    sys_drel_max=np.nan,
                )
            )
            continue

        n_total = len(selected_windows)
        n_flagged = sum(
            1
            for _, window in selected_windows.iterrows()
            if pd.notna(window["p_sys"])
            and window["p_sys"] < alpha_sys
            and window["dSSR_sys"] > delta_min_sys
        )

        system_vote_records.append(
            dict(
                cycle_id=cycle_id,
                sys_vote_frac=n_flagged / n_total,
                n_windows=n_total,
                n_flagged=n_flagged,
                sys_p_min=selected_windows["p_sys"].min(skipna=True),
                sys_drel_max=selected_windows["dSSR_sys"].max(skipna=True),
            )
        )

    system_vote_df = pd.DataFrame(system_vote_records)

    # -------------------------------------------------------------------------
    # 4. Merge the evidence and assign the final class
    # -------------------------------------------------------------------------
    results = (
        cycles[["cycle_id", "truth_clean"]]
        .merge(per_cycle_df, on="cycle_id", how="left")
        .merge(system_vote_df, on="cycle_id", how="left")
    )

    def decide(row: pd.Series) -> str:
        """Convert pump and system evidence into one of the three states."""
        n_samples = int(row["m"]) if pd.notna(row["m"]) else 0
        if n_samples < min_points_per_cycle:
            return "Unknown"

        # Pump fault: nested F-test, gated by the static residual noise floor.
        ssr0 = float(row["SSR0_pump"]) if pd.notna(row["SSR0_pump"]) else 0.0
        pump_flag = False
        if ssr0 > ssr0_floor:
            p_pump = float(row["p_pump"]) if pd.notna(row["p_pump"]) else 1.0
            dssr_pump = float(row["dSSR_pump"]) if pd.notna(row["dSSR_pump"]) else 0.0
            pump_flag = (p_pump < alpha_pump) and (dssr_pump > delta_min_pump)

        # System fault: high enough fraction of supporting super-cycle windows.
        vote_fraction = float(row["sys_vote_frac"]) if pd.notna(row["sys_vote_frac"]) else 0.0
        system_flag = vote_fraction >= sys_vote_frac

        if pump_flag and not system_flag:
            return "Pump Fault"
        if system_flag and not pump_flag:
            return "System Fault"
        if pump_flag and system_flag:
            return "Pump Fault"
        return "Normal"

    results["pred_label"] = results.apply(decide, axis=1)

    # -------------------------------------------------------------------------
    # 5. Evaluate the classifier
    # -------------------------------------------------------------------------
    valid_labels = ["Normal", "Pump Fault", "System Fault"]
    eval_mask = results["truth_clean"].isin(valid_labels) & results["pred_label"].isin(valid_labels)

    cm = confusion_matrix(
        results.loc[eval_mask, "truth_clean"],
        results.loc[eval_mask, "pred_label"],
        labels=valid_labels,
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {label}" for label in valid_labels],
        columns=[f"Pred {label}" for label in valid_labels],
    )

    print("\n" + "=" * 60)
    print("Confusion matrix")
    print("=" * 60)
    print(cm_df.to_string())

    report = classification_report(
        results.loc[eval_mask, "truth_clean"],
        results.loc[eval_mask, "pred_label"],
        labels=valid_labels,
        zero_division=0,
    )
    print(f"\n{report}")

    balanced_accuracy = balanced_accuracy_score(
        results.loc[eval_mask, "truth_clean"],
        results.loc[eval_mask, "pred_label"],
    )
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    misclassified = results[eval_mask & (results["truth_clean"] != results["pred_label"])]
    if misclassified.empty:
        print("\nPerfect classification: 0 errors.")
    else:
        print(f"\n{len(misclassified)} misclassified cycles:")
        for _, row in misclassified.iterrows():
            print(
                f"  Cycle {row['cycle_id']}: "
                f"truth={row['truth_clean']}, pred={row['pred_label']}"
            )

    # -------------------------------------------------------------------------
    # 6. Save detailed tables
    # -------------------------------------------------------------------------
    results_csv = output_dir / "per_cycle_decisions_ftest.csv"
    keep_columns = [
        column
        for column in [
            "cycle_id",
            "true_label",
            "pred_label",
            "m",
            "F_pump",
            "p_pump",
            "dSSR_pump",
            "SSR0_pump",
            "sys_vote_frac",
            "n_windows",
            "n_flagged",
            "sys_p_min",
            "sys_drel_max",
        ]
        if column in results.columns
    ]
    results[keep_columns].to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    windows_csv = output_dir / "supercycle_windows.csv"
    window_df.to_csv(windows_csv, index=False)
    print(f"Window-level results saved to: {windows_csv}")

    return results, cm_df


if __name__ == "__main__":
    main()
# %%
