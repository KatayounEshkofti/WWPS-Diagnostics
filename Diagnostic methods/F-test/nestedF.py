"""
Three-state pump-condition classifier and diagnoser.

The script contains three related workflows for classifying Pump 1 cycles as:
Normal, Pump Fault, or System Fault.

Workflows
---------
1. Per-cycle pump F-test + super-cycle system F-test
   - Keeps the same per-cycle pump test.
   - Uses sliding windows of consecutive cycles for a system-curve F-test.

2. Full diagnoser class
   - End-to-end version with configuration, optional weighted least squares,
     Holm-Bonferroni correction, optional threshold sweep, output files, and plots.

Expected inputs
---------------
The time-series Excel file should include:
    Timestamp
    Pump1_Freq_Hz
    Pump1_Flow_m3h
    Pump1_Head_m

The cycle-label Excel file should include:
    cycle_id
    start_s
    end_s
    truth
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Shared defaults
# -----------------------------------------------------------------------------
default_data_xlsx = Path("data/pump_system_faultsimualtion.xlsx")
default_cycles_xlsx = Path("data/Cycles-Labels.xlsx")
default_output_dir = Path("results/pump_fault_diagnosis")

valid_labels = ["Normal", "Pump Fault", "System Fault"]
all_labels = valid_labels + ["Unknown"]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def to_seconds(series: pd.Series) -> pd.Series:
    """Convert a datetime or numeric time column to seconds from the start."""
    if np.issubdtype(series.dtype, np.datetime64):
        start_time = series.iloc[0]
        return (series - start_time).dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def standardize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return standardized values together with the original mean and std."""
    mean_value = float(np.nanmean(values))
    std_value = float(np.nanstd(values))
    if std_value == 0 or not np.isfinite(std_value):
        std_value = 1.0
    return (values - mean_value) / std_value, mean_value, std_value


def nested_f_test(ssr_null: float, ssr_alt: float, n_samples: int, p_null: int, p_alt: int) -> Tuple[float, float]:
    """Classic nested-model F-test."""
    if n_samples <= p_alt + 1:
        return np.nan, np.nan

    numerator = (ssr_null - ssr_alt) / (p_alt - p_null)
    denominator = ssr_alt / (n_samples - p_alt)

    if denominator <= 0:
        return np.nan, np.nan

    f_value = numerator / denominator
    p_value = 1.0 - stats.f.cdf(f_value, p_alt - p_null, n_samples - p_alt)
    return float(f_value), float(p_value)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    valid_index = [idx for idx, value in enumerate(p_values) if np.isfinite(value)]
    if not valid_index:
        return [np.nan] * len(p_values)

    sorted_pairs = sorted(((p_values[idx], idx) for idx in valid_index), key=lambda item: item[0])
    n_tests = len(sorted_pairs)
    adjusted = [np.nan] * len(p_values)
    previous = 0.0

    for rank, (p_value, original_index) in enumerate(sorted_pairs, start=1):
        corrected = (n_tests - rank + 1) * p_value
        corrected = max(corrected, previous)
        previous = corrected
        adjusted[original_index] = min(1.0, corrected)

    return adjusted


def normalize_cycle_columns(cycles: pd.DataFrame) -> pd.DataFrame:
    """Normalize common cycle-label column names."""
    lower_to_original = {col.lower(): col for col in cycles.columns}

    required = ["cycle_id", "start_s", "end_s"]
    for column in required:
        if column not in lower_to_original:
            raise KeyError(f"Missing required cycle column: {column}")

    rename_map = {
        lower_to_original["cycle_id"]: "cycle_id",
        lower_to_original["start_s"]: "start_s",
        lower_to_original["end_s"]: "end_s",
    }

    if "truth" in lower_to_original:
        rename_map[lower_to_original["truth"]] = "truth"

    cycles = cycles.rename(columns=rename_map)
    if "truth" not in cycles.columns:
        cycles["truth"] = None

    return cycles.sort_values("cycle_id").reset_index(drop=True)


def prepare_nominal_stream(
    data: pd.DataFrame,
    time_col: str,
    freq_col: str,
    flow_col: str,
    head_col: str,
    nominal_freq: float,
    freq_tol: float,
) -> pd.DataFrame:
    """Keep nominal-frequency Pump 1 points and affinity-normalize Q and H."""
    df = data.copy()
    df["t_s"] = to_seconds(df[time_col])

    nominal_mask = (np.abs(df[freq_col] - nominal_freq) <= freq_tol) & (df[freq_col] > 0.0)
    df_nominal = df[nominal_mask].copy()

    scale = nominal_freq / df_nominal[freq_col].replace(0, np.nan)
    df_nominal["Q50"] = df_nominal[flow_col] * scale
    df_nominal["H50"] = df_nominal[head_col] * (scale**2)
    df_nominal["t_rel"] = df_nominal["t_s"]

    return df_nominal


def clean_labels(series: pd.Series) -> pd.Series:
    """Normalize labels and keep unknown labels explicit."""
    labels = series.astype(str).str.strip()
    labels.loc[~labels.isin(valid_labels)] = "Unknown"
    return labels


# -----------------------------------------------------------------------------
# Pump and system model fits
# -----------------------------------------------------------------------------
def fit_pump_h0(Q: np.ndarray, H: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Static pump curve: H = a0 + a1 Q + a2 Q^2."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    design = np.column_stack([np.ones_like(Q), Q, Q**2])

    if weights is not None:
        design_w = design * weights[:, None]
        H_w = H * weights
        beta, *_ = np.linalg.lstsq(design_w, H_w, rcond=None)
        residual = H_w - design_w @ beta
    else:
        beta, *_ = np.linalg.lstsq(design, H, rcond=None)
        residual = H - design @ beta

    ssr = float(np.sum(residual**2))
    return beta, ssr


def fit_pump_h1(Q: np.ndarray, H: np.ndarray, t: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Drifting pump curve with time-varying coefficients."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    t = np.asarray(t, dtype=float)

    t_norm, _, _ = standardize(t)
    design = np.column_stack([
        np.ones_like(Q),
        Q,
        Q**2,
        t_norm,
        t_norm * Q,
        t_norm * (Q**2),
    ])

    if weights is not None:
        design_w = design * weights[:, None]
        H_w = H * weights
        beta, *_ = np.linalg.lstsq(design_w, H_w, rcond=None)
        residual = H_w - design_w @ beta
    else:
        beta, *_ = np.linalg.lstsq(design, H, rcond=None)
        residual = H - design @ beta

    ssr = float(np.sum(residual**2))
    return beta, ssr


def fit_system_h0(Q: np.ndarray, H: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Static system curve: H = Hs + K Q^2."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    design = np.column_stack([np.ones_like(Q), Q**2])

    if weights is not None:
        design_w = design * weights[:, None]
        H_w = H * weights
        beta, *_ = np.linalg.lstsq(design_w, H_w, rcond=None)
        residual = H_w - design_w @ beta
    else:
        beta, *_ = np.linalg.lstsq(design, H, rcond=None)
        residual = H - design @ beta

    ssr = float(np.sum(residual**2))
    return beta, ssr


def fit_system_h1(Q: np.ndarray, H: np.ndarray, t: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Drifting system curve: H = (Hs + b0 t) + (K + b2 t) Q^2."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    t = np.asarray(t, dtype=float)

    t_norm, _, _ = standardize(t)
    design = np.column_stack([
        np.ones_like(Q),
        Q**2,
        t_norm,
        t_norm * (Q**2),
    ])

    if weights is not None:
        design_w = design * weights[:, None]
        H_w = H * weights
        beta, *_ = np.linalg.lstsq(design_w, H_w, rcond=None)
        residual = H_w - design_w @ beta
    else:
        beta, *_ = np.linalg.lstsq(design, H, rcond=None)
        residual = H - design @ beta

    ssr = float(np.sum(residual**2))
    return beta, ssr


def fit_pump_h0_optimized(Q: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, float]:
    """Static pump curve using numerical minimization, kept from the first workflow."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    design = np.column_stack([np.ones_like(Q), Q, Q**2])

    try:
        beta_initial, *_ = np.linalg.lstsq(design, H, rcond=None)
    except Exception:
        beta_initial = np.array([H.mean(), 0.0, 0.0])

    def ssr(params: np.ndarray) -> float:
        a0, a1, a2 = params
        prediction = a0 + a1 * Q + a2 * (Q**2)
        return float(np.sum((H - prediction) ** 2))

    result = optimize.minimize(ssr, beta_initial, method="L-BFGS-B")
    ssr_value = ssr(result.x)
    return result.x, ssr_value


def fit_pump_h1_optimized(Q: np.ndarray, H: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Drifting pump curve using numerical minimization, kept from the first workflow."""
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    t = np.asarray(t, dtype=float)

    t_centered = t - t.mean()
    t_scale = float(t_centered.std()) if t_centered.std() > 0 else 1.0
    t_norm = t_centered / t_scale

    (a0, a1, a2), _ = fit_pump_h0_optimized(Q, H)
    initial = np.array([a0, a1, a2, 0.0, 0.0, 0.0], dtype=float)

    def ssr(params: np.ndarray) -> float:
        a0, a1, a2, al0, al1, al2 = params
        prediction = (a0 + al0 * t_norm) + (a1 + al1 * t_norm) * Q + (a2 + al2 * t_norm) * (Q**2)
        return float(np.sum((H - prediction) ** 2))

    result = optimize.minimize(ssr, initial, method="L-BFGS-B")
    ssr_value = ssr(result.x)
    return result.x, ssr_value


def fit_system_curve(Q: np.ndarray, H: np.ndarray) -> Tuple[float, float, float]:
    """Approximate system curve fit: H = Hs + K Q^2."""
    beta, ssr = fit_system_h0(Q, H)
    head_static, friction = beta
    return float(head_static), float(friction), ssr

# -----------------------------------------------------------------------------
# Pump F-test + super-cycle system F-test
# -----------------------------------------------------------------------------
def run_supercycle_classifier(
    data_xlsx: Path = default_data_xlsx,
    cycles_xlsx: Path = default_cycles_xlsx,
    out_dir: Path = default_output_dir / "supercycle_classifier",
    nominal_freq: float = 50.0,
    freq_tol: float = 0.5,
    min_points_per_cycle: int = 30,
    alpha_pump: float = 0.01,
    delta_min_pump: float = 0.03,
    alpha_sys: float = 0.01,
    delta_min_sys: float = 0.02,
    window_size: int = 6,
    window_stride: int = 2,
    time_col: str = "Timestamp",
    freq_col: str = "Pump1_Freq_Hz",
    flow_col: str = "Pump1_Flow_m3h",
    head_col: str = "Pump1_Head_m",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the super-cycle system F-test version of the classifier."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(data_xlsx)
    cycles = normalize_cycle_columns(pd.read_excel(cycles_xlsx))
    df_nominal = prepare_nominal_stream(data, time_col, freq_col, flow_col, head_col, nominal_freq, freq_tol)

    per_cycle_records = []
    for _, row in cycles.iterrows():
        cycle_id = int(row["cycle_id"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        truth = row.get("truth", None)

        block = df_nominal[(df_nominal["t_s"] >= start_s) & (df_nominal["t_s"] <= end_s)].copy()
        n_samples = block.shape[0]

        if n_samples < min_points_per_cycle:
            per_cycle_records.append({
                "cycle_id": cycle_id,
                "m": n_samples,
                "true_label": truth,
                "F_pump": np.nan,
                "p_pump": np.nan,
                "dSSR_rel_pump": np.nan,
            })
            continue

        Q = block["Q50"].to_numpy(float)
        H = block["H50"].to_numpy(float)
        t = block["t_rel"].to_numpy(float)

        _, ssr0 = fit_pump_h0(Q, H)
        _, ssr1 = fit_pump_h1(Q, H, t)
        f_pump, p_pump = nested_f_test(ssr0, ssr1, n_samples, p_null=3, p_alt=6)
        delta_rel = (ssr0 - ssr1) / max(ssr0, 1e-12)

        per_cycle_records.append({
            "cycle_id": cycle_id,
            "m": n_samples,
            "true_label": truth,
            "F_pump": f_pump,
            "p_pump": p_pump,
            "dSSR_rel_pump": delta_rel,
        })

    per_cycle = pd.DataFrame(per_cycle_records)

    window_records = []
    n_cycles = cycles.shape[0]
    for start_idx in range(0, n_cycles - window_size + 1, window_stride):
        end_idx = start_idx + window_size - 1
        cycle_ids = cycles.loc[start_idx:end_idx, "cycle_id"].tolist()
        start_s = float(cycles.loc[start_idx, "start_s"])
        end_s = float(cycles.loc[end_idx, "end_s"])

        block = df_nominal[(df_nominal["t_s"] >= start_s) & (df_nominal["t_s"] <= end_s)].copy()
        n_samples = block.shape[0]

        if n_samples < min_points_per_cycle * 2:
            window_records.append({
                "win_start": start_idx,
                "win_end": end_idx,
                "cycles": cycle_ids,
                "m": n_samples,
                "F_sys": np.nan,
                "p_sys": np.nan,
                "dSSR_rel_sys": np.nan,
            })
            continue

        Q = block["Q50"].to_numpy(float)
        H = block["H50"].to_numpy(float)
        t = block["t_rel"].to_numpy(float)

        _, ssr0 = fit_system_h0(Q, H)
        _, ssr1 = fit_system_h1(Q, H, t)
        f_sys, p_sys = nested_f_test(ssr0, ssr1, n_samples, p_null=2, p_alt=4)
        delta_rel = (ssr0 - ssr1) / max(ssr0, 1e-12)

        window_records.append({
            "win_start": start_idx,
            "win_end": end_idx,
            "cycles": cycle_ids,
            "m": n_samples,
            "F_sys": f_sys,
            "p_sys": p_sys,
            "dSSR_rel_sys": delta_rel,
        })

    windows = pd.DataFrame(window_records)

    system_agg = []
    for cycle_id in cycles["cycle_id"]:
        selected = windows[windows["cycles"].apply(lambda ids: cycle_id in ids)]
        if selected.empty:
            system_agg.append({"cycle_id": cycle_id, "sys_p_min": np.nan, "sys_drel_max": np.nan})
            continue
        system_agg.append({
            "cycle_id": cycle_id,
            "sys_p_min": selected["p_sys"].min(skipna=True),
            "sys_drel_max": selected["dSSR_rel_sys"].max(skipna=True),
        })

    system_agg = pd.DataFrame(system_agg)

    results = cycles[["cycle_id", "start_s", "end_s", "truth"]].merge(per_cycle, on="cycle_id", how="left").merge(system_agg, on="cycle_id", how="left")

    def decide(row: pd.Series) -> str:
        n_samples = int(row["m"]) if pd.notna(row["m"]) else 0
        if n_samples < min_points_per_cycle:
            return "Unknown"

        pump_flag = (row["p_pump"] < alpha_pump) and (row["dSSR_rel_pump"] > delta_min_pump)
        sys_flag = (row["sys_p_min"] < alpha_sys) and (row["sys_drel_max"] > delta_min_sys)

        if pump_flag and not sys_flag:
            return "Pump Fault"
        if sys_flag and not pump_flag:
            return "System Fault"
        if pump_flag and sys_flag:
            return "Pump Fault" if row["dSSR_rel_pump"] >= row["sys_drel_max"] else "System Fault"
        return "Normal"

    results["pred_label"] = results.apply(decide, axis=1)
    results["truth_clean"] = clean_labels(results["truth"])
    results["pred_clean"] = clean_labels(results["pred_label"])

    labels_all = valid_labels + ["Unknown"]
    cm_all = confusion_matrix(results["truth_clean"], results["pred_clean"], labels=labels_all)
    cm_all_df = pd.DataFrame(cm_all, index=[f"True {label}" for label in labels_all], columns=[f"Pred {label}" for label in labels_all])
    assert cm_all.sum() == len(results)

    known_mask = (results["truth_clean"] != "Unknown") & (results["pred_clean"] != "Unknown")
    cm_known = confusion_matrix(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"], labels=valid_labels)
    cm_known_df = pd.DataFrame(cm_known, index=[f"True {label}" for label in valid_labels], columns=[f"Pred {label}" for label in valid_labels])
    report = classification_report(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"], labels=valid_labels, zero_division=0)

    results_out = out_dir / "per_cycle_decisions_supercycle_sysF.csv"
    windows_out = out_dir / "supercycle_windows_stats.csv"
    results.to_csv(results_out, index=False)
    windows.to_csv(windows_out, index=False)

    print("\nSuper-cycle classifier confusion matrix, including Unknown:")
    print(cm_all_df.to_string())
    print("\nSuper-cycle classifier confusion matrix, known labels only:")
    print(cm_known_df.to_string())
    print("\nSuper-cycle classifier report:")
    print(report)
    print(f"\nSuper-cycle decisions saved to: {results_out}")
    print(f"Super-cycle window statistics saved to: {windows_out}")

    return results, windows


# -----------------------------------------------------------------------------
# Full diagnoser class
# -----------------------------------------------------------------------------
@dataclass
class DiagnoserConfig:
    data_xlsx: Path = default_data_xlsx
    cycles_xlsx: Path = default_cycles_xlsx
    out_dir: Path = default_output_dir / "full_diagnoser"

    time_col: str = "Timestamp"
    freq_col: str = "Pump1_Freq_Hz"
    flow_col: str = "Pump1_Flow_m3h"
    head_col: str = "Pump1_Head_m"

    nominal_freq: float = 50.0
    freq_tol: float = 0.5

    min_points_per_cycle: int = 30
    min_points_per_window: int = 60

    alpha_pump: float = 0.01
    delta_min_pump: float = 0.03

    alpha_sys: float = 0.01
    delta_min_sys: float = 0.02

    window_size: int = 6
    window_stride: int = 2

    holm_bonferroni: bool = True
    use_wls: bool = False
    make_plots: bool = True

    do_sweep: bool = False
    sweep_alphas: Tuple[float, ...] = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2)
    sweep_deltas_p: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
    sweep_deltas_s: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10)

    seed: int = 42


class PumpSystemDiagnoser:
    """End-to-end pump/system fault diagnoser for cycle-bounded data."""

    def __init__(self, config: DiagnoserConfig):
        self.config = config
        np.random.seed(config.seed)
        self.raw_data: Optional[pd.DataFrame] = None
        self.nominal_data: Optional[pd.DataFrame] = None
        self.cycles: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self.windows: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Load the time series and cycle table."""
        self.raw_data = pd.read_excel(self.config.data_xlsx)
        self.cycles = normalize_cycle_columns(pd.read_excel(self.config.cycles_xlsx))

    def preprocess(self) -> None:
        """Prepare the nominal-frequency stream."""
        if self.raw_data is None:
            raise RuntimeError("Data must be loaded before preprocessing.")

        cfg = self.config
        self.nominal_data = prepare_nominal_stream(
            self.raw_data,
            cfg.time_col,
            cfg.freq_col,
            cfg.flow_col,
            cfg.head_col,
            cfg.nominal_freq,
            cfg.freq_tol,
        )

    def per_cycle_tests(self) -> pd.DataFrame:
        """Run the per-cycle pump F-test."""
        if self.nominal_data is None or self.cycles is None:
            raise RuntimeError("Data and cycles must be available before testing.")

        cfg = self.config
        records = []

        for _, row in self.cycles.iterrows():
            cycle_id = int(row["cycle_id"])
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])
            truth = row.get("truth", None)

            block = self.nominal_data[(self.nominal_data["t_s"] >= start_s) & (self.nominal_data["t_s"] <= end_s)].copy()
            n_samples = int(block.shape[0])

            if n_samples < cfg.min_points_per_cycle:
                records.append({
                    "cycle_id": cycle_id,
                    "m": n_samples,
                    "true_label": truth,
                    "F_pump": np.nan,
                    "p_pump": np.nan,
                    "dSSR_rel_pump": np.nan,
                })
                continue

            Q = block["Q50"].to_numpy(float)
            H = block["H50"].to_numpy(float)
            t = block["t_rel"].to_numpy(float)

            weights = 1.0 / np.sqrt(1.0 + Q**2) if cfg.use_wls else None

            _, ssr0 = fit_pump_h0(Q, H, weights=weights)
            _, ssr1 = fit_pump_h1(Q, H, t, weights=weights)
            f_pump, p_pump = nested_f_test(ssr0, ssr1, n_samples, p_null=3, p_alt=6)
            delta_rel = (ssr0 - ssr1) / max(ssr0, 1e-12)

            records.append({
                "cycle_id": cycle_id,
                "m": n_samples,
                "true_label": truth,
                "F_pump": f_pump,
                "p_pump": p_pump,
                "dSSR_rel_pump": delta_rel,
            })

        return pd.DataFrame(records)

    def window_system_tests(self) -> pd.DataFrame:
        """Run system F-tests on sliding windows of cycles."""
        if self.nominal_data is None or self.cycles is None:
            raise RuntimeError("Data and cycles must be available before testing.")

        cfg = self.config
        records = []
        n_cycles = self.cycles.shape[0]

        for start_idx in range(0, n_cycles - cfg.window_size + 1, cfg.window_stride):
            end_idx = start_idx + cfg.window_size - 1
            cycle_ids = self.cycles.loc[start_idx:end_idx, "cycle_id"].tolist()
            start_s = float(self.cycles.loc[start_idx, "start_s"])
            end_s = float(self.cycles.loc[end_idx, "end_s"])

            block = self.nominal_data[(self.nominal_data["t_s"] >= start_s) & (self.nominal_data["t_s"] <= end_s)].copy()
            n_samples = int(block.shape[0])

            if n_samples < max(cfg.min_points_per_window, cfg.min_points_per_cycle * 2):
                records.append({
                    "win_start": start_idx,
                    "win_end": end_idx,
                    "cycles": cycle_ids,
                    "m": n_samples,
                    "F_sys": np.nan,
                    "p_sys": np.nan,
                    "dSSR_rel_sys": np.nan,
                })
                continue

            Q = block["Q50"].to_numpy(float)
            H = block["H50"].to_numpy(float)
            t = block["t_rel"].to_numpy(float)

            weights = 1.0 / np.sqrt(1.0 + Q**2) if cfg.use_wls else None

            _, ssr0 = fit_system_h0(Q, H, weights=weights)
            _, ssr1 = fit_system_h1(Q, H, t, weights=weights)
            f_sys, p_sys = nested_f_test(ssr0, ssr1, n_samples, p_null=2, p_alt=4)
            delta_rel = (ssr0 - ssr1) / max(ssr0, 1e-12)

            records.append({
                "win_start": start_idx,
                "win_end": end_idx,
                "cycles": cycle_ids,
                "m": n_samples,
                "F_sys": f_sys,
                "p_sys": p_sys,
                "dSSR_rel_sys": delta_rel,
            })

        return pd.DataFrame(records)

    def aggregate_system_to_cycles(self, windows: pd.DataFrame) -> pd.DataFrame:
        """Map sliding-window system statistics back to individual cycles."""
        if self.cycles is None:
            raise RuntimeError("Cycles must be available before aggregation.")

        cfg = self.config
        records = []

        for cycle_id in self.cycles["cycle_id"]:
            selected = windows[windows["cycles"].apply(lambda ids: cycle_id in ids)].reset_index(drop=True)

            if selected.empty:
                records.append({
                    "cycle_id": cycle_id,
                    "sys_p_min": np.nan,
                    "sys_p_holm_min": np.nan,
                    "sys_drel_max": np.nan,
                })
                continue

            p_values = selected["p_sys"].tolist()
            if cfg.holm_bonferroni:
                p_adjusted = holm_adjust(p_values)
                p_min_holm = np.nanmin(np.asarray(p_adjusted, dtype=float)) if p_adjusted else np.nan
            else:
                p_min_holm = np.nan

            records.append({
                "cycle_id": cycle_id,
                "sys_p_min": selected["p_sys"].min(skipna=True),
                "sys_p_holm_min": p_min_holm,
                "sys_drel_max": selected["dSSR_rel_sys"].max(skipna=True),
            })

        return pd.DataFrame(records)

    @staticmethod
    def decide_row(
        row: pd.Series,
        alpha_pump: float,
        delta_min_pump: float,
        alpha_sys: float,
        delta_min_sys: float,
        use_holm: bool,
        min_points: int,
    ) -> str:
        """Final rule for one cycle."""
        n_samples = int(row["m"]) if pd.notna(row["m"]) else 0
        if n_samples < min_points:
            return "Unknown"

        p_pump = float(row["p_pump"])
        delta_pump = float(row["dSSR_rel_pump"])
        p_sys_key = "sys_p_holm_min" if use_holm else "sys_p_min"
        p_sys = float(row[p_sys_key]) if pd.notna(row[p_sys_key]) else np.nan
        delta_sys = float(row["sys_drel_max"]) if pd.notna(row["sys_drel_max"]) else np.nan

        pump_flag = (p_pump < alpha_pump) and (delta_pump > delta_min_pump)
        system_flag = (p_sys < alpha_sys) and (delta_sys > delta_min_sys)

        if pump_flag and not system_flag:
            return "Pump Fault"
        if system_flag and not pump_flag:
            return "System Fault"
        if pump_flag and system_flag:
            return "Pump Fault" if delta_pump >= delta_sys else "System Fault"
        return "Normal"

    def threshold_sweep(self, results: pd.DataFrame) -> Dict[str, float]:
        """Search a small threshold grid for the best balanced accuracy and macro F1."""
        cfg = self.config
        best: Optional[Dict[str, float]] = None

        def score_thresholds(alpha_p: float, delta_p: float, alpha_s: float, delta_s: float) -> Tuple[float, float]:
            predictions = results.apply(
                lambda row: self.decide_row(
                    row,
                    alpha_p,
                    delta_p,
                    alpha_s,
                    delta_s,
                    cfg.holm_bonferroni,
                    cfg.min_points_per_cycle,
                ),
                axis=1,
            )
            combined = pd.DataFrame({"truth": results["truth_clean"], "pred": predictions})
            mask = combined["truth"].isin(valid_labels) & combined["pred"].isin(valid_labels)
            if mask.sum() == 0:
                return np.nan, np.nan
            balanced_accuracy = balanced_accuracy_score(combined.loc[mask, "truth"], combined.loc[mask, "pred"])
            macro_f1 = f1_score(combined.loc[mask, "truth"], combined.loc[mask, "pred"], average="macro")
            return balanced_accuracy, macro_f1

        for alpha_p in cfg.sweep_alphas:
            for delta_p in cfg.sweep_deltas_p:
                for alpha_s in cfg.sweep_alphas:
                    for delta_s in cfg.sweep_deltas_s:
                        balanced_accuracy, macro_f1 = score_thresholds(alpha_p, delta_p, alpha_s, delta_s)
                        if not np.isfinite(balanced_accuracy):
                            continue

                        current_score = (balanced_accuracy, macro_f1)
                        if best is None or current_score > (best["balanced_accuracy"], best["macro_f1"]):
                            best = {
                                "alpha_pump": alpha_p,
                                "delta_min_pump": delta_p,
                                "alpha_sys": alpha_s,
                                "delta_min_sys": delta_s,
                                "balanced_accuracy": float(balanced_accuracy),
                                "macro_f1": float(macro_f1),
                            }

        if best is None:
            best = {
                "alpha_pump": cfg.alpha_pump,
                "delta_min_pump": cfg.delta_min_pump,
                "alpha_sys": cfg.alpha_sys,
                "delta_min_sys": cfg.delta_min_sys,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
            }

        return best

    @staticmethod
    def evaluate(results: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, str]:
        """Build confusion matrices and sklearn report."""
        cm_all = confusion_matrix(results["truth_clean"], results["pred_clean"], labels=all_labels)
        cm_all_df = pd.DataFrame(cm_all, index=[f"True {label}" for label in all_labels], columns=[f"Pred {label}" for label in all_labels])

        known_mask = results["truth_clean"].isin(valid_labels) & results["pred_clean"].isin(valid_labels)
        cm_known = confusion_matrix(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"], labels=valid_labels)
        cm_known_df = pd.DataFrame(cm_known, index=[f"True {label}" for label in valid_labels], columns=[f"Pred {label}" for label in valid_labels])

        metrics: Dict[str, float] = {}
        if known_mask.sum() > 0:
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"]))
            metrics["macro_f1"] = float(f1_score(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"], average="macro"))
            report = classification_report(results.loc[known_mask, "truth_clean"], results.loc[known_mask, "pred_clean"], labels=valid_labels, zero_division=0)
        else:
            metrics["balanced_accuracy"] = np.nan
            metrics["macro_f1"] = np.nan
            report = "No known labels overlap to compute a report."

        assert cm_all.sum() == len(results)
        return metrics, cm_all_df, cm_known_df, report

    def run(self) -> Dict[str, Path]:
        """Run the complete diagnoser workflow and write outputs."""
        cfg = self.config
        self.load()
        self.preprocess()

        per_cycle = self.per_cycle_tests()
        windows = self.window_system_tests()
        system_agg = self.aggregate_system_to_cycles(windows)

        if self.cycles is None:
            raise RuntimeError("Cycles were not loaded.")

        results = (
            self.cycles[["cycle_id", "start_s", "end_s", "truth"]]
            .merge(per_cycle, on="cycle_id", how="left")
            .merge(system_agg, on="cycle_id", how="left")
        )

        results["truth_clean"] = clean_labels(results["truth"])

        sweep_summary = None
        if cfg.do_sweep:
            sweep_summary = self.threshold_sweep(results)
            cfg.alpha_pump = sweep_summary["alpha_pump"]
            cfg.delta_min_pump = sweep_summary["delta_min_pump"]
            cfg.alpha_sys = sweep_summary["alpha_sys"]
            cfg.delta_min_sys = sweep_summary["delta_min_sys"]

        results["pred_label"] = results.apply(
            lambda row: self.decide_row(
                row,
                cfg.alpha_pump,
                cfg.delta_min_pump,
                cfg.alpha_sys,
                cfg.delta_min_sys,
                cfg.holm_bonferroni,
                cfg.min_points_per_cycle,
            ),
            axis=1,
        )

        results["pred_clean"] = clean_labels(results["pred_label"])
        p_sys_series = results["sys_p_holm_min"] if cfg.holm_bonferroni else results["sys_p_min"]

        results = results.assign(
            pump_flag=(results["p_pump"] < cfg.alpha_pump) & (results["dSSR_rel_pump"] > cfg.delta_min_pump),
            sys_flag=(p_sys_series < cfg.alpha_sys) & (results["sys_drel_max"] > cfg.delta_min_sys),
            effsize_p=results["dSSR_rel_pump"],
            effsize_s=results["sys_drel_max"],
        )

        self.results = results
        self.windows = windows

        metrics, cm_all_df, cm_known_df, report = self.evaluate(results)

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        results_out = cfg.out_dir / "per_cycle_decisions_supercycle_sysF.csv"
        windows_out = cfg.out_dir / "supercycle_windows_stats.csv"
        metrics_out = cfg.out_dir / "metrics.txt"
        config_out = cfg.out_dir / "config_used.json"
        cm_all_out = cfg.out_dir / "confusion_matrix_all.csv"
        cm_known_out = cfg.out_dir / "confusion_matrix_known.csv"

        results.to_csv(results_out, index=False)
        windows.to_csv(windows_out, index=False)
        cm_all_df.to_csv(cm_all_out)
        cm_known_df.to_csv(cm_known_out)

        with open(config_out, "w", encoding="utf-8") as fp:
            json.dump({key: str(value) for key, value in asdict(cfg).items()}, fp, indent=2)

        with open(metrics_out, "w", encoding="utf-8") as fp:
            fp.write("Configuration\n")
            fp.write("-------------\n")
            for key, value in asdict(cfg).items():
                fp.write(f"{key}: {value}\n")
            if sweep_summary:
                fp.write("\nBest threshold sweep result\n")
                fp.write("---------------------------\n")
                for key, value in sweep_summary.items():
                    fp.write(f"{key}: {value}\n")
            fp.write("\nMetrics\n")
            fp.write("-------\n")
            for key, value in metrics.items():
                fp.write(f"{key}: {value}\n")
            fp.write("\nClassification report, known labels only\n")
            fp.write("----------------------------------------\n")
            fp.write(report)

        if cfg.make_plots:
            self.plot_pvalues_and_labels(results)
            self.plot_confusion_matrix(cm_known_df, "Confusion matrix, known labels", cfg.out_dir / "confusion_matrix_known.png")
            try:
                self.plot_example_drift(results)
            except Exception:
                pass

        print("\nFull diagnoser metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"\nOutputs written to: {cfg.out_dir}")

        return {
            "results_out": results_out,
            "windows_out": windows_out,
            "metrics_out": metrics_out,
            "config_out": config_out,
            "cm_all_out": cm_all_out,
            "cm_known_out": cm_known_out,
        }

    def plot_pvalues_and_labels(self, results: pd.DataFrame) -> None:
        """Plot pump and system p-values across cycles."""
        cfg = self.config
        out_path = cfg.out_dir / "pvalues_timeline.png"

        fig, ax = plt.subplots(figsize=(11, 4))
        cycle_ids = results["cycle_id"].to_numpy()
        pump_p = results["p_pump"].to_numpy(float)
        sys_p = results["sys_p_holm_min"].to_numpy(float) if cfg.holm_bonferroni else results["sys_p_min"].to_numpy(float)

        ax.plot(cycle_ids, pump_p, marker="o", label="Pump p-value")
        ax.plot(cycle_ids, sys_p, marker="x", label="System p-value")
        ax.axhline(cfg.alpha_pump, linestyle="--", label=f"pump alpha={cfg.alpha_pump}")
        ax.axhline(cfg.alpha_sys, linestyle=":", label=f"system alpha={cfg.alpha_sys}")

        for cycle_id, pump_value, system_value, label in zip(cycle_ids, pump_p, sys_p, results["pred_clean"]):
            finite_values = [value for value in [pump_value, system_value] if np.isfinite(value)]
            if finite_values:
                ax.text(cycle_id, max(finite_values) * 1.1, label, rotation=90, va="bottom", fontsize=7)

        ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("p-value")
        ax.legend()
        ax.set_title("Pump and system p-values by cycle")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    def plot_confusion_matrix(self, cm_df: pd.DataFrame, title: str, out_path: Path) -> None:
        """Save a simple confusion-matrix heatmap."""
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        matrix = cm_df.to_numpy()
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(cm_df.shape[1]))
        ax.set_xticklabels(cm_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(cm_df.shape[0]))
        ax.set_yticklabels(cm_df.index, fontsize=8)

        vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "white" if value > 0.5 * vmax else "black"
                ax.text(j, i, int(value), ha="center", va="center", color=text_color, fontsize=8)

        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    def plot_example_drift(self, results: pd.DataFrame) -> None:
        """Plot H-Q drift for the first non-normal predicted cycle."""
        if self.cycles is None or self.nominal_data is None:
            return

        candidates = results.query("pred_clean != 'Normal'").head(1)
        if candidates.empty:
            return

        cfg = self.config
        cycle_id = int(candidates["cycle_id"].iloc[0])
        cycle_row = self.cycles.loc[self.cycles["cycle_id"] == cycle_id].iloc[0]
        block = self.nominal_data[
            (self.nominal_data["t_s"] >= float(cycle_row["start_s"]))
            & (self.nominal_data["t_s"] <= float(cycle_row["end_s"]))
        ].copy()

        Q = block["Q50"].to_numpy(float)
        H = block["H50"].to_numpy(float)
        t = block["t_rel"].to_numpy(float)

        beta_h0, _ = fit_pump_h0(Q, H)
        beta_h1, _ = fit_pump_h1(Q, H, t)
        t_norm, _, _ = standardize(t)
        q_grid = np.linspace(np.nanmin(Q), np.nanmax(Q), 100)

        a0, a1, a2, al0, al1, al2 = beta_h1
        h_early = (a0 + al0 * t_norm.min()) + (a1 + al1 * t_norm.min()) * q_grid + (a2 + al2 * t_norm.min()) * (q_grid**2)
        h_late = (a0 + al0 * t_norm.max()) + (a1 + al1 * t_norm.max()) * q_grid + (a2 + al2 * t_norm.max()) * (q_grid**2)
        h_static = beta_h0[0] + beta_h0[1] * q_grid + beta_h0[2] * (q_grid**2)

        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(Q, H, s=8, alpha=0.5, label="cycle data")
        ax.plot(q_grid, h_early, label="H1 early")
        ax.plot(q_grid, h_late, label="H1 late")
        ax.plot(q_grid, h_static, label="H0")
        ax.legend()
        ax.set_title(f"Cycle {cycle_id}: pump-curve drift")
        ax.set_xlabel("Q50 (m³/h)")
        ax.set_ylabel("H50 (m)")
        fig.tight_layout()
        fig.savefig(cfg.out_dir / f"cycle_{cycle_id}_drift.png", dpi=160)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose pump and system faults from cycle-bounded pump data.")
    parser.add_argument("--data-xlsx", type=Path, default=default_data_xlsx)
    parser.add_argument("--cycles-xlsx", type=Path, default=default_cycles_xlsx)
    parser.add_argument("--out-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--mode",
        choices=["full", "mahalanobis", "supercycle", "all"],
        default="full",
        help="Which workflow to run.",
    )
    parser.add_argument("--alpha-pump", type=float, default=None)
    parser.add_argument("--delta-pump", type=float, default=None)
    parser.add_argument("--alpha-sys", type=float, default=None)
    parser.add_argument("--delta-sys", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--min-points-per-cycle", type=int, default=None)
    parser.add_argument("--min-points-per-window", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--use-wls", action="store_true")
    parser.add_argument("--no-holm", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def build_diagnoser_config(args: argparse.Namespace) -> DiagnoserConfig:
    """Build config from command-line arguments."""
    config = DiagnoserConfig(
        data_xlsx=args.data_xlsx,
        cycles_xlsx=args.cycles_xlsx,
        out_dir=args.out_dir / "full_diagnoser" if args.mode == "all" else args.out_dir,
    )

    if args.alpha_pump is not None:
        config.alpha_pump = args.alpha_pump
    if args.delta_pump is not None:
        config.delta_min_pump = args.delta_pump
    if args.alpha_sys is not None:
        config.alpha_sys = args.alpha_sys
    if args.delta_sys is not None:
        config.delta_min_sys = args.delta_sys
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.window_stride is not None:
        config.window_stride = args.window_stride
    if args.min_points_per_cycle is not None:
        config.min_points_per_cycle = args.min_points_per_cycle
    if args.min_points_per_window is not None:
        config.min_points_per_window = args.min_points_per_window

    config.make_plots = not args.no_plots
    config.use_wls = args.use_wls
    config.holm_bonferroni = not args.no_holm
    config.do_sweep = args.sweep

    return config


def main() -> None:
    args = parse_args()

    if args.mode in {"supercycle", "all"}:
        run_supercycle_classifier(
            data_xlsx=args.data_xlsx,
            cycles_xlsx=args.cycles_xlsx,
            out_dir=args.out_dir / "supercycle_classifier" if args.mode == "all" else args.out_dir,
        )

    if args.mode in {"full", "all"}:
        config = build_diagnoser_config(args)
        diagnoser = PumpSystemDiagnoser(config)
        outputs = diagnoser.run()
        print("\nGenerated files:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
