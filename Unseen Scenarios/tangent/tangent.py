#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online pump fault detection with tangent residuals
==================================================

This script implements a three-state detector for Pump 1:

    - Normal
    - Pump Fault
    - System Fault

The main signal is the tangent residual ratio, ``Iw``. It compares the local
movement of the operating point against the tangent of the learned pump curve
and the tangent of the system curve. This dynamic signal is complemented by a
mean-flow z-score, which catches system faults that have already settled to a
new operating point and therefore have very small derivatives inside each
segment.

The workflow is:
    1. Load the simulation data.
    2. Keep valid full-speed Pump 1 samples.
    3. Split the valid stream into contiguous operating segments.
    4. Fit a healthy Pump 1 baseline curve from early data.
    5. Run the online detector over each segment.
    6. Optionally smooth the predicted states.
    7. Compare predictions with cycle labels when available.
    8. Save the per-segment results and a diagnostic plot.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Configuration
# =============================================================================

data_path = Path("data/pump_system_faultsimualtion.xlsx")
cycles_path = Path("data/Cycle-time.xlsx")
output_dir = Path("results/tangent_residual_detector")

learning_max_time_s = 12 * 60 * 60
learning_min_normal = 20

threshold_1_min_floor = 0.025
threshold_2_min_floor = 0.060
trim_top_quantile = 0.85

# Static operating-point backup. Large values indicate that the mean segment
# flow has shifted far from the learned healthy baseline.
q_deviation_z_threshold = 4.0

full_speed_hz = 49.5
min_flow_m3h = 0.1
freq_stability_window = 5
freq_stability_std_max = 0.5

min_segment_points = 20
min_segment_length = 25

friction_coeff = 0.0003
nominal_frequency_hz = 50.0
small_value = 1e-9

adaptation_window = 1000
adaptation_interval = 50

apply_smoothing = True
smoothing_window = 3

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

valid_labels = ["Normal", "Pump Fault", "System Fault"]


# =============================================================================
# Data preparation
# =============================================================================

def load_pump_data(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the simulation file and prepare the Pump 1 working dataframe."""
    df = pd.read_excel(excel_path)

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    start_time = df["Timestamp"].min()
    df["t_s"] = (df["Timestamp"] - start_time).dt.total_seconds().astype(float)

    pump_1 = df[["t_s", "Pump1_Flow_m3h", "Pump1_Head_m", "Pump1_Freq_Hz"]].copy()
    pump_1.columns = ["t_s", "Q1", "H1", "f1"]

    pump_1 = pump_1[(pump_1["Q1"] > 0) & (pump_1["H1"] > 0) & (pump_1["f1"] > 0)].copy()

    full_speed = pump_1["f1"] >= full_speed_hz
    positive_flow = pump_1["Q1"] > min_flow_m3h
    stable_frequency = (
        pump_1["f1"]
        .rolling(freq_stability_window, center=True)
        .std()
        .fillna(False)
        < freq_stability_std_max
    )

    pump_1["valid"] = full_speed & positive_flow & stable_frequency
    return df, pump_1


def contiguous_segments(mask: pd.Series, min_points: int = min_segment_points) -> list[tuple[int, int]]:
    """Convert a boolean mask into contiguous True runs."""
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = indices[0]
    previous = indices[0]

    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue

        if (previous - start + 1) >= min_points:
            segments.append((start, previous))

        start = index
        previous = index

    if (previous - start + 1) >= min_points:
        segments.append((start, previous))

    return segments


def fit_baseline_pump_curve(pump_1: pd.DataFrame) -> tuple[float, float, float]:
    """Fit the early healthy Pump 1 curve, H = a1 + a2 Q + a3 Q²."""
    baseline = pump_1[(pump_1["t_s"] < 12000) & pump_1["valid"]]

    if len(baseline) > 10:
        design = np.vstack(
            [
                np.ones(len(baseline)),
                baseline["Q1"].values,
                baseline["Q1"].values ** 2,
            ]
        ).T
        coeffs, _, _, _ = np.linalg.lstsq(design, baseline["H1"].values, rcond=None)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    return 10.0, -0.1, -0.001


# =============================================================================
# Signal processing helpers
# =============================================================================

def robust_boxcar(values: np.ndarray, window: int = 11) -> np.ndarray:
    """Median boxcar filter with edge-aware windows."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1

    result = np.zeros_like(values)
    half_window = window // 2
    n_values = len(values)

    for i in range(n_values):
        start = max(0, i - half_window)
        end = min(n_values, i + half_window + 1)
        result[i] = np.median(values[start:end])

    return result


def enhanced_central_diff(values: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Central-difference derivative with edge handling and light smoothing."""
    values = np.asarray(values, dtype=float)
    n_values = len(values)
    derivative = np.zeros_like(values, dtype=float)

    if n_values <= 1:
        return derivative

    if n_values == 2:
        derivative[:] = (values[1] - values[0]) / max(dt, small_value)
        return derivative

    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * max(dt, small_value))
    derivative[0] = (values[1] - values[0]) / max(dt, small_value)
    derivative[-1] = (values[-1] - values[-2]) / max(dt, small_value)

    return robust_boxcar(derivative, 5)


def mad(values) -> float:
    """Median absolute deviation, scaled to match a normal-distribution std."""
    return float(stats.median_abs_deviation(values, scale="normal"))


# =============================================================================
# Detector
# =============================================================================
class EnhancedFaultDetector:
    """
    Online detector with two channels.

    Channel 1 uses the tangent residual ratio. Channel 2 uses the segment mean
    flow z-score as a static operating-point check. Hysteresis is only applied
    to Iw-based system-fault detections.
    """

    def __init__(
        self,
        full_df: pd.DataFrame,
        pump_coeffs: tuple[float, float, float],
    ) -> None:
        self.full_df = full_df
        self.a1, self.a2, self.a3 = pump_coeffs

        self.learning = True
        self.clean_iw: list[float] = []
        self.clean_q: list[float] = []

        self.threshold_1 = threshold_1_min_floor
        self.threshold_2 = threshold_2_min_floor

        self.q_mu: float | None = None
        self.q_std: float | None = None

        self.recent_normal_iw = deque(maxlen=adaptation_window)
        self.recent_normal_q = deque(maxlen=adaptation_window)

        self.cycle_count = 0
        self.consecutive_faults = 0
        self.current_state = "Normal"
        self.results: list[dict] = []

    def _trimmed_thresholds(self, data) -> tuple[float, float]:
        """Compute robust Iw thresholds from a trimmed normal distribution."""
        values = np.sort(np.asarray(data, dtype=float))
        if len(values) < 8:
            return threshold_1_min_floor, threshold_2_min_floor

        trimmed = values[values <= np.quantile(values, trim_top_quantile)]
        if len(trimmed) < 5:
            trimmed = values

        center = float(np.median(trimmed))
        spread = mad(trimmed)

        threshold_1 = max(threshold_1_min_floor, center + 1.5 * spread)
        threshold_2 = max(threshold_2_min_floor, center + 3.0 * spread)

        if threshold_2 <= threshold_1:
            threshold_2 = max(threshold_1 * 1.5, threshold_1 + 0.01)

        return float(threshold_1), float(threshold_2)

    def compute_iw(self, segment: pd.DataFrame) -> tuple[float, float, float, float]:
        """Compute Iw, mean flow, and segment variability features."""
        q1 = segment["Q1"].values.astype(float)
        h1 = segment["H1"].values.astype(float)
        time_s = segment["t_s"].values.astype(float)

        if len(time_s) >= 2:
            dt = float(np.median(np.diff(time_s)))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        else:
            dt = 1.0

        q1_smooth = robust_boxcar(q1, 15)
        h1_smooth = robust_boxcar(h1, 15)

        dq = enhanced_central_diff(q1_smooth, dt)
        dh = enhanced_central_diff(h1_smooth, dt)

        pump_slope = self.a2 + 2.0 * self.a3 * q1_smooth

        total_flow = (
            self.full_df.loc[segment.index, "Pump1_Flow_m3h"]
            + self.full_df.loc[segment.index, "Pump2_Flow_m3h"]
            + self.full_df.loc[segment.index, "Pump3_Flow_m3h"]
        ).values.astype(float)
        total_flow_smooth = robust_boxcar(total_flow, 15)

        system_slope = 2.0 * friction_coeff * total_flow_smooth

        pump_residual = dh - pump_slope * dq
        system_residual = dh - system_slope * dq

        numerator = np.percentile(np.abs(pump_residual), 75)
        denominator = np.percentile(np.abs(pump_residual) + np.abs(system_residual), 75)
        iw = float(numerator / denominator) if denominator > small_value else 0.5

        flow_std = float(np.std(q1_smooth))
        head_std = float(np.std(h1_smooth))
        stability = min(1.0, 2.0 / (flow_std + head_std + 0.1))
        enhanced_iw = iw * stability

        mean_q = float(np.mean(q1))
        return enhanced_iw, mean_q, flow_std, head_std

    def process_segment(self, segment: pd.DataFrame, cycle_id: int) -> dict:
        """Process one segment and return the online detector output."""
        iw, mean_q, flow_std, head_std = self.compute_iw(segment)

        start_s = float(segment["t_s"].iloc[0])
        end_s = float(segment["t_s"].iloc[-1])
        q_z = 0.0

        if self.learning:
            if len(self.clean_iw) >= 6:
                z_score = abs(iw - np.median(self.clean_iw)) / (mad(self.clean_iw) + small_value)
            else:
                z_score = 0.0

            if z_score <= 3.0 and iw < threshold_2_min_floor:
                self.clean_iw.append(iw)
                self.clean_q.append(mean_q)
                state = "Normal"
            elif iw >= threshold_2_min_floor:
                state = "Pump Fault"
            elif iw >= threshold_1_min_floor:
                state = "System Fault"
            else:
                state = "Normal"

            if len(self.clean_iw) >= learning_min_normal or end_s > learning_max_time_s:
                self.threshold_1, self.threshold_2 = self._trimmed_thresholds(self.clean_iw)
                self.q_mu = float(np.mean(self.clean_q))
                self.q_std = max(0.01, float(np.std(self.clean_q)))
                self.learning = False
                print(
                    f"[Learning] t1={self.threshold_1:.4f}, "
                    f"t2={self.threshold_2:.4f}, "
                    f"q_mu={self.q_mu:.1f}, q_std={self.q_std:.3f}, "
                    f"clean_normals={len(self.clean_iw)}"
                )

        else:
            if iw >= self.threshold_2:
                iw_state = "Pump Fault"
            elif iw >= self.threshold_1:
                iw_state = "System Fault"
            else:
                iw_state = "Normal"

            q_z = (mean_q - self.q_mu) / self.q_std if self.q_mu else 0.0
            q_anomaly = abs(q_z) > q_deviation_z_threshold

            if iw_state == "Pump Fault":
                state = "Pump Fault"
                self.consecutive_faults += 1

            elif iw_state == "System Fault":
                state = "System Fault"
                self.consecutive_faults += 1
                if self.current_state == "Normal" and self.consecutive_faults < 2:
                    state = "Normal"
                    self.consecutive_faults -= 1

            elif q_anomaly:
                state = "System Fault"
                self.consecutive_faults += 1

            else:
                state = "Normal"
                self.consecutive_faults = 0

            self.current_state = state

            if state == "Normal":
                self.recent_normal_iw.append(iw)
                self.recent_normal_q.append(mean_q)

            self.cycle_count += 1
            if self.cycle_count % adaptation_interval == 0 and len(self.recent_normal_iw) > 20:
                self.threshold_1, self.threshold_2 = self._trimmed_thresholds(
                    list(self.recent_normal_iw)
                )
                self.q_mu = float(np.mean(list(self.recent_normal_q)))
                self.q_std = max(0.01, float(np.std(list(self.recent_normal_q))))

        if state == "Normal":
            confidence = 1.0 - min(1.0, iw / (self.threshold_1 + small_value))
        elif state == "System Fault":
            confidence = min(
                1.0,
                max(iw - self.threshold_1, abs(q_z) / 10.0)
                / max(self.threshold_2 - self.threshold_1, small_value),
            )
        else:
            confidence = min(1.0, (iw - self.threshold_2) / max(0.5 - self.threshold_2, small_value))

        result = {
            "cycle_id": cycle_id,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "Iw": float(iw),
            "mean_Q": float(mean_q),
            "Q_z": float(q_z),
            "flow_std": float(flow_std),
            "head_std": float(head_std),
            "fault_state": state,
            "confidence": float(confidence),
            "T1": float(self.threshold_1),
            "T2": float(self.threshold_2),
            "learning_phase": self.learning,
        }
        self.results.append(result)
        return result


# =============================================================================
# Evaluation and plotting
# =============================================================================
def smooth_state_sequence(states: pd.Series, window: int = smoothing_window) -> list[str]:
    """Apply a centered rolling majority vote to the state sequence."""
    state_to_int = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    int_to_state = {value: key for key, value in state_to_int.items()}

    state_ints = states.map(state_to_int).values
    smoothed = []

    for i in range(len(state_ints)):
        start = max(0, i - window // 2)
        end = min(len(state_ints), i + window // 2 + 1)
        values, counts = np.unique(state_ints[start:end], return_counts=True)
        smoothed.append(int_to_state[values[np.argmax(counts)]])

    return smoothed


def add_ground_truth(results: pd.DataFrame, cycle_file: Path) -> pd.DataFrame:
    """Attach the best-overlapping cycle label to each detected segment."""
    results = results.copy()

    if not cycle_file.exists():
        results["truth"] = "Unknown"
        return results

    cycles = pd.read_excel(cycle_file)
    cycles = cycles.rename(
        columns={"start_t_sec": "start_s", "stop_t_sec": "end_s", "label": "truth"}
    )
    cycles["cycle_id"] = np.arange(1, len(cycles) + 1)

    truth_labels = []
    for _, row in results.iterrows():
        best_overlap = 0.0
        best_label = "Normal"

        for _, cycle in cycles.iterrows():
            overlap = max(
                0,
                min(row["end_s"], cycle["end_s"]) - max(row["start_s"], cycle["start_s"]),
            ) / max(row["end_s"] - row["start_s"], 1)

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = cycle["truth"]

        truth_labels.append(label_map.get(str(best_label).strip(), best_label))

    results["truth"] = truth_labels
    return results


def print_evaluation(results: pd.DataFrame) -> None:
    """Print confusion matrices and classification reports."""
    mask = results["truth"].isin(valid_labels)
    if not mask.any():
        print("No valid ground-truth labels were found. Skipping evaluation.")
        return

    for column_name, display_name in [
        ("fault_state", "Raw"),
        ("fault_state_smooth", "Smoothed"),
    ]:
        print(f"\n{'=' * 60}")
        print(f"{display_name} predictions")
        print(f"{'=' * 60}")

        cm = confusion_matrix(
            results.loc[mask, "truth"],
            results.loc[mask, column_name],
            labels=valid_labels,
        )
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {label}" for label in valid_labels],
            columns=[f"Pred {label}" for label in valid_labels],
        )
        print(cm_df.to_string())

        print(
            classification_report(
                results.loc[mask, "truth"],
                results.loc[mask, column_name],
                labels=valid_labels,
                zero_division=0,
            )
        )

        balanced_accuracy = balanced_accuracy_score(
            results.loc[mask, "truth"],
            results.loc[mask, column_name],
        )
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")


def save_diagnostic_plot(results: pd.DataFrame, figure_path: Path) -> None:
    """Save a compact diagnostic plot of Iw, Q-z score, and fault states."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    threshold_1_plot = float(results["T1"].iloc[-1])
    threshold_2_plot = float(results["T2"].iloc[-1])

    ax = axes[0]
    ax.plot(results["start_s"], results["Iw"], "b.", alpha=0.7, label="Iw")
    ax.axhline(
        threshold_1_plot,
        color="r",
        linestyle="--",
        label=f"T1 ({threshold_1_plot:.4f})",
    )
    ax.axhline(
        threshold_2_plot,
        color="g",
        linestyle="--",
        label=f"T2 ({threshold_2_plot:.4f})",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Iw")
    ax.set_title("Tangent residual ratio")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(results["start_s"], results["Q_z"], "r.-", alpha=0.7, label="Q z-score")
    ax.axhline(
        q_deviation_z_threshold,
        color="gray",
        linestyle="--",
        label=f"±{q_deviation_z_threshold}",
    )
    ax.axhline(-q_deviation_z_threshold, color="gray", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Q z-score")
    ax.set_title("Operating-point deviation")
    ax.legend()
    ax.grid(True)

    state_map = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    ax = axes[2]

    if "truth" in results.columns and results["truth"].isin(valid_labels).any():
        truth_states = [state_map[state] for state in results["truth"]]
        ax.step(
            results["start_s"],
            truth_states,
            where="post",
            color="red",
            linestyle=":",
            label="Truth",
        )

    predicted_states = [state_map[state] for state in results["fault_state"]]
    ax.step(
        results["start_s"],
        predicted_states,
        where="post",
        color="black",
        label="Predicted",
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(valid_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("Classification")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main script
# =============================================================================
def main() -> pd.DataFrame:
    """Run the complete detection workflow."""
    start_wall_time = time.time()

    print("Loading simulation data...")
    full_df, pump_1 = load_pump_data(data_path)
    print(f"Prepared {len(pump_1)} Pump 1 samples; {int(pump_1['valid'].sum())} are valid.")

    segments = contiguous_segments(pump_1["valid"])
    print(f"Found {len(segments)} valid operating segments.")

    pump_coeffs = fit_baseline_pump_curve(pump_1)
    print(
        "Baseline pump curve: "
        f"H = {pump_coeffs[0]:.4f} + {pump_coeffs[1]:.6f} Q "
        f"+ {pump_coeffs[2]:.8f} Q²"
    )

    detector = EnhancedFaultDetector(full_df, pump_coeffs)
    print("Running enhanced fault detection...")

    for cycle_id, (start_idx, end_idx) in enumerate(segments):
        segment = pump_1.iloc[start_idx : end_idx + 1].copy()
        if len(segment) < min_segment_length:
            continue
        detector.process_segment(segment, cycle_id)

    results = pd.DataFrame(detector.results)
    print(f"Processed {len(results)} segments.")

    if apply_smoothing and len(results):
        results["fault_state_smooth"] = smooth_state_sequence(results["fault_state"])
    elif len(results):
        results["fault_state_smooth"] = results["fault_state"]

    if len(results):
        results = add_ground_truth(results, cycles_path)
        print_evaluation(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "enhanced_fault_detection_results.csv"
    output_plot = output_dir / "enhanced_fault_detection_plot.png"

    results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    if len(results):
        save_diagnostic_plot(results, output_plot)
        print(f"Plot saved to: {output_plot}")

    elapsed_s = time.time() - start_wall_time
    print(f"Finished in {elapsed_s:.1f} seconds.")
    return results


if __name__ == "__main__":
    main()
