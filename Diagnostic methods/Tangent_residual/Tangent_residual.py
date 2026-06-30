#!/usr/bin/env python3
# %%
"""
Online tangent-residual fault detection for pump-station data.

The script detects three operating states from Pump 1 data:
Normal, System Fault, and Pump Fault.

Two detector variants are kept in the file:

1. ``ContaminationSafeDetector``
   The original contamination-safe learner. It learns thresholds from clean
   normal segments, excludes suspected contaminated segments during learning,
   and adapts from recent normal segments.

2. ``VarianceGateDetector``
   The modified version. It keeps the same tangent-residual method and adds a
   variance gate near the lower threshold to better separate mild System Fault
   behavior from Normal operation. This is the default variant.

The detector uses the tangent-residual ratio ``Iw`` computed from smoothed
within-segment derivatives of flow and head. Ground-truth labels are assigned
from the same fault windows used in the original experiment.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Paths
input_file = Path("data/pump_system_faultsimualtion.xlsx")
output_dir = Path("results/tangent_residual_detection")
output_dir.mkdir(parents=True, exist_ok=True)

# Choose the detector used by main(): "v1" or "v2".
detector_variant = "v2"

# Learning and adaptation
learning_max_time_s = 12 * 60 * 60
learning_min_normal_cycles_v1 = 12
learning_min_normal_cycles_v2 = 20
learning_outlier_guard_v1 = 3.5
learning_outlier_guard_v2 = 3.0

adaptation_window_v1 = 500
adaptation_window_v2 = 1_000
adaptation_interval = 50

# Provisional thresholds used before learning has enough clean normals
provisional_t1_default = 0.020
provisional_t2_default = 0.060

# Threshold computation
trim_top_q_v1 = 0.90
tim_top_q_v2 = 0.85
t1_floor_v1 = 0.010
t2_floor_v1 = 0.050
t1_floor_v2 = 0.025
t2_floor_v2 = 0.060

# Variance gate used by the modified detector
variance_gate_rel_margin = 0.20
std_z_up = 2.5
std_z_down = -0.5

# Valid operating segments
full_speed_hz = 49.5
min_flow_m3h = 0.1
freq_stability_roll = 5
freq_stability_std_max = 0.5
min_segment_points_to_keep = 20
min_segment_len_to_process = 25

# Known system curve constant
friction_coeff = 0.0006
nominal_freq_hz = 50.0

# Ground-truth fault windows, in seconds from the first timestamp
pump_fault_window = (12_000.0, 21_000.0)
system_fault_window = (86_400.0 + 40_000.0, 86_400.0 + 61_600.0)

# Post-processing and run behavior
apply_state_smoothing = True
smooth_window = 3
processing_delay_s = 0.01

eps = 1e-9
seed = 13


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def load_pump_data(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the simulation table and prepare valid Pump 1 operating samples."""
    df = pd.read_excel(file_path)

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    start_time = df["Timestamp"].min()
    df["t_s"] = (df["Timestamp"] - start_time).dt.total_seconds().astype(float)

    pump_1 = df[["t_s", "Pump1_Flow_m3h", "Pump1_Head_m", "Pump1_Freq_Hz"]].copy()
    pump_1.columns = ["t_s", "q1", "h1", "f1"]

    pump_1 = pump_1[(pump_1["q1"] > 0) & (pump_1["h1"] > 0) & (pump_1["f1"] > 0)].copy()

    full_speed = pump_1["f1"] >= full_speed_hz
    positive_flow = pump_1["q1"] > min_flow_m3h
    stable_frequency = (
        pump_1["f1"].rolling(freq_stability_roll, center=True).std() < freq_stability_std_max
    )
    pump_1["valid"] = full_speed & positive_flow & stable_frequency.fillna(False)

    return df, pump_1


def contiguous_segments(mask: pd.Series, min_points: int = min_segment_points_to_keep) -> list[tuple[int, int]]:
    """Convert a boolean mask into consecutive True segments."""
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = idx[0]
    previous = idx[0]

    for current in idx[1:]:
        if current == previous + 1:
            previous = current
            continue

        if previous - start + 1 >= min_points:
            segments.append((start, previous))
        start = current
        previous = current

    if previous - start + 1 >= min_points:
        segments.append((start, previous))

    return segments


def fit_baseline_pump_curve(pump_1: pd.DataFrame) -> tuple[float, float, float]:
    """Fit the healthy baseline pump curve H = a1 + a2*Q + a3*Q^2."""
    baseline = pump_1[(pump_1["t_s"] < 12_000) & pump_1["valid"]]

    if len(baseline) > 10:
        x_pump = np.vstack([
            np.ones(len(baseline)),
            baseline["q1"].values,
            baseline["q1"].values ** 2,
        ]).T
        y_pump = baseline["h1"].values
        coeffs, *_ = np.linalg.lstsq(x_pump, y_pump, rcond=None)
        return tuple(float(v) for v in coeffs)

    return 10.0, -0.1, -0.001


# -----------------------------------------------------------------------------
# Signal processing helpers
# -----------------------------------------------------------------------------
def robust_boxcar(x: np.ndarray, window: int = 11) -> np.ndarray:
    """Median boxcar filter with edge handling."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1

    result = np.zeros_like(x)
    half = window // 2
    n = len(x)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(x[start:end])

    return result


def enhanced_central_diff(x: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Central-difference derivative with edge handling and light median smoothing."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    derivative = np.zeros_like(x, dtype=float)

    if n == 1:
        return derivative
    if n == 2:
        derivative[:] = (x[1] - x[0]) / max(dt, eps)
        return derivative

    derivative[1:-1] = (x[2:] - x[:-2]) / (2.0 * max(dt, eps))
    derivative[0] = (x[1] - x[0]) / max(dt, eps)
    derivative[-1] = (x[-1] - x[-2]) / max(dt, eps)

    return robust_boxcar(derivative, 5)


def mad(values: Iterable[float]) -> float:
    """Median absolute deviation scaled to match normal-distribution sigma."""
    return float(stats.median_abs_deviation(values, scale="normal"))


def robust_z(value: float, reference_values: Iterable[float]) -> float:
    """MAD-based z-score for one value relative to a reference distribution."""
    ref = np.asarray(list(reference_values), dtype=float)
    if ref.size < 6:
        return 0.0
    center = float(np.median(ref))
    scale = mad(ref) + eps
    return float((value - center) / scale)


# -----------------------------------------------------------------------------
# Detector variants
# -----------------------------------------------------------------------------
class ContaminationSafeDetector:
    """
    Contamination-safe tangent-residual detector.

    The learning phase excludes suspected contaminated segments, uses provisional
    thresholds for early detection, then computes final thresholds from trimmed
    clean-normal Iw values. Adaptation after learning is also based on recent
    normal segments only.
    """

    def __init__(
        self,
        full_data: pd.DataFrame,
        pump_coeffs: tuple[float, float, float],
        learning_min_normal_cycles: int = learning_min_normal_cycles_v1,
        learning_outlier_guard: float = learning_outlier_guard_v1,
        adaptation_window: int = adaptation_window_v1,
        trim_top_q: float = trim_top_q_v1,
        t1_floor: float = t1_floor_v1,
        t2_floor: float = t2_floor_v1,
    ):
        self.full_data = full_data
        self.a1, self.a2, self.a3 = pump_coeffs

        self.learning_min_normal_cycles = learning_min_normal_cycles
        self.learning_outlier_guard = learning_outlier_guard
        self.trim_top_q = trim_top_q
        self.t1_floor = t1_floor
        self.t2_floor = t2_floor

        self.learning_phase = True
        self.clean_normal_iw: list[float] = []
        self.provisional_t1 = provisional_t1_default
        self.provisional_t2 = provisional_t2_default
        self.t1 = np.nan
        self.t2 = np.nan

        self.recent_normal_iw = deque(maxlen=adaptation_window)
        self.cycle_count = 0
        self.consecutive_faults = 0
        self.current_state = "Normal"

        self.results: list[dict] = []
        self.suspect_excluded = 0

    def _trimmed_thresholds(self, iw_data: Iterable[float]) -> tuple[float, float]:
        """Compute T1 and T2 from trimmed clean-normal Iw values."""
        x = np.asarray(list(iw_data), dtype=float)
        if x.size < 8:
            t1 = max(self.t1_floor, float(self.provisional_t1))
            t2 = max(self.t2_floor, float(self.provisional_t2))
            if t2 <= t1:
                t2 = max(t1 * 1.5, t1 + 0.01)
            return float(t1), float(t2)

        x_sorted = np.sort(x)
        cut = np.quantile(x_sorted, self.trim_top_q)
        trimmed = x_sorted[x_sorted <= cut]
        if trimmed.size < 5:
            trimmed = x_sorted

        center = np.median(trimmed)
        spread = mad(trimmed)
        t1 = center + 1.5 * spread
        t2 = center + 3.0 * spread

        if t2 <= t1:
            t2 = max(t1 * 1.5, t1 + 0.01)

        t1 = max(self.t1_floor, t1)
        t2 = max(self.t2_floor, t2)
        return float(t1), float(t2)

    def _update_provisional_thresholds(self) -> None:
        """Refresh provisional thresholds once enough clean normals are available."""
        if len(self.clean_normal_iw) >= 6:
            self.provisional_t1, self.provisional_t2 = self._trimmed_thresholds(
                self.clean_normal_iw
            )

    def calculate_iw(self, segment: pd.DataFrame) -> tuple[float, float, float]:
        """Compute enhanced Iw, flow standard deviation, and head standard deviation."""
        q1 = segment["q1"].values.astype(float)
        h1 = segment["h1"].values.astype(float)
        t = segment["t_s"].values.astype(float)

        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
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
            self.full_data.loc[segment.index, "Pump1_Flow_m3h"]
            + self.full_data.loc[segment.index, "Pump2_Flow_m3h"]
            + self.full_data.loc[segment.index, "Pump3_Flow_m3h"]
        ).values.astype(float)
        total_flow_smooth = robust_boxcar(total_flow, 15)

        system_slope = 2.0 * friction_coeff * total_flow_smooth

        psi_pump = dh - pump_slope * dq
        psi_system = dh - system_slope * dq

        numerator = np.percentile(np.abs(psi_pump), 75)
        denominator = np.percentile(np.abs(psi_pump) + np.abs(psi_system), 75)
        iw = float(numerator / denominator) if denominator > eps else 0.5

        flow_std = float(np.std(q1_smooth))
        head_std = float(np.std(h1_smooth))
        stability = min(1.0, 2.0 / (flow_std + head_std + 0.1))
        enhanced_iw = iw * stability

        return float(enhanced_iw), flow_std, head_std

    def _learn_or_classify(self, iw: float, flow_std: float, head_std: float, end_s: float) -> tuple[str, float]:
        """Run the learning-phase decision logic."""
        self._update_provisional_thresholds()

        if len(self.clean_normal_iw) >= 6:
            center = np.median(self.clean_normal_iw)
            spread = mad(self.clean_normal_iw) + eps
            z_score = abs(iw - center) / spread
        else:
            z_score = 0.0

        is_suspect = False
        if len(self.clean_normal_iw) >= 6 and z_score > self.learning_outlier_guard:
            is_suspect = True
        if iw >= max(self.provisional_t2, provisional_t2_default):
            is_suspect = True

        if is_suspect:
            self.suspect_excluded += 1
            if iw >= self.provisional_t2:
                state = "Pump Fault"
                self.consecutive_faults += 1
            elif iw >= self.provisional_t1:
                state = "System Fault"
                self.consecutive_faults += 1
            else:
                state = "Normal"
                self.consecutive_faults = 0
        else:
            self._accept_clean_normal(iw, flow_std, head_std)
            state = "Normal"
            self.consecutive_faults = 0

        confidence = self._confidence(state, iw, self.provisional_t1, self.provisional_t2)

        if len(self.clean_normal_iw) >= self.learning_min_normal_cycles or end_s > learning_max_time_s:
            self.t1, self.t2 = self._trimmed_thresholds(self.clean_normal_iw)
            self.learning_phase = False
            print(
                "Learning complete. "
                f"Clean normals={len(self.clean_normal_iw)}, "
                f"excluded suspects={self.suspect_excluded}. "
                f"Set T1={self.t1:.4f}, T2={self.t2:.4f}"
            )

        return state, confidence

    def _online_classify(self, iw: float, flow_std: float, head_std: float) -> tuple[str, float]:
        """Run the post-learning online decision logic."""
        if iw < self.t1:
            new_state = "Normal"
            self.consecutive_faults = 0
        elif self.t1 <= iw < self.t2:
            new_state = "System Fault"
            self.consecutive_faults += 1
        else:
            new_state = "Pump Fault"
            self.consecutive_faults += 1

        if new_state != self.current_state and self.consecutive_faults < 2:
            new_state = self.current_state

        self.current_state = new_state
        state = self.current_state
        confidence = self._confidence(state, iw, self.t1, self.t2)

        if state == "Normal":
            self._accept_recent_normal(iw, flow_std, head_std)

        self.cycle_count += 1
        if self.cycle_count % adaptation_interval == 0 and len(self.recent_normal_iw) > 20:
            self.t1, self.t2 = self._trimmed_thresholds(list(self.recent_normal_iw))

        return state, confidence

    @staticmethod
    def _confidence(state: str, iw: float, t1: float, t2: float) -> float:
        """Confidence score based on the active thresholds."""
        if state == "Normal":
            return float(1.0 - min(1.0, iw / (t1 + eps)))
        if state == "System Fault":
            return float(min(1.0, (iw - t1) / max(t2 - t1, eps)))
        return float(min(1.0, (iw - t2) / max(0.5 - t2, eps)))

    def _accept_clean_normal(self, iw: float, flow_std: float, head_std: float) -> None:
        """Store one clean normal segment during learning."""
        self.clean_normal_iw.append(float(iw))

    def _accept_recent_normal(self, iw: float, flow_std: float, head_std: float) -> None:
        """Store one normal segment for adaptation after learning."""
        self.recent_normal_iw.append(float(iw))

    def process_segment(self, segment: pd.DataFrame, cycle_id: int) -> dict:
        """Process one segment and return the detector result."""
        iw, flow_std, head_std = self.calculate_iw(segment)

        start_s = float(segment["t_s"].iloc[0])
        end_s = float(segment["t_s"].iloc[-1])
        duration_s = end_s - start_s

        if self.learning_phase:
            state, confidence = self._learn_or_classify(iw, flow_std, head_std, end_s)
        else:
            state, confidence = self._online_classify(iw, flow_std, head_std)

        result = {
            "cycle_id": cycle_id,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "iw": float(iw),
            "flow_std": float(flow_std),
            "head_std": float(head_std),
            "fault_state": state,
            "confidence": float(confidence),
            "t1": float(self.t1 if not self.learning_phase else self.provisional_t1),
            "t2": float(self.t2 if not self.learning_phase else self.provisional_t2),
            "learning_phase": self.learning_phase,
        }
        self.results.append(result)
        return result


class VarianceGateDetector(ContaminationSafeDetector):
    """Modified detector with the variance gate near T1."""

    def __init__(self, full_data: pd.DataFrame, pump_coeffs: tuple[float, float, float]):
        super().__init__(
            full_data=full_data,
            pump_coeffs=pump_coeffs,
            learning_min_normal_cycles=learning_min_normal_cycles_v2,
            learning_outlier_guard=learning_outlier_guard_v2,
            adaptation_window=adaptation_window_v2,
            trim_top_q=tim_top_q_v2,
            t1_floor=t1_floor_v2,
            t2_floor=t2_floor_v2,
        )
        self.clean_normal_std_sum: list[float] = []
        self.recent_normal_std_sum = deque(maxlen=adaptation_window_v2)

    def _accept_clean_normal(self, iw: float, flow_std: float, head_std: float) -> None:
        self.clean_normal_iw.append(float(iw))
        self.clean_normal_std_sum.append(float(flow_std + head_std))

    def _accept_recent_normal(self, iw: float, flow_std: float, head_std: float) -> None:
        self.recent_normal_iw.append(float(iw))
        self.recent_normal_std_sum.append(float(flow_std + head_std))

    def _variance_gate_adjustment(
        self,
        iw: float,
        flow_std: float,
        head_std: float,
        base_state: str,
    ) -> tuple[str, float]:
        """Adjust Normal/System Fault decisions near T1 using segment variability."""
        if not np.isfinite(self.t1) or self.t1 <= 0:
            return base_state, 0.0

        margin = variance_gate_rel_margin * self.t1
        if abs(iw - self.t1) > margin:
            return base_state, 0.0

        reference = list(self.recent_normal_std_sum) or self.clean_normal_std_sum
        if len(reference) < 6:
            return base_state, 0.0

        gate_score = robust_z(flow_std + head_std, reference)

        if gate_score >= std_z_up and base_state == "Normal":
            return "System Fault", gate_score
        if gate_score <= std_z_down and base_state == "System Fault":
            return "Normal", gate_score

        return base_state, gate_score

    def _online_classify(self, iw: float, flow_std: float, head_std: float) -> tuple[str, float]:
        if iw < self.t1:
            new_state = "Normal"
            self.consecutive_faults = 0
        elif self.t1 <= iw < self.t2:
            new_state = "System Fault"
            self.consecutive_faults += 1
        else:
            new_state = "Pump Fault"
            self.consecutive_faults += 1

        new_state, _ = self._variance_gate_adjustment(iw, flow_std, head_std, new_state)

        if new_state != self.current_state and self.current_state == "Normal":
            if self.consecutive_faults < 2 and new_state in ("System Fault", "Pump Fault"):
                new_state = self.current_state

        self.current_state = new_state
        state = self.current_state
        confidence = self._confidence(state, iw, self.t1, self.t2)

        if state == "Normal":
            self._accept_recent_normal(iw, flow_std, head_std)

        self.cycle_count += 1
        if self.cycle_count % adaptation_interval == 0 and len(self.recent_normal_iw) > 20:
            self.t1, self.t2 = self._trimmed_thresholds(list(self.recent_normal_iw))

        return state, confidence


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return the fraction of interval a covered by interval b."""
    if a1 <= a0:
        return 0.0
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0.0, right - left) / (a1 - a0)


def assign_truth_labels(results: pd.DataFrame) -> pd.Series:
    """Assign truth labels from the predefined pump and system fault windows."""
    truth_labels = []
    for _, row in results.iterrows():
        start_s = row["start_s"]
        end_s = row["end_s"]
        if overlap_ratio(start_s, end_s, *pump_fault_window) > 0.5:
            truth_labels.append("Pump Fault")
        elif overlap_ratio(start_s, end_s, *system_fault_window) > 0.5:
            truth_labels.append("System Fault")
        else:
            truth_labels.append("Normal")
    return pd.Series(truth_labels, index=results.index)


def smooth_states(results: pd.DataFrame, window: int = smooth_window) -> pd.Series:
    """Apply rolling majority smoothing to the predicted state sequence."""
    state_to_int = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    int_to_state = {value: key for key, value in state_to_int.items()}
    state_values = results["fault_state"].map(state_to_int).values

    smoothed = []
    for i in range(len(state_values)):
        start = max(0, i - window // 2)
        end = min(len(state_values), i + window // 2 + 1)
        values, counts = np.unique(state_values[start:end], return_counts=True)
        smoothed.append(int_to_state[values[np.argmax(counts)]])

    return pd.Series(smoothed, index=results.index)


def confusion_df(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str]) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame."""
    cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for truth, pred in zip(y_true, y_pred):
        if truth in labels and pred in labels:
            cm.loc[truth, pred] += 1
    return cm


def print_precision_recall_f1(cm: pd.DataFrame, labels: list[str], title: str = "") -> None:
    """Print accuracy, precision, recall, and F1 for each class."""
    if title:
        print(title)

    accuracy = cm.values.trace() / max(cm.values.sum(), 1)
    print(f"Accuracy: {accuracy:.3f}")

    for label in labels:
        true_positive = cm.loc[label, label]
        false_positive = cm[label].sum() - true_positive
        false_negative = cm.loc[label].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        print(f"{label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
    """Apply a compact journal-style matplotlib setup."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_detection_summary(results: pd.DataFrame, labels: list[str], output_path: Path) -> None:
    """Save the three-panel Iw/state/indicator diagnostic plot."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    t1_plot = float(results["t1"].iloc[-1]) if len(results) else np.nan
    t2_plot = float(results["t2"].iloc[-1]) if len(results) else np.nan

    ax = axes[0]
    ax.plot(results["start_s"], results["iw"], "b.", alpha=0.7, label="Iw values")
    if np.isfinite(t1_plot):
        ax.axhline(t1_plot, color="r", linestyle="--", label=f"T1 ({t1_plot:.4f})")
    if np.isfinite(t2_plot):
        ax.axhline(t2_plot, color="g", linestyle="--", label=f"T2 ({t2_plot:.4f})")
    ax.axvspan(pump_fault_window[0], pump_fault_window[1], alpha=0.2, color="red", label="Pump Fault region")
    ax.axvspan(system_fault_window[0], system_fault_window[1], alpha=0.2, color="orange", label="System Fault region")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Iw value")
    ax.set_title("Tangent-residual fault detection")
    ax.legend()

    state_map = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    raw_states = [state_map[s] for s in results["fault_state"]]
    smooth_values = [state_map[s] for s in results["fault_state_smooth"]]
    truth_states = [state_map[s] for s in results["truth"]]
    confidences = results["confidence"].values if len(results) else []

    ax = axes[1]
    for i in range(len(raw_states) - 1):
        x0 = results["start_s"].iloc[i]
        x1 = results["start_s"].iloc[i + 1]
        y = raw_states[i]
        ax.plot([x0, x1], [y, y], color="blue", alpha=float(confidences[i]), linewidth=3)
    ax.step(results["start_s"], smooth_values, where="post", color="black", alpha=0.6, label="Smoothed state")
    ax.step(results["start_s"], truth_states, where="post", color="red", linestyle=":", alpha=0.7, label="Ground truth")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fault state")
    ax.set_title("Predicted state and ground truth")
    ax.legend()

    ax = axes[2]
    ax.plot(results["start_s"], results["flow_std"], "g-", alpha=0.7, label="Flow std")
    ax.plot(results["start_s"], results["head_std"], "m-", alpha=0.7, label="Head std")
    ax.axvspan(pump_fault_window[0], pump_fault_window[1], alpha=0.2, color="red")
    ax.axvspan(system_fault_window[0], system_fault_window[1], alpha=0.2, color="orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Standard deviation")
    ax.set_title("Additional segment indicators")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fault_states(results: pd.DataFrame, output_pdf: Path, output_eps: Path) -> None:
    """Save the compact fault-state comparison figure."""
    set_plot_style()

    fig, ax = plt.subplots(figsize=(6, 3.2))
    state_map = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    raw_states = [state_map[s] for s in results["fault_state"]]
    smooth_states_values = [state_map[s] for s in results["fault_state_smooth"]]
    truth_states = [state_map[s] for s in results["truth"]]
    confidences = results["confidence"].values if len(results) else []

    for i in range(len(raw_states) - 1):
        x0 = results["start_s"].iloc[i]
        x1 = results["start_s"].iloc[i + 1]
        y = raw_states[i]
        ax.plot([x0, x1], [y, y], color="C0", alpha=float(confidences[i]), linewidth=3, solid_capstyle="butt")

    ax.step(results["start_s"], truth_states, where="post", color="red", linestyle=":", alpha=0.9, label="True labels")
    ax.step(results["start_s"], smooth_states_values, where="post", color="black", linestyle="-", alpha=0.7, label="Predicted state")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Normal", "System Fault", "Pump Fault"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fault state")
    ax.legend(frameon=True, loc="upper right")

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)

    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    with mpl.rc_context({"text.usetex": False}):
        fig.savefig(output_eps, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run_detector(
    full_data: pd.DataFrame,
    pump_1: pd.DataFrame,
    segments: list[tuple[int, int]],
    pump_coeffs: tuple[float, float, float],
) -> pd.DataFrame:
    """Run the selected detector variant over all valid segments."""
    if detector_variant.lower() == "v1":
        detector: ContaminationSafeDetector = ContaminationSafeDetector(full_data, pump_coeffs)
    else:
        detector = VarianceGateDetector(full_data, pump_coeffs)

    online_results = []
    print(f"Starting tangent-residual fault detection ({detector_variant})...")

    for cycle_id, (start_idx, end_idx) in enumerate(segments):
        segment = pump_1.iloc[start_idx:end_idx + 1].copy()
        if len(segment) < min_segment_len_to_process:
            continue

        online_results.append(detector.process_segment(segment, cycle_id))

        if processing_delay_s > 0:
            time.sleep(processing_delay_s)

    print("Processing complete.")
    results = pd.DataFrame(online_results)
    results.attrs["detector"] = detector
    return results


def main() -> None:
    full_data, pump_1 = load_pump_data(input_file)
    segments = contiguous_segments(pump_1["valid"], min_points=min_segment_points_to_keep)
    pump_coeffs = fit_baseline_pump_curve(pump_1)

    print(f"Loaded {len(full_data):,} rows and found {len(segments):,} valid segments.")
    print(
        "Baseline pump curve: "
        f"H = {pump_coeffs[0]:.4f} + {pump_coeffs[1]:.6f} Q + {pump_coeffs[2]:.8f} Q^2"
    )

    results = run_detector(full_data, pump_1, segments, pump_coeffs)

    if results.empty:
        print("No valid segments were processed.")
        return

    if apply_state_smoothing:
        results["fault_state_smooth"] = smooth_states(results, smooth_window)
    else:
        results["fault_state_smooth"] = results["fault_state"]

    results["truth"] = assign_truth_labels(results)

    labels = ["Normal", "System Fault", "Pump Fault"]
    cm_raw = confusion_df(results["truth"], results["fault_state"], labels)
    cm_smooth = confusion_df(results["truth"], results["fault_state_smooth"], labels)

    raw_accuracy = cm_raw.values.trace() / max(cm_raw.values.sum(), 1)
    smooth_accuracy = cm_smooth.values.trace() / max(cm_smooth.values.sum(), 1)

    print(f"Detection accuracy (raw): {raw_accuracy:.3f}")
    print("Confusion matrix (raw):")
    print(cm_raw)
    print(f"\nDetection accuracy (smoothed): {smooth_accuracy:.3f}")
    print("Confusion matrix (smoothed):")
    print(cm_smooth)

    detector = results.attrs["detector"]
    print("\nDetection summary:")
    print(f"Learning phase max duration: {learning_max_time_s / 3600:.1f} hours")
    if detector.learning_phase:
        print(
            "Learning did not finish within the data span. "
            f"Using provisional thresholds: T1={detector.provisional_t1:.4f}, "
            f"T2={detector.provisional_t2:.4f}"
        )
    else:
        print(f"Final thresholds: T1={detector.t1:.4f}, T2={detector.t2:.4f}")
    print(
        f"Clean normals collected: {len(detector.clean_normal_iw)} | "
        f"Suspect segments excluded during learning: {detector.suspect_excluded}"
    )

    print_precision_recall_f1(cm_raw, labels, title="\nRaw predictions")
    print_precision_recall_f1(cm_smooth, labels, title="\nSmoothed predictions")

    output_csv = output_dir / "enhanced_fault_detection_results.csv"
    output_excel = output_dir / "enhanced_fault_detection_results.xlsx"
    output_plot = output_dir / "enhanced_fault_detection_plot.png"
    output_state_pdf = output_dir / "fault_states.pdf"
    output_state_eps = output_dir / "fault_states.eps"

    results.to_csv(output_csv, index=False)
    results.to_excel(output_excel, index=False)
    plot_detection_summary(results, labels, output_plot)
    plot_fault_states(results, output_state_pdf, output_state_eps)

    print("\nResults saved to:")
    print(f"  CSV:   {output_csv}")
    print(f"  Excel: {output_excel}")
    print(f"  Plot:  {output_plot}")
    print(f"  State: {output_state_pdf}")


if __name__ == "__main__":
    main()

# %%
