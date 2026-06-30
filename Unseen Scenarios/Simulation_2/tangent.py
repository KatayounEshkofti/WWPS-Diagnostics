#!/usr/bin/env python3
# %% -*- coding: utf-8 -*-
"""
Enhanced online pump fault detection
====================================

Three-state classifier: Normal, Pump Fault, System Fault.

Method: Tangent residual ratio (Iw) with adaptive operating-point backup.

Architecture
------------
Two complementary detection channels run in parallel:

  Channel 1 — Iw (dynamic tangent residual):
      Computes time-derivatives dQ/dt, dH/dt within each operating segment
      and projects them onto the pump-curve tangent (ψ_p) vs system-curve
      tangent (ψ_s). The ratio Iw = P75(|ψ_p|) / P75(|ψ_p| + |ψ_s|) is
      near 0 for Normal, moderate for System Fault, and high for Pump Fault.

  Channel 2 — Q̄ z-score (static operating-point deviation):
      The mean flow rate Q̄ per segment is compared to the learned Normal
      baseline. A z-score exceeding ±4 indicates the operating point has
      shifted — flagging system faults that Iw misses.

Modeling details
--------------------------------------
  Q-deviation backup:
      Adds Channel 2 to catch static system faults invisible to Iw.

  Adaptive Q-baseline reset:
      After stable_reset_count consecutive segments where Iw < t1 (no
      dynamic fault signature) and Q is stable (std < q_stable_std_max),
      the Q baseline is reset to the current operating point. This allows
      the detector to accept a "new normal" after recovery, while remaining
      safe during actual faults (where Q drops ~7 m³/h per cycle, producing
      std >> 1.0 over any 2-cycle window).

  Fault persistence:
      When Q-deviation fires and the previous fault type was Pump Fault,
      the classification is maintained as Pump Fault (not downgraded to
      System Fault). Q-deviation only promotes Normal → System Fault.

Hysteresis policy
-----------------
  - Iw-based detections: require 2 consecutive non-Normal segments before
    switching away from Normal (transient noise protection).
  - Q-deviation detections: No hysteresis (Q shift is unambiguous and
    persistent, not a transient artifact).
  - Pump Fault from Iw: always accepted immediately (strong dynamic signal).

Online adaptation
-----------------
  - Iw thresholds (t1, t2) are periodically recomputed from a trimmed
    buffer of recent Normal Iw values (every adapt_interval cycles).
  - Flow baseline (mean and standard deviation) adapts both from normal segments AND via the
    stability-gated reset mechanism.

"""

import warnings
warnings.filterwarnings("ignore")

import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)


# =============================================================================
# Configuration
# =============================================================================

# Project paths
data_path = Path("data/pump_system_faultsimualtion.xlsx")
cycles_path = Path("data/Cycle-time.xlsx")
output_dir = Path("results/enhanced_fault_detection")

# Learning phase
learning_max_time_s = 12 * 60 * 60   # max wall-time for learning (seconds)
learning_min_normal_cycles = 20            # collect this many clean normals
learning_outlier_guard = 3.0           # MAD z-score to reject learning outliers

# Iw threshold estimation
provisional_t1_default = 0.020             # used before learning completes
provisional_t2_default = 0.060
t1_min_floor = 0.025             # hard floor for t1
t2_min_floor = 0.060             # hard floor for t2
trim_top_q = 0.85              # drop top 15% when computing thresholds

# Operating-point deviation backup
q_deviation_z_thresh = 4.0                 # z-score to flag Q anomaly

# Adaptive flow-baseline reset
stable_reset_count = 2                   # consecutive stable low-Iw segments to reset
q_stable_std_max = 1.0                 # max flow std across reset window (m³/h)
                                            # (faults produce std ≈ 3–8, normal ≈ 0.1–0.3)

# Valid operating segment requirements
full_speed_hz = 49.5               # minimum frequency for "full speed"
min_flow_m3h = 0.1                # minimum flow to consider valid
freq_stab_roll = 5                  # rolling window for frequency stability check
freq_stab_std_max = 0.5                # max std of frequency in rolling window

# Segment length requirements
min_seg_points_to_keep = 20               # minimum contiguous valid points
min_seg_len_to_process = 25               # ignore segments shorter than this

# System curve setting
friction_coeff = 0.0003                    # k_f for system curve slope: ms = 2·k_f·Qtot
f_nom = 50.0                                # nominal pump frequency (Hz)

# Online adaptation
adapt_window = 1000                     # max recent-normal buffer size
adapt_interval = 50                       # recompute thresholds every N cycles

# Optional post-processing
apply_state_smoothing = True
smooth_window = 3                 # rolling majority window

# Reproducibility and numerical safety
eps = 1e-9
seed = 42

# Label mapping used by the evaluation step
label_map = {
    "normal":               "Normal",
    "Normal":               "Normal",
    "pump_blockage_window": "Pump Fault",
    "pump_blockage":        "Pump Fault",
    "pump_fault":           "Pump Fault",
    "pump fault":           "Pump Fault",
    "Pump Fault":           "Pump Fault",
    "system_fault_window":  "System Fault",
    "system_fault":         "System Fault",
    "system fault":         "System Fault",
    "System Fault":         "System Fault",
}


# =============================================================================
# 2) Data loading and preparation
# =============================================================================
print("Loading data...")
df = pd.read_excel(data_path)

# Ensure datetime and create seconds-since-start column
if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
t0 = df["Timestamp"].min()
df["t_s"] = (df["Timestamp"] - t0).dt.total_seconds().astype(float)

# Extract Pump 1 signals
p1 = df[["t_s", "Pump1_Flow_m3h", "Pump1_Head_m", "Pump1_Freq_Hz"]].copy()
p1.columns = ["t_s", "Q1", "H1", "f1"]

# Basic physical validity filter
p1 = p1[(p1["Q1"] > 0) & (p1["H1"] > 0) & (p1["f1"] > 0)].copy()

# Valid operation mask: full-speed + positive flow + stable frequency
fullspeed = p1["f1"] >= full_speed_hz
positive_flow = p1["Q1"] > min_flow_m3h
freq_stability = p1["f1"].rolling(freq_stab_roll, center=True).std().fillna(999)
stable_operation = freq_stability < freq_stab_std_max
p1["valid"] = fullspeed & positive_flow & stable_operation

print(f"  Total samples: {len(p1)}")
print(f"  Valid samples: {p1['valid'].sum()}")


# =============================================================================
# 3) Build contiguous valid segments
# =============================================================================
def contiguous_segments(mask: pd.Series, min_points: int = min_seg_points_to_keep):
    """
    Convert boolean mask into (start_idx, end_idx) pairs of consecutive
    True runs, keeping only those with at least `min_points` samples.
    """
    idx = np.where(mask.values)[0]
    if len(idx) == 0:
        return []

    segs = []
    start = idx[0]
    prev  = idx[0]

    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        # end of a contiguous run
        if (prev - start + 1) >= min_points:
            segs.append((start, prev))
        start = k
        prev  = k

    # last run
    if (prev - start + 1) >= min_points:
        segs.append((start, prev))

    return segs


segments = contiguous_segments(p1["valid"], min_points=min_seg_points_to_keep)
print(f"  Valid segments: {len(segments)}")


# =============================================================================
# 4) Baseline pump curve: H ≈ a1 + a2·Q + a3·Q²
#    Fitted from early valid data (before 12 000 s) = healthy pump behavior
# =============================================================================
baseline_data = p1[(p1["t_s"] < 12000) & p1["valid"]]

if len(baseline_data) > 10:
    X_baseline = np.vstack([
        np.ones(len(baseline_data)),
        baseline_data["Q1"].values,
        baseline_data["Q1"].values ** 2,
    ]).T
    a_hat, _, _, _ = np.linalg.lstsq(X_baseline, baseline_data["H1"].values, rcond=None)
    a1, a2, a3 = a_hat
else:
    # Fallback if insufficient early data
    a1, a2, a3 = 10.0, -0.1, -0.001

print(f"  Pump curve: H = {a1:.4f} + {a2:.6f}·Q + {a3:.8f}·Q²")


# =============================================================================
# 5) Signal processing helpers
# =============================================================================
def robust_boxcar(x: np.ndarray, w: int = 11) -> np.ndarray:
    """
    Median (boxcar) filter with odd window. Robust to outliers.
    Handles edges by naturally shrinking the window.
    """
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    if w % 2 == 0:
        w += 1

    result = np.zeros_like(x)
    half = w // 2
    n = len(x)

    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        result[i] = np.median(x[s:e])

    return result


def enhanced_central_diff(x: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Central-difference derivative with edge handling + light median smoothing.
    Assumes uniform time spacing `dt`.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = np.zeros_like(x, dtype=float)

    if n <= 1:
        return d
    if n == 2:
        d[:] = (x[1] - x[0]) / max(dt, eps)
        return d

    # Central differences for interior points
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * max(dt, eps))
    # Forward/backward for edges
    d[0]  = (x[1] - x[0])   / max(dt, eps)
    d[-1] = (x[-1] - x[-2]) / max(dt, eps)

    # Light smoothing to suppress derivative noise
    d = robust_boxcar(d, 5)
    return d


def mad(x) -> float:
    """Median absolute deviation (scaled to match std for normal data)."""
    return float(stats.median_abs_deviation(np.asarray(x), scale='normal'))


def robust_z(x: float, ref_values) -> float:
    """MAD-based z-score of scalar x relative to reference distribution."""
    ref = np.asarray(ref_values, dtype=float)
    if ref.size < 6:
        return 0.0
    m = float(np.median(ref))
    s = mad(ref) + eps
    return (x - m) / s


# =============================================================================
# 6) Enhanced Fault Detector class
# =============================================================================
class EnhancedFaultDetector:
    """
    Online fault detector with two complementary channels:

    Channel 1 — Iw (dynamic tangent residual):
        Catches faults with active within-segment drift (pump degradation
        progressing during the ~4-min segment).

    Channel 2 — Q̄ z-score (static operating-point deviation):
        Catches faults where the system has settled to a new steady-state
        operating point that Iw cannot distinguish from Normal.

    Three fixes integrated:
        Q-deviation backup for static system faults
        Adaptive Q-baseline reset after stable operation
        Fault persistence: Q-deviation preserves a pump-fault state
    """

    def __init__(self):
        # ── Phase control ──
        self.learning_phase = True

        # ── Learning buffers (clean normals only) ──
        self.clean_normal_iw  = []      # Iw values accepted as clean Normal
        self.clean_normal_q   = []      # corresponding mean Q values
        self.clean_normal_std_sum = []  # flow_std + head_std (for variance gate)

        # ── Provisional thresholds (during learning) ──
        self.provisional_t1 = provisional_t1_default
        self.provisional_t2 = provisional_t2_default

        # ── Final thresholds (set when learning completes) ──
        self.t1 = np.nan
        self.t2 = np.nan

        # ── Q-deviation baseline  ──
        self.q_mu  = None   # mean Q from Normal segments
        self.q_std = None   # std Q from Normal segments

        # ── Adaptive Q-baseline reset state  ──
        self.consec_low_iw   = 0       # consecutive segments with Iw < t1
        self.low_iw_q_buffer = []      # Q values during current low-Iw streak

        # ── Fault persistence state  ──
        self.prev_fault_type = "Normal"  # last non-Normal fault type seen

        # ── Adaptation buffers (post-learning) ──
        self.recent_normal_iw  = deque(maxlen=adapt_window)
        self.recent_normal_q   = deque(maxlen=adapt_window)
        self.recent_normal_std_sum = deque(maxlen=adapt_window)

        # ── State machine ──
        self.cycle_count        = 0
        self.consecutive_faults = 0
        self.current_state      = "Normal"

        # ── Diagnostics ──
        self.results          = []
        self.suspect_excluded = 0

    # ─────────────────────────────────────────────────────────────────────
    # Threshold computation (trimmed MAD-based)
    # ─────────────────────────────────────────────────────────────────────
    def _trimmed_thresholds(self, iw_data):
        """
        Compute t1 and t2 from a trimmed distribution of clean-Normal Iw values.
        Drops the top (1 - trim_top_q) fraction to avoid residual contamination.
        Uses median + MAD with hard floors to ensure robustness.
        """
        x = np.asarray(iw_data, dtype=float)

        if x.size < 8:
            t1 = max(t1_min_floor, float(self.provisional_t1))
            t2 = max(t2_min_floor, float(self.provisional_t2))
            if t2 <= t1:
                t2 = max(t1 * 1.5, t1 + 0.01)
            return float(t1), float(t2)

        # Trim top fraction
        x_sorted = np.sort(x)
        cut = np.quantile(x_sorted, trim_top_q)
        trimmed = x_sorted[x_sorted <= cut]
        if trimmed.size < 5:
            trimmed = x_sorted

        med = float(np.median(trimmed))
        m   = mad(trimmed)

        t1 = med + 1.5 * m
        t2 = med + 3.0 * m

        # Enforce separation and floors
        if t2 <= t1:
            t2 = max(t1 * 1.5, t1 + 0.01)
        t1 = max(t1_min_floor, t1)
        t2 = max(t2_min_floor, t2)

        return float(t1), float(t2)

    def _update_provisional_thresholds(self):
        """Refresh provisional thresholds from current clean-Normal buffer."""
        if len(self.clean_normal_iw) >= 6:
            t1, t2 = self._trimmed_thresholds(self.clean_normal_iw)
            self.provisional_t1 = t1
            self.provisional_t2 = t2

    # ─────────────────────────────────────────────────────────────────────
    # Iw computation (tangent residual ratio)
    # ─────────────────────────────────────────────────────────────────────
    def compute_enhanced_iw(self, seg: pd.DataFrame):
        """
        Compute the tangent residual ratio Iw and auxiliary features for
        one operating segment.

        Steps:
            1. Robust-smooth Q and H with median boxcar filter
            2. Compute time derivatives dQ/dt, dH/dt via central differences
            3. Project onto pump-curve tangent: ψ_p = dH - m_p · dQ
               where m_p = dH_pump/dQ = a2 + 2·a3·Q (from baseline curve)
            4. Project onto system-curve tangent: ψ_s = dH - m_s · dQ
               where m_s = 2·k_f·Q_total (from system curve)
            5. Compute Iw = P75(|ψ_p|) / P75(|ψ_p| + |ψ_s|)
            6. Apply stability dampening: Iw_enhanced = Iw · stability_factor

        Returns:
            enhanced_iw : float — stability-dampened Iw value
            mean_q      : float — mean flow rate of the segment
            flow_std    : float — standard deviation of smoothed flow
            head_std    : float — standard deviation of smoothed head
        """
        Q1 = seg["Q1"].values.astype(float)
        H1 = seg["H1"].values.astype(float)
        t  = seg["t_s"].values.astype(float)

        # Estimate sampling interval dt
        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        else:
            dt = 1.0

        # Robust smoothing
        Q1s = robust_boxcar(Q1, 15)
        H1s = robust_boxcar(H1, 15)

        # Time derivatives
        dQ = enhanced_central_diff(Q1s, dt)
        dH = enhanced_central_diff(H1s, dt)

        # Pump-curve tangent slope: m_p = dH_pump/dQ = a2 + 2·a3·Q
        mp = a2 + 2.0 * a3 * Q1s

        # Total system flow (all three pumps)
        Qtot = (
            df.loc[seg.index, "Pump1_Flow_m3h"]
            + df.loc[seg.index, "Pump2_Flow_m3h"]
            + df.loc[seg.index, "Pump3_Flow_m3h"]
        ).values.astype(float)
        Qtot_s = robust_boxcar(Qtot, 15)

        # System-curve tangent slope: m_s = 2·k_f·Q_total
        ms = 2.0 * friction_coeff * Qtot_s

        # Tangent residuals
        psi_p = dH - mp * dQ   # pump residual
        psi_s = dH - ms * dQ   # system residual

        # Robust percentile-based ratio
        num = np.percentile(np.abs(psi_p), 75)
        den = np.percentile(np.abs(psi_p) + np.abs(psi_s), 75)
        Iw = float(num / den) if den > eps else 0.5

        # Stability dampening (reduces Iw in high-variability segments
        # where derivative noise could produce spurious residuals)
        flow_std = float(np.std(Q1s))
        head_std = float(np.std(H1s))
        stability = min(1.0, 2.0 / (flow_std + head_std + 0.1))
        enhanced_iw = Iw * stability

        mean_q = float(np.mean(Q1))

        return enhanced_iw, mean_q, flow_std, head_std

    # ─────────────────────────────────────────────────────────────────────
    # Per-segment processing (learning + online phases)
    # ─────────────────────────────────────────────────────────────────────
    def process_segment(self, seg: pd.DataFrame, cid: int) -> dict:
        """
        Process one operating segment and return a classification result.

        During the learning phase:
            - Collects clean Normal Iw and Q values (outlier-guarded)
            - Uses provisional thresholds for early fault detection
            - Ends learning when enough clean normals are collected

        During the online phase:
            - Channel 1: Iw-based classification (dynamic faults)
            - Channel 2: Q-deviation check (static faults)
            - Adaptive Q-baseline reset (new normal detection)
            - Fault persistence (pump fault preservation)
            - Hysteresis (Iw-based only, not Q-deviation)
            - Periodic threshold adaptation from recent normals
        """
        # ── Compute features ──
        Iw, mean_q, flow_std, head_std = self.compute_enhanced_iw(seg)

        s0  = float(seg["t_s"].iloc[0])
        s1  = float(seg["t_s"].iloc[-1])
        dur = s1 - s0

        q_z = 0.0  # will be set in online phase

        # ================================================================
        # Learning phase
        # ================================================================
        if self.learning_phase:

            # Keep provisional thresholds fresh as normals accumulate
            self._update_provisional_thresholds()

            # ── Outlier guard (MAD z-score) ──
            if len(self.clean_normal_iw) >= 6:
                med = float(np.median(self.clean_normal_iw))
                m   = mad(self.clean_normal_iw) + eps
                z   = abs(Iw - med) / m
            else:
                z = 0.0

            # Determine if this segment is suspect (possibly contaminated)
            is_suspect = False
            if len(self.clean_normal_iw) >= 6 and z > learning_outlier_guard:
                is_suspect = True
            if Iw >= max(self.provisional_t2, provisional_t2_default):
                is_suspect = True

            # ── Classify during learning ──
            if is_suspect:
                self.suspect_excluded += 1

                if Iw >= self.provisional_t2:
                    fault_state = "Pump Fault"
                    self.consecutive_faults += 1
                elif Iw >= self.provisional_t1:
                    fault_state = "System Fault"
                    self.consecutive_faults += 1
                else:
                    fault_state = "Normal"
                    self.consecutive_faults = 0
            else:
                # Accept as clean Normal for learning
                self.clean_normal_iw.append(Iw)
                self.clean_normal_q.append(mean_q)
                self.clean_normal_std_sum.append(flow_std + head_std)
                fault_state = "Normal"
                self.consecutive_faults = 0

            # ── Confidence (provisional) ──
            if fault_state == "Normal":
                confidence = 1.0 - min(1.0, Iw / (self.provisional_t1 + eps))
            elif fault_state == "System Fault":
                confidence = min(1.0, (Iw - self.provisional_t1) /
                                 max(self.provisional_t2 - self.provisional_t1, eps))
            else:
                confidence = min(1.0, (Iw - self.provisional_t2) /
                                 max(0.5 - self.provisional_t2, eps))

            # ── End learning if criteria met ──
            if (len(self.clean_normal_iw) >= learning_min_normal_cycles
                    or s1 > learning_max_time_s):
                self.t1, self.t2 = self._trimmed_thresholds(self.clean_normal_iw)

                # Initialize Q baseline
                self.q_mu  = float(np.mean(self.clean_normal_q))
                self.q_std = max(0.01, float(np.std(self.clean_normal_q)))

                self.learning_phase = False
                print(f"[Learning complete] clean_normals={len(self.clean_normal_iw)}, "
                      f"suspects_excluded={self.suspect_excluded}")
                print(f"  t1={self.t1:.4f}, t2={self.t2:.4f}")
                print(f"  q_mu={self.q_mu:.1f}, q_std={self.q_std:.3f}")

        # ================================================================
        # Online phase
        # ================================================================
        else:
            # ── Channel 1: Iw-based classification (dynamic) ──
            if Iw >= self.t2:
                iw_state = "Pump Fault"
            elif Iw >= self.t1:
                iw_state = "System Fault"
            else:
                iw_state = "Normal"

            # ── Channel 2: Q-deviation check (static)  ──
            q_z = (mean_q - self.q_mu) / self.q_std if self.q_mu else 0.0
            q_anomaly = abs(q_z) > q_deviation_z_thresh

            # ── Track consecutive low-Iw segments  ──
            if Iw < self.t1:
                self.consec_low_iw += 1
                self.low_iw_q_buffer.append(mean_q)
            else:
                self.consec_low_iw   = 0
                self.low_iw_q_buffer = []

            # ── Adaptive Q-baseline reset  ──
            # If N consecutive segments have Iw ≈ 0 (no dynamic fault)
            # AND their Q values are stable (not dropping like during fault),
            # then the system has settled to a new normal operating point.
            if (self.consec_low_iw >= stable_reset_count
                    and len(self.low_iw_q_buffer) >= stable_reset_count):

                recent_q = np.array(self.low_iw_q_buffer[-stable_reset_count:])

                if np.std(recent_q) < q_stable_std_max:
                    # Accept as new normal baseline
                    _previous_q_mu = self.q_mu
                    self.q_mu  = float(np.mean(recent_q))
                    self.q_std = max(0.01, float(np.std(recent_q)))

                    # Recompute q_z with updated baseline
                    q_z = (mean_q - self.q_mu) / self.q_std
                    q_anomaly = abs(q_z) > q_deviation_z_thresh

                    # Refill adaptation buffer
                    self.recent_normal_q.clear()
                    for q in recent_q:
                        self.recent_normal_q.append(q)

                    # Also reset fault persistence when entering new normal
                    self.prev_fault_type = "Normal"

            # ── Combined decision logic ──
            if iw_state == "Pump Fault":
                # ── Strong dynamic pump fault signal → always accept ──
                fault_state = "Pump Fault"
                self.consecutive_faults += 1

            elif iw_state == "System Fault":
                # ── Dynamic system fault → apply hysteresis ──
                fault_state = "System Fault"
                self.consecutive_faults += 1
                # Require 2 consecutive non-Normal before switching from Normal
                if self.current_state == "Normal" and self.consecutive_faults < 2:
                    fault_state = "Normal"
                    self.consecutive_faults -= 1

            elif q_anomaly:
                # ── Iw says Normal but Q has shifted far from baseline ──
                # This is a static fault (settled to new faulty operating point).
                # No hysteresis — Q shift is unambiguous and persistent.

                # : If previous fault was Pump Fault and Q is still
                # anomalous, maintain Pump Fault (don't downgrade to System Fault).
                # Q-deviation only promotes Normal → System Fault.
                if self.prev_fault_type == "Pump Fault":
                    fault_state = "Pump Fault"
                else:
                    fault_state = "System Fault"
                self.consecutive_faults += 1

            else:
                # ── Normal: both Iw and Q-deviation agree ──
                fault_state = "Normal"
                self.consecutive_faults = 0

            # ── Update state machine ──
            self.current_state = fault_state

            # ── Track fault persistence  ──
            if fault_state != "Normal":
                self.prev_fault_type = fault_state
            # Reset fault persistence after confirmed stable normal
            if self.consec_low_iw >= stable_reset_count and not q_anomaly:
                self.prev_fault_type = "Normal"

            # ── Feed adaptation buffers with Normal-only data ──
            if fault_state == "Normal":
                self.recent_normal_iw.append(Iw)
                self.recent_normal_q.append(mean_q)
                self.recent_normal_std_sum.append(flow_std + head_std)

            # ── Periodic threshold adaptation ──
            self.cycle_count += 1
            if (self.cycle_count % adapt_interval == 0
                    and len(self.recent_normal_iw) > 20):
                self.t1, self.t2 = self._trimmed_thresholds(
                    list(self.recent_normal_iw)
                )
                # Also gently adapt Q baseline from recent normals
                if len(self.recent_normal_q) > 10:
                    self.q_mu  = float(np.mean(list(self.recent_normal_q)))
                    self.q_std = max(0.01, float(np.std(list(self.recent_normal_q))))

            # ── Confidence ──
            if fault_state == "Normal":
                confidence = 1.0 - min(1.0, Iw / (self.t1 + eps))
            elif fault_state == "System Fault":
                # Confidence from whichever channel is stronger
                iw_conf = (Iw - self.t1) / max(self.t2 - self.t1, eps)
                qz_conf = abs(q_z) / 10.0
                confidence = min(1.0, max(iw_conf, qz_conf))
            else:  # Pump Fault
                confidence = min(1.0, (Iw - self.t2) / max(0.5 - self.t2, eps))

        # ── Store result ──
        result = {
            "cycle_id":       cid,
            "start_s":        s0,
            "end_s":          s1,
            "duration_s":     dur,
            "Iw":             float(Iw),
            "mean_q":         float(mean_q),
            "q_z":            float(q_z),
            "flow_std":       float(flow_std),
            "head_std":       float(head_std),
            "fault_state":    fault_state,
            "confidence":     float(confidence),
            "t1":             float(self.t1 if not self.learning_phase
                                    else self.provisional_t1),
            "t2":             float(self.t2 if not self.learning_phase
                                    else self.provisional_t2),
            "q_mu":           float(self.q_mu) if self.q_mu else np.nan,
            "learning_phase": self.learning_phase,
        }
        self.results.append(result)
        return result


# =============================================================================
# 7) Run detector over all segments
# =============================================================================
detector = EnhancedFaultDetector()
online_results = []

print("\nStarting enhanced fault detection...")
for cid, (i0, i1) in enumerate(segments):
    seg = p1.iloc[i0 : i1 + 1].copy()
    if len(seg) < min_seg_len_to_process:
        continue
    result = detector.process_segment(seg, cid)
    online_results.append(result)

print("Processing complete!")

# Collect results
res_online = pd.DataFrame(online_results)
print(f"  Segments processed: {len(res_online)}")


# =============================================================================
# 8) Optional post-processing: rolling majority smoothing
# =============================================================================
if apply_state_smoothing and len(res_online) > 0:
    state_to_int = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}
    int_to_state = {v: k for k, v in state_to_int.items()}

    state_ints = res_online["fault_state"].map(state_to_int).values
    smoothed = []

    for i in range(len(state_ints)):
        s = max(0, i - smooth_window // 2)
        e = min(len(state_ints), i + smooth_window // 2 + 1)
        vals, counts = np.unique(state_ints[s:e], return_counts=True)
        smoothed.append(int_to_state[vals[np.argmax(counts)]])

    res_online["fault_state_smooth"] = smoothed
else:
    res_online["fault_state_smooth"] = res_online.get(
        "fault_state", pd.Series(dtype=object)
    )


# =============================================================================
# 9) Ground truth from cycle labels
# =============================================================================
if cycles_path.exists():
    cycles_df = pd.read_excel(cycles_path)
    cycles_df = cycles_df.rename(columns={
        "start_t_sec": "start_s",
        "stop_t_sec":  "end_s",
        "label":       "truth",
    })
    if "cycle_id" not in cycles_df.columns:
        cycles_df["cycle_id"] = np.arange(1, len(cycles_df) + 1)

    print(f"\nGround truth loaded: {len(cycles_df)} cycles")
    print(f"  Labels: {cycles_df['truth'].value_counts().to_dict()}")

    # Map each detected segment to its best-overlapping ground-truth label
    truth_labels = []
    for _, row in res_online.iterrows():
        s0, s1 = row["start_s"], row["end_s"]
        best_overlap = 0.0
        best_label   = "Normal"

        for _, c in cycles_df.iterrows():
            left  = max(s0, c["start_s"])
            right = min(s1, c["end_s"])
            overlap = max(0.0, right - left) / max(s1 - s0, 1.0)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label   = c["truth"]

        truth_labels.append(label_map.get(str(best_label).strip(), best_label))

    res_online["truth"] = truth_labels
else:
    print("\n[Warning] No cycle-labels file found; skipping evaluation.")
    res_online["truth"] = "Unknown"


# =============================================================================
# 10) Evaluation metrics
# =============================================================================
valid_labels = ["Normal", "Pump Fault", "System Fault"]

if res_online["truth"].isin(valid_labels).any():
    mask = res_online["truth"].isin(valid_labels)

    for col_name, display_name in [("fault_state", "Raw"),
                                    ("fault_state_smooth", "Smoothed")]:
        print(f"\n{'=' * 60}")
        print(f"{display_name} Predictions")
        print(f"{'=' * 60}")

        y_true = res_online.loc[mask, "truth"]
        y_pred = res_online.loc[mask, col_name]

        cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {l}" for l in valid_labels],
            columns=[f"Pred {l}" for l in valid_labels],
        )
        print(cm_df.to_string())

        report = classification_report(
            y_true, y_pred, labels=valid_labels, zero_division=0
        )
        print(f"\n{report}")

        ba = balanced_accuracy_score(y_true, y_pred)
        print(f"Balanced accuracy: {ba:.4f}")

    # Show misclassified segments
    mis = res_online[mask & (res_online["truth"] != res_online["fault_state"])]
    if mis.empty:
        print("\nPerfect classification — 0 errors!")
    else:
        print(f"\n{len(mis)} misclassified segment(s):")
        for _, r in mis.iterrows():
            print(f"  Seg {r['cycle_id']:>3}: truth={r['truth']:>14}, "
                  f"pred={r['fault_state']:>14}, "
                  f"Iw={r['Iw']:.4f}, Q={r['mean_q']:.1f}, q_z={r['q_z']:.1f}")
else:
    print("\n[Warning] No valid ground-truth labels for evaluation.")


# =============================================================================
# 11) Summary
# =============================================================================
print(f"\n{'=' * 60}")
print("Detection Summary")
print(f"{'=' * 60}")
print(f"Learning phase max duration: {learning_max_time_s / 3600:.1f} hours")

if detector.learning_phase:
    print(f"Learning did not complete within data span.")
    print(f"  Using provisional: t1={detector.provisional_t1:.4f}, "
          f"t2={detector.provisional_t2:.4f}")
else:
    print(f"Final thresholds: t1={detector.t1:.4f}, t2={detector.t2:.4f}")
    print(f"Q baseline: q_mu={detector.q_mu:.1f}, q_std={detector.q_std:.3f}")

print(f"Clean normals collected: {len(detector.clean_normal_iw)}")
print(f"Suspects excluded during learning: {detector.suspect_excluded}")
print(f"Segments processed: {len(res_online)}")


# =============================================================================
# 12) Save outputs
# =============================================================================
output_dir.mkdir(parents=True, exist_ok=True)

out_csv = output_dir / "enhanced_fault_detection_results.csv"
out_excel = output_dir / "enhanced_fault_detection_results.xlsx"
out_png = output_dir / "enhanced_fault_detection_plot.png"

res_online.to_csv(out_csv, index=False)
res_online.to_excel(out_excel, index=False)
print(f"\nResults saved to:")
print(f"  CSV:   {out_csv}")
print(f"  Excel: {out_excel}")


# =============================================================================
# 13) Visualization
# =============================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

t1_plot = float(res_online["t1"].iloc[-1]) if len(res_online) else np.nan
t2_plot = float(res_online["t2"].iloc[-1]) if len(res_online) else np.nan

# ── Plot 1: Iw values and thresholds ──
ax = axes[0]
ax.plot(res_online["start_s"], res_online["Iw"], 'b.', alpha=0.7, label='Iw')
if np.isfinite(t1_plot):
    ax.axhline(t1_plot, color='r', linestyle='--', label=f't1 ({t1_plot:.4f})')
if np.isfinite(t2_plot):
    ax.axhline(t2_plot, color='g', linestyle='--', label=f't2 ({t2_plot:.4f})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Iw value')
ax.set_title('Channel 1: Tangent Residual Iw (dynamic fault detection)')
ax.legend()
ax.grid(True)

# ── Plot 2: Q z-score (operating-point deviation) ──
ax = axes[1]
ax.plot(res_online["start_s"], res_online["q_z"], 'r.-', alpha=0.7, label='Q z-score')
ax.axhline(q_deviation_z_thresh, color='gray', linestyle='--',
           label=f'±{q_deviation_z_thresh} threshold')
ax.axhline(-q_deviation_z_thresh, color='gray', linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Q z-score')
ax.set_title('Channel 2: Q̄ Operating-Point Deviation (static fault detection + adaptive reset)')
ax.legend()
ax.grid(True)

# ── Plot 3: Q baseline tracking ──
ax = axes[2]
ax.plot(res_online["start_s"], res_online["mean_q"], 'b.-', alpha=0.7, label='Segment mean Q')
ax.plot(res_online["start_s"], res_online["q_mu"], 'k--', alpha=0.7, label='Q baseline (μ)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flow rate (m³/h)')
ax.set_title('Adaptive Q Baseline Tracking')
ax.legend()
ax.grid(True)

# ── Plot 4: Fault states (predicted vs truth) ──
ax = axes[3]
state_map = {"Normal": 0, "System Fault": 1, "Pump Fault": 2}

if "truth" in res_online.columns and res_online["truth"].isin(valid_labels).any():
    truth_states = [state_map.get(s, 0) for s in res_online["truth"]]
    ax.step(res_online["start_s"], truth_states, where='post',
            color='red', linestyle=':', alpha=0.9, label='Ground Truth')

pred_states = [state_map.get(s, 0) for s in res_online["fault_state"]]
ax.step(res_online["start_s"], pred_states, where='post',
        color='black', linestyle='-', alpha=0.7, label='Predicted')

if "fault_state_smooth" in res_online.columns:
    smooth_states = [state_map.get(s, 0) for s in res_online["fault_state_smooth"]]
    ax.step(res_online["start_s"], smooth_states, where='post',
            color='blue', linestyle='--', alpha=0.5, label='Smoothed')

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(valid_labels)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Fault State')
ax.set_title('Classification Results')
ax.legend()
ax.grid(True)

plt.tight_layout()
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"  Plot:  {out_png}")

plt.show()
print("\nDone.")
# %%
