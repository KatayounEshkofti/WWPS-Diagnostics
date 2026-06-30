# %%
"""
SCADA-calibrated pump-station verification simulator.

This script replays a measured inflow profile, simulates the pump-station
hydraulics, and verifies the simulated level, three-phase currents, and power
against SCADA measurements. The electrical model is calibrated from steady-state
SCADA data and can replay measured voltage traces when they are available.

The main checks include level RMSE/MAE, per-pump electrical error metrics,
per-cycle current and power comparison, SCADA-synchronized electrical replay,
and validation figures.
"""

# Libraries
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Final

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import gc

rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
# Wet-well geometry
cross_section_area = 8.0  # m²

# Frequency & speed
pump_min_freq = 25.0   # Hz (used only during soft-start)
pump_max_freq = 50.0   # Hz (nominal full-speed)
nominal_freq  = 50.0   # Hz
nominal_rpm   = 1460.0 # rpm at 50 Hz

# Level control set-points
level_start  = 1.60  # m – if no pump running, start lead pump
level_stop   = 0.5   # m – emergency stop all
level_start2 = 1.8   # m – start second pump when one already running
level_stop2  = 0.80  # m – stop second pump when two running

# Values for simulating abnormal flows
peak_rate      = 0.0005  # Events per second (≈18 events/day)
peak_magnitude = 60.0    # Extra flow during peaks (m³/h)
peak_duration  = 900     # Peak duration in seconds (15 minutes)

# 50 Hz pump curve (US gpm, m)
pump_curve_data_50hz: List[Tuple[float, float]] = [
    (0, 32.0), (400, 29.9), (800, 27.4),
    (1200, 24.4), (1600, 21.3), (2000, 18.3),
]

# System curve parameters: H = H_s + K·Q²  (Q in m³/h, H in m)
static_head    = 2.0
friction_coeff = 0.0006

# Keep copies of the original system-curve constants:
friction_coeff_base      = friction_coeff
static_head_base         = static_head

# When (during day 2) the pipe-clogging ramp starts/stops:
system_fault_start           = 86400 + 40000
system_fault_end             = 86400 + 61600
system_friction_increase     = 1.0
system_static_head_increase  = 0.5

# Physics & efficiency
pump_efficiency  = 0.90
water_density    = 1000.0  # kg/m³
gravity          = 9.81    # m/s²

# ---------------------------------------------------------------------------
# Three-phase electrical settings calibrated from SCADA data
# ---------------------------------------------------------------------------
# Nominal per-phase voltages (measured means from SCADA)
voltage_phase1_mean = 236.42   # V
voltage_phase2_mean = 236.64   # V
voltage_phase3_mean = 238.16   # V
voltage_noise_std   = 2.3      # V (typical spread observed in SCADA)

# Nominal steady-state per-phase current (from SCADA ~27 A)
nominal_current_per_phase = 27.0  # A
current_noise_std         = 0.14  # A (SCADA steady-state std)

# Power factor used in the SCADA formula
power_factor_scada = 0.78  # (V1*I1 + V2*I2 + V3*I3) * 0.78 / 1000

# Soft-start / soft-stop timing (from SCADA transient analysis)
soft_start_steps = 10  # seconds – linear ramp 0 → nominal current
soft_stop_steps  = 9   # seconds – linear ramp nominal current → 0

# Original constants kept for the hydraulic model
nominal_voltage  = 400     # V (three-phase line-to-line, used in old formula)
nominal_current  = 30.0    # A (average – used in old formula only)
power_factor     = 0.9     # old formula power factor

# Unit conversion
usgpm_to_m3h = 0.2271

# Soft-start/stop slopes (kept for frequency ramp logic)
soft_start_time = float(soft_start_steps)  # s
soft_stop_time  = float(soft_stop_steps)   # s

# Simulation horizon
dt = 1.0  # s, discrete time-step

# Measurement noise (sensor realism)
level_meas_noise_std      = 0.02  # ±2 cm
flow_noise_relative_std   = 0.01  # 1 %
power_noise_relative_std  = 0.01  # 1 %

# ------------------------------------------------------------------
# Inflow data – use recorded values + their time stamps
# ------------------------------------------------------------------
flow_csv_path: Final[Path] = Path("data/flow_rates.csv")

try:
    df_flow = pd.read_csv(flow_csv_path, parse_dates=["time"])
except FileNotFoundError as e:
    raise SystemExit(f"[FATAL] Could not find {flow_csv_path}") from e

# 1) Keep only positive (incoming) flow; treat negatives as zero inflow
df_flow["flow_in_m3s"] = df_flow["flow_rate_m3_s"].clip(lower=0.0)

# 2) Build a *uniform 1-second* time grid that covers the whole file
t0 = df_flow["time"].iloc[0]
df_flow["t_rel_s"] = (df_flow["time"] - t0).dt.total_seconds().astype("int64")

# 3) Create a dense series indexed by every second, forward-filled
max_t = int(df_flow["t_rel_s"].iloc[-1])
full_index = pd.RangeIndex(start=0, stop=max_t + 1, step=1)
profile_m3s = (
    df_flow.set_index("t_rel_s")["flow_in_m3s"]
    .reindex(full_index)
    .ffill()
    .to_numpy()
)
profile_m3h = profile_m3s * 3600.0
profile_len = profile_m3h.size
print(f"[INIT] Loaded exact inflow profile: {profile_len:,} s "
      f"({profile_len/86400:.1f} days) from {t0}.")

# -- LEVEL column (for validation) --
level_column_candidates = ["level_m", "water_level_m", "level_meter", "level"]
for col in level_column_candidates:
    if col in df_flow.columns:
        df_flow["level_meas_m"] = df_flow[col].astype(float)
        print(f"[INIT] Using '{col}' as measured level.")
        break
else:
    raise SystemExit("[FATAL] No water-level column found in CSV; "
                     "cannot validate simulation.")

profile_level = (
    df_flow.set_index("t_rel_s")["level_meas_m"]
    .reindex(full_index)
    .ffill()
    .to_numpy()
)

# -- Finalise time parameters --
sim_time = profile_len - 1
print(f"[INIT] Horizon set to {sim_time/86400:.2f} days "
      f"({sim_time:,} s).")

# Free the flow DataFrame (we have the numpy arrays already)
del df_flow
import gc; gc.collect()


# ------------------------------------------------------------------
# Load SCADA electrical data for verification
# ------------------------------------------------------------------
print("[INIT] Loading SCADA electrical data for verification …")

scada_xlsx_path = Path("data/combined_timeseries_softstart_softstop_modified.xlsx")

output_dir = Path("results/scada_electrical_validation")
figures_dir = output_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
use_latex = False

# Read all sheets at once
scada_sheets = pd.read_excel(scada_xlsx_path, sheet_name=None)

# Combine the three sheets vertically
df_scada = pd.concat(
    [scada_sheets[sheet] for sheet in scada_sheets],
    ignore_index=True
)

# Parse and clean time
df_scada["time_parsed"] = pd.to_datetime(df_scada["time"], errors="coerce")
df_scada = df_scada.dropna(subset=["time_parsed"]).copy()

# Sort by time in case the sheets are not already perfectly ordered
df_scada = df_scada.sort_values("time_parsed").reset_index(drop=True)

# Optional but recommended: remove duplicate timestamps if any exist
df_scada = df_scada.drop_duplicates(subset=["time_parsed"], keep="first")

scada_t0 = df_scada["time_parsed"].iloc[0]
df_scada["t_rel_s"] = (
    df_scada["time_parsed"] - scada_t0
).dt.total_seconds().astype("int64")

# Build lookup dict: t_rel_s → row index for O(1) access
scada_max_t = int(df_scada["t_rel_s"].iloc[-1])
scada_valid_times = set(df_scada["t_rel_s"].values)

# Store SCADA data as dict-of-arrays indexed by t_rel_s for memory efficiency
_scada_t = df_scada["t_rel_s"].values
_scada_cols_needed = [
    "pump1_current1", "pump1_current2", "pump1_current3", "pump1_power",
    "pump2_current1", "pump2_current2", "pump2_current3", "pump2_power",
    "pump3_current1", "pump3_current2", "pump3_current3", "pump3_power",
    "voltage_1", "voltage_2", "voltage_3"
]

scada_columns = {}
for col_name in _scada_cols_needed:
    arr = np.full(scada_max_t + 1, np.nan, dtype=np.float32)
    if col_name in df_scada.columns:
        arr[_scada_t] = df_scada[col_name].values.astype(np.float32)
    else:
        print(f"[WARN] Column '{col_name}' not found in SCADA workbook.")
    scada_columns[col_name] = arr

# Free the heavy DataFrame
del df_scada, scada_sheets
gc.collect()

print(f"[INIT] SCADA electrical data loaded: {len(scada_valid_times):,} valid timestamps, "
      f"{scada_max_t:,} s span from {len(scada_columns)} stored signals.")
# ------------------------------------------------------------------
# V3: Per-pump calibration from SCADA steady-state data
# ------------------------------------------------------------------
def calibrate_per_pump_params(scada_cols, s_max_t):
    """Extract per-pump, per-phase current/voltage means from SCADA steady state."""
    params = {}
    for pidx, prefix in enumerate(["pump1", "pump2", "pump3"]):
        c1 = scada_cols[f"{prefix}_current1"]
        c2 = scada_cols[f"{prefix}_current2"]
        c3 = scada_cols[f"{prefix}_current3"]
        ss = (c1 > 25.0) & (c2 > 25.0) & (c3 > 25.0) & np.isfinite(c1)
        n_ss = int(ss.sum())
        if n_ss > 100:
            v1, v2, v3 = scada_cols["voltage_1"], scada_cols["voltage_2"], scada_cols["voltage_3"]
            params[pidx] = {
                "i1_mean": float(np.nanmean(c1[ss])), "i2_mean": float(np.nanmean(c2[ss])),
                "i3_mean": float(np.nanmean(c3[ss])),
                "i1_std": float(np.nanstd(c1[ss])), "i2_std": float(np.nanstd(c2[ss])),
                "i3_std": float(np.nanstd(c3[ss])),
                "v1_mean": float(np.nanmean(v1[ss])), "v2_mean": float(np.nanmean(v2[ss])),
                "v3_mean": float(np.nanmean(v3[ss])),
                "v1_std": float(np.nanstd(v1[ss])), "v2_std": float(np.nanstd(v2[ss])),
                "v3_std": float(np.nanstd(v3[ss])),
                "n_samples": n_ss,
            }
            p = params[pidx]
            print(f"  [CAL] Pump {pidx+1}: I=({p['i1_mean']:.2f}, {p['i2_mean']:.2f}, {p['i3_mean']:.2f}) A  "
                  f"V=({p['v1_mean']:.1f}, {p['v2_mean']:.1f}, {p['v3_mean']:.1f}) V  ({n_ss:,} ss)")
        else:
            params[pidx] = {
                "i1_mean": 27.0, "i2_mean": 27.0, "i3_mean": 27.0,
                "i1_std": 0.14, "i2_std": 0.14, "i3_std": 0.14,
                "v1_mean": voltage_phase1_mean, "v2_mean": voltage_phase2_mean,
                "v3_mean": voltage_phase3_mean,
                "v1_std": 2.3, "v2_std": 2.3, "v3_std": 2.3, "n_samples": 0,
            }
            print(f"  [CAL] Pump {pidx+1}: insufficient data, using defaults")
    return params

print("[INIT] Calibrating per-pump parameters from SCADA …")
pump_cal_params = calibrate_per_pump_params(scada_columns, scada_max_t)


# ------------------------------------------------------------------
# Lookup function (constant-time and no pandas during run)
# ------------------------------------------------------------------
@np.vectorize
def inflow_from_profile(t_sec: int) -> float:
    """Deterministic inflow (m³ h⁻¹) from the logger file."""
    if t_sec < profile_len:
        return profile_m3h[t_sec]
    return profile_m3h[-1]


# ---------------------------------------------------------------------------
# Pump and system-curve helpers
# ---------------------------------------------------------------------------
def piecewise_linear_interpolation(x: float, pts: List[Tuple[float, float]]) -> float:
    """Linear interpolation on a rate-head dataset (x assumed monotonic)."""
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x0 <= x <= x1:
                break
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def get_pump_head_at_flow(flow_usgpm: float, speed_ratio: float) -> float:
    """Affinity-scaled pump curve: H ∝ N², Q ∝ N."""
    if speed_ratio <= 0.0:
        return 0.0
    flow_nominal = flow_usgpm / speed_ratio
    h_nominal = piecewise_linear_interpolation(flow_nominal, pump_curve_data_50hz)
    return h_nominal * speed_ratio ** 2


def system_head(flow_m3h: float) -> float:
    """System curve H(Q)."""
    return static_head + friction_coeff * flow_m3h ** 2


def find_operating_flow(rpm: float) -> float:
    """Bisection search for flow (m³/h) where pump-head = system-head."""
    if rpm < 50.0:
        return 0.0
    ratio = rpm / nominal_rpm
    lo, hi = 0.0, 3000.0
    eps = 0.1
    while hi - lo > eps:
        mid = 0.5 * (lo + hi)
        pump_h = get_pump_head_at_flow(mid, ratio)
        sys_h = system_head(mid * usgpm_to_m3h)
        (lo, hi) = (mid, hi) if pump_h > sys_h else (lo, mid)
    flow_usgpm = 0.5 * (lo + hi)
    return flow_usgpm * usgpm_to_m3h


def approximate_power_kW(flow_m3h: float, eff: float) -> float:
    """Shaft power P = ρ·g·Q·H / (η·1000)."""
    if flow_m3h <= 0.0 or eff < 1e-9:
        return 0.0
    head_m = system_head(flow_m3h)
    flow_m3s = flow_m3h / 3600.0
    return water_density * gravity * flow_m3s * head_m / (1000.0 * eff)


# ---------------------------------------------------------------------------
# Pump model: soft start/stop, hydraulics, and three-phase electrical output
# ---------------------------------------------------------------------------
class Pump:
    """One centrifugal pump with VFD soft-start/stop and three-phase
    electrical model calibrated per-pump from SCADA measurements."""

    def __init__(self, name: str, pump_index: int = 0,
                 calibration: dict = None):
        self.name: str = name
        self.pump_index: int = pump_index

        # Per-pump calibration
        cal = calibration or {}
        self._i1_nom = cal.get("i1_mean", nominal_current_per_phase)
        self._i2_nom = cal.get("i2_mean", nominal_current_per_phase)
        self._i3_nom = cal.get("i3_mean", nominal_current_per_phase)
        self._i1_std = cal.get("i1_std", current_noise_std)
        self._i2_std = cal.get("i2_std", current_noise_std)
        self._i3_std = cal.get("i3_std", current_noise_std)
        self._v1_nom = cal.get("v1_mean", voltage_phase1_mean)
        self._v2_nom = cal.get("v2_mean", voltage_phase2_mean)
        self._v3_nom = cal.get("v3_mean", voltage_phase3_mean)
        self._v1_std = cal.get("v1_std", voltage_noise_std)
        self._v2_std = cal.get("v2_std", voltage_noise_std)
        self._v3_std = cal.get("v3_std", voltage_noise_std)

        # Real-time states
        self.is_running: bool = False
        self.current_freq: float = 0.0
        self.blockage_factor: float = 1.0
        self.efficiency: float = pump_efficiency

        # Soft-start/stop bookkeeping
        self._soft_starting: bool = False
        self._soft_stopping: bool = False
        self._soft_start_elapsed: float = 0.0
        self._soft_stop_elapsed: float = 0.0

        # Hydraulics & energy
        self.flow_m3h: float = 0.0
        self.head_m: float   = 0.0
        self.shaft_kW: float = 0.0
        self.input_kW: float = 0.0
        self.energy_kWh: float = 0.0

        # Three-phase electrical outputs
        self.current_phase1: float = 0.0
        self.current_phase2: float = 0.0
        self.current_phase3: float = 0.0
        self.voltage_phase1: float = 0.0
        self.voltage_phase2: float = 0.0
        self.voltage_phase3: float = 0.0
        self.power_3ph_kW: float   = 0.0
        self._current_ramp_frac: float = 0.0

    # -----------------------------------------------------------------
    # Control interface
    # -----------------------------------------------------------------
    def start(self, t: float) -> None:
        if not self.is_running:
            self.is_running = True
            self._soft_starting = True
            self._soft_start_elapsed = 0.0
            self._current_ramp_frac = 0.0
            self.current_freq = pump_min_freq
            print(f"[{t:.1f}s] {self.name} START (soft-start)")

    def stop(self, t: float) -> None:
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            # Capture the current ramp fraction at the moment of stop command
            # (normally 1.0 if fully running)
            print(f"[{t:.1f}s] {self.name} STOP (soft-stop)")

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------
    def _effective_rpm(self) -> float:
        """Actual hydraulic speed after blockage scaling."""
        rpm_nom = (self.current_freq / nominal_freq) * nominal_rpm
        return rpm_nom * self.blockage_factor

    def _update_soft_ramps(self, dt: float) -> None:
        # Soft-start
        if self._soft_starting:
            self._soft_start_elapsed += dt
            frac = min(1.0, self._soft_start_elapsed / soft_start_time)
            self.current_freq = pump_min_freq + (pump_max_freq - pump_min_freq) * frac
            # Current ramp: linear from 1/N to 1.0 over SOFT_START_STEPS
            self._current_ramp_frac = min(1.0, self._soft_start_elapsed / soft_start_steps)
            if frac >= 1.0:
                self._soft_starting = False
                self._current_ramp_frac = 1.0

        # Soft-stop
        if self._soft_stopping:
            self._soft_stop_elapsed += dt
            # Current ramp: linear from 1.0 to 0.0 over SOFT_STOP_STEPS
            stop_frac = min(1.0, self._soft_stop_elapsed / soft_stop_steps)
            self._current_ramp_frac = max(0.0, 1.0 - stop_frac)

            # Frequency follows the same ramp
            self.current_freq = pump_max_freq * self._current_ramp_frac
            if self.current_freq <= 0.1 or stop_frac >= 1.0:
                self.current_freq = 0.0
                self._soft_stopping = False
                self.is_running = False
                self._current_ramp_frac = 0.0
                print("  … fully stopped")

        # Normal running – hold at full freq if neither ramp active
        if self.is_running and not (self._soft_starting or self._soft_stopping):
            self.current_freq = pump_max_freq
            self._current_ramp_frac = 1.0

    def _compute_three_phase_electrical(self, t_sec: int = -1) -> None:
        """Compute per-phase currents, voltages, and power.
        V3: per-pump calibration + SCADA voltage replay."""
        if self._current_ramp_frac <= 0.0 and not self.is_running:
            self.current_phase1 = self.current_phase2 = self.current_phase3 = 0.0
            self.voltage_phase1 = self.voltage_phase2 = self.voltage_phase3 = 0.0
            self.power_3ph_kW = 0.0
            return

        # --- Voltages: replay from SCADA when available ---
        use_scada_v = False
        if 0 <= t_sec <= scada_max_t:
            v1s = scada_columns["voltage_1"][t_sec]
            v2s = scada_columns["voltage_2"][t_sec]
            v3s = scada_columns["voltage_3"][t_sec]
            if np.isfinite(v1s) and np.isfinite(v2s) and np.isfinite(v3s):
                self.voltage_phase1 = float(v1s)
                self.voltage_phase2 = float(v2s)
                self.voltage_phase3 = float(v3s)
                use_scada_v = True
        if not use_scada_v:
            self.voltage_phase1 = self._v1_nom + random.gauss(0.0, self._v1_std)
            self.voltage_phase2 = self._v2_nom + random.gauss(0.0, self._v2_std)
            self.voltage_phase3 = self._v3_nom + random.gauss(0.0, self._v3_std)

        # --- Currents: per-phase nominal with calibrated noise ---
        frac = self._current_ramp_frac
        self.current_phase1 = max(0.0, self._i1_nom * frac + random.gauss(0.0, self._i1_std * frac))
        self.current_phase2 = max(0.0, self._i2_nom * frac + random.gauss(0.0, self._i2_std * frac))
        self.current_phase3 = max(0.0, self._i3_nom * frac + random.gauss(0.0, self._i3_std * frac))

        # --- Power (SCADA formula) ---
        self.power_3ph_kW = (
            self.voltage_phase1 * self.current_phase1
            + self.voltage_phase2 * self.current_phase2
            + self.voltage_phase3 * self.current_phase3
        ) * power_factor_scada / 1000.0

    # -----------------------------------------------------------------
    # Public update called each time-step
    # -----------------------------------------------------------------
    def update(self, dt: float, t: float) -> None:
        self._update_soft_ramps(dt)

        if not (self.is_running or self._soft_stopping):
            self.flow_m3h = self.head_m = self.shaft_kW = self.input_kW = 0.0
            self._compute_three_phase_electrical(t_sec=int(t))
            return

        # Hydraulics
        rpm_eff = self._effective_rpm()
        raw_flow = find_operating_flow(rpm_eff)
        flow_noise = 1.0 + random.gauss(0.0, flow_noise_relative_std)
        self.flow_m3h = max(0.0, raw_flow * flow_noise)
        flow_usgpm = self.flow_m3h / usgpm_to_m3h
        speed_ratio = rpm_eff / nominal_rpm
        self.head_m = get_pump_head_at_flow(flow_usgpm, speed_ratio)

        # Energetics
        self.shaft_kW = approximate_power_kW(self.flow_m3h, self.efficiency)
        if self.current_freq > 0:
            current_ratio = self.current_freq / pump_max_freq
            I_effective = min(nominal_current * current_ratio / self.blockage_factor,
                              5 * nominal_current)
            self.input_kW = (math.sqrt(3) * nominal_voltage * I_effective
                             * power_factor / 1000)
            self.input_kW *= 1.0 + random.gauss(0.0, power_noise_relative_std)
        else:
            self.input_kW = 0.0
        self.energy_kWh += self.input_kW * dt / 3600.0

        # Three-phase electrical (pass timestamp for SCADA voltage replay)
        self._compute_three_phase_electrical(t_sec=int(t))


# ---------------------------------------------------------------------------
# Pump-station controller
# ---------------------------------------------------------------------------
class PumpStation:
    """Three identical pumps, round-robin lead-pump rotation."""

    def __init__(self):
        self.pumps = [
            Pump(f"Pump {i+1}", pump_index=i,
                 calibration=pump_cal_params.get(i, {}))
            for i in range(3)
        ]
        self.level: float = float(profile_level[0])
        self.time_s: float = 0.0
        self.lead_index: int = 0
        self.running_stack: List[int] = []

        # Statistics
        self.daily_starts: dict[int, List[int]]   = {}
        self.daily_runtime: dict[int, List[float]] = {}
        self.hourly_energy: dict[int, List[float]] = {}
        self.active_peaks = []
        self.next_peak = random.expovariate(peak_rate)

    # ---------------------------------------------------------------
    # Inflow model
    # ---------------------------------------------------------------
    def inflow_m3h_peak(self) -> float:
        return self.inflow_m3h()

    def inflow_m3h(self) -> float:
        return float(inflow_from_profile(int(self.time_s)))

    # ---------------------------------------------------------------
    # Lead-pump rotation helper
    # ---------------------------------------------------------------
    def _next_lead(self) -> int:
        self.lead_index = (self.lead_index + 1) % len(self.pumps)
        return self.lead_index

    # ---------------------------------------------------------------
    # Control logic
    # ---------------------------------------------------------------
    def _start_pump(self, idx: int):
        self.pumps[idx].start(self.time_s)
        self.running_stack.append(idx)
        day = int(self.time_s // 86400)
        self.daily_starts.setdefault(day, [0, 0, 0])[idx] += 1
        self._next_lead()

    def _stop_last_pump(self):
        if self.running_stack:
            idx = self.running_stack[-1]
            if self.pumps[idx].is_running:
                self.pumps[idx].stop(self.time_s)

    def _stop_all(self):
        for idx in reversed(self.running_stack):
            if self.pumps[idx].is_running:
                self.pumps[idx].stop(self.time_s)

    def control(self):
        measured_level = self.level + random.gauss(0.0, level_meas_noise_std)
        n_run = len(self.running_stack)

        if measured_level >= level_start and n_run == 0:
            self._start_pump(self.lead_index)
        elif n_run == 1 and measured_level >= level_start2:
            self._start_pump(self.lead_index)

        if n_run == 2 and measured_level <= level_stop2:
            self._stop_last_pump()
        if measured_level <= level_stop:
            self._stop_all()

    # ---------------------------------------------------------------
    # Simulation step
    # ---------------------------------------------------------------
    def step(self, dt: float):
        # 1) Update pumps
        for p in self.pumps:
            p.update(dt, self.time_s)
        self.running_stack = [i for i in self.running_stack
                              if self.pumps[i].is_running]

        # 2) Hydraulics – volume balance
        inflow  = self.inflow_m3h()
        outflow = sum(p.flow_m3h for p in self.pumps)
        dV = (inflow - outflow) / 3600 * dt
        self.level += dV / cross_section_area

        # 3) Control
        self.control()

        # 4) Stats accumulation
        day = int(self.time_s // 86400)
        self.daily_runtime.setdefault(day, [0.0, 0.0, 0.0])
        dt_h = dt / 3600
        for i, p in enumerate(self.pumps):
            if p.is_running:
                self.daily_runtime[day][i] += dt_h
        hour = int(self.time_s // 3600)
        self.hourly_energy.setdefault(hour, [0.0, 0.0, 0.0])
        for i, p in enumerate(self.pumps):
            self.hourly_energy[hour][i] += p.input_kW * dt_h

        # 5) Advance time
        self.time_s += dt


# ---------------------------------------------------------------------------
# Run the simulation and verification workflow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sns.set(style="whitegrid")
    station = PumpStation()

    # Logs for later analysis
    t_log: List[float]     = []
    level_log: List[float] = []
    inflow_log: List[float]  = []
    outflow_log: List[float] = []
    level_meas_log: List[float] = []

    flow_logs      = [[] for _ in station.pumps]
    head_logs      = [[] for _ in station.pumps]
    freq_logs      = [[] for _ in station.pumps]
    input_logs     = [[] for _ in station.pumps]
    blockage_logs  = [[] for _ in station.pumps]
    power_logs     = [[] for _ in station.pumps]

    _alloc = sim_time  # max entries
    current1_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    current2_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    current3_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    voltage1_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    voltage2_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    voltage3_logs  = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    power_3ph_logs = [np.zeros(_alloc, dtype=np.float32) for _ in range(3)]
    _log_idx = 0  # shared write index for pre-allocated arrays

    print("Starting verification …")
    _progress_interval = sim_time // 10
    while station.time_s < sim_time:
        # Collect before step
        t_log.append(station.time_s)
        level_log.append(station.level)
        t_idx = int(station.time_s)
        if t_idx % _progress_interval == 0 and t_idx > 0:
            print(f"  … {t_idx/sim_time*100:.0f}% ({t_idx:,} s)", flush=True)
        level_meas_log.append(profile_level[t_idx] if t_idx < len(profile_level)
                              else profile_level[-1])
        inflow_log.append(station.inflow_m3h())
        outflow_log.append(sum(p.flow_m3h for p in station.pumps))

        # Step
        station.step(dt)

        # Log per-pump data
        for i, p in enumerate(station.pumps):
            flow_logs[i].append(p.flow_m3h)
            head_logs[i].append(p.head_m)
            freq_logs[i].append(p.current_freq)
            input_logs[i].append(p.input_kW)
            blockage_logs[i].append(p.blockage_factor)
            power_logs[i].append(p.shaft_kW)
            current1_logs[i][_log_idx]  = p.current_phase1
            current2_logs[i][_log_idx]  = p.current_phase2
            current3_logs[i][_log_idx]  = p.current_phase3
            voltage1_logs[i][_log_idx]  = p.voltage_phase1
            voltage2_logs[i][_log_idx]  = p.voltage_phase2
            voltage3_logs[i][_log_idx]  = p.voltage_phase3
            power_3ph_logs[i][_log_idx] = p.power_3ph_kW
        _log_idx += 1

    # ---------------------------------------------------------------------------
    # Level validation metrics
    # ---------------------------------------------------------------------------
    sim_arr  = np.asarray(level_log)
    meas_arr = np.asarray(level_meas_log)
    rmse = float(np.sqrt(np.mean((sim_arr - meas_arr)**2)))
    mae  = float(np.mean(np.abs(sim_arr - meas_arr)))
    print(f"Level validation → RMSE = {rmse:.3f} m | MAE = {mae:.3f} m")
    print("Simulation complete. Producing plots …")

    # Convert simulation logs to NumPy for easy slicing
    t_arr        = np.asarray(t_log)
    inflow_arr   = np.asarray(inflow_log)
    outflow_arr  = np.asarray(outflow_log)

    # Trim pre-allocated arrays to actual length
    actual_len = _log_idx
    sim_current1 = [current1_logs[i][:actual_len] for i in range(3)]
    sim_current2 = [current2_logs[i][:actual_len] for i in range(3)]
    sim_current3 = [current3_logs[i][:actual_len] for i in range(3)]
    sim_power_3ph = [power_3ph_logs[i][:actual_len] for i in range(3)]
    sim_voltage1 = [voltage1_logs[i][:actual_len] for i in range(3)]
    sim_voltage2 = [voltage2_logs[i][:actual_len] for i in range(3)]
    sim_voltage3 = [voltage3_logs[i][:actual_len] for i in range(3)]
    # Free pre-allocated buffers
    del current1_logs, current2_logs, current3_logs
    del voltage1_logs, voltage2_logs, voltage3_logs, power_3ph_logs
    import gc; gc.collect()

    # ==================================================================
    # Part A: Level validation plot
    # ==================================================================
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    sl = slice(5000, min(30000, len(t_log)))
    ax.plot(t_arr[sl], meas_arr[sl], label="Measured",  color="#3a86ff", lw=1.6)
    ax.plot(t_arr[sl], sim_arr[sl],  label="Simulated", color="#ff006e", lw=1.4, alpha=0.8)
    ax.axhline(level_start, color="black", ls="--", lw=0.9, label="Start/Stop levels")
    ax.axhline(level_stop,  color="black", ls="--", lw=0.9)
    ax.set_ylabel("Water level (m)")
    ax.set_xlabel("Time (s from start)")
    ax.minorticks_on()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "Level_Validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Level_Validation.pdf saved.")

    # ==================================================================
    # Part B – Inflow rate plot
    # ==================================================================
    mask_start, mask_stop = 0, min(25000, len(t_log))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(t_arr[mask_start:mask_stop], inflow_arr[mask_start:mask_stop],
            color="#fb5607", lw=1.6, label="Inflow rate")
    ax.set_ylabel(r"Inflow rate (m$^3$/h)")
    ax.set_xlabel("Time (s)")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(figures_dir / "Inflow_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Inflow_rate.pdf saved.")

    # ==================================================================
    # Part C – Three-phase current verification
    # ==================================================================

    print("\n" + "="*70)
    print("THREE-PHASE CURRENT & POWER VERIFICATION")
    print("="*70)

    # Build aligned arrays – only at valid SCADA timestamps
    compare_len = min(len(t_arr), scada_max_t + 1)
    valid_indices = np.array([t for t in range(compare_len) if t in scada_valid_times],
                             dtype=np.int32)
    n_valid = len(valid_indices)
    print(f"[VERIFY] {n_valid:,} valid SCADA timestamps in overlap region "
          f"({compare_len:,} s).")

    # ------------------------------------------------------------------
    # Per-pump current & power metrics
    # ------------------------------------------------------------------
    pump_prefixes = ["pump1", "pump2", "pump3"]

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        print(f"\n--- {station.pumps[pump_idx].name} ---")

        for phase_label, sim_arr_full, scada_key in [
            ("I_L1", sim_current1[pump_idx], f"{prefix}_current1"),
            ("I_L2", sim_current2[pump_idx], f"{prefix}_current2"),
            ("I_L3", sim_current3[pump_idx], f"{prefix}_current3"),
            ("Power", sim_power_3ph[pump_idx], f"{prefix}_power"),
        ]:
            s_arr = sim_arr_full[valid_indices]
            r_arr = scada_columns[scada_key][valid_indices]
            finite = np.isfinite(r_arr)
            s_arr, r_arr = s_arr[finite], r_arr[finite]

            rmse_v = float(np.sqrt(np.mean((s_arr - r_arr)**2)))
            mae_v  = float(np.mean(np.abs(s_arr - r_arr)))
            mask_nz = r_arr > 1.0
            mape = (float(np.mean(np.abs(s_arr[mask_nz] - r_arr[mask_nz])
                                  / r_arr[mask_nz]) * 100)
                    if mask_nz.sum() > 0 else float("nan"))
            print(f"  {phase_label:6s}  RMSE={rmse_v:7.3f}  MAE={mae_v:7.3f}  "
                  f"MAPE={mape:5.1f}%")

    # ==================================================================
    # Part D – Pump-cycle detection & per-cycle comparison
    # ==================================================================
    print("\n" + "="*70)
    print("PER-CYCLE CURRENT & POWER COMPARISON (complete cycles only)")
    print("="*70)

    cycle_metrics_all = []  # collect for later summary

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        # Detect cycles from SCADA data using the dense array
        scada_run = np.nan_to_num(scada_columns[f"{prefix}_current1"][:compare_len], nan=0.0)
        is_on = scada_run > 0.5

        diff = np.diff(is_on.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        stops  = np.where(diff == -1)[0] + 1

        if len(stops) > 0 and (len(starts) == 0 or stops[0] < starts[0]):
            stops = stops[1:]
        if len(starts) > len(stops):
            starts = starts[:len(stops)]

        n_cycles = len(starts)
        complete_count = 0
        cycle_rmse_i = []
        cycle_rmse_p = []
        cycle_mae_i  = []
        cycle_mae_p  = []

        for ci in range(n_cycles):
            t_on, t_off = int(starts[ci]), int(stops[ci])
            cycle_len = t_off - t_on

            if cycle_len < 5:
                continue

            # Check all timestamps present (vectorized)
            cycle_range = np.arange(t_on, t_off)
            scada_vals = scada_columns[f"{prefix}_current1"][t_on:t_off]
            if np.any(np.isnan(scada_vals)):
                continue  # skip cycles with gaps

            if t_off > len(sim_current1[pump_idx]):
                continue

            complete_count += 1

            # Sim average current across 3 phases
            s_i_avg = (sim_current1[pump_idx][t_on:t_off].astype(np.float64) +
                       sim_current2[pump_idx][t_on:t_off].astype(np.float64) +
                       sim_current3[pump_idx][t_on:t_off].astype(np.float64)) / 3.0
            r_i_avg = (scada_columns[f"{prefix}_current1"][t_on:t_off].astype(np.float64) +
                       scada_columns[f"{prefix}_current2"][t_on:t_off].astype(np.float64) +
                       scada_columns[f"{prefix}_current3"][t_on:t_off].astype(np.float64)) / 3.0
            s_pw = sim_power_3ph[pump_idx][t_on:t_off].astype(np.float64)
            r_pw = scada_columns[f"{prefix}_power"][t_on:t_off].astype(np.float64)

            rmse_i = float(np.sqrt(np.mean((s_i_avg - r_i_avg)**2)))
            rmse_p = float(np.sqrt(np.mean((s_pw - r_pw)**2)))
            mae_i  = float(np.mean(np.abs(s_i_avg - r_i_avg)))
            mae_p  = float(np.mean(np.abs(s_pw - r_pw)))

            cycle_rmse_i.append(rmse_i)
            cycle_rmse_p.append(rmse_p)
            cycle_mae_i.append(mae_i)
            cycle_mae_p.append(mae_p)

            cycle_metrics_all.append({
                "pump": pump_name, "cycle": complete_count,
                "t_start": t_on, "t_end": t_off, "duration_s": cycle_len,
                "current_RMSE": rmse_i, "current_MAE": mae_i,
                "power_RMSE": rmse_p, "power_MAE": mae_p,
            })

        print(f"\n  {pump_name}: {n_cycles} total cycles, "
              f"{complete_count} complete (no SCADA gaps)")
        if complete_count > 0:
            print(f"    Avg current RMSE : {np.mean(cycle_rmse_i):.3f} A")
            print(f"    Avg current MAE  : {np.mean(cycle_mae_i):.3f} A")
            print(f"    Avg power RMSE   : {np.mean(cycle_rmse_p):.3f} kW")
            print(f"    Avg power MAE    : {np.mean(cycle_mae_p):.3f} kW")

    # Save cycle metrics to Excel
    if cycle_metrics_all:
        df_cycles = pd.DataFrame(cycle_metrics_all)
        df_cycles.to_excel(output_dir / "Cycle_Metrics.xlsx", index=False)
        print("\n[SAVE] Cycle_Metrics.xlsx saved.")

    # ==================================================================
    # Part D_2 – SCADA-synchronized electrical verification
    # ==================================================================
    print("\n" + "=" * 70)
    print("SCADA-SYNCHRONIZED ELECTRICAL VERIFICATION")
    print("(eliminates timing mismatch – pure electrical model accuracy)")
    print("=" * 70)

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        cal = pump_cal_params.get(pump_idx, {})
        i1n = cal.get("i1_mean", 27.0)
        i2n = cal.get("i2_mean", 27.0)
        i3n = cal.get("i3_mean", 27.0)

        # Detect ON/OFF from SCADA
        c1_raw = np.nan_to_num(scada_columns[f"{prefix}_current1"][:compare_len], nan=0.0)
        is_on = c1_raw > 0.5
        diff = np.diff(is_on.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        stops  = np.where(diff == -1)[0] + 1
        if len(stops) > 0 and (len(starts) == 0 or stops[0] < starts[0]):
            stops = stops[1:]
        if len(starts) > len(stops):
            starts = starts[:len(stops)]

        # Generate synthetic trace using SCADA timing + calibrated model
        synth_c1 = np.zeros(compare_len, dtype=np.float32)
        synth_c2 = np.zeros(compare_len, dtype=np.float32)
        synth_c3 = np.zeros(compare_len, dtype=np.float32)
        synth_pw = np.zeros(compare_len, dtype=np.float32)

        for ci in range(len(starts)):
            t_on, t_off = int(starts[ci]), int(stops[ci])
            for t in range(t_on, min(t_off + soft_stop_steps + 1, compare_len)):
                el_start = t - t_on
                el_stop  = t - t_off
                if t < t_off:
                    frac = min(1.0, (el_start + 1) / soft_start_steps) if el_start < soft_start_steps else 1.0
                else:
                    frac = max(0.0, 1.0 - (el_stop + 1) / soft_stop_steps) if el_stop < soft_stop_steps else 0.0

                if frac > 0:
                    # Use SCADA voltage
                    if t <= scada_max_t:
                        v1 = scada_columns["voltage_1"][t]
                        v2 = scada_columns["voltage_2"][t]
                        v3 = scada_columns["voltage_3"][t]
                        if not (np.isfinite(v1) and np.isfinite(v2) and np.isfinite(v3)):
                            v1, v2, v3 = cal.get("v1_mean", 236.4), cal.get("v2_mean", 236.6), cal.get("v3_mean", 238.2)
                        else:
                            v1, v2, v3 = float(v1), float(v2), float(v3)
                    else:
                        v1, v2, v3 = cal.get("v1_mean", 236.4), cal.get("v2_mean", 236.6), cal.get("v3_mean", 238.2)

                    synth_c1[t] = i1n * frac
                    synth_c2[t] = i2n * frac
                    synth_c3[t] = i3n * frac
                    synth_pw[t] = (v1 * synth_c1[t] + v2 * synth_c2[t] + v3 * synth_c3[t]) * 0.78 / 1000.0

        # Metrics
        sc1 = scada_columns[f"{prefix}_current1"][:compare_len]
        sc2 = scada_columns[f"{prefix}_current2"][:compare_len]
        sc3 = scada_columns[f"{prefix}_current3"][:compare_len]
        spw = scada_columns[f"{prefix}_power"][:compare_len]
        valid = np.isfinite(sc1) & np.isfinite(sc2) & np.isfinite(sc3) & np.isfinite(spw)

        both_run = (synth_c1 > 0.5) & (np.nan_to_num(sc1, nan=0.0) > 0.5) & valid
        both_ss  = (synth_c1 > 25.0) & (np.nan_to_num(sc1, nan=0.0) > 25.0) & valid

        print(f"\n  Pump {pump_idx + 1} (SCADA-sync):")
        for label, syn, real in [("I_L1", synth_c1, sc1), ("I_L2", synth_c2, sc2),
                                  ("I_L3", synth_c3, sc3), ("Power", synth_pw, spw)]:
            m = valid & np.isfinite(real)
            s, r = syn[m].astype(np.float64), real[m].astype(np.float64)
            _rmse_sc = float(np.sqrt(np.mean((s - r) ** 2)))
            _mae_sc  = float(np.mean(np.abs(s - r)))
            nz = r > 1.0
            mape = float(np.mean(np.abs(s[nz] - r[nz]) / r[nz]) * 100) if nz.sum() > 0 else float("nan")
            print(f"    {label:6s} Overall  RMSE={_rmse_sc:7.3f}  MAE={_mae_sc:7.3f}  MAPE={mape:5.1f}%")

        if both_run.sum() > 0:
            s_iavg = ((synth_c1 + synth_c2 + synth_c3) / 3.0)[both_run].astype(np.float64)
            r_iavg = ((np.nan_to_num(sc1, 0) + np.nan_to_num(sc2, 0) + np.nan_to_num(sc3, 0)) / 3.0)[both_run].astype(np.float64)
            rmse_i = float(np.sqrt(np.mean((s_iavg - r_iavg) ** 2)))
            mape_i = float(np.mean(np.abs(s_iavg - r_iavg) / np.maximum(r_iavg, 0.1)) * 100)
            s_p = synth_pw[both_run].astype(np.float64)
            r_p = spw[both_run].astype(np.float64)
            rmse_p = float(np.sqrt(np.mean((s_p - r_p) ** 2)))
            mape_p = float(np.mean(np.abs(s_p - r_p) / np.maximum(r_p, 0.1)) * 100)
            print(f"    I_avg  BothRun  RMSE={rmse_i:7.3f}  MAPE={mape_i:5.1f}%  ({both_run.sum():,} pts)")
            print(f"    Power  BothRun  RMSE={rmse_p:7.3f}  MAPE={mape_p:5.1f}%")

        if both_ss.sum() > 0:
            s_iavg = ((synth_c1 + synth_c2 + synth_c3) / 3.0)[both_ss].astype(np.float64)
            r_iavg = ((np.nan_to_num(sc1, 0) + np.nan_to_num(sc2, 0) + np.nan_to_num(sc3, 0)) / 3.0)[both_ss].astype(np.float64)
            rmse_i = float(np.sqrt(np.mean((s_iavg - r_iavg) ** 2)))
            mape_i = float(np.mean(np.abs(s_iavg - r_iavg) / np.maximum(r_iavg, 0.1)) * 100)
            s_p = synth_pw[both_ss].astype(np.float64)
            r_p = spw[both_ss].astype(np.float64)
            rmse_p = float(np.sqrt(np.mean((s_p - r_p) ** 2)))
            mape_p = float(np.mean(np.abs(s_p - r_p) / np.maximum(r_p, 0.1)) * 100)
            print(f"    I_avg  StdySt   RMSE={rmse_i:7.3f}  MAPE={mape_i:5.2f}%  ({both_ss.sum():,} pts)")
            print(f"    Power  StdySt   RMSE={rmse_p:7.3f}  MAPE={mape_p:5.2f}%")
    # ------------------------------------------------------------------
    # Common plotting window and styles
    # ------------------------------------------------------------------
    plot_start_hour = 0.0
    plot_end_hour   = 24.0

    # Nice contrasting colors
    color_scada = "#1f77b4"      # muted blue
    color_sim   = "#d95f02"      # warm orange
    color_meas  = "#2a9d8f"      # teal (for measured level if needed)
    color_extra = "#6c757d"      # gray for thresholds etc.

    def get_hour_window_mask(t_seconds, start_hour=plot_start_hour, end_hour=plot_end_hour):
        """Return boolean mask for a selected hour window."""
        t_hours = np.asarray(t_seconds) / 3600.0
        return (t_hours >= start_hour) & (t_hours <= end_hour)

    def style_legend_below(ax, ncol=2, yshift=-0.28):
        """Place legend centered below x-axis."""
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, yshift),
            ncol=ncol,
            frameon=True,
            borderaxespad=0.0
    )
    t_arr = np.asarray(t_log)
    inflow_arr = np.asarray(inflow_log)
    outflow_arr = np.asarray(outflow_log)
    plot_mask = get_hour_window_mask(t_arr)
    t_hours_plot = t_arr[plot_mask] / 3600.0
    # ==================================================================
    # PART E – Long-duration current comparison (first 24h, downsampled)
    # ==================================================================
    ds = 10
    day_len = min(86400, compare_len)

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        t_ds = np.arange(0, day_len, ds)
        s_avg = ((sim_current1[pump_idx][:day_len:ds].astype(np.float64) +
                  sim_current2[pump_idx][:day_len:ds].astype(np.float64) +
                  sim_current3[pump_idx][:day_len:ds].astype(np.float64)) / 3.0)
        r_avg = (np.nan_to_num(scada_columns[f"{prefix}_current1"][:day_len:ds], nan=0.0).astype(np.float64) +
                 np.nan_to_num(scada_columns[f"{prefix}_current2"][:day_len:ds], nan=0.0).astype(np.float64) +
                 np.nan_to_num(scada_columns[f"{prefix}_current3"][:day_len:ds], nan=0.0).astype(np.float64)) / 3.0

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(t_ds / 3600, r_avg, color="#3a86ff", lw=0.8,
                label="SCADA avg current", alpha=0.9)
        ax.plot(t_ds / 3600, s_avg, color="#ff006e", lw=0.7,
                label="Sim avg current", alpha=0.75)
        ax.set_ylabel("Average Phase Current (A)")
        ax.set_xlabel("Time (hours)")
        ax.set_title(f"{pump_name}", fontsize=12)
        ax.minorticks_on()
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fname = f"Current_24h_{pump_name.replace(' ', '_')}.pdf"
        fig.savefig(figures_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {fname} saved.")

    # ==================================================================
    # Global setting for Matplotlib
    # ==================================================================
    mpl.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    # ==================================================================
    # PART F – level validation
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    ax.plot(
        t_arr[plot_mask] / 3600.0,
        meas_arr[plot_mask],
        label="Measured",
        color=color_meas,
        lw=1.8
    )
    ax.plot(
        t_arr[plot_mask] / 3600.0,
        sim_arr[plot_mask],
        label="Simulated",
        color=color_sim,
        lw=1.6,
        alpha=0.9
    )

    ax.axhline(level_start, color=color_extra, ls="--", lw=1.0, label="Start/Stop levels")
    ax.axhline(level_stop,  color=color_extra, ls="--", lw=1.0)

    ax.set_ylabel("Water level (m)")
    ax.set_xlabel("Time (hours)")
    ax.set_xlim(plot_start_hour, plot_end_hour)
    #ax.set_title(f"Water level overview ({PLOT_START_HOUR:.0f}–{PLOT_END_HOUR:.0f} h)")
    ax.minorticks_on()

    style_legend_below(ax, ncol=3, yshift=-0.30)

    fig.tight_layout()
    fig.savefig(figures_dir / "Level_Validation.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "Level_Validation.eps", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Level_Validation.pdf saved.")

    # ==================================================================
    # PART G – Inflow rate
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    ax.plot(
        t_arr[plot_mask] / 3600.0,
        inflow_arr[plot_mask],
        color="#4c956c",
        lw=1.6,
        label="Inflow rate"
    )

    ax.set_ylabel(r"Inflow rate ($\mathrm{m^3/h}$)")
    ax.set_xlabel("Time (hours)")
    ax.set_xlim(plot_start_hour, plot_end_hour)
    #ax.set_title(f"Inflow overview ({PLOT_START_HOUR:.0f}–{PLOT_END_HOUR:.0f} h)")
    ax.minorticks_on()

    style_legend_below(ax, ncol=1, yshift=-0.26)

    fig.tight_layout()
    fig.savefig(figures_dir / "Inflow_rate.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "Inflow_rate.eps", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Inflow_rate.pdf saved.")
   # ==================================================================
    # PART H – long-duration current comparison (selected hour window)
    # ==================================================================
    ds = 10
    day_len = min(86400, compare_len)

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        t_ds = np.arange(0, day_len, ds)
        t_ds_h = t_ds / 3600.0
        mask_ds = (t_ds_h >= plot_start_hour) & (t_ds_h <= plot_end_hour)

        s_avg = (
            sim_current1[pump_idx][:day_len:ds].astype(np.float64) +
            sim_current2[pump_idx][:day_len:ds].astype(np.float64) +
            sim_current3[pump_idx][:day_len:ds].astype(np.float64)
        ) / 3.0

        r_avg = (
            np.nan_to_num(scada_columns[f"{prefix}_current1"][:day_len:ds], nan=0.0).astype(np.float64) +
            np.nan_to_num(scada_columns[f"{prefix}_current2"][:day_len:ds], nan=0.0).astype(np.float64) +
            np.nan_to_num(scada_columns[f"{prefix}_current3"][:day_len:ds], nan=0.0).astype(np.float64)
        ) / 3.0

        fig, ax = plt.subplots(figsize=(10, 3.8))

        ax.plot(
            t_ds_h[mask_ds], r_avg[mask_ds],
            color=color_scada, lw=1.1, alpha=0.95,
            label=r"SCADA avg. current"
        )
        ax.plot(
            t_ds_h[mask_ds], s_avg[mask_ds],
            color=color_sim, lw=1.1, alpha=0.90,
            label=r"Simulated avg. current"
        )

        ax.set_ylabel(r"Average phase current (A)")
        ax.set_xlabel(r"Time (hours)")
        ax.set_xlim(plot_start_hour, plot_end_hour)
        ax.set_title(rf"{pump_name}",
                    fontsize=13)
        ax.minorticks_on()

        style_legend_below(ax, ncol=2, yshift=-0.30)

        fig.tight_layout()
        fname = f"Current_{int(plot_start_hour)}to{int(plot_end_hour)}h_{pump_name.replace(' ', '_')}.pdf"
        fname_1 = f"Current_{int(plot_start_hour)}to{int(plot_end_hour)}h_{pump_name.replace(' ', '_')}.eps"
        fig.savefig(figures_dir / fname, bbox_inches="tight")
        fig.savefig(figures_dir / fname_1, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {fname} saved.")

    # ==================================================================
    # PART H2 – Long duration power compariosn (selected hour window)
    # ==================================================================
    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        t_ds = np.arange(0, day_len, ds)
        t_ds_h = t_ds / 3600.0
        mask_ds = (t_ds_h >= plot_start_hour) & (t_ds_h <= plot_end_hour)

        s_pw = sim_power_3ph[pump_idx][:day_len:ds].astype(np.float64)
        r_pw = np.nan_to_num(scada_columns[f"{prefix}_power"][:day_len:ds], nan=0.0).astype(np.float64)

        fig, ax = plt.subplots(figsize=(10, 3.8))

        ax.plot(
            t_ds_h[mask_ds], r_pw[mask_ds],
            color=color_meas, lw=1.1, alpha=0.95,
            label=r"SCADA power"
        )
        ax.plot(
            t_ds_h[mask_ds], s_pw[mask_ds],
            color=color_sim, lw=1.1, alpha=0.90,
            label=r"Simulated power"
        )

        ax.set_ylabel(r"Three-phase power (kW)")
        ax.set_xlabel(r"Time (hours)")
        ax.set_xlim(plot_start_hour, plot_end_hour)
        ax.set_title(rf"{pump_name}",
                    fontsize=13)
        ax.minorticks_on()

        style_legend_below(ax, ncol=2, yshift=-0.30)

        fig.tight_layout()
        fname = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_{pump_name.replace(' ', '_')}.pdf"
        fname_1 = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_{pump_name.replace(' ', '_')}.eps"
        fig.savefig(figures_dir / fname, bbox_inches="tight")
        fig.savefig(figures_dir / fname_1, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {fname} saved.")
        
    

    # ==================================================================
    # PART I – Scatter plot: Current (Sim vs SCADA)
    # ==================================================================
    _scatter_len = min(compare_len, actual_len, len(scada_columns[f"pump1_current1"]))
    subsample = np.arange(0, _scatter_len, 5)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for pump_idx, ax in enumerate(axes):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        s_avg = ((sim_current1[pump_idx][subsample].astype(np.float64) +
                  sim_current2[pump_idx][subsample].astype(np.float64) +
                  sim_current3[pump_idx][subsample].astype(np.float64)) / 3.0)
        r_avg = (np.nan_to_num(scada_columns[f"{prefix}_current1"][subsample], nan=0.0).astype(np.float64) +
                 np.nan_to_num(scada_columns[f"{prefix}_current2"][subsample], nan=0.0).astype(np.float64) +
                 np.nan_to_num(scada_columns[f"{prefix}_current3"][subsample], nan=0.0).astype(np.float64)) / 3.0

        mask = (s_avg > 0.5) & (r_avg > 0.5)
        if mask.sum() > 0:
            ax.scatter(r_avg[mask], s_avg[mask], s=1, alpha=0.3, color="#6a4c93")
            lims = [0, max(r_avg[mask].max(), s_avg[mask].max()) * 1.05]
            ax.plot(lims, lims, 'k--', lw=0.8, label=r"1\!:\!1 line")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        ax.set_xlabel(r"SCADA current (A)")
        ax.set_ylabel(r"Simulated current (A)")
        ax.set_title(rf"{pump_name}", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(r"Current scatter: Simulated vs.\ SCADA (phase average)",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "Current_Scatter_All_Pumps.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "Current_Scatter_All_Pumps.eps", bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Current_Scatter_All_Pumps.pdf saved.")
    # ==================================================================
    # PART J – Export full verification data
    # ==================================================================
    print("\n[EXPORT] Writing full verification data …")
    try:
        timestamps = [t0 + timedelta(seconds=int(s)) for s in t_log]
        data = {
            "Timestamp": timestamps,
            "Water_Level_m": level_log,
            "Total_Inflow_m3h": inflow_log,
            "Total_Outflow_m3h": outflow_log,
        }
        for i, pump in enumerate(station.pumps, start=1):
            data[f"Pump{i}_Flow_m3h"]    = flow_logs[i - 1]
            data[f"Pump{i}_Head_m"]      = head_logs[i - 1]
            data[f"Pump{i}_Freq_Hz"]     = freq_logs[i - 1]
            data[f"Pump{i}_Input_kW"]    = input_logs[i - 1]
            # V2 three-phase outputs
            data[f"Pump{i}_Current_L1_A"]  = sim_current1[i - 1].tolist()
            data[f"Pump{i}_Current_L2_A"]  = sim_current2[i - 1].tolist()
            data[f"Pump{i}_Current_L3_A"]  = sim_current3[i - 1].tolist()
            data[f"Pump{i}_Voltage_L1_V"]  = sim_voltage1[i - 1].tolist()
            data[f"Pump{i}_Voltage_L2_V"]  = sim_voltage2[i - 1].tolist()
            data[f"Pump{i}_Voltage_L3_V"]  = sim_voltage3[i - 1].tolist()
            data[f"Pump{i}_Power_3ph_kW"]  = sim_power_3ph[i - 1].tolist()

        df_out = pd.DataFrame(data)
        del data; gc.collect()
        
        if "Timestamp" in df_out.columns:
            if pd.api.types.is_datetime64_any_dtype(df_out["Timestamp"]):
                try:
                    df_out["Timestamp"] = df_out["Timestamp"].dt.tz_localize(None)
                except TypeError:
                    pass
            else:
                df_out["Timestamp"] = pd.to_datetime(df_out["Timestamp"]).dt.tz_localize(None)

        # Save as CSV instead of XLSX to reduce memory usage
        output_path = output_dir / "Verification_V2.csv"
        df_out.to_csv(output_path, index=False)
        del df_out; gc.collect()
        print(f"[SAVE] Results saved → {output_path.resolve()}")
    except MemoryError:
        print("[WARN] Not enough memory for full export; skipping.")

    # ==================================================================
    # PART K – summary metrics 
    # ==================================================================
    print("\n" + "="*70)
    print("SUMMARY OF ALL VERIFICATION METRICS")
    print("="*70)
    print(f"  Level  RMSE = {rmse:.3f} m  |  MAE = {mae:.3f} m")

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name
        p_cycles = [m for m in cycle_metrics_all if m["pump"] == pump_name]
        if p_cycles:
            avg_ci = np.mean([m["current_RMSE"] for m in p_cycles])
            avg_pi = np.mean([m["power_RMSE"]   for m in p_cycles])
            print(f"  {pump_name}  Current cycle-RMSE = {avg_ci:.3f} A  |  "
                  f"Power cycle-RMSE = {avg_pi:.3f} kW  "
                  f"({len(p_cycles)} complete cycles)")
        else:
            print(f"  {pump_name}  No complete cycles found for comparison.")
    #%% # ==================================================================
    # Part L: Current Comparison-Whole 
    #=========================================================================
    ds = 10
    day_len = min(86400, compare_len)

    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(10, 9),
        sharex=True
    )

    for pump_idx, ax in enumerate(axes):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        t_ds = np.arange(0, day_len, ds)

        s_avg = (
            sim_current1[pump_idx][:day_len:ds].astype(np.float64) +
            sim_current2[pump_idx][:day_len:ds].astype(np.float64) +
            sim_current3[pump_idx][:day_len:ds].astype(np.float64)
        ) / 3.0

        r_avg = (
            np.nan_to_num(scada_columns[f"{prefix}_current1"][:day_len:ds], nan=0.0).astype(np.float64) +
            np.nan_to_num(scada_columns[f"{prefix}_current2"][:day_len:ds], nan=0.0).astype(np.float64) +
            np.nan_to_num(scada_columns[f"{prefix}_current3"][:day_len:ds], nan=0.0).astype(np.float64)
        ) / 3.0

        # Plot real first, then simulated
        ax.plot(
            t_ds / 3600, r_avg,
            color="#3a86ff", lw=1.2, alpha=0.9,
            label="Real"
        )
        ax.plot(
            t_ds / 3600, s_avg,
            color="#ff006e", lw=1.1, alpha=0.75,
            label="Simulated"
        )

        ax.set_ylabel("Avg Phase\nCurrent (A)")
        ax.set_title(pump_name, fontsize=12)
        ax.minorticks_on()

    # Only bottom subplot gets x-label
    axes[-1].set_xlabel("Time (hours)")

    # Create one common legend under the x-axis
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 0]   # Simulated first, then Real
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )
    # Leave space at bottom for legend
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fname = "Current_24h_All_Pumps_Subplots.pdf"
    fname_1 = "Current_24h_All_Pumps_Subplots.eps"
    fig.savefig(figures_dir / fname, bbox_inches="tight")
    fig.savefig(figures_dir / fname_1, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {fname} saved.")
    
    #%% ==================================================================
    #M : Power Comparison: Whole 
    #======================================================================

    fig, axes = plt.subplots(
    nrows=3, ncols=1,
    figsize=(10, 10),
    sharex=True
    )

    for pump_idx, ax in enumerate(axes):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        t_ds = np.arange(0, day_len, ds)
        t_ds_h = t_ds / 3600.0
        mask_ds = (t_ds_h >= plot_start_hour) & (t_ds_h <= plot_end_hour)

        s_pw = sim_power_3ph[pump_idx][:day_len:ds].astype(np.float64)
        r_pw = np.nan_to_num(
            scada_columns[f"{prefix}_power"][:day_len:ds],
            nan=0.0
        ).astype(np.float64)

        # Plot real and simulated
        ax.plot(
            t_ds_h[mask_ds], r_pw[mask_ds],
            color=color_meas, lw=1.1, alpha=0.95,
            label="Real"
        )
        ax.plot(
            t_ds_h[mask_ds], s_pw[mask_ds],
            color=color_sim, lw=1.1, alpha=0.90,
            label="Simulated"
        )
        
        ax.set_ylabel("Three-phase\npower (kW)")
        ax.set_xlim(plot_start_hour, plot_end_hour)
        ax.set_title(pump_name, fontsize=13)
        ax.minorticks_on()

    # Only the bottom subplot gets the x-label
    axes[-1].set_xlabel("Time (hours)")

    # One common legend under all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 0]   # Simulated first, then Real
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    # Leave space at bottom for the legend
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fname = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_All_Pumps.pdf"
    fname_1 = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_All_Pumps.eps"

    fig.savefig(figures_dir / fname, bbox_inches="tight")
    fig.savefig(figures_dir / fname_1, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] {fname} saved.")
    print(f"[PLOT] {fname_1} saved.")
# =================================================================
    # ------------------------------------------------------------------
    # Concordant analysis: metrics when both sim & SCADA agree on state
    # ------------------------------------------------------------------
    print("\n" + "-"*70)
    print("CONCORDANT ANALYSIS (both sim & SCADA agree pump is running)")
    print("-"*70)

    for pump_idx in range(3):
        prefix = pump_prefixes[pump_idx]
        pump_name = station.pumps[pump_idx].name

        sim_c1 = sim_current1[pump_idx][:compare_len].astype(np.float64)
        sim_pw = sim_power_3ph[pump_idx][:compare_len].astype(np.float64)
        r_c1 = np.nan_to_num(scada_columns[f"{prefix}_current1"][:compare_len], nan=0.0).astype(np.float64)
        r_pw = np.nan_to_num(scada_columns[f"{prefix}_power"][:compare_len], nan=0.0).astype(np.float64)

        both_running = (sim_c1 > 0.5) & (r_c1 > 0.5)
        both_off     = (sim_c1 < 0.5) & (r_c1 < 0.5)
        both_ss      = (sim_c1 > 25.0) & (r_c1 > 25.0)

        n = compare_len
        print(f"\n  {pump_name}:")
        print(f"    State agreement: {(both_running.sum()+both_off.sum())/n*100:.1f}%  "
              f"(both-run={both_running.sum():,}, both-off={both_off.sum():,}, "
              f"mismatch={(n-both_running.sum()-both_off.sum()):,})")

        if both_running.sum() > 0:
            rmse_i = float(np.sqrt(np.mean((sim_c1[both_running] - r_c1[both_running])**2)))
            mape_i = float(np.mean(np.abs(sim_c1[both_running] - r_c1[both_running])
                                   / r_c1[both_running]) * 100)
            rmse_p = float(np.sqrt(np.mean((sim_pw[both_running] - r_pw[both_running])**2)))
            mape_p = float(np.mean(np.abs(sim_pw[both_running] - r_pw[both_running])
                                   / np.maximum(r_pw[both_running], 0.1)) * 100)
            print(f"    [Both running] Current RMSE={rmse_i:.3f} A, MAPE={mape_i:.1f}%")
            print(f"    [Both running] Power   RMSE={rmse_p:.3f} kW, MAPE={mape_p:.1f}%")

        if both_ss.sum() > 0:
            rmse_i = float(np.sqrt(np.mean((sim_c1[both_ss] - r_c1[both_ss])**2)))
            mape_i = float(np.mean(np.abs(sim_c1[both_ss] - r_c1[both_ss])
                                   / r_c1[both_ss]) * 100)
            rmse_p = float(np.sqrt(np.mean((sim_pw[both_ss] - r_pw[both_ss])**2)))
            mape_p = float(np.mean(np.abs(sim_pw[both_ss] - r_pw[both_ss])
                                   / np.maximum(r_pw[both_ss], 0.1)) * 100)
            print(f"    [Steady-state] Current RMSE={rmse_i:.3f} A, MAPE={mape_i:.2f}%  "
                  f"({both_ss.sum():,} samples)")
            print(f"    [Steady-state] Power   RMSE={rmse_p:.3f} kW, MAPE={mape_p:.2f}%")

    print("\nDone.")

#%% ==================================================================
# Part N: Current Comparison — Whole, with error track
# ====================================================================
import matplotlib.gridspec as gridspec

ds = 10
day_len = min(86400, compare_len)

fig = plt.figure(figsize=(10, 11))
gs = gridspec.GridSpec(
    nrows=6, ncols=1,
    height_ratios=[1, 0.45, 1, 0.45, 1, 0.45],
    hspace=0.15
)

axes_main = []
axes_err  = []

for pump_idx in range(3):
    prefix = pump_prefixes[pump_idx]
    pump_name = station.pumps[pump_idx].name

    ax_main = fig.add_subplot(gs[pump_idx * 2, 0])
    ax_err  = fig.add_subplot(gs[pump_idx * 2 + 1, 0], sharex=ax_main)
    axes_main.append(ax_main)
    axes_err.append(ax_err)

    t_ds = np.arange(0, day_len, ds)
    t_ds_h = t_ds / 3600.0

    s_avg = (
        sim_current1[pump_idx][:day_len:ds].astype(np.float64) +
        sim_current2[pump_idx][:day_len:ds].astype(np.float64) +
        sim_current3[pump_idx][:day_len:ds].astype(np.float64)
    ) / 3.0

    r_avg = (
        np.nan_to_num(scada_columns[f"{prefix}_current1"][:day_len:ds], nan=0.0).astype(np.float64) +
        np.nan_to_num(scada_columns[f"{prefix}_current2"][:day_len:ds], nan=0.0).astype(np.float64) +
        np.nan_to_num(scada_columns[f"{prefix}_current3"][:day_len:ds], nan=0.0).astype(np.float64)
    ) / 3.0

    err = s_avg - r_avg

    # --- Main panel: Real vs Simulated ---
    ax_main.plot(t_ds_h, r_avg, color="#3a86ff", lw=1.2, alpha=0.9, label="Real")
    ax_main.plot(t_ds_h, s_avg, color="#ff006e", lw=1.1, alpha=0.75, label="Simulated")
    ax_main.set_ylabel("Avg Phase\nCurrent (A)")
    ax_main.set_title(pump_name, fontsize=12)
    ax_main.minorticks_on()
    ax_main.tick_params(labelbottom=False)  # hide x-tick labels on main axis

    # --- Error panel ---
    ax_err.fill_between(t_ds_h, 0, err, color="#6c757d", alpha=0.55,
                        linewidth=0, label="Error (Sim − Real)")
    ax_err.axhline(0.0, color="black", lw=0.6, ls="-")
    ax_err.set_ylabel("Error (A)")
    ax_err.minorticks_on()

    # Symmetric y-limits around zero, capped to avoid extreme spikes dominating
    err_abs_max = float(np.nanpercentile(np.abs(err), 99.5))
    if err_abs_max < 1.0:
        err_abs_max = 1.0
    ax_err.set_ylim(-err_abs_max * 1.1, err_abs_max * 1.1)

    # Add RMSE/MAE annotation in the error panel
    rmse_v = float(np.sqrt(np.mean(err**2)))
    mae_v  = float(np.mean(np.abs(err)))
    ax_err.text(
        0.99, 0.92,
        f"RMSE = {rmse_v:.2f} A   MAE = {mae_v:.2f} A",
        transform=ax_err.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="0.7", alpha=0.85)
    )

    # Hide x-tick labels on all but the bottom error panel
    if pump_idx < 2:
        ax_err.tick_params(labelbottom=False)

axes_err[-1].set_xlabel("Time (hours)")

# One legend at the bottom
handles_m, labels_m = axes_main[0].get_legend_handles_labels()
handles_e, labels_e = axes_err[0].get_legend_handles_labels()
order = [1, 0]  # Simulated, Real
fig.legend(
    [handles_m[i] for i in order] + handles_e,
    [labels_m[i] for i in order] + labels_e,
    loc="lower center", ncol=3, frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)
fig.tight_layout(rect=[0, 0.05, 1, 1])

fname   = "Current_24h_All_Pumps_Subplots.pdf"
fname_1 = "Current_24h_All_Pumps_Subplots.eps"
fig.savefig(figures_dir / fname, bbox_inches="tight")
fig.savefig(figures_dir / fname_1, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] {fname} saved (with error track).")


#%% ==================================================================
# Part O: Power Comparison — Whole, with error track
# ====================================================================
fig = plt.figure(figsize=(10, 11))
gs = gridspec.GridSpec(
    nrows=6, ncols=1,
    height_ratios=[1, 0.45, 1, 0.45, 1, 0.45],
    hspace=0.15
)

axes_main = []
axes_err  = []

for pump_idx in range(3):
    prefix = pump_prefixes[pump_idx]
    pump_name = station.pumps[pump_idx].name

    ax_main = fig.add_subplot(gs[pump_idx * 2, 0])
    ax_err  = fig.add_subplot(gs[pump_idx * 2 + 1, 0], sharex=ax_main)
    axes_main.append(ax_main)
    axes_err.append(ax_err)

    t_ds = np.arange(0, day_len, ds)
    t_ds_h = t_ds / 3600.0
    mask_ds = (t_ds_h >= plot_start_hour) & (t_ds_h <= plot_end_hour)

    s_pw = sim_power_3ph[pump_idx][:day_len:ds].astype(np.float64)
    r_pw = np.nan_to_num(scada_columns[f"{prefix}_power"][:day_len:ds], nan=0.0).astype(np.float64)

    err = s_pw - r_pw

    ax_main.plot(t_ds_h[mask_ds], r_pw[mask_ds],
                 color=color_meas, lw=1.1, alpha=0.95, label="Real")
    ax_main.plot(t_ds_h[mask_ds], s_pw[mask_ds],
                 color=color_sim, lw=1.1, alpha=0.90, label="Simulated")
    ax_main.set_ylabel("Three-phase\npower (kW)")
    ax_main.set_title(pump_name, fontsize=12)
    ax_main.set_xlim(plot_start_hour, plot_end_hour)
    ax_main.minorticks_on()
    ax_main.tick_params(labelbottom=False)

    ax_err.fill_between(t_ds_h[mask_ds], 0, err[mask_ds],
                        color="#6c757d", alpha=0.55, linewidth=0,
                        label="Error (Sim − Real)")
    ax_err.axhline(0.0, color="black", lw=0.6)
    ax_err.set_ylabel("Error (kW)")
    ax_err.minorticks_on()

    err_abs_max = float(np.nanpercentile(np.abs(err[mask_ds]), 99.5))
    if err_abs_max < 0.5:
        err_abs_max = 0.5
    ax_err.set_ylim(-err_abs_max * 1.1, err_abs_max * 1.1)

    rmse_v = float(np.sqrt(np.mean(err[mask_ds]**2)))
    mae_v  = float(np.mean(np.abs(err[mask_ds])))
    ax_err.text(
        0.99, 0.92,
        f"RMSE = {rmse_v:.2f} kW   MAE = {mae_v:.2f} kW",
        transform=ax_err.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="0.7", alpha=0.85)
    )

    if pump_idx < 2:
        ax_err.tick_params(labelbottom=False)

axes_err[-1].set_xlabel("Time (hours)")

handles_m, labels_m = axes_main[0].get_legend_handles_labels()
handles_e, labels_e = axes_err[0].get_legend_handles_labels()
order = [1, 0]
fig.legend(
    [handles_m[i] for i in order] + handles_e,
    [labels_m[i] for i in order] + labels_e,
    loc="lower center", ncol=3, frameon=True,
    bbox_to_anchor=(0.5, 0.01)
)
fig.tight_layout(rect=[0, 0.05, 1, 1])

fname   = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_All_Pumps.pdf"
fname_1 = f"Power_{int(plot_start_hour)}to{int(plot_end_hour)}h_All_Pumps.eps"
fig.savefig(figures_dir / fname, bbox_inches="tight")
fig.savefig(figures_dir / fname_1, bbox_inches="tight")
plt.close(fig)
print(f"[PLOT] {fname} saved (with error track).")
