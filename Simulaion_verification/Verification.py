# %%
"""Validate the pump-station simulator against recorded sump-level data.

The script replays a measured inflow profile, compares the simulated
wet-well level with the measured level, and saves the validation plots
and summary tables used for manuscript figures.
"""

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Final


rng = np.random.default_rng(seed=42)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Wet‑well geometry
cross_section_area = 8.0  # m²

# Frequency & speed
pump_min_freq = 25.0      # Hz (used only during soft‑start)
pump_max_freq = 50.0      # Hz (nominal full‑speed)
nominal_freq  = 50.0       # Hz
nominal_rpm   = 1460.0     # rpm at 50 Hz

# Level control set‑points
level_start  = 1.60  # m – if no pump running, start lead pump
level_stop   = 0.5  # m – emergency stop all
level_start_2 = 1.8  # m – start second pump when one already running
level_stop_2  = 0.80  # m – stop second pump when two running

# Peak-flow settings used by the validation scenario
peak_rate = 0.0005  # Events per second (≈18 events/day)
peak_magnitude = 60.0  # Extra flow during peaks (m³/h)
peak_duration = 900  # Peak duration in seconds (15 minutes)

# 50 Hz pump curve (US gpm, m)
pump_curve_data_50hz: List[Tuple[float, float]] = [
    (0,    32.0),
    (400,  29.9),
    (800,  27.4),
    (1200, 24.4),
    (1600, 21.3),
    (2000, 18.3),
]

# System curve parameters: H = H_s + K·Q² (Q in m³/h, H in m)
static_head    = 2.0
friction_coeff = 0.0006

# Reference system-curve constants
friction_coeff_base = friction_coeff
static_head_base    = static_head

# When (during day 2) the pipe-clogging ramp starts/stops (for fault simulation purpose):
system_fault_start       = 86400 + 40000    # day 2,  40 000 s into it
system_fault_end         = 86400 + 61600    # day 2,  61 600 s into it

# End-of-ramp changes for the fault scenario
system_friction_increase = 1.0             # +100% friction
system_static_head_increase = 0.5           # +0.5 m static head

# Physics & efficiency
pump_efficiency = 0.90
water_density   = 1000.0  # kg/m³
gravity         = 9.81    # m/s²

# Electrical specs (approx.)
nominal_voltage = 400      # V (three‑phase)
nominal_current = 30.0     # A (average)
power_factor    = 0.9

# Unit conversion
usgpm_to_m3h = 0.2271

# Soft‑start/stop slopes
soft_start_time = 10.0  # s
soft_stop_time  = 10.0  # s

# Simulation horizon
dt       = 1.0        # s, discrete time‑step

# Measurement noise
level_meas_noise_std     = 0.02   # ±2 cm
flow_noise_relative_std  = 0.01   # 1 %
power_noise_relative_std = 0.01   # 1 %




# ------------------------------------------------------------------
# Inflow profile from the recorded data
# ------------------------------------------------------------------
# CSV exported by the flow-rate preprocessing script
flow_csv_path: Final[Path] = Path("data/flow_rates.csv")
real_starts_path: Final[Path] = Path("data/Real_starts.xlsx")
real_runtime_path: Final[Path] = Path("data/Real_Runtime.xlsx")

output_dir = Path("results/simulator_validation")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"

for folder in (output_dir, figures_dir, tables_dir):
    folder.mkdir(parents=True, exist_ok=True)
# ── Load & prepare ────────────────────────────────────────────────
try:
    df_flow = pd.read_csv(flow_csv_path, parse_dates=["time"])
except FileNotFoundError as e:
    raise SystemExit(f"Could not find {flow_csv_path}") from e

# Keep incoming flow only; negative values are treated as zero inflow.
df_flow["flow_in_m3s"] = df_flow["flow_rate_m3_s"].clip(lower=0.0)

# Build a uniform 1-second time grid covering the whole file.
t0           = df_flow["time"].iloc[0]
df_flow["t_rel_s"] = (df_flow["time"] - t0).dt.total_seconds().astype("int64")

# Create a dense, second-by-second profile using forward fill.
max_t        = int(df_flow["t_rel_s"].iloc[-1])
full_index   = pd.RangeIndex(start=0, stop=max_t + 1, step=1)
profile_m3s  = (
    df_flow.set_index("t_rel_s")["flow_in_m3s"]
           .reindex(full_index)               # add missing seconds
           .ffill()            # hold last value
           .to_numpy()                        # → NumPy for speed
)
profile_m3h  = profile_m3s * 3600.0           # simulator works in m³ h⁻¹
profile_len  = profile_m3h.size

print(f"Loaded exact inflow profile: {profile_len:,} s "
      f"({profile_len/86400:.1f} days) from {t0:%Y-%m-%d %H:%M:%S}.")

# ── Measured level column used for validation ─────────────────────────
level_column_candidates = ["level_m", "water_level_m", "level_meter", "level"]

for col in level_column_candidates:          # pick first one that exists
    if col in df_flow.columns:
        df_flow["level_meas_m"] = df_flow[col].astype(float)
        print(f"Using '{col}' as measured level.")
        break
else:
    raise SystemExit("No water-level column found in CSV; "
                     "cannot validate simulation.")

profile_level = (
    df_flow.set_index("t_rel_s")["level_meas_m"]
           .reindex(full_index)
           .ffill()            # hold last value
           .to_numpy()
)

# ── Simulation horizon from the recorded profile ──────────────────────
profile_len = profile_m3h.size               # seconds of data
sim_time    = profile_len - 1
                                   # run the sim for exactly that long
dt          = 1.0                            # leave as 1 s

print(f"Simulation horizon set to {sim_time/86400:.2f} days "
      f"({sim_time:,} s).")

# ------------------------------------------------------------------
# Lookup function (constant-time and no pandas during run)
# ------------------------------------------------------------------
@np.vectorize                       # allows array input if needed
def inflow_from_profile(t_sec: int) -> float:
    """
    Deterministic inflow (m³ h⁻¹) taken directly from the logger file.
    • 0 ≤ t_sec < profile_len   → exact recorded value
    • t_sec ≥ profile_len       → reuse the last known value
    """
    if t_sec < profile_len:
        return profile_m3h[t_sec]
    return profile_m3h[-1]


# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------

def piecewise_linear_interpolation(x: float, pts: List[Tuple[float, float]]) -> float:
    """Linear interpolation on a rate‑head dataset (x assumed monotonic)."""
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
    """Affinity‑scaled pump curve: H ∝ N², Q ∝ N."""
    if speed_ratio <= 0.0:
        return 0.0
    flow_nominal = flow_usgpm / speed_ratio  # back‑scale to nominal curve
    h_nominal    = piecewise_linear_interpolation(flow_nominal, pump_curve_data_50hz)
    return h_nominal * speed_ratio ** 2


def system_head(flow_m3h: float) -> float:
    """System curve H(Q)."""
    return static_head + friction_coeff * flow_m3h ** 2


def find_operating_flow(rpm: float) -> float:
    """Bisection search for flow (m³/h) where pump‑head = system‑head."""
    if rpm < 50.0:  # tiny speed – treat as no flow
        return 0.0

    ratio = rpm / nominal_rpm
    lo, hi = 0.0, 3000.0  # US gpm bounds
    eps = 0.1
    while hi - lo > eps:
        mid = 0.5 * (lo + hi)
        pump_h = get_pump_head_at_flow(mid, ratio)
        sys_h  = system_head(mid * usgpm_to_m3h)
        (lo, hi) = (mid, hi) if pump_h > sys_h else (lo, mid)

    flow_usgpm = 0.5 * (lo + hi)
    return flow_usgpm * usgpm_to_m3h


def approximate_power_kW(flow_m3h: float, eff: float) -> float:
    """Shaft power P = ρ·g·Q·H / (η·1000)."""
    if flow_m3h <= 0.0 or eff < 1e-9:
        return 0.0
    head_m  = system_head(flow_m3h)
    flow_m3s = flow_m3h / 3600.0
    return water_density * gravity * flow_m3s * head_m / (1000.0 * eff)

# -----------------------------------------------------------------------------
# 3. Pump model
# -----------------------------------------------------------------------------
class Pump:
    """One centrifugal pump with VFD for soft‑starting/stopping."""

    def __init__(self, name: str):
        self.name: str = name

        # Real‑time states
        self.is_running: bool = False
        self.current_freq: float = 0.0      # Hz – what the VFD is producing
        self.blockage_factor: float = 1.0   # 1.0 = clean; <1.0 lowers effective speed
        self.efficiency: float = pump_efficiency

        # Soft‑start/stop bookkeeping
        self._soft_starting: bool = False
        self._soft_stopping: bool = False
        self._soft_start_elapsed: float = 0.0
        self._soft_stop_elapsed: float = 0.0

        # Hydraulics & energy
        self.flow_m3h: float = 0.0
        self.head_m: float = 0.0
        self.shaft_kW: float = 0.0  # hydraulic power
        self.input_kW: float = 0.0  # electrical input
        self.energy_kWh: float = 0.0

    # ---------------------------------------------------------------------
    # Control interface
    # ---------------------------------------------------------------------
    def start(self, t: float) -> None:
        if not self.is_running:
            self.is_running = True
            self._soft_starting = True
            self._soft_start_elapsed = 0.0
            self.current_freq = pump_min_freq
            print(f"[{t:.1f}s] {self.name} START (soft‑start)")

    def stop(self, t: float) -> None:
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            print(f"[{t:.1f}s] {self.name} STOP (soft‑stop)")

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _effective_rpm(self) -> float:
        """Actual hydraulic speed after blockage scaling."""
        rpm_nom = (self.current_freq / nominal_freq) * nominal_rpm
        return rpm_nom * self.blockage_factor

    def _update_soft_ramps(self, dt: float) -> None:
        # Soft‑start
        if self._soft_starting:
            self._soft_start_elapsed += dt
            frac = min(1.0, self._soft_start_elapsed / soft_start_time)
            self.current_freq = pump_min_freq + (pump_max_freq - pump_min_freq) * frac
            if frac >= 1.0:
                self._soft_starting = False
        # Soft‑stop
        if self._soft_stopping:
            self._soft_stop_elapsed += dt
            frac = min(1.0, self._soft_stop_elapsed / soft_stop_time)
            self.current_freq *= 1.0 - frac
            if self.current_freq <= 0.1:
                self.current_freq = 0.0
                self._soft_stopping = False
                self.is_running = False
                print("        … fully stopped")
        # Normal running – hold at full freq if neither ramp active
        if self.is_running and not (self._soft_starting or self._soft_stopping):
            self.current_freq = pump_max_freq

    # ---------------------------------------------------------------------
    # Public update called each time‑step
    # ---------------------------------------------------------------------
    def update(self, dt: float, t: float) -> None:
        self._update_soft_ramps(dt)

        # If stopped, clear hydraulics & energy and quit
        if not (self.is_running or self._soft_stopping):
            self.flow_m3h = self.head_m = self.shaft_kW = self.input_kW = 0.0
            return

        # ---------------- Hydraulics ----------------
        rpm_eff = self._effective_rpm()
        raw_flow = find_operating_flow(rpm_eff)
        # add realistic sensor noise
        flow_noise = 1.0 + random.gauss(0.0, flow_noise_relative_std)
        self.flow_m3h = max(0.0, raw_flow * flow_noise)

        flow_usgpm   = self.flow_m3h / usgpm_to_m3h
        speed_ratio  = rpm_eff / nominal_rpm
        self.head_m  = get_pump_head_at_flow(flow_usgpm, speed_ratio)

        # --------------- Energetics ------------------
        self.shaft_kW = approximate_power_kW(self.flow_m3h, self.efficiency)
        # crude electrical input estimate proportional to freq
        if self.current_freq > 0:
            current_ratio = self.current_freq / pump_max_freq
            I_effective = min(nominal_current * current_ratio / self.blockage_factor,
                               5 * nominal_current)
            self.input_kW = math.sqrt(3) * nominal_voltage * I_effective * power_factor / 1000
            self.input_kW *= 1.0 + random.gauss(0.0, power_noise_relative_std)
        else:
            self.input_kW = 0.0
        self.energy_kWh += self.input_kW * dt / 3600.0

# -----------------------------------------------------------------------------
# 4. Pump-station controller
# -----------------------------------------------------------------------------

class PumpStation:
    """Three identical pumps, round‑robin lead‑pump rotation."""

    def __init__(self):
        self.pumps = [Pump(f"Pump {i+1}") for i in range(3)]
        self.level: float = float(profile_level[0])   # start from measured level
        self.time_s: float = 0.0
        self.lead_index: int = 0
        self.running_stack: List[int] = []  # indices in start order

        # Statistics
        self.daily_starts: dict[int, List[int]] = {}
        self.daily_runtime: dict[int, List[float]] = {}
        self.hourly_energy: dict[int, List[float]] = {}
        self.active_peaks = []  # Track ongoing peak events (end times)
        self.next_peak = random.expovariate(peak_rate)  # Time until next peak

# ---------------------------------------------------------------
# Inflow model — use recorded profile (no peaks, no randomness)
# ---------------------------------------------------------------
    def inflow_m3h_peak(self) -> float:
        """Deprecated: kept for backward compatibility but unused."""
        return self.inflow_m3h()          # call the deterministic one

    def inflow_m3h(self) -> float:
        """
        Deterministic inflow lookup.
        Units: m³ h⁻¹   (simulation uses m³/h everywhere).
        """
        return float(inflow_from_profile(int(self.time_s)))

    # ---------------------------------------------------------------
    # Lead‑pump rotation helper
    # ---------------------------------------------------------------
    def _next_lead(self) -> int:
        self.lead_index = (self.lead_index + 1) % len(self.pumps)
        return self.lead_index

    # ---------------------------------------------------------------
    # Control logic (start/stop commands)
    # ---------------------------------------------------------------
    def _start_pump(self, idx: int):
        self.pumps[idx].start(self.time_s)
        self.running_stack.append(idx)
        # log stats
        day = int(self.time_s // 86400)
        self.daily_starts.setdefault(day, [0, 0, 0])[idx] += 1
        # rotate lead for next event
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

        # ------- start logic -------
        if measured_level >= level_start and n_run == 0:
            self._start_pump(self.lead_index)
        elif n_run == 1 and measured_level >= level_start_2:
            self._start_pump(self.lead_index)

        # ------- stop logic --------
        if n_run == 2 and measured_level <= level_stop_2:
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

        # purge stack of any pumps that completed soft‑stop
        self.running_stack = [i for i in self.running_stack if self.pumps[i].is_running]

        # 2) Hydraulics – volume balance
        inflow  = self.inflow_m3h()
        outflow = sum(p.flow_m3h for p in self.pumps)
        dV = (inflow - outflow) / 3600 * dt
        self.level += dV / cross_section_area

        # 4) Control – start/stop commands
        self.control()

        # 5) Stats accumulation
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

        # 6) advance time
        self.time_s += dt

# -----------------------------------------------------------------------------
# 5. Run validation and create outputs
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sns.set(style="whitegrid")

    station = PumpStation()

    # Logs used for validation and plotting
    t_log: List[float] = []
    level_log: List[float] = []
    inflow_log: List[float] = []
    outflow_log: List[float] = []
    level_meas_log: List[float] = []


    flow_logs  = [[] for _ in station.pumps]
    head_logs  = [[] for _ in station.pumps]
    freq_logs  = [[] for _ in station.pumps]
    input_logs = [[] for _ in station.pumps]
    blockage_logs = [[] for _ in station.pumps]
    efficiency_logs = [[] for _ in station.pumps]
    input_power_logs = [[] for _ in station.pumps]
    power_logs = [[] for _ in station.pumps]

    print("Starting verification …")
    while station.time_s < sim_time:
        # Record one sample before advancing the station.
        t_log.append(station.time_s)
        level_log.append(station.level)
        level_meas_log.append(profile_level[int(station.time_s)])
        inflow_log.append(station.inflow_m3h())
        outflow_log.append(sum(p.flow_m3h for p in station.pumps))
        station.step(dt)

        for i, p in enumerate(station.pumps):
            flow_logs[i].append(p.flow_m3h)
            head_logs[i].append(p.head_m)
            freq_logs[i].append(p.current_freq)
            input_logs[i].append(p.input_kW)
            blockage_logs[i].append(p.blockage_factor)
            power_logs[i].append(p.shaft_kW)
        
    sim_arr  = np.asarray(level_log)
    meas_arr = np.asarray(level_meas_log)

    rmse = float(np.sqrt(np.mean((sim_arr - meas_arr)**2)))
    mae  = float(np.mean(np.abs(sim_arr - meas_arr)))

    print(f"Level validation →  RMSE = {rmse:.3f} m   |   MAE = {mae:.3f} m")

    print("Simulation complete.  Producing plots …")

    # ---------------------------------------------------------------------
    # A) Wet‑well level & flows (first 25 000 s for clarity)
    # ---------------------------------------------------------------------
    idx5k = [i for i, t in enumerate(t_log) if  (system_fault_start - 2000) <= t <= (system_fault_end + 2000)]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(np.array(t_log)[idx5k], np.array(level_log)[idx5k])
    ax[0].axhline(level_start, c="r", ls="--", label="Start 1.6 m")
    ax[0].axhline(level_stop,  c="g", ls="--", label="Stop 0.5 m")
    ax[0].set_ylabel("Level [m]")
    ax[0].legend()

    ax[1].plot(np.array(t_log)[idx5k], np.array(outflow_log)[idx5k], label="Outflow")
    ax[1].set_ylabel("m³/h")
    ax[2].plot(np.array(t_log)[idx5k], np.array(inflow_log)[idx5k], label="Inflow", color="#f7a072")
    ax[2].set_ylabel("m³/h")
    ax[2].set_xlabel("Time [s]")
    for a in ax:
        a.grid(True)
    plt.tight_layout()
    plt.show()
    mask_start = t_log[0]
    mask_stop = t_log[25000]
    mask_5k = [i for i,ts in enumerate(t_log) if   mask_start <= ts <= mask_stop]
    t_5k    = [t_log[i] for i in mask_5k]
    lvl_5k  = [level_log[i] for i in mask_5k]
    inflw_5k= [inflow_log[i] for i in mask_5k]
    outfw_5k= [outflow_log[i] for i in mask_5k]

    freq_5k   = [[] for _ in station.pumps]
    power_5k  = [[] for _ in station.pumps]
    flow_5k   = [[] for _ in station.pumps]
    head_5k   = [[] for _ in station.pumps]
    input_power_5k  = [[] for _ in station.pumps]


    for i in range(3):
        freq_5k[i]  = [freq_logs[i][j]  for j in mask_5k]
        flow_5k[i]  = [flow_logs[i][j]  for j in mask_5k]
        head_5k[i]  = [head_logs[i][j]  for j in mask_5k]
        power_5k[i] = [input_logs[i][j] for j in mask_5k]
    sns.set(style="whitegrid")
    
    #%% ###############################
    # Separate validation plots
    ###################################
    ## Level Error 
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ---------- Global style (journal aesthetic) ----------
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
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
    plt.rcParams['pdf.fonttype'] = 42   # Embed fonts properly
    plt.rcParams['ps.fonttype'] = 42

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    # Measured vs simulated
    ax.plot(t_log[5000:30000], meas_arr[5000:30000],
            label="Measured", color="#3a86ff", lw=1.6)
    ax.plot(t_log[5000:30000], sim_arr[5000:30000],
            label="Simulated", color="#ff006e", lw=1.4, alpha=0.8)

    # Start/stop thresholds
    ax.axhline(level_start, color="black", linestyle="--", lw=0.9, label="Start/Stop levels")
    ax.axhline(level_stop,  color="black", linestyle="--", lw=0.9)

    # Labels
    ax.set_ylabel("Water level")
    ax.set_xlabel("Time (s from start)")

    # Clean spines
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)

    ax.minorticks_on()

    # Legend below plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(figures_dir / "level_validation.pdf", bbox_inches="tight")
    with mpl.rc_context({"text.usetex": False}):
        fig.savefig(figures_dir / "level_validation.eps", bbox_inches="tight")

    plt.show()
#%% ##############################################
# Inflow pattern
##################################################
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ---------- Global style (journal aesthetic) ----------
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
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
    plt.rcParams['pdf.fonttype'] = 42   # Embed fonts properly
    plt.rcParams['ps.fonttype'] = 42

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    ax.plot(t_5k, inflw_5k, color="#fb5607", lw=1.6, label="Inflow rate")

    # Labels
    ax.set_ylabel(r"Inflow rate (m$^3$/h)")
    ax.set_xlabel("Time (s)")

    # Clean spines
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)

    ax.minorticks_on()
        # Legend below plot
    fig.tight_layout()
    fig.savefig(figures_dir / "inflow_rate.pdf", bbox_inches="tight")
    with mpl.rc_context({"text.usetex": False}):
        fig.savefig(figures_dir / "inflow_rate.eps", bbox_inches="tight")

    plt.show()


    #%% #########################################  
    # Total Outflow
    #############################################
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS

    plt.plot(t_5k, outfw_5k, 'c-', color = '#02c39a')
    plt.ylabel(r"Outflow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "total_outflow.eps", format="eps", dpi=300)
    plt.close()

    #%% ###############################################
    # Pump Heads
    ###################################################
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS

    plt.plot(t_5k, head_5k[0], 'b-', label="Pump 1")
    plt.plot(t_5k, head_5k[1], 'r-', label="Pump 2")
    plt.plot(t_5k, head_5k[2], 'g-', label="Pump 3")
    plt.ylabel(r"Head (m)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pump_heads.eps", format="eps", dpi=300)
    plt.close()

    #%% ##############################################
    # Pump Power
    ##################################################
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS
    plt.plot(t_5k, power_5k[0], 'b-', label="Pump 1")
    plt.plot(t_5k, power_5k[1], 'r-', label="Pump 2")
    plt.plot(t_5k, power_5k[2], 'g-', label="Pump 3")
    plt.ylabel(r"Power (kW)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pump_power.eps", format="eps", dpi=300)
    plt.close()

    #%% ##############################################
    # Pump Flow
    ##################################################
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS

    plt.plot(t_5k, flow_5k[0], 'b-', label="Pump 1")
    plt.plot(t_5k, flow_5k[1], 'r-', label="Pump 2")
    plt.plot(t_5k, flow_5k[2], 'g-', label="Pump 3")
    plt.ylabel(r"Flow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pump_flow.eps", format="eps", dpi=300)
    plt.close()
# %%###########################
# Save the verification time series to Excel
################################
t0 = datetime.now().replace(tzinfo=None)

timestamps = [t0 + timedelta(seconds=int(s)) for s in t_log]

data = {
    "Timestamp":          timestamps,
    "Water_Level_m":      level_log,
    "Total_Inflow_m3h":   inflow_log,
    "Total_Outflow_m3h":  outflow_log,
}

for i, pump in enumerate(station.pumps, start=1):
    data[f"Pump{i}_Flow_m3h"] = flow_logs[i - 1]
    data[f"Pump{i}_Head_m"]   = head_logs[i - 1]
    data[f"Pump{i}_Freq_Hz"]  = freq_logs[i - 1]
    data[f"Pump{i}_Input_kW"] = input_logs[i - 1]

df = pd.DataFrame(data)

if "Timestamp" in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        try:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
        except TypeError:
            pass
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)

max_rows_per_sheet = 1_048_500
output_path = tables_dir / "verification.xlsx"

total_rows = len(df)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    start = 0
    sheet_index = 1

    while start < total_rows:
        end = min(start + max_rows_per_sheet, total_rows)
        df_chunk = df.iloc[start:end]

        # Name sheets as Part_1, Part_2, ...
        sheet_name = f"Part_{sheet_index}"

        # Write the current chunk to its corresponding sheet
        df_chunk.to_excel(writer, sheet_name=sheet_name, index=False)

        # Prepare for next chunk
        sheet_index += 1
        start = end
print(f"Results saved → {output_path.resolve()}")
# %% ###################################################
# Runtime and number of starts
#######################################################
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["text.usetex"] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

days_rt = sorted(station.daily_runtime.keys())
p1 = [station.daily_runtime[d][0] for d in days_rt]
p2 = [station.daily_runtime[d][1] for d in days_rt]
p3 = [station.daily_runtime[d][2] for d in days_rt]

days_strt = sorted(station.daily_starts.keys())
v1 = [station.daily_starts[d][0] for d in days_strt]
v2 = [station.daily_starts[d][1] for d in days_strt]
v3 = [station.daily_starts[d][2] for d in days_strt]

# Define consistent colors for all pumps
colors = sns.color_palette("pastel")[:3]  # Use only 3 distinct colors

# Create the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
label_params = {'fontsize': 14, 'fontweight': 'bold', 'ha': 'left', 'va': 'top'}

# Plot 1: Runtime
x = np.arange(len(days_rt))
w = 0.25
axes[0].bar(x - w, p1, width=w, label="Pump 1", color=colors[0], edgecolor='black')
axes[0].bar(x,     p2, width=w, label="Pump 2", color=colors[1], edgecolor='black')
axes[0].bar(x + w, p3, width=w, label="Pump 3", color=colors[2], edgecolor='black')

axes[0].set_ylabel("Runtime (hours)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Day {d+1}" for d in days_rt], rotation=45, ha='right')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].text(-0.06, 0.98, '(a)', transform=axes[0].transAxes, **label_params)  # Added label

# Plot 2: Starts (volume-like)
x2 = np.arange(len(days_strt))
w2 = 0.25
axes[1].bar(x2 - w2, v1, width=w2, color=colors[0], edgecolor='black')
axes[1].bar(x2,     v2, width=w2, color=colors[1], edgecolor='black')
axes[1].bar(x2 + w2, v3, width=w2, color=colors[2], edgecolor='black')

axes[1].set_ylabel("Starts (count)")
axes[1].set_xticks(x2)
axes[1].set_xticklabels([f"Day {d+1}" for d in days_strt], rotation=45, ha='right')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].text(-0.1, 0.98, '(b)', transform=axes[1].transAxes, **label_params)  # Added label

# Unified legend below both plots
fig.legend(["Pump 1", "Pump 2", "Pump 3"], loc='lower center', ncol=3, frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, -0.08))
plt.savefig(
    figures_dir / "station_daily_metrics.pdf",
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.05
)
plt.savefig(figures_dir / "station_daily_metrics.eps", format="eps", dpi=300, bbox_inches="tight")
# Overall title
plt.show()
# %% ---------------------------------------------------------
# Save daily runtime table
# ---------------------------------------------------------
runtime_df = pd.DataFrame({
    "Day":        [f"Day {d+1}" for d in days_rt],
    "Pump1_h":    p1,
    "Pump2_h":    p2,
    "Pump3_h":    p3,
})
runtime_fname = tables_dir / "daily_runtime.xlsx"
runtime_df.to_excel(runtime_fname, index=False)
print(f"Pump runtime table saved to {runtime_fname.resolve()}")

# ---------------------------------------------------------
# Save daily starts table
# ---------------------------------------------------------
starts_df = pd.DataFrame({
    "Day":        [f"Day {d+1}" for d in days_strt],
    "Pump1_cnt":  v1,
    "Pump2_cnt":  v2,
    "Pump3_cnt":  v3,
})
starts_fname = tables_dir / "daily_starts.xlsx"
starts_df.to_excel(starts_fname, index=False)
print(f"Pump starts table saved to {starts_fname.resolve()}")

# %%#####################################################
# Start-count discrepancy analysis
#########################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ------------------------------------------------------------------
# 1) Read the two workbooks
# ------------------------------------------------------------------
df_daily = pd.read_excel(tables_dir / "daily_starts.xlsx")   # simulated results
df_real  = pd.read_excel(real_starts_path)    # real measurements

# ------------------------------------------------------------------
# 2) Extract the day number, drop Day 27, and sort
#    (works whether the Day column looks like “Day 1” or just 1)
# ------------------------------------------------------------------
for df in (df_daily, df_real):
    df["Day_num"] = df["Day"].astype(str).str.extract(r"(\d+)").astype(int)
    df.sort_values("Day_num", inplace=True)

df_daily = df_daily[df_daily["Day_num"] < 27]   # keep Days 1-26
df_real  = df_real[df_real["Day_num"] < 27]

# ------------------------------------------------------------------
# 3) Reshape to long format
# ------------------------------------------------------------------
daily_long = df_daily.melt(
    id_vars=["Day_num"],
    value_vars=["Pump1_cnt", "Pump2_cnt", "Pump3_cnt"],
    var_name="Pump",
    value_name="Simulated"
)
daily_long["Pump"] = daily_long["Pump"].str.replace("_cnt", "")

real_long = df_real.melt(
    id_vars=["Day_num"],
    value_vars=["Pump1_cnt", "Pump2_cnt", "Pump3_cnt"],
    var_name="Pump",
    value_name="Real"
)
real_long["Pump"] = real_long["Pump"].str.replace("_cnt", "")

# ------------------------------------------------------------------
# 4) Merge an continue plotting
# ------------------------------------------------------------------
df_merged = pd.merge(daily_long, real_long, on=["Day_num", "Pump"])

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Global style (journal aesthetic) ----------
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Custom palette ----------
colors = {
    "Simulated": "#1f77b4",  # deep blue
    "Real": "#d62728",       # vermillion
    "Deviation": "salmon"    # highlight
}

# ---------- Plot ----------
fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.5), sharex=True)

for i, (ax, pump) in enumerate(zip(axes, ['Pump1', 'Pump2', 'Pump3']), 1):
    pump_data = df_merged[df_merged['Pump'] == pump]
    
    # Simulated vs Real
    ax.plot(pump_data['Day_num'], pump_data['Simulated'],
            marker='o', markersize=4, label="Simulated",
            linewidth=1.8, color=colors["Simulated"])
    ax.plot(pump_data['Day_num'], pump_data['Real'],
            marker='s', markersize=4, label="Real",
            linewidth=1.8, linestyle="--", color=colors["Real"])
    
    # Highlight deviations
    for _, row in pump_data.iterrows():
        diff = abs(row['Simulated'] - row['Real'])
        if diff >= 3:
            ax.axvline(x=row['Day_num'], color=colors["Deviation"],
                       alpha=0.35, linestyle="-", ymin=0.05, ymax=0.95)
    
    # Formatting
    ax.set_title(pump, fontsize=12, weight="bold", pad=6)
    ax.set_ylabel("Start count")
    ax.set_ylim(30, 55)
    ax.minorticks_on()
    
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
    
    # Legend only once
    if i == 3:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=2, frameon=True)

# Shared x-axis
axes[-1].set_xlabel("Day")
axes[-1].set_xticks(range(1, 27))

fig.tight_layout(pad=2.0)
fig.subplots_adjust(top=0.95)

# ---------- Save ----------
fig.savefig(figures_dir / "pump_simulation_analysis.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(figures_dir / "pump_simulation_analysis.eps", bbox_inches="tight")

plt.show()


# %%#######################################
# Helper for real-runtime conversion
##############################################
def convert_runtime(time_str):
    import re
    # Extract hours, minutes, and seconds using regex
    match = re.match(r'(\d+)h\s*(\d+)min\s*(\d+)s', time_str)
    if not match:
        raise ValueError("Invalid time format")
    
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    
    # Calculate total decimal hours
    total_hours = hours + minutes/60 + seconds/3600
    
    # Format to 9 decimal places with comma separator
    return f"{total_hours:.9f}".replace('.', ',')

#%% Example usage:
real_measurement = "1h 54min 20s"
converted = convert_runtime(real_measurement) 
print(converted)

# %% ########################################
# Runtime
#############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Read simulation data
sim = pd.read_excel(tables_dir / "daily_runtime.xlsx")
sim = sim[sim['Day'] != 'Day 27']  # Remove day 27
sim['Day'] = sim['Day'].str.extract(r'(\d+)').astype(int)  # Extract day number

# Read real measurements
real = pd.read_excel(real_runtime_path)
real = real[real['Day'] != 'Day 27']  # Remove day 27
real['Day'] = real['Day'].str.extract(r'(\d+)').astype(int)  # Extract day number

# Melt data to long format for plotting
sim_long = sim.melt(id_vars='Day', var_name='Pump', value_name='Simulated')
real_long = real.melt(id_vars='Day', var_name='Pump', value_name='Real')

# Merge datasets
df = pd.merge(sim_long, real_long, on=['Day', 'Pump'])
df['Difference'] = df['Simulated'] - df['Real']

# ---------- Global style (journal aesthetic) ----------
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Colors ----------
colors = {
    "Sim": "#1f77b4",      # deep blue
    "Real": "#d62728",     # vermillion
    "Over": "#1f77b4",     # same hue as Sim for overestimation fill
    "Under": "#d62728",    # same hue as Real for underestimation fill
    "Star": "#e66101"      # muted orange for significant diffs
}

# ---------- Figure / axes ----------
fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.6), sharex=True)

for i, pump in enumerate(['Pump1_h', 'Pump2_h', 'Pump3_h']):
    ax = axes[i]
    pump_data = df[df['Pump'] == pump].copy()

    # Lines
    ax.plot(pump_data['Day'], pump_data['Simulated'],
            marker='o', markersize=4, linewidth=1.8,
            color=colors["Sim"], label='Simulated')
    ax.plot(pump_data['Day'], pump_data['Real'],
            marker='s', markersize=4, linewidth=1.8, linestyle='--',
            color=colors["Real"], label='Real')

    # Fills (Over/Under)
    ax.fill_between(pump_data['Day'],
                    pump_data['Simulated'], pump_data['Real'],
                    where=(pump_data['Simulated'] > pump_data['Real']),
                    interpolate=True, alpha=0.15, color=colors["Over"], label='Overestimation')
    ax.fill_between(pump_data['Day'],
                    pump_data['Simulated'], pump_data['Real'],
                    where=(pump_data['Simulated'] < pump_data['Real']),
                    interpolate=True, alpha=0.15, color=colors["Under"], label='Underestimation')

    # Significant differences (|Δ| > 0.5 h)
    if 'Difference' in pump_data.columns:
        sig = pump_data[pump_data['Difference'].abs() > 0.5]
        if not sig.empty:
            ax.scatter(sig['Day'], (sig[['Simulated','Real']].max(axis=1))*1.02,
                       marker='*', s=80, color=colors["Star"], zorder=5)

    # Formatting
    ax.set_title(pump.replace("_h", ""), fontsize=12, weight='bold', pad=6)
    ax.set_ylabel('Runtime (hours)')
    ax.set_xlim(0.8, 26.2)
    ax.set_ylim(0.8, 3.1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.minorticks_on()

    # Clean spines
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)

# Shared x-label
axes[-1].set_xlabel('Day')

# ---------- Legend below the plot ----------
legend_handles = [
    Line2D([0], [0], color=colors["Sim"], lw=1.8, marker='o', markersize=4, label='Simulated'),
    Line2D([0], [0], color=colors["Real"], lw=1.8, ls='--', marker='s', markersize=4, label='Real'),
    Patch(facecolor=colors["Over"], alpha=0.15, edgecolor='none', label='Overestimation'),
    Patch(facecolor=colors["Under"], alpha=0.15, edgecolor='none', label='Underestimation'),
]
fig.legend(
    legend_handles, [h.get_label() for h in legend_handles],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.05),   # move slightly up
    ncol=4,
    frameon=True
)

# Layout and save
fig.tight_layout(pad=2.0)
fig.subplots_adjust(top=0.95, bottom=0.12)

fig.savefig(figures_dir / "pump_runtime_analysis.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(figures_dir / "pump_runtime_analysis.eps", bbox_inches="tight")

plt.show()
# %%
