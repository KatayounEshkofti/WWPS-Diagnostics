# %%
"""
Pump-station simulation with empirical inflow sampling and two fault scenarios.

The script simulates a three-pump wet-well station with soft starts/stops,
round-robin lead-pump rotation, a temporary system-curve fault, and a gradual
Pump 1 blockage fault. It also saves the simulated time series and produces the
figures used for inspection and reporting.
"""

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(seed=42)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

# Wet-well geometry
cross_section_area = 8.0  # m²

# Frequency and speed
pump_min_freq = 25.0       # Hz, used during soft start
pump_max_freq = 50.0       # Hz, nominal full speed
nominal_freq = 50.0        # Hz
nominal_rpm = 1460.0       # rpm at 50 Hz

# Level-control set points
level_start = 1.60         # m, start the lead pump when no pump is running
level_stop = 0.50          # m, stop all pumps
level_start_2 = 1.80       # m, start the second pump when one pump is running
level_stop_2 = 0.80        # m, stop the second pump when two pumps are running

# Optional inflow peaks
peak_rate = 0.0005         # events per second, roughly 18 events/day
peak_magnitude = 0         # additional flow during peaks, m³/h
peak_duration = 900        # seconds, 15 minutes

# Pump curve at 50 Hz: flow in US gpm, head in m
pump_curve_data_50hz: List[Tuple[float, float]] = [
    (0, 32.0),
    (400, 29.9),
    (800, 27.4),
    (1200, 24.4),
    (1600, 21.3),
    (2000, 18.3),
]

# System curve: H = H_s + K * Q², with Q in m³/h and H in m
static_head = 2.0
friction_coeff = 0.0006

# Baseline values used to restore the system curve after the fault period
friction_coeff_base = friction_coeff
static_head_base = static_head

# System-fault window and ramp severity
system_fault_start = 60000
system_fault_end = 80000
system_friction_increase = 1.2
system_static_head_increase = 0.5

# Pump-fault window and blockage severity
pump_fault_start = 86400 + 25000
pump_fault_end = 86400 + 40000
pump_blockage_final_factor = 0.3
pump_efficiency_final_ratio = 0.7

# Physics and efficiency
pump_efficiency = 0.90
water_density = 1000.0     # kg/m³
gravity = 9.81             # m/s²

# Approximate electrical specifications
nominal_voltage = 400      # V, three-phase
nominal_current = 30.0     # A
power_factor = 0.9

# Unit conversion
usgpm_to_m3h = 0.2271

# Soft start/stop timing
soft_start_time = 10.0     # s
soft_stop_time = 10.0      # s

# Simulation horizon
sim_time = 86400 * 2       # two days
time_step = 1.0            # s

# Measurement noise
level_meas_noise_std = 0.02
flow_noise_relative_std = 0.01
power_noise_relative_std = 0.01

# Project paths
project_root = Path(__file__).resolve().parent
data_dir = project_root / "data"
results_dir = project_root / "results"
figures_dir = results_dir / "figures"
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# CSV produced by the flow-rate distribution analysis script
flow_csv_path: Final[Path] = data_dir / "flow_rates.csv"


# -----------------------------------------------------------------------------
# Inflow distribution
# -----------------------------------------------------------------------------

try:
    flow_df = pd.read_csv(flow_csv_path)
except FileNotFoundError as exc:
    raise SystemExit(
        f"Could not find {flow_csv_path}. Run the preprocessing script first "
        "or update flow_csv_path."
    ) from exc

positive_inflows = flow_df["flow_rate_m3_s"]
positive_inflows = positive_inflows[positive_inflows > 0.0].to_numpy(dtype=np.float64)
positive_inflows.sort()
n_positive = positive_inflows.size
print(f"Loaded {n_positive:,} positive inflow samples for the ECDF sampler.")


def ecdf_draw_m3s(size: int = 1, rng_: np.random.Generator | None = None) -> np.ndarray:
    """
    Draw random inflow samples from the empirical distribution.

    Parameters
    ----------
    size : int
        Number of samples to draw.
    rng_ : np.random.Generator or None
        Optional NumPy random generator. If omitted, a new generator is used.

    Returns
    -------
    np.ndarray
        Inflow samples in m³/s.
    """
    rng_ = rng_ or np.random.default_rng()
    quantiles = rng_.random(size)
    return np.quantile(positive_inflows, quantiles)


# -----------------------------------------------------------------------------
# Pump and system-curve helpers
# -----------------------------------------------------------------------------

def piecewise_linear_interpolation(x: float, points: List[Tuple[float, float]]) -> float:
    """Linearly interpolate a monotonic flow-head curve."""
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= x <= x1:
                break

    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def get_pump_head_at_flow(flow_usgpm: float, speed_ratio: float) -> float:
    """Return the affinity-scaled pump head for a given flow and speed ratio."""
    if speed_ratio <= 0.0:
        return 0.0

    flow_nominal = flow_usgpm / speed_ratio
    head_nominal = piecewise_linear_interpolation(flow_nominal, pump_curve_data_50hz)
    return head_nominal * speed_ratio ** 2


def system_head(flow_m3h: float) -> float:
    """Evaluate the current system curve."""
    return static_head + friction_coeff * flow_m3h ** 2


def find_operating_flow(rpm: float) -> float:
    """Find the pump operating flow where the pump and system heads match."""
    if rpm < 50.0:
        return 0.0

    ratio = rpm / nominal_rpm
    lower, upper = 0.0, 3000.0  # US gpm bounds
    tolerance = 0.1

    while upper - lower > tolerance:
        mid = 0.5 * (lower + upper)
        pump_h = get_pump_head_at_flow(mid, ratio)
        sys_h = system_head(mid * usgpm_to_m3h)
        lower, upper = (mid, upper) if pump_h > sys_h else (lower, mid)

    flow_usgpm = 0.5 * (lower + upper)
    return flow_usgpm * usgpm_to_m3h


def approximate_power_kw(flow_m3h: float, eff: float) -> float:
    """Estimate hydraulic shaft power from flow, head, and efficiency."""
    if flow_m3h <= 0.0 or eff < 1e-9:
        return 0.0

    head_m = system_head(flow_m3h)
    flow_m3s = flow_m3h / 3600.0
    return water_density * gravity * flow_m3s * head_m / (1000.0 * eff)


# -----------------------------------------------------------------------------
# Pump model
# -----------------------------------------------------------------------------


class Pump:
    """Single centrifugal pump with a VFD soft start/stop model."""

    def __init__(self, name: str):
        self.name = name

        # Operating state
        self.is_running = False
        self.current_freq = 0.0
        self.blockage_factor = 1.0
        self.efficiency = pump_efficiency

        # Soft-start/stop state
        self._soft_starting = False
        self._soft_stopping = False
        self._soft_start_elapsed = 0.0
        self._soft_stop_elapsed = 0.0

        # Hydraulics and energy
        self.flow_m3h = 0.0
        self.head_m = 0.0
        self.shaft_kw = 0.0
        self.input_kw = 0.0
        self.energy_kwh = 0.0

    def start(self, t: float) -> None:
        """Start the pump through the soft-start ramp."""
        if not self.is_running:
            self.is_running = True
            self._soft_starting = True
            self._soft_start_elapsed = 0.0
            self.current_freq = pump_min_freq
            print(f"[{t:.1f}s] {self.name} start: soft-start ramp")

    def stop(self, t: float) -> None:
        """Stop the pump through the soft-stop ramp."""
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            print(f"[{t:.1f}s] {self.name} stop: soft-stop ramp")

    def _effective_rpm(self) -> float:
        """Hydraulic speed after blockage scaling."""
        rpm_nominal = (self.current_freq / nominal_freq) * nominal_rpm
        return rpm_nominal * self.blockage_factor

    def _update_soft_ramps(self, dt: float) -> None:
        if self._soft_starting:
            self._soft_start_elapsed += dt
            ramp_fraction = min(1.0, self._soft_start_elapsed / soft_start_time)
            self.current_freq = pump_min_freq + (pump_max_freq - pump_min_freq) * ramp_fraction
            if ramp_fraction >= 1.0:
                self._soft_starting = False

        if self._soft_stopping:
            self._soft_stop_elapsed += dt
            ramp_fraction = min(1.0, self._soft_stop_elapsed / soft_stop_time)
            self.current_freq *= 1.0 - ramp_fraction
            if self.current_freq <= 0.1:
                self.current_freq = 0.0
                self._soft_stopping = False
                self.is_running = False
                print("        fully stopped")

        if self.is_running and not (self._soft_starting or self._soft_stopping):
            self.current_freq = pump_max_freq

    def update(self, dt: float, t: float) -> None:
        """Advance the pump state by one simulation step."""
        self._update_soft_ramps(dt)

        if not (self.is_running or self._soft_stopping):
            self.flow_m3h = self.head_m = self.shaft_kw = self.input_kw = 0.0
            return

        # Hydraulics
        rpm_eff = self._effective_rpm()
        raw_flow = find_operating_flow(rpm_eff)
        flow_noise = 1.0 + random.gauss(0.0, flow_noise_relative_std)
        self.flow_m3h = max(0.0, raw_flow * flow_noise)

        flow_usgpm = self.flow_m3h / usgpm_to_m3h
        speed_ratio = rpm_eff / nominal_rpm
        self.head_m = get_pump_head_at_flow(flow_usgpm, speed_ratio)

        # Energy
        self.shaft_kw = approximate_power_kw(self.flow_m3h, self.efficiency)
        if self.current_freq > 0:
            current_ratio = self.current_freq / pump_max_freq
            effective_current = min(
                nominal_current * current_ratio / self.blockage_factor,
                5 * nominal_current,
            )
            self.input_kw = math.sqrt(3) * nominal_voltage * effective_current * power_factor / 1000
            self.input_kw *= 1.0 + random.gauss(0.0, power_noise_relative_std)
        else:
            self.input_kw = 0.0

        self.energy_kwh += self.input_kw * dt / 3600.0


# -----------------------------------------------------------------------------
# Pump-station controller
# -----------------------------------------------------------------------------


class PumpStation:
    """Three identical pumps with round-robin lead-pump rotation."""

    def __init__(self):
        self.pumps = [Pump(f"Pump {i + 1}") for i in range(3)]
        self.level = 0.8
        self.time_s = 0.0
        self.lead_index = 0
        self.running_stack: List[int] = []

        self.daily_starts: dict[int, List[int]] = {}
        self.daily_runtime: dict[int, List[float]] = {}
        self.hourly_energy: dict[int, List[float]] = {}
        self.active_peaks = []
        self.next_peak = random.expovariate(peak_rate)

    def inflow_m3h_peak(self) -> float:
        """Draw an empirical inflow sample and add any active synthetic peak."""
        q_m3s_base = float(ecdf_draw_m3s())
        base_m3h = q_m3s_base * 3600.0

        peak_flow = 0.0
        self.next_peak -= time_step
        if self.next_peak <= 0:
            self.active_peaks.append(self.time_s + peak_duration)
            self.next_peak = random.expovariate(peak_rate)
            print(f"[{self.time_s:.0f}s] inflow peak started")

        for end_time in list(self.active_peaks):
            if self.time_s > end_time:
                self.active_peaks.remove(end_time)
            else:
                peak_flow += peak_magnitude

        return base_m3h + peak_flow

    def inflow_m3h(self) -> float:
        """Return one empirical inflow sample in m³/h."""
        q_m3s = float(ecdf_draw_m3s())
        return q_m3s * 3600.0

    def _next_lead(self) -> int:
        self.lead_index = (self.lead_index + 1) % len(self.pumps)
        return self.lead_index

    def _start_pump(self, idx: int) -> None:
        self.pumps[idx].start(self.time_s)
        self.running_stack.append(idx)

        day = int(self.time_s // 86400)
        self.daily_starts.setdefault(day, [0, 0, 0])[idx] += 1
        self._next_lead()

    def _stop_last_pump(self) -> None:
        if self.running_stack:
            idx = self.running_stack[-1]
            if self.pumps[idx].is_running:
                self.pumps[idx].stop(self.time_s)

    def _stop_all(self) -> None:
        for idx in reversed(self.running_stack):
            if self.pumps[idx].is_running:
                self.pumps[idx].stop(self.time_s)

    def control(self) -> None:
        measured_level = self.level + random.gauss(0.0, level_meas_noise_std)
        n_running = len(self.running_stack)

        if measured_level >= level_start and n_running == 0:
            self._start_pump(self.lead_index)
        elif n_running == 1 and measured_level >= level_start_2:
            self._start_pump(self.lead_index)

        if n_running == 2 and measured_level <= level_stop_2:
            self._stop_last_pump()
        if measured_level <= level_stop:
            self._stop_all()

    def step(self, dt: float) -> None:
        """Advance the station by one time step."""
        global friction_coeff, static_head

        # System fault is applied before computing pump hydraulics.
        if self.time_s < system_fault_start:
            friction_coeff = friction_coeff_base
            static_head = static_head_base
        elif self.time_s < system_fault_end:
            progress = (self.time_s - system_fault_start) / (system_fault_end - system_fault_start)
            friction_coeff = friction_coeff_base * (1 + system_friction_increase * progress)
            static_head = static_head_base + system_static_head_increase * progress
        else:
            friction_coeff = friction_coeff_base
            static_head = static_head_base

        for pump in self.pumps:
            pump.update(dt, self.time_s)

        self.running_stack = [i for i in self.running_stack if self.pumps[i].is_running]

        inflow = self.inflow_m3h_peak()
        outflow = sum(pump.flow_m3h for pump in self.pumps)
        volume_change = (inflow - outflow) / 3600 * dt
        self.level += volume_change / cross_section_area

        # Pump 1 gradual blockage.
        pump_1 = self.pumps[0]
        if pump_fault_start <= self.time_s < pump_fault_end:
            progress = (self.time_s - pump_fault_start) / (pump_fault_end - pump_fault_start)
            progress = np.clip(progress, 0.0, 1.0)
            pump_1.blockage_factor = 1.0 - 0.7 * progress
            pump_1.efficiency = pump_efficiency * (1.0 - 0.3 * progress)
        elif self.time_s >= pump_fault_end:
            pump_1.blockage_factor = pump_blockage_final_factor
            pump_1.efficiency = pump_efficiency * pump_efficiency_final_ratio
        else:
            pump_1.blockage_factor = 1.0
            pump_1.efficiency = pump_efficiency

        self.control()

        day = int(self.time_s // 86400)
        self.daily_runtime.setdefault(day, [0.0, 0.0, 0.0])
        dt_h = dt / 3600
        for i, pump in enumerate(self.pumps):
            if pump.is_running:
                self.daily_runtime[day][i] += dt_h

        hour = int(self.time_s // 3600)
        self.hourly_energy.setdefault(hour, [0.0, 0.0, 0.0])
        for i, pump in enumerate(self.pumps):
            self.hourly_energy[hour][i] += pump.input_kw * dt_h

        self.time_s += dt


# -----------------------------------------------------------------------------
# Simulation and plots
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    sns.set(style="whitegrid")

    station = PumpStation()

    # Logs for later analysis
    t_log: List[float] = []
    level_log: List[float] = []
    inflow_log: List[float] = []
    outflow_log: List[float] = []

    flow_logs = [[] for _ in station.pumps]
    head_logs = [[] for _ in station.pumps]
    freq_logs = [[] for _ in station.pumps]
    input_logs = [[] for _ in station.pumps]
    blockage_logs = [[] for _ in station.pumps]
    efficiency_logs = [[] for _ in station.pumps]
    input_power_logs = [[] for _ in station.pumps]
    power_logs = [[] for _ in station.pumps]

    print("Starting 48-hour simulation...")
    while station.time_s < sim_time:
        station.step(time_step)

        t_log.append(station.time_s)
        level_log.append(station.level)
        inflow_log.append(station.inflow_m3h_peak())
        outflow_log.append(sum(pump.flow_m3h for pump in station.pumps))

        for i, pump in enumerate(station.pumps):
            flow_logs[i].append(pump.flow_m3h)
            head_logs[i].append(pump.head_m)
            freq_logs[i].append(pump.current_freq)
            input_logs[i].append(pump.input_kw)
            blockage_logs[i].append(pump.blockage_factor)
            power_logs[i].append(pump.shaft_kw)

    print("Simulation complete. Producing plots...")

    # -------------------------------------------------------------------------
    # Wet-well level and station flows for a representative interval
    # -------------------------------------------------------------------------
    idx_5k = [i for i, t in enumerate(t_log) if 10000 <= t <= 22000]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(np.array(t_log)[idx_5k], np.array(level_log)[idx_5k])
    ax[0].axhline(level_start, c="r", ls="--", label="Start 1.6 m")
    ax[0].axhline(level_stop, c="g", ls="--", label="Stop 0.5 m")
    ax[0].set_ylabel("Level [m]")
    ax[0].legend()

    ax[1].plot(np.array(t_log)[idx_5k], np.array(outflow_log)[idx_5k], label="Outflow")
    ax[1].set_ylabel("m³/h")
    ax[2].plot(np.array(t_log)[idx_5k], np.array(inflow_log)[idx_5k], label="Inflow", color="#f7a072")
    ax[2].set_ylabel("m³/h")
    ax[2].set_xlabel("Time [s]")

    for axis in ax:
        axis.grid(True)

    plt.tight_layout()
    plt.show()

    mask_5k = [i for i, ts in enumerate(t_log) if 10000 <= ts <= 22000]
    t_5k = [t_log[i] for i in mask_5k]
    lvl_5k = [level_log[i] for i in mask_5k]
    inflw_5k = [inflow_log[i] for i in mask_5k]
    outfw_5k = [outflow_log[i] for i in mask_5k]

    freq_5k = [[] for _ in station.pumps]
    power_5k = [[] for _ in station.pumps]
    flow_5k = [[] for _ in station.pumps]
    head_5k = [[] for _ in station.pumps]
    input_power_5k = [[] for _ in station.pumps]

    for i in range(3):
        freq_5k[i] = [freq_logs[i][j] for j in mask_5k]
        flow_5k[i] = [flow_logs[i][j] for j in mask_5k]
        head_5k[i] = [head_logs[i][j] for j in mask_5k]
        power_5k[i] = [input_logs[i][j] for j in mask_5k]

    sns.set(style="whitegrid")

    # %% Separate plots
    # Water level
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, lvl_5k, "b-", label="Water Level [m]")
    plt.axhline(level_start, color="r", ls="--", label="Start at 1.6 m")
    plt.axhline(level_stop, color="g", ls="--", label="Stop at 0.5 m")
    plt.ylabel(r"Level (m)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Water_Level.eps", format="eps", dpi=300)
    plt.close()

    # %% Inflow pattern
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, inflw_5k, "b-", label="Water Level [m]", color="#fb5607")
    plt.ylabel(r"Inflow rate (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Inflow_rate.eps", format="eps", dpi=300)
    plt.close()

    # %% Total outflow
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, outfw_5k, "c-")
    plt.ylabel(r"Outflow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Total_Outflow.eps", format="eps", dpi=300)
    plt.close()

    # %% Pump heads
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, head_5k[0], "b-", label="Pump 1")
    plt.plot(t_5k, head_5k[1], "r-", label="Pump 2")
    plt.plot(t_5k, head_5k[2], "g-", label="Pump 3")
    plt.ylabel(r"Head (m)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Pump_Heads.eps", format="eps", dpi=300)
    plt.close()

    # %% Pump power
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, power_5k[0], "b-", label="Pump 1")
    plt.plot(t_5k, power_5k[1], "r-", label="Pump 2")
    plt.plot(t_5k, power_5k[2], "g-", label="Pump 3")
    plt.ylabel(r"Power (kW)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Pump_Power.eps", format="eps", dpi=300)
    plt.close()

    # %% Pump flow
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(t_5k, flow_5k[0], "b-", label="Pump 1")
    plt.plot(t_5k, flow_5k[1], "r-", label="Pump 2")
    plt.plot(t_5k, flow_5k[2], "g-", label="Pump 3")
    plt.ylabel(r"Flow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "Pump_Flow.eps", format="eps", dpi=300)
    plt.close()

    # %% Operating-point trajectory on the system curve
    flow_axis = np.linspace(0, 400, 250)
    system_curve = system_head(flow_axis)

    plt.figure(figsize=(9, 7), layout="constrained")
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.plot(flow_axis, system_curve, "k-", lw=3, alpha=0.9, label="System curve")

    beta_values = np.linspace(1.0, 0.6, 5)
    curve_colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
    for i, beta in enumerate(beta_values):
        pump_curve = [get_pump_head_at_flow(q / usgpm_to_m3h, beta) for q in flow_axis]
        plt.plot(
            flow_axis,
            pump_curve,
            "--",
            lw=2,
            alpha=0.8,
            color=curve_colors[i],
            label=fr"$\beta$={beta:.2f}",
        )

    plt.scatter(
        flow_logs[0],
        head_logs[0],
        s=25,
        alpha=0.7,
        color="#E63946",
        edgecolor="k",
        linewidth=0.5,
        label="Pump 1 operating points",
        zorder=10,
    )

    plt.xlabel(r"$Q$ (m$^3$/h)", fontsize=14)
    plt.ylabel(r"$H$ (m)", fontsize=14)
    plt.legend(
        fontsize=12,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
        facecolor="white",
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(0, 400)
    plt.ylim(0, 1.1 * max(system_curve))
    plt.xticks(np.arange(0, 401, 50))
    plt.yticks(np.arange(0, int(1.1 * max(system_curve)) + 10, 10))
    plt.savefig(figures_dir / "Trajectory.eps", format="eps", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "Trajectory.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    # %% Save the full simulated data set
    ts0 = datetime.now()
    timestamps = [ts0 + timedelta(seconds=int(s)) for s in t_log]
    data = {
        "Timestamp": timestamps,
        "Water_Level_m": level_log,
        "Total_Inflow_m3h": inflow_log,
        "Total_Outflow_m3h": outflow_log,
    }

    for i, pump in enumerate(station.pumps, 1):
        data[f"Pump{i}_Flow_m3h"] = flow_logs[i - 1]
        data[f"Pump{i}_Head_m"] = head_logs[i - 1]
        data[f"Pump{i}_Freq_Hz"] = freq_logs[i - 1]
        data[f"Pump{i}_Input_kW"] = input_logs[i - 1]
        data[f"Pump{i}_BlockageFactor"] = blockage_logs[i - 1]

    df = pd.DataFrame(data)
    output_file = results_dir / "pump_system_faultsimualtion.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file.resolve()}")

    # %% Hourly accumulated energy consumption for each pump
    if hasattr(station, "hourly_energy") and station.hourly_energy:
        hour_indices = sorted(station.hourly_energy.keys())
        hours = hour_indices

        pump_energies = [
            [station.hourly_energy[h][pump_idx] for h in hour_indices[0:25]]
            for pump_idx in range(3)
        ]
        cumulative_energies = [np.cumsum(energies) for energies in pump_energies]

        plt.figure(figsize=(8, 4))

        colors = ["b", "r", "g"]
        markers = ["o", "s", "^"]
        labels = ["Pump 1", "Pump 2", "Pump 3"]

        for i in range(3):
            plt.plot(
                hours[0:25],
                cumulative_energies[i][0:25],
                color=colors[i],
                marker=markers[i],
                linewidth=2,
                markersize=6,
                label=labels[i],
            )

        sns.set(style="whitegrid", font_scale=1.2)
        plt.rcParams["text.usetex"] = True
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        plt.xlabel(r"Time (hours)")
        plt.ylabel(r"Accumulated Energy (kWh)")
        plt.legend()
        plt.grid(True)
        plt.xticks(hours)
        plt.xlim(0, 25)
        plt.tight_layout()
        plt.savefig(figures_dir / "AccEnergy.eps", format="eps", dpi=300)
        plt.close()
    else:
        print("No hourly energy data recorded.")

    # %% Runtime and number of starts
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    days_rt = sorted(station.daily_runtime.keys())
    p1 = [station.daily_runtime[d][0] for d in days_rt]
    p2 = [station.daily_runtime[d][1] for d in days_rt]
    p3 = [station.daily_runtime[d][2] for d in days_rt]

    days_strt = sorted(station.daily_starts.keys())
    v1 = [station.daily_starts[d][0] for d in days_strt]
    v2 = [station.daily_starts[d][1] for d in days_strt]
    v3 = [station.daily_starts[d][2] for d in days_strt]

    colors = sns.color_palette("pastel")[:3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    label_params = {"fontsize": 14, "fontweight": "bold", "ha": "left", "va": "top"}

    x = np.arange(len(days_rt))
    bar_width = 0.25
    axes[0].bar(x - bar_width, p1, width=bar_width, label="Pump 1", color=colors[0], edgecolor="black")
    axes[0].bar(x, p2, width=bar_width, label="Pump 2", color=colors[1], edgecolor="black")
    axes[0].bar(x + bar_width, p3, width=bar_width, label="Pump 3", color=colors[2], edgecolor="black")
    axes[0].set_ylabel("Runtime (hours)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Day {d + 1}" for d in days_rt], rotation=45, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].text(-0.06, 0.98, "(a)", transform=axes[0].transAxes, **label_params)

    x2 = np.arange(len(days_strt))
    bar_width_2 = 0.25
    axes[1].bar(x2 - bar_width_2, v1, width=bar_width_2, color=colors[0], edgecolor="black")
    axes[1].bar(x2, v2, width=bar_width_2, color=colors[1], edgecolor="black")
    axes[1].bar(x2 + bar_width_2, v3, width=bar_width_2, color=colors[2], edgecolor="black")
    axes[1].set_ylabel("Starts (count)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"Day {d + 1}" for d in days_strt], rotation=45, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].text(-0.1, 0.98, "(b)", transform=axes[1].transAxes, **label_params)

    fig.legend(
        ["Pump 1", "Pump 2", "Pump 3"],
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.08),
    )
    plt.savefig(
        figures_dir / "Station_Daily_Metrics.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.savefig(figures_dir / "Station_Daily_Metrics.eps", format="eps", dpi=300, bbox_inches="tight")
    plt.show()

    # %% Combined plots: level, head, flow, and energy
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    label_params = {"fontsize": 14, "fontweight": "bold", "ha": "left", "va": "top"}

    axis = axes[0, 0]
    axis.plot(t_5k, lvl_5k, "b-")
    axis.axhline(level_start, color="r", ls="--", label="Start at 1.6 m")
    axis.axhline(level_stop, color="g", ls="--", label="Stop at 0.5 m")
    axis.set_ylabel(r"Level (m)")
    axis.set_xlabel(r"Time (s)")
    axis.grid(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    axis.text(-0.08, 0.93, "(a)", transform=axis.transAxes, **label_params)

    axis = axes[0, 1]
    axis.plot(t_5k, head_5k[0], "b-", label="Pump 1")
    axis.plot(t_5k, head_5k[1], "r-", label="Pump 2")
    axis.plot(t_5k, head_5k[2], "g-", label="Pump 3")
    axis.set_ylabel(r"Head (m)")
    axis.set_xlabel(r"Time (s)")
    axis.grid(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    axis.text(-0.08, 0.93, "(b)", transform=axis.transAxes, **label_params)

    axis = axes[1, 0]
    axis.plot(t_5k, flow_5k[0], "b-", label="Pump 1")
    axis.plot(t_5k, flow_5k[1], "r-", label="Pump 2")
    axis.plot(t_5k, flow_5k[2], "g-", label="Pump 3")
    axis.set_ylabel(r"Flow (m$^3$/h)")
    axis.set_xlabel(r"Time (s)")
    axis.grid(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    axis.text(-0.08, 0.98, "(c)", transform=axis.transAxes, **label_params)

    axis = axes[1, 1]
    hour_indices = sorted(station.hourly_energy.keys())
    hours = hour_indices
    pump_energies = [
        [station.hourly_energy[h][pump_idx] for h in hour_indices[0:25]]
        for pump_idx in range(3)
    ]
    cumulative_energies = [np.cumsum(energies) for energies in pump_energies]

    colors = ["b", "r", "g"]
    markers = ["o", "s", "^"]
    labels = ["Pump 1", "Pump 2", "Pump 3"]

    for i in range(3):
        axis.plot(
            hours[0:25],
            cumulative_energies[i][0:25],
            color=colors[i],
            marker=markers[i],
            linewidth=2,
            markersize=6,
            label=labels[i],
        )

    axis.set_xlabel(r"Time (hours)")
    axis.set_ylabel(r"Accumulated Energy (kWh)")
    axis.legend(loc="best")
    axis.grid(True)
    axis.set_xticks(hours)
    axis.set_xlim(0, 25)
    axis.text(-0.09, 0.98, "(d)", transform=axis.transAxes, **label_params)

    plt.savefig(figures_dir / "Combined_Plots.eps", format="eps", dpi=300)
    plt.close()

# %%
