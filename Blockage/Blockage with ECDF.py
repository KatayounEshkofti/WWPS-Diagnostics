# %%
"""
Pump-station simulation with an ECDF inflow model and Pump 1 blockage.

The script simulates a three-pump wet-well station for two days. Inflow is
sampled from an empirical flow-rate distribution, short peak events can be
superimposed, and Pump 1 is gradually degraded during a fixed fault window.

The outputs are intended for paper figures and downstream diagnosis scripts:
    - full simulation results as an Excel file
    - water-level and inflow plots
    - operating-point trajectory plot
    - daily runtime/start-count plots
    - a combined four-panel summary plot
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator


# =============================================================================
# Configuration
# =============================================================================

random_seed = 42
random.seed(random_seed)
rng = np.random.default_rng(random_seed)

# Paths
flow_csv_path: Final[Path] = Path("data/flow_rates.csv")
results_dir: Final[Path] = Path("results/blockage_simulation")
figures_dir: Final[Path] = results_dir / "figures"

# Wet-well geometry
cross_section_area = 8.0  # m²

# Frequency and speed
pump_min_freq = 25.0      # Hz, used only during soft start
pump_max_freq = 50.0      # Hz, nominal full speed
nominal_freq = 50.0       # Hz
nominal_rpm = 1460.0      # rpm at 50 Hz

# Level-control set points
level_start = 1.60        # m, start lead pump when no pump is running
level_stop = 0.50         # m, stop all pumps
level_start_2 = 1.80      # m, start second pump when one is already running
level_stop_2 = 0.80       # m, stop second pump when two pumps are running

# Inflow peaks
peak_rate = 0.0005        # events per second, approximately 18 events/day
peak_magnitude = 30.0     # extra flow during peaks, m³/h
peak_duration = 900       # seconds

# Pump curve at 50 Hz: (flow in US gpm, head in m)
pump_curve_data_50hz: list[tuple[float, float]] = [
    (0.0, 32.0),
    (400.0, 29.9),
    (800.0, 27.4),
    (1200.0, 24.4),
    (1600.0, 21.3),
    (2000.0, 18.3),
]

# System curve: H = Hs + K Q², with Q in m³/h and H in m
static_head = 2.0
friction_coeff = 0.0006

# Physics and efficiency
pump_efficiency = 0.90
water_density = 1000.0    # kg/m³
gravity = 9.81            # m/s²

# Electrical model
nominal_voltage = 400.0   # V, three-phase
nominal_current = 30.0    # A
power_factor = 0.9

# Unit conversion
usgpm_to_m3h = 0.2271

# Soft-start and soft-stop ramps
soft_start_time = 10.0    # s
soft_stop_time = 10.0     # s

# Simulation horizon
simulation_time = 86400 * 2  # two days
time_step = 1.0              # s

# Measurement noise
level_meas_noise_std = 0.02      # m
flow_noise_relative_std = 0.01   # relative std
power_noise_relative_std = 0.01  # relative std

# Pump 1 blockage scenario
pump_fault_start = 12000.0
pump_fault_end = 21000.0
pump_blockage_drop = 0.4          # 1.0 -> 0.6 during the ramp
pump_efficiency_drop = 0.3        # 0.90 -> 0.63 during the ramp
post_fault_blockage_factor = 0.9  # intentionally kept as in the original script
post_fault_efficiency_factor = 0.7


# =============================================================================
# Data loading and sampling
# =============================================================================

def load_positive_inflows(path: Path) -> np.ndarray:
    """Load positive inflow samples from the flow-rate analysis output."""
    try:
        df_flow = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Could not find {path}. Run the flow-rate preprocessing first, "
            "or update flow_csv_path."
        ) from exc

    positive = df_flow["flow_rate_m3_s"]
    positive = positive[positive > 0.0].to_numpy(dtype=np.float64)
    positive.sort()

    if positive.size == 0:
        raise SystemExit("No positive inflow samples were found in the flow-rate file.")

    print(f"Loaded {positive.size:,} positive inflow samples for ECDF sampling.")
    return positive


def ecdf_draw_m3s(
    positive_inflows: np.ndarray,
    size: int = 1,
    sample_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw inflow samples in m³/s from the empirical distribution."""
    sample_rng = sample_rng or rng
    u = sample_rng.random(size)
    return np.quantile(positive_inflows, u)


# =============================================================================
# Pump and system curves
# =============================================================================

def piecewise_linear_interpolation(
    x: float,
    points: list[tuple[float, float]],
) -> float:
    """Linearly interpolate a monotone flow-head curve."""
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
    """Pump head from affinity scaling of the 50 Hz curve."""
    if speed_ratio <= 0.0:
        return 0.0

    flow_nominal = flow_usgpm / speed_ratio
    head_nominal = piecewise_linear_interpolation(
        flow_nominal,
        pump_curve_data_50hz,
    )
    return head_nominal * speed_ratio**2


def system_head(flow_m3h: float | np.ndarray) -> float | np.ndarray:
    """System curve head for a given flow in m³/h."""
    return static_head + friction_coeff * flow_m3h**2


def find_operating_flow(rpm: float) -> float:
    """Find the flow where the pump curve intersects the system curve."""
    if rpm < 50.0:
        return 0.0

    speed_ratio = rpm / nominal_rpm
    lo, hi = 0.0, 3000.0  # US gpm
    tolerance = 0.1

    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        pump_h = get_pump_head_at_flow(mid, speed_ratio)
        sys_h = system_head(mid * usgpm_to_m3h)
        lo, hi = (mid, hi) if pump_h > sys_h else (lo, mid)

    flow_usgpm = 0.5 * (lo + hi)
    return flow_usgpm * usgpm_to_m3h


def approximate_power_kw(flow_m3h: float, efficiency: float) -> float:
    """Hydraulic shaft power estimate in kW."""
    if flow_m3h <= 0.0 or efficiency < 1e-9:
        return 0.0

    head_m = system_head(flow_m3h)
    flow_m3s = flow_m3h / 3600.0
    return water_density * gravity * flow_m3s * head_m / (1000.0 * efficiency)


# =============================================================================
# Pump and station models
# =============================================================================

class Pump:
    """One centrifugal pump with soft-start and soft-stop behavior."""

    def __init__(self, name: str):
        self.name = name

        self.is_running = False
        self.current_freq = 0.0
        self.blockage_factor = 1.0
        self.efficiency = pump_efficiency

        self._soft_starting = False
        self._soft_stopping = False
        self._soft_start_elapsed = 0.0
        self._soft_stop_elapsed = 0.0

        self.flow_m3h = 0.0
        self.head_m = 0.0
        self.shaft_kw = 0.0
        self.input_kw = 0.0
        self.energy_kwh = 0.0

    def start(self, time_s: float) -> None:
        """Start the pump with a soft-start ramp."""
        if not self.is_running:
            self.is_running = True
            self._soft_starting = True
            self._soft_start_elapsed = 0.0
            self.current_freq = pump_min_freq
            print(f"[{time_s:.1f}s] {self.name} start")

    def stop(self, time_s: float) -> None:
        """Stop the pump with a soft-stop ramp."""
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            print(f"[{time_s:.1f}s] {self.name} stop")

    def _effective_rpm(self) -> float:
        """Hydraulic speed after blockage scaling."""
        rpm_nominal = (self.current_freq / nominal_freq) * nominal_rpm
        return rpm_nominal * self.blockage_factor

    def _update_soft_ramps(self, dt: float) -> None:
        if self._soft_starting:
            self._soft_start_elapsed += dt
            frac = min(1.0, self._soft_start_elapsed / soft_start_time)
            self.current_freq = pump_min_freq + (pump_max_freq - pump_min_freq) * frac
            if frac >= 1.0:
                self._soft_starting = False

        if self._soft_stopping:
            self._soft_stop_elapsed += dt
            frac = min(1.0, self._soft_stop_elapsed / soft_stop_time)
            self.current_freq *= 1.0 - frac
            if self.current_freq <= 0.1:
                self.current_freq = 0.0
                self._soft_stopping = False
                self.is_running = False
                print("        fully stopped")

        if self.is_running and not (self._soft_starting or self._soft_stopping):
            self.current_freq = pump_max_freq

    def update(self, dt: float, time_s: float) -> None:
        """Advance the pump model by one time step."""
        self._update_soft_ramps(dt)

        if not (self.is_running or self._soft_stopping):
            self.flow_m3h = 0.0
            self.head_m = 0.0
            self.shaft_kw = 0.0
            self.input_kw = 0.0
            return

        rpm_eff = self._effective_rpm()
        raw_flow = find_operating_flow(rpm_eff)
        flow_noise = 1.0 + random.gauss(0.0, flow_noise_relative_std)
        self.flow_m3h = max(0.0, raw_flow * flow_noise)

        flow_usgpm = self.flow_m3h / usgpm_to_m3h
        speed_ratio = rpm_eff / nominal_rpm
        self.head_m = get_pump_head_at_flow(flow_usgpm, speed_ratio)

        self.shaft_kw = approximate_power_kw(self.flow_m3h, self.efficiency)

        if self.current_freq > 0:
            current_ratio = self.current_freq / pump_max_freq
            effective_current = min(
                nominal_current * current_ratio / self.blockage_factor,
                5.0 * nominal_current,
            )
            self.input_kw = (
                math.sqrt(3.0)
                * nominal_voltage
                * effective_current
                * power_factor
                / 1000.0
            )
            self.input_kw *= 1.0 + random.gauss(0.0, power_noise_relative_std)
        else:
            self.input_kw = 0.0

        self.energy_kwh += self.input_kw * dt / 3600.0


class PumpStation:
    """Three-pump wet-well station with round-robin lead-pump rotation."""

    def __init__(self, positive_inflows_m3s: np.ndarray):
        self.positive_inflows_m3s = positive_inflows_m3s
        self.pumps = [Pump(f"Pump {i + 1}") for i in range(3)]

        self.level = 0.8
        self.time_s = 0.0
        self.lead_index = 0
        self.running_stack: list[int] = []

        self.daily_starts: dict[int, list[int]] = {}
        self.daily_runtime: dict[int, list[float]] = {}
        self.hourly_energy: dict[int, list[float]] = {}
        self.active_peaks: list[float] = []
        self.next_peak = random.expovariate(peak_rate)

    def inflow_m3h_peak(self) -> float:
        """Draw ECDF inflow and add any active peak contribution."""
        q_m3s_base = float(ecdf_draw_m3s(self.positive_inflows_m3s))
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
        """Draw one ECDF inflow value and return it in m³/h."""
        q_m3s = float(ecdf_draw_m3s(self.positive_inflows_m3s))
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

    def _update_pump_fault(self) -> None:
        """Apply the Pump 1 blockage scenario."""
        pump_1 = self.pumps[0]

        if pump_fault_start <= self.time_s < pump_fault_end:
            progress = (self.time_s - pump_fault_start) / (pump_fault_end - pump_fault_start)
            pump_1.blockage_factor = 1.0 - pump_blockage_drop * progress
            pump_1.efficiency = pump_efficiency * (1.0 - pump_efficiency_drop * progress)
        elif self.time_s >= pump_fault_end:
            pump_1.blockage_factor = post_fault_blockage_factor
            pump_1.efficiency = pump_efficiency * post_fault_efficiency_factor
        else:
            pump_1.blockage_factor = 1.0
            pump_1.efficiency = pump_efficiency

    def step(self, dt: float) -> None:
        """Advance the station by one simulation step."""
        for pump in self.pumps:
            pump.update(dt, self.time_s)

        self.running_stack = [i for i in self.running_stack if self.pumps[i].is_running]

        inflow = self.inflow_m3h_peak()
        outflow = sum(pump.flow_m3h for pump in self.pumps)
        d_volume = (inflow - outflow) / 3600.0 * dt
        self.level += d_volume / cross_section_area

        self._update_pump_fault()
        self.control()

        day = int(self.time_s // 86400)
        self.daily_runtime.setdefault(day, [0.0, 0.0, 0.0])
        dt_h = dt / 3600.0
        for i, pump in enumerate(self.pumps):
            if pump.is_running:
                self.daily_runtime[day][i] += dt_h

        hour = int(self.time_s // 3600)
        self.hourly_energy.setdefault(hour, [0.0, 0.0, 0.0])
        for i, pump in enumerate(self.pumps):
            self.hourly_energy[hour][i] += pump.input_kw * dt_h

        self.time_s += dt


# =============================================================================
# Plotting and export helpers
# =============================================================================

def apply_journal_style(font_size: int = 11) -> None:
    """Use a compact journal-style Matplotlib configuration."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 1,
        "legend.fontsize": max(font_size - 1, 8),
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


def save_eps_without_tex(fig: plt.Figure, path: Path) -> None:
    """Save EPS without requiring LaTeX during the EPS export."""
    with mpl.rc_context({"text.usetex": False}):
        fig.savefig(path, bbox_inches="tight")


def make_output_dirs() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def plot_wetwell_overview(
    time_log: list[float],
    level_log: list[float],
    inflow_log: list[float],
    outflow_log: list[float],
) -> None:
    """Plot water level, inflow, and outflow over the blockage interval."""
    idx_window = [i for i, t in enumerate(time_log) if 10000 <= t <= 22000]
    time_arr = np.asarray(time_log)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time_arr[idx_window], np.asarray(level_log)[idx_window])
    axes[0].axhline(level_start, color="r", linestyle="--", label="Start 1.6 m")
    axes[0].axhline(level_stop, color="g", linestyle="--", label="Stop 0.5 m")
    axes[0].set_ylabel("Level [m]")
    axes[0].legend()

    axes[1].plot(time_arr[idx_window], np.asarray(outflow_log)[idx_window], label="Outflow")
    axes[1].set_ylabel("m³/h")

    axes[2].plot(
        time_arr[idx_window],
        np.asarray(inflow_log)[idx_window],
        label="Inflow",
        color="#f7a072",
    )
    axes[2].set_ylabel("m³/h")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(figures_dir / "wetwell_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_windowed_logs(
    time_log: list[float],
    level_log: list[float],
    inflow_log: list[float],
    outflow_log: list[float],
    flow_logs: list[list[float]],
    head_logs: list[list[float]],
    input_logs: list[list[float]],
) -> dict[str, object]:
    """Collect the 10,000-22,000 s window used in the paper figures."""
    mask = [i for i, t in enumerate(time_log) if 10000 <= t <= 22000]

    return {
        "time": [time_log[i] for i in mask],
        "level": [level_log[i] for i in mask],
        "inflow": [inflow_log[i] for i in mask],
        "outflow": [outflow_log[i] for i in mask],
        "flow": [[flow_logs[j][i] for i in mask] for j in range(3)],
        "head": [[head_logs[j][i] for i in mask] for j in range(3)],
        "input_power": [[input_logs[j][i] for i in mask] for j in range(3)],
    }


def plot_water_level(window: dict[str, object]) -> None:
    apply_journal_style(font_size=11)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(window["time"], window["level"], "b-", label="Water level")
    ax.axhline(level_start, color="r", linestyle="--", label="Start at 1.6 m")
    ax.axhline(level_stop, color="g", linestyle="--", label="Stop at 0.5 m")
    ax.set_ylabel(r"Level (m)")
    ax.set_xlabel(r"Time (s)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    save_eps_without_tex(fig, figures_dir / "Water_Level.eps")
    plt.close(fig)


def plot_inflow(window: dict[str, object]) -> None:
    apply_journal_style(font_size=11)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    ax.plot(window["time"], window["inflow"], color="#d62728", lw=1.6, label="Inflow rate")
    ax.set_ylabel(r"Inflow rate (m$^3$/h)")
    ax.set_xlabel("Time (s)")

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig(figures_dir / "Inflow_rate.pdf", bbox_inches="tight")
    save_eps_without_tex(fig, figures_dir / "Inflow_rate.eps")
    plt.close(fig)


def plot_operating_trajectory(
    flow_logs: list[list[float]],
    head_logs: list[list[float]],
) -> None:
    apply_journal_style(font_size=11)
    mpl.rcParams.update({
        "path.simplify": True,
        "path.simplify_threshold": 0.4,
        "agg.path.chunksize": 10000,
    })

    flow_axis = np.linspace(0, 400, 250)
    system_curve = system_head(flow_axis)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    ax.plot(flow_axis, system_curve, color="black", lw=2.2, alpha=0.9, label="System curve")

    betas = np.linspace(1.00, 0.60, 5)
    base_color = np.array([31 / 255, 119 / 255, 180 / 255])
    dash_styles = [(None, None), (4, 2), (6, 2), (2, 2), (8, 3)]

    for i, beta in enumerate(betas):
        color = tuple(base_color * (0.6 + 0.08 * i))
        pump_curve = [get_pump_head_at_flow(q / usgpm_to_m3h, beta) for q in flow_axis]
        ax.plot(
            flow_axis,
            pump_curve,
            color=color,
            lw=1.6,
            alpha=0.9,
            dashes=dash_styles[i],
            label=fr"$\beta={beta:.2f}$",
        )

    ax.scatter(
        flow_logs[0],
        head_logs[0],
        s=14,
        alpha=0.7,
        color="#d62728",
        edgecolors="k",
        label="Pump 1 operating points",
        zorder=10,
        rasterized=True,
    )

    ax.set_xlabel(r"$Q$ (m$^{3}$/h)")
    ax.set_ylabel(r"$H$ (m)")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.1 * float(np.max(system_curve)))
    ax.set_xticks(np.arange(0, 401, 50))
    ax.set_yticks(np.arange(0, int(1.1 * np.max(system_curve)) + 10, 10))

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
    ax.minorticks_on()
    ax.legend(loc="upper left", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(figures_dir / "Trajectory.pdf", bbox_inches="tight")
    save_eps_without_tex(fig, figures_dir / "Trajectory.eps")
    plt.close(fig)


def save_simulation_excel(
    station: PumpStation,
    time_log: list[float],
    level_log: list[float],
    inflow_log: list[float],
    outflow_log: list[float],
    flow_logs: list[list[float]],
    head_logs: list[list[float]],
    freq_logs: list[list[float]],
    input_logs: list[list[float]],
    blockage_logs: list[list[float]],
) -> None:
    """Save the full simulated time series to Excel."""
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=int(t)) for t in time_log]

    data = {
        "Timestamp": timestamps,
        "Water_Level_m": level_log,
        "Total_Inflow_m3h": inflow_log,
        "Total_Outflow_m3h": outflow_log,
    }

    for i, _pump in enumerate(station.pumps, start=1):
        data[f"Pump{i}_Flow_m3h"] = flow_logs[i - 1]
        data[f"Pump{i}_Head_m"] = head_logs[i - 1]
        data[f"Pump{i}_Freq_Hz"] = freq_logs[i - 1]
        data[f"Pump{i}_Input_kW"] = input_logs[i - 1]
        data[f"Pump{i}_BlockageFactor"] = blockage_logs[i - 1]

    out_df = pd.DataFrame(data)
    output_path = results_dir / "pump_simulation_results_blockage_fixed.xlsx"
    out_df.to_excel(output_path, index=False)
    print(f"Simulation results saved to {output_path.resolve()}")


def plot_daily_metrics(station: PumpStation) -> None:
    apply_journal_style(font_size=10)

    days_runtime = sorted(station.daily_runtime.keys())
    p1_runtime = [station.daily_runtime[day][0] for day in days_runtime]
    p2_runtime = [station.daily_runtime[day][1] for day in days_runtime]
    p3_runtime = [station.daily_runtime[day][2] for day in days_runtime]

    days_starts = sorted(station.daily_starts.keys())
    p1_starts = [station.daily_starts[day][0] for day in days_starts]
    p2_starts = [station.daily_starts[day][1] for day in days_starts]
    p3_starts = [station.daily_starts[day][2] for day in days_starts]

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
    label_params = {"fontsize": 11, "fontweight": "bold", "ha": "left", "va": "bottom"}

    x = np.arange(len(days_runtime))
    width = 0.25
    axes[0].bar(x - width, p1_runtime, width=width, color=colors[0], edgecolor="black", lw=0.5)
    axes[0].bar(x, p2_runtime, width=width, color=colors[1], edgecolor="black", lw=0.5)
    axes[0].bar(x + width, p3_runtime, width=width, color=colors[2], edgecolor="black", lw=0.5)
    axes[0].set_ylabel("Runtime (hours)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Day {day + 1}" for day in days_runtime], rotation=45, ha="right")
    axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    axes[0].text(-0.08, 1.02, "(a)", transform=axes[0].transAxes, **label_params)

    x2 = np.arange(len(days_starts))
    axes[1].bar(x2 - width, p1_starts, width=width, color=colors[0], edgecolor="black", lw=0.5)
    axes[1].bar(x2, p2_starts, width=width, color=colors[1], edgecolor="black", lw=0.5)
    axes[1].bar(x2 + width, p3_starts, width=width, color=colors[2], edgecolor="black", lw=0.5)
    axes[1].set_ylabel("Starts (count)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"Day {day + 1}" for day in days_starts], rotation=45, ha="right")
    axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    axes[1].text(-0.08, 1.02, "(b)", transform=axes[1].transAxes, **label_params)

    for ax in axes:
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)

    fig.legend(
        ["Pump 1", "Pump 2", "Pump 3"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=True,
    )

    fig.savefig(figures_dir / "Station_Daily_Metrics.pdf", bbox_inches="tight")
    save_eps_without_tex(fig, figures_dir / "Station_Daily_Metrics.eps")
    plt.close(fig)


def plot_combined_summary(station: PumpStation, window: dict[str, object]) -> None:
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.0,
    })
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.8), constrained_layout=True)
    label_params = {"fontsize": 11, "fontweight": "bold", "ha": "left", "va": "bottom"}

    def clean_axis(ax: plt.Axes) -> None:
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
        ax.minorticks_on()

    ax = axes[0, 0]
    ax.plot(window["time"], window["level"], color=colors[0], lw=1.6)
    ax.axhline(level_start, color="gray", lw=0.9, ls="--")
    ax.axhline(level_stop, color="gray", lw=0.9, ls="--")
    ax.set_ylabel(r"Level (m)")
    ax.set_xlabel(r"Time (s)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    clean_axis(ax)
    ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes, **label_params)

    ax = axes[0, 1]
    for i, color in enumerate(colors):
        ax.plot(window["time"], window["head"][i], color=color, label=f"Pump {i + 1}")
    ax.set_ylabel(r"Head (m)")
    ax.set_xlabel(r"Time (s)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    clean_axis(ax)
    ax.text(-0.08, 1.02, "(b)", transform=ax.transAxes, **label_params)

    ax = axes[1, 0]
    for i, color in enumerate(colors):
        ax.plot(window["time"], window["flow"][i], color=color, label=f"Pump {i + 1}")
    ax.set_ylabel(r"Flow (m$^3$/h)")
    ax.set_xlabel(r"Time (s)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    clean_axis(ax)
    ax.text(-0.08, 1.02, "(c)", transform=ax.transAxes, **label_params)

    ax = axes[1, 1]
    hour_indices = sorted(station.hourly_energy.keys())
    hours = hour_indices[:25]
    pump_energies = [[station.hourly_energy[h][i] for h in hours] for i in range(3)]
    cumulative_energy = [np.cumsum(energy) for energy in pump_energies]
    markers = ["o", "s", "^"]

    for energy, color, marker in zip(cumulative_energy, colors, markers):
        ax.plot(hours, energy, color=color, marker=marker, markersize=4, lw=1.6)

    ax.set_xlabel(r"Time (hours)")
    ax.set_ylabel(r"Accumulated Energy (kWh)")
    ax.set_xlim(min(hours), max(hours))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    clean_axis(ax)
    ax.text(-0.08, 1.02, "(d)", transform=ax.transAxes, **label_params)

    handles = [Line2D([0], [0], color=color, lw=1.8, label=f"Pump {i + 1}") for i, color in enumerate(colors)]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        frameon=True,
    )

    fig.savefig(figures_dir / "Combined_Plots.pdf", bbox_inches="tight")
    save_eps_without_tex(fig, figures_dir / "Combined_Plots.eps")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================

def run_simulation(station: PumpStation) -> dict[str, object]:
    """Run the 48-hour simulation and collect all time-series logs."""
    time_log: list[float] = []
    level_log: list[float] = []
    inflow_log: list[float] = []
    outflow_log: list[float] = []

    flow_logs: list[list[float]] = [[] for _ in station.pumps]
    head_logs: list[list[float]] = [[] for _ in station.pumps]
    freq_logs: list[list[float]] = [[] for _ in station.pumps]
    input_logs: list[list[float]] = [[] for _ in station.pumps]
    blockage_logs: list[list[float]] = [[] for _ in station.pumps]
    efficiency_logs: list[list[float]] = [[] for _ in station.pumps]
    shaft_power_logs: list[list[float]] = [[] for _ in station.pumps]

    print("Starting 48-hour simulation.")
    while station.time_s < simulation_time:
        station.step(time_step)

        time_log.append(station.time_s)
        level_log.append(station.level)
        # Kept as a separate draw, matching the original logging behavior.
        inflow_log.append(station.inflow_m3h_peak())
        outflow_log.append(sum(pump.flow_m3h for pump in station.pumps))

        for i, pump in enumerate(station.pumps):
            flow_logs[i].append(pump.flow_m3h)
            head_logs[i].append(pump.head_m)
            freq_logs[i].append(pump.current_freq)
            input_logs[i].append(pump.input_kw)
            blockage_logs[i].append(pump.blockage_factor)
            efficiency_logs[i].append(pump.efficiency)
            shaft_power_logs[i].append(pump.shaft_kw)

    print("Simulation complete. Preparing outputs.")
    return {
        "time": time_log,
        "level": level_log,
        "inflow": inflow_log,
        "outflow": outflow_log,
        "flow_logs": flow_logs,
        "head_logs": head_logs,
        "freq_logs": freq_logs,
        "input_logs": input_logs,
        "blockage_logs": blockage_logs,
        "efficiency_logs": efficiency_logs,
        "shaft_power_logs": shaft_power_logs,
    }


def main() -> None:
    make_output_dirs()
    sns.set(style="whitegrid")

    positive_inflows = load_positive_inflows(flow_csv_path)
    station = PumpStation(positive_inflows)
    logs = run_simulation(station)

    plot_wetwell_overview(
        logs["time"],
        logs["level"],
        logs["inflow"],
        logs["outflow"],
    )

    window = extract_windowed_logs(
        logs["time"],
        logs["level"],
        logs["inflow"],
        logs["outflow"],
        logs["flow_logs"],
        logs["head_logs"],
        logs["input_logs"],
    )

    plot_water_level(window)
    plot_inflow(window)
    plot_operating_trajectory(logs["flow_logs"], logs["head_logs"])

    save_simulation_excel(
        station,
        logs["time"],
        logs["level"],
        logs["inflow"],
        logs["outflow"],
        logs["flow_logs"],
        logs["head_logs"],
        logs["freq_logs"],
        logs["input_logs"],
        logs["blockage_logs"],
    )

    plot_daily_metrics(station)
    plot_combined_summary(station, window)

    print(f"All outputs are in {results_dir.resolve()}")


if __name__ == "__main__":
    main()

# %%
