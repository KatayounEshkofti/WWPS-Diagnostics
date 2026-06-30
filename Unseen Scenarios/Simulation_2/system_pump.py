#!/usr/bin/env python3
# %% -*- coding: utf-8 -*-
"""
Pump-station simulation with two fault scenarios
================================================

This script simulates a three-pump wet-well station over 48 hours. It uses an
empirical inflow distribution, soft-start/soft-stop pump dynamics, a system-curve
model, and two injected fault scenarios:

1. A temporary system fault represented by a ramp in friction and static head.
2. A gradual Pump 1 blockage during the second day of simulation.

The script saves the simulated time series to Excel and generates the figures
used for inspection and reporting.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(seed=42)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Project paths
flow_csv_path: Final[Path] = Path("data/flow_rates.csv")
results_dir = Path("results")
figures_dir = results_dir / "figures"

# Wet-well geometry
cross_section_area = 8.0  # m²

# Pump frequency and speed
pump_min_freq = 25.0      # Hz, used during soft-start
pump_max_freq = 50.0      # Hz, nominal full speed
nominal_freq = 50.0       # Hz
nominal_rpm = 1460.0      # rpm at 50 Hz

# Level-control set points
level_start = 1.60   # m, start lead pump if no pump is running
level_stop = 0.50    # m, emergency stop level
level_start2 = 1.80  # m, start the second pump if one pump is already running
level_stop2 = 0.80   # m, stop the second pump when two pumps are running

# Inflow peak settings
peak_rate = 0.0005      # events per second, about 18 events/day
peak_magnitude = 0.0    # extra flow during peaks, m³/h
peak_duration = 900.0   # seconds, 15 minutes

# Pump curve at 50 Hz: flow in US gpm, head in m
pump_curve_data_50hz: list[tuple[float, float]] = [
    (0, 32.0),
    (400, 29.9),
    (800, 27.4),
    (1200, 24.4),
    (1600, 21.3),
    (2000, 18.3),
]

# System curve: H = Hs + K Q², with Q in m³/h and H in m
static_head = 2.0
friction_coeff = 0.0006

# Nominal system-curve values, used for fault recovery
friction_coeff_base = friction_coeff
static_head_base = static_head

# Temporary system fault
system_fault_start = 50_000
system_fault_end = 65_000
system_friction_increase = 0.8
system_static_head_increase = 0.5

# Pump 1 blockage fault
pump_fault_start = 86_400 + 10_000
pump_fault_end = 86_400 + 40_000
pump_blockage_drop = 0.5
pump_efficiency_drop = 0.5
pump_recovered_blockage_factor = 0.9
pump_recovered_efficiency_factor = 0.7

# Physics and efficiency
pump_efficiency = 0.90
water_density = 1000.0  # kg/m³
gravity = 9.81          # m/s²

# Electrical approximation
nominal_voltage = 400.0   # V, three-phase
nominal_current = 30.0    # A
power_factor = 0.9

# Unit conversion
usgpm_to_m3h = 0.2271

# Soft-start/stop durations
soft_start_time = 10.0  # s
soft_stop_time = 10.0   # s

# Simulation horizon
sim_time = 86_400 * 2  # 2 days
dt = 1.0               # s

# Measurement noise
level_meas_noise_std = 0.02      # roughly ±2 cm
flow_noise_relative_std = 0.01   # 1%
power_noise_relative_std = 0.01  # 1%


# -----------------------------------------------------------------------------
# Inflow data and empirical sampler
# -----------------------------------------------------------------------------

def load_positive_inflows(path: Path) -> np.ndarray:
    """Load positive inflow samples from the preprocessing output."""
    try:
        flow_df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Could not find {path}. Run the flow-rate preprocessing script first "
            "or update flow_csv_path."
        ) from exc

    positive = flow_df["flow_rate_m3_s"]
    positive = positive[positive > 0.0].to_numpy(dtype=np.float64)
    positive.sort()

    if positive.size == 0:
        raise SystemExit("No positive inflow samples were found in the flow-rate file.")

    print(f"Loaded {positive.size:,} positive inflow samples for ECDF sampling.")
    return positive


positive_inflow_samples = load_positive_inflows(flow_csv_path)


def ecdf_draw_m3s(size: int = 1, sampler: np.random.Generator | None = None) -> np.ndarray:
    """Draw inflow values from the empirical distribution, returned in m³/s."""
    sampler = sampler or np.random.default_rng()
    u = sampler.random(size)
    return np.quantile(positive_inflow_samples, u)


# -----------------------------------------------------------------------------
# Pump and system-curve helpers
# -----------------------------------------------------------------------------

def piecewise_linear_interpolation(
    x_value: float,
    points: list[tuple[float, float]],
) -> float:
    """Interpolate linearly over a monotonic flow-head table."""
    if x_value <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif x_value >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= x_value <= x1:
                break

    return y0 + (x_value - x0) * (y1 - y0) / (x1 - x0)


def get_pump_head_at_flow(flow_usgpm: float, speed_ratio: float) -> float:
    """Evaluate the pump curve with the standard affinity-law scaling."""
    if speed_ratio <= 0.0:
        return 0.0

    flow_nominal = flow_usgpm / speed_ratio
    head_nominal = piecewise_linear_interpolation(flow_nominal, pump_curve_data_50hz)
    return head_nominal * speed_ratio**2


def system_head(flow_m3h: float | np.ndarray) -> float | np.ndarray:
    """Return the system head for a given flow rate."""
    return static_head + friction_coeff * flow_m3h**2


def find_operating_flow(rpm: float) -> float:
    """Find the operating flow where pump head and system head intersect."""
    if rpm < 50.0:
        return 0.0

    speed_ratio = rpm / nominal_rpm
    lo, hi = 0.0, 3000.0  # US gpm bounds
    tolerance = 0.1

    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        pump_head = get_pump_head_at_flow(mid, speed_ratio)
        station_head = system_head(mid * usgpm_to_m3h)
        lo, hi = (mid, hi) if pump_head > station_head else (lo, mid)

    flow_usgpm = 0.5 * (lo + hi)
    return flow_usgpm * usgpm_to_m3h


def approximate_power_kw(flow_m3h: float, efficiency: float) -> float:
    """Approximate hydraulic shaft power in kW."""
    if flow_m3h <= 0.0 or efficiency < 1e-9:
        return 0.0

    head_m = system_head(flow_m3h)
    flow_m3s = flow_m3h / 3600.0
    return water_density * gravity * flow_m3s * head_m / (1000.0 * efficiency)


# -----------------------------------------------------------------------------
# Pump model
# -----------------------------------------------------------------------------

class Pump:
    """A centrifugal pump with a VFD soft-start/soft-stop profile."""

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
        """Start the pump with the soft-start ramp."""
        if not self.is_running:
            self.is_running = True
            self._soft_starting = True
            self._soft_start_elapsed = 0.0
            self.current_freq = pump_min_freq
            print(f"[{time_s:.1f}s] {self.name} start, soft-start ramp")

    def stop(self, time_s: float) -> None:
        """Request a soft stop."""
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            print(f"[{time_s:.1f}s] {self.name} stop, soft-stop ramp")

    def _effective_rpm(self) -> float:
        """Hydraulic speed after blockage scaling."""
        rpm_nominal = (self.current_freq / nominal_freq) * nominal_rpm
        return rpm_nominal * self.blockage_factor

    def _update_soft_ramps(self, step_s: float) -> None:
        if self._soft_starting:
            self._soft_start_elapsed += step_s
            fraction = min(1.0, self._soft_start_elapsed / soft_start_time)
            self.current_freq = pump_min_freq + (pump_max_freq - pump_min_freq) * fraction
            if fraction >= 1.0:
                self._soft_starting = False

        if self._soft_stopping:
            self._soft_stop_elapsed += step_s
            fraction = min(1.0, self._soft_stop_elapsed / soft_stop_time)
            self.current_freq *= 1.0 - fraction
            if self.current_freq <= 0.1:
                self.current_freq = 0.0
                self._soft_stopping = False
                self.is_running = False
                print("        fully stopped")

        if self.is_running and not (self._soft_starting or self._soft_stopping):
            self.current_freq = pump_max_freq

    def update(self, step_s: float, time_s: float) -> None:
        """Update pump hydraulics and power for one simulation step."""
        self._update_soft_ramps(step_s)

        if not (self.is_running or self._soft_stopping):
            self.flow_m3h = 0.0
            self.head_m = 0.0
            self.shaft_kw = 0.0
            self.input_kw = 0.0
            return

        effective_rpm = self._effective_rpm()
        raw_flow = find_operating_flow(effective_rpm)
        flow_noise = 1.0 + random.gauss(0.0, flow_noise_relative_std)
        self.flow_m3h = max(0.0, raw_flow * flow_noise)

        flow_usgpm = self.flow_m3h / usgpm_to_m3h
        speed_ratio = effective_rpm / nominal_rpm
        self.head_m = get_pump_head_at_flow(flow_usgpm, speed_ratio)

        self.shaft_kw = approximate_power_kw(self.flow_m3h, self.efficiency)

        if self.current_freq > 0:
            current_ratio = self.current_freq / pump_max_freq
            effective_current = min(
                nominal_current * current_ratio / self.blockage_factor,
                5 * nominal_current,
            )
            self.input_kw = (
                math.sqrt(3)
                * nominal_voltage
                * effective_current
                * power_factor
                / 1000
            )
            self.input_kw *= 1.0 + random.gauss(0.0, power_noise_relative_std)
        else:
            self.input_kw = 0.0

        self.energy_kwh += self.input_kw * step_s / 3600.0


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
        self.running_stack: list[int] = []

        self.daily_starts: dict[int, list[int]] = {}
        self.daily_runtime: dict[int, list[float]] = {}
        self.hourly_energy: dict[int, list[float]] = {}
        self.active_peaks: list[float] = []
        self.next_peak = random.expovariate(peak_rate)

    def inflow_m3h_peak(self) -> float:
        """Draw inflow from the ECDF and add active peak-flow events."""
        base_m3s = float(ecdf_draw_m3s())
        base_m3h = base_m3s * 3600.0

        peak_flow = 0.0
        self.next_peak -= dt
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
        """Draw one inflow value from the empirical distribution in m³/h."""
        flow_m3s = float(ecdf_draw_m3s())
        return flow_m3s * 3600.0

    def _next_lead(self) -> int:
        self.lead_index = (self.lead_index + 1) % len(self.pumps)
        return self.lead_index

    def _start_pump(self, pump_index: int) -> None:
        self.pumps[pump_index].start(self.time_s)
        self.running_stack.append(pump_index)

        day = int(self.time_s // 86_400)
        self.daily_starts.setdefault(day, [0, 0, 0])[pump_index] += 1
        self._next_lead()

    def _stop_last_pump(self) -> None:
        if self.running_stack:
            pump_index = self.running_stack[-1]
            if self.pumps[pump_index].is_running:
                self.pumps[pump_index].stop(self.time_s)

    def _stop_all(self) -> None:
        for pump_index in reversed(self.running_stack):
            if self.pumps[pump_index].is_running:
                self.pumps[pump_index].stop(self.time_s)

    def control(self) -> None:
        """Apply wet-well level control logic."""
        measured_level = self.level + random.gauss(0.0, level_meas_noise_std)
        n_running = len(self.running_stack)

        if measured_level >= level_start and n_running == 0:
            self._start_pump(self.lead_index)
        elif n_running == 1 and measured_level >= level_start2:
            self._start_pump(self.lead_index)

        if n_running == 2 and measured_level <= level_stop2:
            self._stop_last_pump()
        if measured_level <= level_stop:
            self._stop_all()

    def step(self, step_s: float) -> None:
        """Advance the station simulation by one time step."""
        global friction_coeff, static_head

        # System fault and recovery are applied before pump hydraulics.
        if self.time_s < system_fault_start:
            friction_coeff = friction_coeff_base
            static_head = static_head_base
        elif self.time_s < system_fault_end:
            progress = (self.time_s - system_fault_start) / (
                system_fault_end - system_fault_start
            )
            friction_coeff = friction_coeff_base * (1 + system_friction_increase * progress)
            static_head = static_head_base + system_static_head_increase * progress
        else:
            friction_coeff = friction_coeff_base
            static_head = static_head_base

        for pump in self.pumps:
            pump.update(step_s, self.time_s)

        self.running_stack = [
            i for i in self.running_stack if self.pumps[i].is_running
        ]

        inflow = self.inflow_m3h_peak()
        outflow = sum(pump.flow_m3h for pump in self.pumps)
        volume_change = (inflow - outflow) / 3600 * step_s
        self.level += volume_change / cross_section_area

        pump_1 = self.pumps[0]
        if pump_fault_start <= self.time_s < pump_fault_end:
            progress = (self.time_s - pump_fault_start) / (pump_fault_end - pump_fault_start)
            progress = np.clip(progress, 0.0, 1.0)
            pump_1.blockage_factor = 1.0 - pump_blockage_drop * progress
            pump_1.efficiency = pump_efficiency * (1.0 - pump_efficiency_drop * progress)
        elif self.time_s >= pump_fault_end:
            pump_1.blockage_factor = pump_recovered_blockage_factor
            pump_1.efficiency = pump_efficiency * pump_recovered_efficiency_factor
        else:
            pump_1.blockage_factor = 1.0
            pump_1.efficiency = pump_efficiency

        self.control()

        day = int(self.time_s // 86_400)
        self.daily_runtime.setdefault(day, [0.0, 0.0, 0.0])
        step_h = step_s / 3600
        for i, pump in enumerate(self.pumps):
            if pump.is_running:
                self.daily_runtime[day][i] += step_h

        hour = int(self.time_s // 3600)
        self.hourly_energy.setdefault(hour, [0.0, 0.0, 0.0])
        for i, pump in enumerate(self.pumps):
            self.hourly_energy[hour][i] += pump.input_kw * step_h

        self.time_s += step_s


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def configure_plot_style(font_scale: float = 1.2) -> None:
    """Apply the plotting style used throughout the script."""
    sns.set(style="whitegrid", font_scale=font_scale)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_figure(filename: str, *, figure: plt.Figure | None = None, **kwargs) -> None:
    """Save a figure inside the configured figures directory."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    target = figures_dir / filename
    active_figure = figure or plt.gcf()
    active_figure.savefig(target, **kwargs)


# -----------------------------------------------------------------------------
# Main simulation and reporting
# -----------------------------------------------------------------------------

def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid")
    station = PumpStation()

    t_log: list[float] = []
    level_log: list[float] = []
    inflow_log: list[float] = []
    outflow_log: list[float] = []

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
        station.step(dt)

        # Keep the original logging behavior: draw/log inflow after each step.
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

    # ------------------------------------------------------------------
    # A) Wet-well level and flow overview
    # ------------------------------------------------------------------
    idx_window = [i for i, time_value in enumerate(t_log) if 10_000 <= time_value <= 22_000]
    time_array = np.array(t_log)
    level_array = np.array(level_log)
    inflow_array = np.array(inflow_log)
    outflow_array = np.array(outflow_log)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time_array[idx_window], level_array[idx_window])
    axes[0].axhline(level_start, c="r", ls="--", label="Start 1.6 m")
    axes[0].axhline(level_stop, c="g", ls="--", label="Stop 0.5 m")
    axes[0].set_ylabel("Level [m]")
    axes[0].legend()

    axes[1].plot(time_array[idx_window], outflow_array[idx_window], label="Outflow")
    axes[1].set_ylabel("m³/h")

    axes[2].plot(time_array[idx_window], inflow_array[idx_window], label="Inflow", color="#f7a072")
    axes[2].set_ylabel("m³/h")
    axes[2].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True)

    fig.tight_layout()
    plt.show()

    mask_window = [i for i, time_value in enumerate(t_log) if 10_000 <= time_value <= 22_000]
    t_window = [t_log[i] for i in mask_window]
    level_window = [level_log[i] for i in mask_window]
    inflow_window = [inflow_log[i] for i in mask_window]
    outflow_window = [outflow_log[i] for i in mask_window]

    freq_window = [[] for _ in station.pumps]
    power_window = [[] for _ in station.pumps]
    flow_window = [[] for _ in station.pumps]
    head_window = [[] for _ in station.pumps]
    input_power_window = [[] for _ in station.pumps]

    for i in range(3):
        freq_window[i] = [freq_logs[i][j] for j in mask_window]
        flow_window[i] = [flow_logs[i][j] for j in mask_window]
        head_window[i] = [head_logs[i][j] for j in mask_window]
        power_window[i] = [input_logs[i][j] for j in mask_window]

    # ------------------------------------------------------------------
    # Separate figures used in the paper/report
    # ------------------------------------------------------------------
    configure_plot_style()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, level_window, "b-", label="Water Level [m]")
    plt.axhline(level_start, color="r", ls="--", label="Start at 1.6 m")
    plt.axhline(level_stop, color="g", ls="--", label="Stop at 0.5 m")
    plt.ylabel(r"Level (m)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    save_figure("Water_Level.eps", format="eps", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, inflow_window, "b-", label="Inflow", color="#fb5607")
    plt.ylabel(r"Inflow rate (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    save_figure("Inflow_rate.eps", format="eps", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, outflow_window, "c-")
    plt.ylabel(r"Outflow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure("Total_Outflow.eps", format="eps", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, head_window[0], "b-", label="Pump 1")
    plt.plot(t_window, head_window[1], "r-", label="Pump 2")
    plt.plot(t_window, head_window[2], "g-", label="Pump 3")
    plt.ylabel(r"Head (m)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    save_figure("Pump_Heads.eps", format="eps", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, power_window[0], "b-", label="Pump 1")
    plt.plot(t_window, power_window[1], "r-", label="Pump 2")
    plt.plot(t_window, power_window[2], "g-", label="Pump 3")
    plt.ylabel(r"Power (kW)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure("Pump_Power.eps", format="eps", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_window, flow_window[0], "b-", label="Pump 1")
    plt.plot(t_window, flow_window[1], "r-", label="Pump 2")
    plt.plot(t_window, flow_window[2], "g-", label="Pump 3")
    plt.ylabel(r"Flow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    save_figure("Pump_Flow.eps", format="eps", dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # B) Operating-point trajectory on the system curve
    # ------------------------------------------------------------------
    flow_axis = np.linspace(0, 400, 250)
    system_head_values = system_head(flow_axis)

    plt.figure(figsize=(9, 7), layout="constrained")
    configure_plot_style()

    plt.plot(flow_axis, system_head_values, "k-", lw=3, alpha=0.9, label="System curve")

    beta_values = np.linspace(1.0, 0.6, 5)
    beta_colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
    for i, beta in enumerate(beta_values):
        pump_head_values = [get_pump_head_at_flow(q / usgpm_to_m3h, beta) for q in flow_axis]
        plt.plot(
            flow_axis,
            pump_head_values,
            "--",
            lw=2,
            alpha=0.8,
            color=beta_colors[i],
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
    plt.ylim(0, 1.1 * max(system_head_values))
    plt.xticks(np.arange(0, 401, 50))
    plt.yticks(np.arange(0, int(1.1 * max(system_head_values)) + 10, 10))
    save_figure("Trajectory.eps", format="eps", dpi=300, bbox_inches="tight")
    save_figure(
        "Trajectory.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close()

    # ------------------------------------------------------------------
    # C) Save the full simulated time series
    # ------------------------------------------------------------------
    start_timestamp = datetime.now()
    timestamps = [start_timestamp + timedelta(seconds=int(s)) for s in t_log]

    data = {
        "Timestamp": timestamps,
        "Water_Level_m": level_log,
        "Total_Inflow_m3h": inflow_log,
        "Total_Outflow_m3h": outflow_log,
    }
    for i, _pump in enumerate(station.pumps, 1):
        data[f"Pump{i}_Flow_m3h"] = flow_logs[i - 1]
        data[f"Pump{i}_Head_m"] = head_logs[i - 1]
        data[f"Pump{i}_Freq_Hz"] = freq_logs[i - 1]
        data[f"Pump{i}_Input_kW"] = input_logs[i - 1]
        data[f"Pump{i}_BlockageFactor"] = blockage_logs[i - 1]

    sim_df = pd.DataFrame(data)
    output_xlsx = results_dir / "pump_system_faultsimualtion.xlsx"
    sim_df.to_excel(output_xlsx, index=False)
    print(f"Simulation data saved to {output_xlsx.resolve()}")

    # ------------------------------------------------------------------
    # Hourly accumulated energy consumption
    # ------------------------------------------------------------------
    if station.hourly_energy:
        hour_indices = sorted(station.hourly_energy.keys())
        hours = hour_indices
        pump_energies = [
            [station.hourly_energy[h][pump_index] for h in hour_indices[0:25]]
            for pump_index in range(3)
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

        configure_plot_style()
        plt.xlabel(r"Time (hours)")
        plt.ylabel(r"Accumulated Energy (kWh)")
        plt.legend()
        plt.grid(True)
        plt.xticks(hours)
        plt.xlim(0, 25)
        plt.tight_layout()
        save_figure("AccEnergy.eps", format="eps", dpi=300)
        plt.close()
    else:
        print("No hourly energy data recorded.")

    # ------------------------------------------------------------------
    # Daily runtime and starts
    # ------------------------------------------------------------------
    configure_plot_style()

    days_runtime = sorted(station.daily_runtime.keys())
    pump1_runtime = [station.daily_runtime[d][0] for d in days_runtime]
    pump2_runtime = [station.daily_runtime[d][1] for d in days_runtime]
    pump3_runtime = [station.daily_runtime[d][2] for d in days_runtime]

    days_starts = sorted(station.daily_starts.keys())
    pump1_starts = [station.daily_starts[d][0] for d in days_starts]
    pump2_starts = [station.daily_starts[d][1] for d in days_starts]
    pump3_starts = [station.daily_starts[d][2] for d in days_starts]

    bar_colors = sns.color_palette("pastel")[:3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    label_params = {"fontsize": 14, "fontweight": "bold", "ha": "left", "va": "top"}

    x_runtime = np.arange(len(days_runtime))
    bar_width = 0.25
    axes[0].bar(x_runtime - bar_width, pump1_runtime, width=bar_width, label="Pump 1", color=bar_colors[0], edgecolor="black")
    axes[0].bar(x_runtime, pump2_runtime, width=bar_width, label="Pump 2", color=bar_colors[1], edgecolor="black")
    axes[0].bar(x_runtime + bar_width, pump3_runtime, width=bar_width, label="Pump 3", color=bar_colors[2], edgecolor="black")
    axes[0].set_ylabel("Runtime (hours)")
    axes[0].set_xticks(x_runtime)
    axes[0].set_xticklabels([f"Day {d + 1}" for d in days_runtime], rotation=45, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].text(-0.06, 0.98, "(a)", transform=axes[0].transAxes, **label_params)

    x_starts = np.arange(len(days_starts))
    axes[1].bar(x_starts - bar_width, pump1_starts, width=bar_width, color=bar_colors[0], edgecolor="black")
    axes[1].bar(x_starts, pump2_starts, width=bar_width, color=bar_colors[1], edgecolor="black")
    axes[1].bar(x_starts + bar_width, pump3_starts, width=bar_width, color=bar_colors[2], edgecolor="black")
    axes[1].set_ylabel("Starts (count)")
    axes[1].set_xticks(x_starts)
    axes[1].set_xticklabels([f"Day {d + 1}" for d in days_starts], rotation=45, ha="right")
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
    save_figure(
        "Station_Daily_Metrics.pdf",
        figure=fig,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    save_figure(
        "Station Daily Metrics.eps",
        figure=fig,
        format="eps",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # ------------------------------------------------------------------
    # Combined panel: level, head, flow, energy
    # ------------------------------------------------------------------
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    label_params = {"fontsize": 14, "fontweight": "bold", "ha": "left", "va": "top"}

    ax = axes[0, 0]
    ax.plot(t_window, level_window, "b-")
    ax.axhline(level_start, color="r", ls="--", label="Start at 1.6 m")
    ax.axhline(level_stop, color="g", ls="--", label="Stop at 0.5 m")
    ax.set_ylabel(r"Level (m)")
    ax.set_xlabel(r"Time (s)")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    ax.text(-0.08, 0.93, "(a)", transform=ax.transAxes, **label_params)

    ax = axes[0, 1]
    ax.plot(t_window, head_window[0], "b-", label="Pump 1")
    ax.plot(t_window, head_window[1], "r-", label="Pump 2")
    ax.plot(t_window, head_window[2], "g-", label="Pump 3")
    ax.set_ylabel(r"Head (m)")
    ax.set_xlabel(r"Time (s)")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    ax.text(-0.08, 0.93, "(b)", transform=ax.transAxes, **label_params)

    ax = axes[1, 0]
    ax.plot(t_window, flow_window[0], "b-", label="Pump 1")
    ax.plot(t_window, flow_window[1], "r-", label="Pump 2")
    ax.plot(t_window, flow_window[2], "g-", label="Pump 3")
    ax.set_ylabel(r"Flow (m$^3$/h)")
    ax.set_xlabel(r"Time (s)")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    ax.text(-0.08, 0.98, "(c)", transform=ax.transAxes, **label_params)

    ax = axes[1, 1]
    hour_indices = sorted(station.hourly_energy.keys())
    hours = hour_indices
    pump_energies = [
        [station.hourly_energy[h][pump_index] for h in hour_indices[0:25]]
        for pump_index in range(3)
    ]
    cumulative_energies = [np.cumsum(energies) for energies in pump_energies]
    colors = ["b", "r", "g"]
    markers = ["o", "s", "^"]
    labels = ["Pump 1", "Pump 2", "Pump 3"]

    for i in range(3):
        ax.plot(
            hours[0:25],
            cumulative_energies[i][0:25],
            color=colors[i],
            marker=markers[i],
            linewidth=2,
            markersize=6,
            label=labels[i],
        )

    ax.set_xlabel(r"Time (hours)")
    ax.set_ylabel(r"Accumulated Energy (kWh)")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_xticks(hours)
    ax.set_xlim(0, 25)
    ax.text(-0.09, 0.98, "(d)", transform=ax.transAxes, **label_params)

    save_figure("Combined_Plots.eps", figure=fig, format="eps", dpi=300)
    plt.close()

    _ = efficiency_logs, input_power_logs, freq_window, input_power_window

    print(f"Figures saved in {figures_dir.resolve()}")


if __name__ == "__main__":
    main()

# %%
