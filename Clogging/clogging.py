#!/usr/bin/env python3
# %%
"""
Pump-station simulation with a gradual system fault.

The model represents a three-pump wet-well station with soft-start/soft-stop
control, a synthetic sinusoidal inflow profile with random peak events, and a
gradual system-curve fault during the second simulated day. The script exports
the simulated time series and the figures used to inspect the fault response.
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

# Reproducibility
random_seed = 42

# Seed the Python random generator used by the simulation.
random.seed(random_seed)                       # Python's random

# Project-local output folders.
output_dir = Path("results/system_fault_simulation")
figure_dir = output_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------------
# 1. Simulation parameters
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
level_stop   = 0.50  # m – emergency stop all
level_start2 = 1.8  # m – start second pump when one already running
level_stop2  = 0.80  # m – stop second pump when two running


peak_rate = 0.0005  # Events per second (≈18 events/day)
peak_magnitude = 50.0  # Extra flow during peaks (m³/h)
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

# Keep copies of the original system-curve constants:
friction_coeff_base = friction_coeff
static_head_base    = static_head

# When (during day 2) the pipe-clogging ramp starts/stops:
system_fault_start       = 86400 + 40000    # day 2,  40 000 s into it
system_fault_end         = 86400 + 61600    # day 2,  61 600 s into it
# How much to increase by end of ramp (fractional / absolute):
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
sim_time = 86400 * 2  # 2 days
dt       = 1.0        # s, discrete time‑step

# Measurement noise (sensor realism)
level_meas_noise_std     = 0.02   # ±2 cm
flow_noise_relative_std  = 0.01   # 1 %
power_noise_relative_std = 0.01   # 1 %

# -----------------------------------------------------------------------------
# 2. Helper functions: pump curves and power
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
            print(f"[{t:.1f}s] {self.name} start")

    def stop(self, t: float) -> None:
        if self.is_running and not self._soft_stopping:
            self._soft_stopping = True
            self._soft_stop_elapsed = 0.0
            print(f"[{t:.1f}s] {self.name} stop")

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
                print("        fully stopped")
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
        self.level: float = 0.8  # m – initial wet‑well level
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
    # Inflow model (simple sinusoid + noise, tweak as desired)
    # ---------------------------------------------------------------
    def inflow_m3h(self) -> float:
        # Base sinusoidal pattern
        period = 86400.0
        t_day = self.time_s % period
        base = 60.0 + 20.0 * math.sin(2 * math.pi * t_day / period)
        
        # Add Poisson-distributed peaks
        peak_flow = 0.0
        self.next_peak -= dt  # Count down to next event
        
        # Trigger new peak event
        if self.next_peak <= 0:
            self.active_peaks.append(self.time_s + peak_duration)
            self.next_peak = random.expovariate(peak_rate)
            print(f"[{self.time_s:.0f}s] inflow peak started")
        
        # Process active peaks
        for end_time in list(self.active_peaks):
            if self.time_s > end_time:
                self.active_peaks.remove(end_time)  # Remove expired peaks
            else:
                peak_flow += peak_magnitude  # Add peak flow
        
        # Combine base + noise + peaks
        noise = random.uniform(-5, 5)
        return max(0.0, base + noise + peak_flow)
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
        elif n_run == 1 and measured_level >= level_start2:
            self._start_pump(self.lead_index)

        # ------- stop logic --------
        if n_run == 2 and measured_level <= level_stop2:
            self._stop_last_pump()
        if measured_level <= level_stop:
            self._stop_all()

    # ---------------------------------------------------------------
    # Simulation step
    # ---------------------------------------------------------------
    def step(self, dt: float):
        global friction_coeff, static_head

        if system_fault_start <= self.time_s < system_fault_end:
            prog = (self.time_s - system_fault_start) / (system_fault_end - system_fault_start)
            # ramp from base → base*(1+system_friction_increase)
            friction_coeff = friction_coeff_base * (1 + system_friction_increase * prog)
            # ramp static head up by system_static_head_increase
            static_head    = static_head_base  + system_static_head_increase * prog

        elif self.time_s >= system_fault_end:
            # after the ramp, keep the "clogged" curve
            friction_coeff = friction_coeff_base * (1 + system_friction_increase)
            static_head    = static_head_base  + system_static_head_increase

        else:
            # before day 2 fault, use the original system curve
            friction_coeff = friction_coeff_base
            static_head    = static_head_base
        
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

        """ # 3) Blockage scenario – Pump 1 gradual clog 12 000 s → 15 000 s
        p1 = self.pumps[0]
        if 12000 <= self.time_s < 21000:
            prog = (self.time_s - 12000) / 9000  # 0 → 1
            p1.blockage_factor = 1.0 - 0.4 * prog  # 1 → 0.6
            p1.efficiency = pump_efficiency * (1.0 - 0.3 * prog)
        elif self.time_s >= 21000:
            p1.blockage_factor = 1.0  # 0.6 if we want to keep degradation afterwards 
            p1.efficiency = pump_efficiency
        else:
            p1.blockage_factor = 1.0
            p1.efficiency = pump_efficiency """

        # 4) Control – start/stop commands (after blockage update!)
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
# 5. Run simulation and create figures
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sns.set(style="whitegrid")

    station = PumpStation()

    # logs for later analysis
    t_log: List[float] = []
    level_log: List[float] = []
    inflow_log: List[float] = []
    outflow_log: List[float] = []

    flow_logs  = [[] for _ in station.pumps]
    head_logs  = [[] for _ in station.pumps]
    freq_logs  = [[] for _ in station.pumps]
    input_logs = [[] for _ in station.pumps]
    blockage_logs = [[] for _ in station.pumps]
    efficiency_logs = [[] for _ in station.pumps]
    input_power_logs = [[] for _ in station.pumps]
    power_logs = [[] for _ in station.pumps]

    print("Starting 48-hour system-fault simulation...")
    while station.time_s < sim_time:
        station.step(dt)

        # collect
        t_log.append(station.time_s)
        level_log.append(station.level)
        inflow_log.append(station.inflow_m3h())
        outflow_log.append(sum(p.flow_m3h for p in station.pumps))
        for i, p in enumerate(station.pumps):
            flow_logs[i].append(p.flow_m3h)
            head_logs[i].append(p.head_m)
            freq_logs[i].append(p.current_freq)
            input_logs[i].append(p.input_kW)
            blockage_logs[i].append(p.blockage_factor)
            power_logs[i].append(p.shaft_kW)

    print("Simulation complete. Creating figures...")

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
    mask_start = system_fault_start - 2000
    mask_stop = system_fault_end + 2000
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
    
    #%% Separate plots 
    # Water Level
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
                "font.size": 16,          # Base font size
                "axes.titlesize": 18,     # Axes title
                "axes.labelsize": 16,     # Axes labels
                "xtick.labelsize": 16,    # X tick labels
                "ytick.labelsize": 16,    # Y tick labels
                "legend.fontsize": 16,    # Legend font size
            })
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS
    plt.plot(t_5k, lvl_5k, 'b-', label="Water Level [m]")
    plt.axhline(level_start, color='r', ls='--', label="Start at 1.6 m")
    plt.axhline(level_stop,  color='g', ls='--', label="Stop at 0.5 m")
    plt.ylabel(r"Level (m)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "water_level.eps", format='eps', dpi=300)
    plt.close()
    
    #%%
    # Inflow pattern
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
            "font.size": 16,          # Base font size
            "axes.titlesize": 18,     # Axes title
            "axes.labelsize": 16,     # Axes labels
            "xtick.labelsize": 16,    # X tick labels
            "ytick.labelsize": 16,    # Y tick labels
            "legend.fontsize": 16,    # Legend font size
        })
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS
    plt.plot(t_5k, inflw_5k, 'b-', label="Inflow rate", color="#fb5607")
    plt.ylabel(r"Inflow rate (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "inflow_rate.eps", format='eps', dpi=300)
    plt.close()

    #%%  Total Outflow
    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
                "font.size": 16,          # Base font size
                "axes.titlesize": 18,     # Axes title
                "axes.labelsize": 16,     # Axes labels
                "xtick.labelsize": 16,    # X tick labels
                "ytick.labelsize": 16,    # Y tick labels
                "legend.fontsize": 16,    # Legend font size
            })
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS
    plt.plot(t_5k, outfw_5k, 'c-', color='#02c39a')
    plt.ylabel(r"Outflow (m$^3$/h)")
    plt.xlabel(r"Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_dir / "total_outflow.eps", format='eps', dpi=300)
    plt.close()

    #%% Pump Heads
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
    plt.savefig(figure_dir / "pump_heads.eps", format='eps', dpi=300)
    plt.close()

    #%% Pump Power
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
    plt.savefig(figure_dir / "pump_power.eps", format='eps', dpi=300)
    plt.close()

    #%% Pump Flow
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
    plt.savefig(figure_dir / "pump_flow.eps", format='eps', dpi=300)
    plt.close()

    #%% ---------------------------------------------------------------------
    # C) Save full data to Excel
    # ---------------------------------------------------------------------
    ts0 = datetime.now()
    timestamps = [ts0 + timedelta(seconds=int(s)) for s in t_log]
    data = {
        "Timestamp": timestamps,
        "Water_Level_m": level_log,
        "Total_Inflow_m3h": inflow_log,
        "Total_Outflow_m3h": outflow_log,
    }
    for i, p in enumerate(station.pumps, 1):
        data[f"Pump{i}_Flow_m3h"]      = flow_logs[i - 1]
        data[f"Pump{i}_Head_m"]        = head_logs[i - 1]
        data[f"Pump{i}_Freq_Hz"]       = freq_logs[i - 1]
        data[f"Pump{i}_Input_kW"]      = input_logs[i - 1]
        data[f"Pump{i}_BlockageFactor"] = blockage_logs[i - 1]

    df = pd.DataFrame(data)
    fname = output_dir / "pump_simulation_results_system_fault.xlsx"
    df.to_excel(fname, index=False)
    print(f"Results saved → {fname.resolve()}")

# %%
# ----------------------------------------------------------------------------------
# Plot Hourly Accumulated Energy Consumption for Each Pump
# ----------------------------------------------------------------------------------
if hasattr(station, 'hourly_energy') and station.hourly_energy:
    hour_indices = sorted(station.hourly_energy.keys())
    # Extract 24-hour window (hours 25-48 inclusive)
    plot_hours = hour_indices[25:49]
    
    if not plot_hours:  # Handle case where slice is empty
        print("Insufficient data for 24-hour period.")
    else:
        # Extract energy data for selected hours
        pump_energies = [
            [station.hourly_energy[h][pump_idx] for h in plot_hours]
            for pump_idx in range(3)
        ]
        cumulative_energies = [np.cumsum(energies) for energies in pump_energies]

        plt.figure(figsize=(8, 4))
        colors = ['b', 'r', 'g']
        markers = ['o', 's', '^']
        labels = ['Pump 1', 'Pump 2', 'Pump 3']

        for i in range(3):
                    plt.plot(plot_hours, cumulative_energies[i],
                     color=colors[i], marker=markers[i],
                     linewidth=2, markersize=6, label=labels[i])

        sns.set(style="whitegrid", font_scale=1.2)
        plt.rcParams.update({
            "font.size": 16,          # Base font size
            "axes.titlesize": 18,     # Axes title
            "axes.labelsize": 16,     # Axes labels
            "xtick.labelsize": 16,    # X tick labels
            "ytick.labelsize": 16,    # Y tick labels
            "legend.fontsize": 16,    # Legend font size
        })
        plt.rcParams["text.usetex"] = True
        plt.rcParams['pdf.fonttype'] = 42  # Ensure type 1 fonts in PDF
        plt.rcParams['ps.fonttype'] = 42   # Ensure type 1 fonts in EPS

        plt.xlabel(r"Time (hours)")
        plt.ylabel(r"Accumulated Energy (kWh)")
        plt.legend()
        plt.grid(True)
        
        # Set ticks every 4 hours to avoid crowding
        plt.xticks(plot_hours[::4])
        plt.xlim(plot_hours[0], plot_hours[-1])  # Auto-limit to data range
        
        plt.tight_layout()
        plt.savefig(figure_dir / "accumulated_energy.eps", format='eps', dpi=300)
        plt.close()
else:
    print("No hourly energy data recorded.")

# %% ###################################################
# Runtime and Number of starts 
#######################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------- Global style (journal aesthetic) ----------
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})
plt.rcParams["text.usetex"] = True
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---------- Data ----------
days_rt = sorted(station.daily_runtime.keys())
p1 = [station.daily_runtime[d][0] for d in days_rt]
p2 = [station.daily_runtime[d][1] for d in days_rt]
p3 = [station.daily_runtime[d][2] for d in days_rt]

days_strt = sorted(station.daily_starts.keys())
v1 = [station.daily_starts[d][0] for d in days_strt]
v2 = [station.daily_starts[d][1] for d in days_strt]
v3 = [station.daily_starts[d][2] for d in days_strt]

# ---------- Colors (consistent across paper) ----------
colors = ["#1f77b4", "#d62728", "#2ca02c"]  # Pump1, Pump2, Pump3

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
label_params = {"fontsize": 11, "fontweight": "bold", "ha": "left", "va": "bottom"}

# --- (a) Runtime ---
x = np.arange(len(days_rt))
w = 0.25
axes[0].bar(x - w, p1, width=w, color=colors[0], edgecolor="black", lw=0.5, label="Pump 1")
axes[0].bar(x,     p2, width=w, color=colors[1], edgecolor="black", lw=0.5, label="Pump 2")
axes[0].bar(x + w, p3, width=w, color=colors[2], edgecolor="black", lw=0.5, label="Pump 3")
axes[0].set_ylabel("Runtime (hours)")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Day {d+1}" for d in days_rt], rotation=45, ha="right")
axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
for s in ["top", "right", "bottom", "left"]:
    axes[0].spines[s].set_visible(True)
axes[0].text(-0.08, 1.02, "(a)", transform=axes[0].transAxes, **label_params)

# --- (b) Starts ---
x2 = np.arange(len(days_strt))
w2 = 0.25
axes[1].bar(x2 - w2, v1, width=w2, color=colors[0], edgecolor="black", lw=0.5)
axes[1].bar(x2,      v2, width=w2, color=colors[1], edgecolor="black", lw=0.5)
axes[1].bar(x2 + w2, v3, width=w2, color=colors[2], edgecolor="black", lw=0.5)
axes[1].set_ylabel("Starts (count)")
axes[1].set_xticks(x2)
axes[1].set_xticklabels([f"Day {d+1}" for d in days_strt], rotation=45, ha="right")
axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
for s in ["top", "right", "bottom", "left"]:
    axes[1].spines[s].set_visible(True)
axes[1].text(-0.08, 1.02, "(b)", transform=axes[1].transAxes, **label_params)

# ---------- Shared legend (below both plots) ----------
fig.legend(["Pump 1", "Pump 2", "Pump 3"],
           loc="upper center", bbox_to_anchor=(0.5, -0.10),
           ncol=3, frameon=True)

# ---------- Save ----------
fig.savefig(figure_dir / "station_daily_metrics.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(figure_dir / "station_daily_metrics.eps", bbox_inches="tight")

plt.show()
# %%#########################################################
# Combined plots: level, head, flow, energy
############################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------- Global journal aesthetic ----------
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
plt.rcParams["ps.fonttype"]  = 42

# ---------- Colors ----------
C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"   # Pump1, Pump2, Pump3

# ---------- Figure ----------
fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.8), constrained_layout=True)
label_params = {"fontsize": 11, "fontweight": "bold", "ha": "left", "va": "bottom"}

# Helper: clean axis
def clean(ax):
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(True)
    ax.minorticks_on()

# -------------------- (a) Water level --------------------
ax = axes[0, 0]
ax.plot(t_5k, lvl_5k, color=C1, lw=1.6)

# Optional: shaded operating 
# for start/stop (clearer than two lines)
y0, y1 = min(level_stop, level_start), max(level_stop, level_start)
#ax.axhspan(y0, y1, color=C1, alpha=0.08, zorder=0, label="_nolegend_")

# Still show reference lines (thin)
ax.axhline(level_start, color="gray", lw=0.9, ls="--")
ax.axhline(level_stop,  color="gray", lw=0.9, ls="--")

ax.set_ylabel(r"Level (m)")
ax.set_xlabel(r"Time (s)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
clean(ax)
ax.text(-0.08, 1.02, "(a)", transform=ax.transAxes, **label_params)

# -------------------- (b) Pump heads --------------------
ax = axes[0, 1]
ax.plot(t_5k, head_5k[0], color=C1, label="Pump 1")
ax.plot(t_5k, head_5k[1], color=C2, label="Pump 2")
ax.plot(t_5k, head_5k[2], color=C3, label="Pump 3")
ax.set_ylabel(r"Head (m)")
ax.set_xlabel(r"Time (s)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
clean(ax)
ax.text(-0.08, 1.02, "(b)", transform=ax.transAxes, **label_params)

# -------------------- (c) Pump flows --------------------
ax = axes[1, 0]
ax.plot(t_5k, flow_5k[0], color=C1, label="Pump 1")
ax.plot(t_5k, flow_5k[1], color=C2, label="Pump 2")
ax.plot(t_5k, flow_5k[2], color=C3, label="Pump 3")
ax.set_ylabel(r"Flow (m$^3$/h)")
ax.set_xlabel(r"Time (s)")
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
clean(ax)
ax.text(-0.08, 1.02, "(c)", transform=ax.transAxes, **label_params)

# -------------------- (d) Accumulated energy --------------------
ax = axes[1, 1]
hour_indices = sorted(station.hourly_energy.keys())
hours = hour_indices[:25]
pump_energies = [[station.hourly_energy[h][i] for h in hours] for i in range(3)]
cumE = [np.cumsum(e) for e in pump_energies]

markers = ["o", "s", "^"]
for i, (col, mk) in enumerate(zip([C1, C2, C3], markers)):
    ax.plot(hours, cumE[i], color=col, marker=mk, markersize=4, lw=1.6)

ax.set_xlabel(r"Time (hours)")
ax.set_ylabel(r"Accumulated Energy (kWh)")
ax.set_xlim(min(hours), max(hours))
ax.xaxis.set_major_locator(MultipleLocator(2))          # every 2 hours
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
clean(ax)
ax.text(-0.08, 1.02, "(d)", transform=ax.transAxes, **label_params)

# ---------- Shared legend (bottom) ----------
handles = [
    Line2D([0], [0], color=C1, lw=1.8, label="Pump 1"),
    Line2D([0], [0], color=C2, lw=1.8, label="Pump 2"),
    Line2D([0], [0], color=C3, lw=1.8, label="Pump 3"),
]
fig.legend(handles, [h.get_label() for h in handles],
           loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=True)

# ---------- Save ----------
fig.savefig(figure_dir / "combined_plots.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(figure_dir / "combined_plots.eps", bbox_inches="tight")

plt.show()
# %%-------------------------------------------------------------------------
# C) Pump‐curve + 5 system curves during system fault (Pump 2 at 50 Hz only)
# -------------------------------------------------------------------------
# 1) Build mask for fault window when Pump 2 is full-speed:
mask_fault = [
    i for i, t in enumerate(t_log)
    if system_fault_start <= t <= system_fault_end
    and freq_logs[1][i] == pump_max_freq
]

flow_fault = [flow_logs[1][i] for i in mask_fault]
head_fault = [head_logs[1][i] for i in mask_fault]
time_fault = [t_log[i] - system_fault_start for i in mask_fault]

# 2) Prepare pump curve
Q = np.linspace(0, 400, 250)
H_pump_nom = [get_pump_head_at_flow(q/usgpm_to_m3h, 1.0) for q in Q]

# 3) Plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------- Global journal aesthetic ----------
mpl.rcParams.update({
        # Aesthetic
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        # Vector PDF performance
        "path.simplify": True,              # simplify long paths
        "path.simplify_threshold": 0.4,     # more aggressive simplification
        "agg.path.chunksize": 10000,        # break long paths into chunks
    })
plt.rcParams["text.usetex"] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Figure (single-column size) ----------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
# Pump curve (reference)
ax.plot(Q, H_pump_nom, "k", lw=2, label=r"Pump curve ($\beta=1.0$)")

# System curves with progressive loading
progs  = np.linspace(0.0, 1.0, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(progs)))
for i, (prog, col) in enumerate(zip(progs, colors)):
    H_sys = (
        static_head_base
        + system_static_head_increase * prog
        + friction_coeff_base
          * (1.0 + system_friction_increase * prog)
          * Q**2
    )
    lbl = "System curve" if i == 0 else "_nolegend_"
    ax.plot(Q, H_sys, "--", lw=1.4, color=col, label=lbl)

# Operating points during fault
pts = ax.scatter(
    flow_fault, head_fault,
    c=time_fault, cmap="plasma",
    s=18, edgecolor="k", linewidth=0.4,
    label="Pump 2 operating points", zorder=10
)

# Colorbar (scaled smaller for single-column width)
cbar = fig.colorbar(pts, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Seconds since fault start", fontsize=9)

# Axes labels + ticks
ax.set_xlabel(r"$Q$ (m$^3$/h)")
ax.set_ylabel(r"$H$ (m)")
ax.set_xlim(0, 400)
ax.set_ylim(0, 1.1 * max(H_pump_nom))
ax.set_xticks(np.arange(0, 401, 100))
ax.set_yticks(np.arange(0, int(1.1 * max(H_pump_nom)) + 10, 20))

# Clean spines
for side in ["top", "right", "bottom","left"]:
    ax.spines[side].set_visible(True)
ax.minorticks_on()

# Legend
ax.legend(loc="lower right", frameon=True, ncol=1)

fig.tight_layout()

# Save
# Turn off LaTeX rendering for this one to avoid font issues in the bitmap
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(
        figure_dir / "pump2_trajectory_with_system_curves.png",
        dpi=600,                    # high resolution for print
        bbox_inches="tight"
    )

plt.show()

# %%
# Set global aesthetics

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Global journal aesthetic ----------
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "lines.linewidth": 1.0,
})
plt.rcParams["text.usetex"] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Wider two-column figure ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)

# --- (a) Inflow ---
ax1.plot(t_5k, inflw_5k, color="#1f77b4")
ax1.set_ylabel(r"Inflow rate (m$^3$/h)")
ax1.set_xlabel(r"Time (s)")
for side in ["top", "right", "bottom", "left"]:
    ax1.spines[side].set_visible(True)
ax1.minorticks_on()
ax1.text(-0.12, 1.02, "(a)", transform=ax1.transAxes,
         fontsize=11, fontweight="bold", ha="left", va="bottom")

# --- (b) Outflow ---
ax2.plot(t_5k, outfw_5k, color="#d62728")
ax2.set_ylabel(r"Outflow (m$^3$/h)")
ax2.set_xlabel(r"Time (s)")
for side in ["top", "right", "bottom", "left"]:
    ax2.spines[side].set_visible(True)
ax2.minorticks_on()
ax2.text(-0.12, 1.02, "(b)", transform=ax2.transAxes,
         fontsize=11, fontweight="bold", ha="left", va="bottom")

# ---------- Save ----------
fig.savefig(figure_dir / "inflow_outflow_wide.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig(figure_dir / "inflow_outflow_wide.eps", bbox_inches="tight")

plt.show()
print(f"All outputs saved under: {output_dir.resolve()}")

# %%
