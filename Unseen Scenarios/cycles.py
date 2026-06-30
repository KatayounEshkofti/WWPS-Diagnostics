#%%
"""
Build Pump 1 cycle-time labels from the unseen two-fault simulation.

The script detects Pump 1 ON/OFF cycles from the frequency signal, assigns
each cycle to a normal or fault-related time window, and saves the cycle table
together with a short summary sheet.
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------
input_file = Path("data/unseen_simulation/two_faults/pump_system_faultsimualtion.xlsx")
output_dir = Path("results/unseen_simulation/two_faults")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "Cycle-time.xlsx"


# ---------------------------------------------------------------------
# Load the simulation data
# ---------------------------------------------------------------------
df = pd.read_excel(input_file, parse_dates=["Timestamp"])


# ---------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------
start_time = df["Timestamp"].iloc[0]
df["t_sec"] = (df["Timestamp"] - start_time).dt.total_seconds().astype(int)


# ---------------------------------------------------------------------
# Detect Pump 1 operating cycles
# ---------------------------------------------------------------------
# Frequency is used as the ON/OFF indicator because it is less sensitive
# to short zero-flow transients.
pump_is_on = df["Pump1_Freq_Hz"].fillna(0) > 0.1

# Rising edges mark cycle starts, and falling edges mark cycle stops.
state_change = pump_is_on.astype(int).diff().fillna(pump_is_on.iloc[0].astype(int))
cycle_starts = df.index[state_change == 1].tolist()
cycle_stops = df.index[state_change == -1].tolist()

# If the signal starts while the pump is already running, the first row
# is the beginning of the first cycle.
if pump_is_on.iloc[0]:
    cycle_starts = [0] + cycle_starts

# If the final cycle does not have a falling edge, close it at the last row.
if len(cycle_stops) == 0 or (len(cycle_starts) > len(cycle_stops)):
    cycle_stops = cycle_stops + [len(df) - 1]

# Keep matched start-stop pairs only.
n_cycles = min(len(cycle_starts), len(cycle_stops))
cycle_starts, cycle_stops = cycle_starts[:n_cycles], cycle_stops[:n_cycles]


# ---------------------------------------------------------------------
# Build the cycle table
# ---------------------------------------------------------------------
cycles = pd.DataFrame({
    "start_idx": cycle_starts,
    "stop_idx": cycle_stops,
})

cycles["start_time"] = df.loc[cycles["start_idx"], "Timestamp"].values
cycles["stop_time"] = df.loc[cycles["stop_idx"], "Timestamp"].values
cycles["start_t_sec"] = df.loc[cycles["start_idx"], "t_sec"].values
cycles["stop_t_sec"] = df.loc[cycles["stop_idx"], "t_sec"].values
cycles["duration_s"] = (cycles["stop_t_sec"] - cycles["start_t_sec"]).astype(int)
cycles["duration_min"] = cycles["duration_s"] / 60.0


# ---------------------------------------------------------------------
# Fault windows
# ---------------------------------------------------------------------
# Time intervals are expressed in seconds from the beginning of the
# simulation.
block_start, block_end = 86_400 + 25_000, 86_400 + 40_000
system_start, system_end = 60_000, 80_000


def overlap(a_start, a_end, b_start, b_end):
    """Return True when two closed time intervals overlap."""
    return (a_start <= b_end) and (b_start <= a_end)


labels = []
for _, row in cycles.iterrows():
    cycle_start = int(row["start_t_sec"])
    cycle_end = int(row["stop_t_sec"])

    if overlap(cycle_start, cycle_end, block_start, block_end):
        labels.append("pump_blockage_window")
    elif overlap(cycle_start, cycle_end, system_start, system_end):
        labels.append("system_fault_window")
    else:
        labels.append("normal")

cycles["label"] = labels


# ---------------------------------------------------------------------
# Save the cycle table and summary
# ---------------------------------------------------------------------
summary = pd.DataFrame({
    "total_cycles": [len(cycles)],
    "avg_duration_s": [cycles["duration_s"].mean()],
    "median_duration_s": [cycles["duration_s"].median()],
    "min_duration_s": [cycles["duration_s"].min()],
    "max_duration_s": [cycles["duration_s"].max()],
    "cycles_in_blockage_win": [(cycles["label"] == "pump_blockage_window").sum()],
    "cycles_in_system_fault_win": [(cycles["label"] == "system_fault_window").sum()],
})

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    cycles.to_excel(writer, index=False, sheet_name="Pump1_Cycles")
    summary.to_excel(writer, index=False, sheet_name="Summary")


# ---------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------
print(f"Cycle table saved to: {output_file}")
output_file

# %%
