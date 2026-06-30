# %%
"""
Flow-rate Distribution Analysis for Sump Level Data
===================================================
This script reads one or many Excel files that contain the water **level** (in metres) of a sump, 
computes the instantaneous **flow-rate** (m³ s⁻¹) from the derivative of level, and produces
both *summary statistics* and *histograms/KDE plots* describing the distribution of flow-rate.

------------------------------------------------------------------

Assumptions & conventions
-------------------------
* **Cross-sectional area** of the sump (`AREA_M2`) is **8 m²**.
* Each workbook is named `output_nov_dec_part_<N>.xlsx` where `<N>` runs from 1 … 180.
* Inside each workbook the first sheet holds the data, containing at minimum:
  - a timestamp column, called exactly **"time"** (case insensitive) and parsable by `pandas.to_datetime`;
  - a numeric **"level"** column (metres).
  Other columns (Pump, Voltage, etc.) are ignored.
* Timestamps may repeat inside a second, so rows are first **sorted** and then **deduplicated**
  (keeping the *last* observation per timestamp) before the derivative is taken.
* The calculation uses the *central difference* for internal points and a *forward/backward* difference
  at the boundaries.
* Output units:
  - `flow_rate_m3_s`   – instantaneous m³ s⁻¹ (signed: positive = inflow / level rising).
  - `flow_rate_L_s`    – instantaneous L s⁻¹ (×1000 for human-friendliness).

Outputs
-------
1. **CSV** – `flow_rates.csv` containing the cleaned time series with both units.
2. **PNG** – `flow_rate_histogram.png` showing a histogram overlaid with a Gaussian-kernel density curve.
3. **TXT** – `flow_rate_summary.txt` with full descriptive statistics (count, mean, stdev, quantiles…).

"""

import os
import re
import glob
import logging
from datetime import timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar (pip install tqdm)

# ══════════════════════════════════════════════════════
# 1. User-adjustable configuration
# ══════════════════════════════════════════════════════

AREA_M2: float = 8.0                  # sump cross-sectional area [m²]
DATA_DIR: str = "H:\Dataset\Sump watcher dataset\Högdalen 2024-11-20 to 2024-12-16"                   # directory holding the *.xlsx files ("." = current)
FILE_GLOB: str = "output_nov_dec_part_*.xlsx"  # glob pattern – leave as-is unless file names differ
OUTPUT_DIR: str = "analysis_results"   # where to drop CSV / PNG / TXT
TIMEZONE = "Europe/Stockholm"          # set to None to keep naive timestamps

# Create the output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# 2. Logging setup – prints progress & debug info
# ══════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# 3. Helper functions
# ══════════════════════════════════════════════════════

def numeric_sort_key(path: str) -> int:
    """Extract the integer suffix ( …_part_**N**.xlsx ) so globbed files sort naturally."""
    m = re.search(r"part_(\d+)\.xlsx$", os.path.basename(path))
    return int(m.group(1)) if m else 0

def load_single_file(path: str) -> pd.DataFrame:
    """Read one workbook, extract the time/level series, clean, and return a DataFrame."""

    # Read entire sheet – dtype guessing is fine; use engine that supports xlsx
    df = pd.read_excel(path, sheet_name=0)

    # Standardise column names to lower case for robustness
    df.columns = [c.lower() for c in df.columns]

    # Require at least ‘time’ and ‘level’
    if not {"time", "level"}.issubset(df.columns):
        raise ValueError(f"File {path} does not have the required 'time' and 'level' columns!")

    # Parse timestamps; coerce errors → NaT then drop
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "level"]).copy()

    # Optionally localise
    if TIMEZONE:
        df["time"] = df["time"].dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(TIMEZONE)

    # Sort and deduplicate (keep last sample per timestamp)
    df = df.sort_values("time").drop_duplicates(subset="time", keep="last")

    # Keep only what we need
    return df[["time", "level"]]

def compute_flow_rate(df_level: pd.DataFrame) -> pd.DataFrame:
    """Given a time-indexed DataFrame with a 'level' column, compute flow rate & return copy."""
    df = df_level.copy()
    df = df.set_index("time")  # easier with time index

    # Calculate time delta in seconds (Series of same length; first entry NaN)
    delta_t = df.index.to_series().diff().dt.total_seconds()

    # Level difference (metres)
    delta_l = df["level"].diff()

    # Flow rate: Q = A * dL/dt   (m³ s⁻¹)
    df["flow_rate_m3_s"] = AREA_M2 * delta_l / delta_t

    # Convert to litres per second (optional, easier for small values)
    df["flow_rate_L_s"] = df["flow_rate_m3_s"] * 1000.0

    # Clean – remove first sample (NaN) and any inf / insane spikes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["flow_rate_m3_s"], inplace=True)

    # Remove values where |delta_t| < 0.1 s (likely duplicates / noise)
    df = df[delta_t >= 0.1]

    return df.reset_index()

# ══════════════════════════════════════════════════════
# 4. Main work – iterate through all files
# ══════════════════════════════════════════════════════

def main() -> None:
    #  Gather & natural-sort the workbook list
    pattern = os.path.join(DATA_DIR, FILE_GLOB)
    paths = sorted(glob.glob(pattern), key=numeric_sort_key)

    if not paths:
        logger.error(f"No files matched {pattern!r}. Check DATA_DIR and FILE_GLOB.")
        return

    logger.info(f"Found {len(paths)} workbook(s). Starting load …")

    flow_parts = []

    for pth in tqdm(paths, desc="Processing files", unit="file"):
        try:
            df_level = load_single_file(pth)
            df_flow  = compute_flow_rate(df_level)
            flow_parts.append(df_flow)
        except Exception as exc:
            logger.exception(f"⚠️  Skipping {pth} because of: {exc}")

    if not flow_parts:
        logger.error("No valid data extracted!")
        return

    #  Concatenate all pieces into one big table
    df_all = pd.concat(flow_parts, ignore_index=True)
    logger.info(f"Total valid flow-rate samples: {len(df_all):,}")

    # ═════ Save cleaned time-series ═════
    csv_path = os.path.join(OUTPUT_DIR, "flow_rates.csv")
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Cleaned flow-rate time series written → {csv_path}")

    # ═════ Summary stats ═════
    stats = df_all["flow_rate_m3_s"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    txt_path = os.path.join(OUTPUT_DIR, "flow_rate_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("# Flow-rate descriptive statistics (m³ s⁻¹)\n\n")
        fp.write(stats.to_string())
    logger.info(f"Summary statistics written → {txt_path}")

    # ═════ Distribution plot ═════
    plt.figure(figsize=(10, 6))
    plt.hist(df_all["flow_rate_m3_s"], bins=200, density=True, alpha=0.6, label="Histogram")
    df_all["flow_rate_m3_s"].plot(kind="kde", linewidth=2, label="KDE")
    plt.xlabel("Flow rate [m³ s⁻¹]")
    plt.ylabel("Probability density")
    plt.title("Sump flow-rate distribution (Nov–Dec)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    png_path = os.path.join(OUTPUT_DIR, "flow_rate_histogram.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    logger.info(f"Histogram saved → {png_path}")

    logger.info("All done! ✅")

# ═════════════════════════════════════════════════════=
# 5. ☝️ Run only when executed as a script (not on import)
# ═════════════════════════════════════════════════════=

if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────
# Empirical-CDF sampler for inflow (Option C)
# ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from pathlib import Path

# 1.  Load the cleaned flow-rate file you already created
FLOW_CSV = Path(r"H:\Dataset\Sump watcher dataset\analysis_results\flow_rates.csv")
df_flow   = pd.read_csv(FLOW_CSV)

# 2.  Use *positive* flows only → inflow
positive  = df_flow["flow_rate_m3_s"][df_flow["flow_rate_m3_s"] > 0].to_numpy()
positive.sort()                     # <= key step: sorted array is all we need

# 3.  Build a reusable sampler
def ecdf_sampler(n=1, rng=None):
    """
    Draw `n` random samples (in m³ s⁻¹) from the *empirical* distribution
    represented by `positive`.  Uses inverse-CDF with numpy.quantile.
    """
    rng = rng or np.random.default_rng()   # default RNG if user didn't pass one
    u   = rng.random(n)                    # n uniform(0,1) numbers
    return np.quantile(positive, u)        # inverse-CDF → samples

#%% ▶ EXAMPLE ───────────────────────────────
if __name__ == "__main__":
    # draw 5 random inflow values and print them
    q100 = ecdf_sampler(500)
    print("5 ECDF draws [m³/s]:", q100)

    # optional quick visual check (needs matplotlib)
    import matplotlib.pyplot as plt
    import seaborn as sns 
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["text.usetex"] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 5))
    plt.hist(positive, bins=200, density=True, alpha=0.6, label="Empirical")
    plt.hist(ecdf_sampler(100_000), bins=200, density=True, alpha=0.4, label="ECDF draws")
    plt.xlabel("Flow rate [m³ s⁻¹]"); plt.ylabel("Probability density")
    plt.legend(); plt.grid(True, ls=":", lw=0.5); plt.tight_layout()
    plt.show()

# %%
# === Publication figure: outlined (step) histograms, no fill ===
import matplotlib as mpl
import matplotlib.pyplot as plt

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
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(6.4, 3.8))

# Outlined histograms (Freedman–Diaconis bins), thick lines, no fill
ax.hist(positive, bins="fd", density=True,
        histtype="step", linewidth=2.2, color="#1f77b4", label="Empirical")
ax.hist(ecdf_sampler(1000_000), bins="fd", density=True,
        histtype="step", linewidth=2.2, color="#d62728", label="ECDF draws")
#ECDF Original Value#100_000
ax.set_xlabel(r"Flow rate (m$^{3}$ s$^{-1}$)")
ax.set_ylabel("Density")
ax.set_xlim(0.0, 0.06)

# Clean spines
for side in ["top", "right", "left", "bottom"]:
    ax.spines[side].set_visible(True)
ax.minorticks_on()

# Light y-grid only (x-grid off to avoid clutter)
ax.grid(axis="y", which="major", linestyle="--", alpha=0.25)
ax.grid(axis="x", visible=False)

ax.legend(frameon=True, loc="upper right")

fig.tight_layout()
fig.savefig("ECDF.pdf", bbox_inches="tight")
with mpl.rc_context({"text.usetex": False}):
    fig.savefig("ECDF.eps", bbox_inches="tight")

plt.show()


# %%
