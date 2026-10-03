"""Strouhal number vs. blockage ratio D/H, extrapolated to an unconfined cylinder.

Usage: python compare_blockage.py results/run_re100.npz results/ny240/run_re100.npz ...
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import INK, MUTED, SERIES, ST_REF, style, zero_crossing_frequency

SKIP = 8000  # all runs reach saturated shedding by step 8000

rows = []
for path in sys.argv[1:]:
    d = np.load(path)
    D, U = float(d["D"]), float(d["U"])
    ny = d["u"].shape[2]
    f, periods = zero_crossing_frequency(d["probe_v"][SKIP:].astype(float))
    rows.append(dict(ny=ny, blockage=D / ny, St=f * D / U, cycles=len(periods)))
rows.sort(key=lambda r: r["blockage"])

b = np.array([r["blockage"] for r in rows])
st = np.array([r["St"] for r in rows])
slope, st0 = np.polyfit(b, st, 1)

print(f"{'H (cells)':>9} {'D/H':>7} {'St':>7} {'cycles':>6}")
for r in rows:
    print(f"{r['ny']:9d} {r['blockage']:7.4f} {r['St']:7.4f} {r['cycles']:6d}")
print(f"Linear extrapolation to D/H -> 0: St = {st0:.4f} "
      f"({100 * (st0 - ST_REF) / ST_REF:+.1f}% vs {ST_REF})")

with open("results/blockage_study.json", "w") as fh:
    json.dump(dict(runs=rows, St_extrapolated=round(st0, 4), St_reference=ST_REF), fh, indent=2)

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
bb = np.linspace(0, b.max() * 1.05, 50)
ax.plot(bb, st0 + slope * bb, color=MUTED, lw=1, ls=":")
ax.plot(b, st, "o", color=SERIES, ms=8, mec="white", mew=2)
for r in rows:
    ax.annotate(f"{r['St']:.3f}", (r["blockage"], r["St"]), xytext=(8, -4),
                textcoords="offset points", color=INK, fontsize=9)
ax.plot(0, st0, "o", mfc="white", mec=SERIES, mew=1.5, ms=8)
ax.annotate(f"extrapolated {st0:.3f}", (0, st0), xytext=(8, 6),
            textcoords="offset points", color=INK, fontsize=9)
ax.axhline(ST_REF, color=MUTED, ls="--", lw=1.2)
ax.text(b.max() * 1.05, ST_REF, f"experiment {ST_REF}", color=MUTED, fontsize=9,
        ha="right", va="bottom")
ax.set_xlim(-0.005, b.max() * 1.1)
ax.set_xlabel("blockage ratio  D / H", color=INK)
ax.set_ylabel("Strouhal number", color=INK)
ax.set_title("Confinement raises the shedding frequency (Re = 100)", color=INK, loc="left")
style(ax)
fig.savefig("results/blockage_re100.png", dpi=150)
