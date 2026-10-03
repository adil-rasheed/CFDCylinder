"""Extract the Strouhal number from a cylinder_lbm.py run and make plots.

Usage: python analyze.py results/run_re100.npz [--skip 10000]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ST_REF = 0.164          # Williamson (1996), Re = 100
INK, MUTED, SERIES = "#1f2328", "#6e7781", "#2a5fa8"


def zero_crossing_frequency(sig):
    """Mean frequency from upward zero crossings (linearly interpolated)."""
    s = sig - sig.mean()
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    t = idx + s[idx] / (s[idx] - s[idx + 1])
    if len(t) < 3:
        raise ValueError("Fewer than 3 shedding cycles in the analysis window")
    periods = np.diff(t)
    return 1.0 / periods.mean(), periods


def fft_frequency(sig, pad=16):
    """Peak frequency of the Hann-windowed, zero-padded spectrum."""
    s = (sig - sig.mean()) * np.hanning(len(sig))
    n = pad * len(s)
    amp = np.abs(np.fft.rfft(s, n))
    freqs = np.fft.rfftfreq(n)
    k = np.argmax(amp[1:]) + 1
    a, b, c = np.log(amp[k - 1:k + 2])
    shift = 0.5 * (a - c) / (a - 2 * b + c)   # parabolic peak interpolation
    return (k + shift) / n, freqs, amp


def vorticity(u):
    return np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color="#d0d7de", linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="results/run_re100.npz")
    p.add_argument("--skip", type=int, default=10000, help="transient steps to discard")
    a = p.parse_args()

    d = np.load(a.run)
    D, U, re = float(d["D"]), float(d["U"]), float(d["re"])
    v = d["probe_v"].astype(float)
    sig = v[a.skip:]

    f_zc, periods = zero_crossing_frequency(sig)
    f_fft, freqs, amp = fft_frequency(sig)
    st_zc, st_fft = f_zc * D / U, f_fft * D / U
    err = 100 * (st_zc - ST_REF) / ST_REF

    summary = dict(Re=re, St_zero_crossing=round(st_zc, 4), St_fft=round(st_fft, 4),
                   St_reference=ST_REF, error_percent=round(err, 2),
                   cycles_analyzed=len(periods),
                   period_steps=round(float(periods.mean()), 1),
                   period_std_steps=round(float(periods.std()), 1),
                   passes=bool(0.15 <= st_zc <= 0.18))
    print(json.dumps(summary, indent=2))

    out = os.path.dirname(a.run) or "."
    tag = f"re{re:g}"
    with open(os.path.join(out, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Vorticity field
    w = vorticity(d["u"].astype(float))
    w = np.ma.masked_where(d["solid"], w)
    lim = np.percentile(np.abs(w.compressed()), 99)
    nx, ny = w.shape
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.set_facecolor("#8c959f")      # cylinder shows through the mask
    im = ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=(0, nx / D, 0, ny / D))
    px, py = d["probe"]
    ax.plot(px / D, py / D, "o", ms=6, mfc="none", mec=INK, mew=1.5)
    ax.annotate("probe", (px / D, py / D), xytext=(6, 6), textcoords="offset points",
                color=INK, fontsize=9)
    ax.set_xlabel("x / D", color=INK)
    ax.set_ylabel("y / D", color=INK)
    ax.set_title(f"Vorticity, Re = {re:g}  (Kármán vortex street)", color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("ω (lattice units)", color=INK)
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, f"vorticity_{tag}.png"), dpi=150)
    plt.close(fig)

    # Probe signal and spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True,
                                   gridspec_kw=dict(width_ratios=[1.6, 1]))
    t = np.arange(len(v)) * U / D
    ax1.axvspan(0, a.skip * U / D, color="#eaeef2", lw=0)
    ax1.text(a.skip * U / D / 2, 0.92, "transient (discarded)", transform=ax1.get_xaxis_transform(),
             ha="center", color=MUTED, fontsize=9)
    ax1.plot(t, v / U, color=SERIES, lw=1.2)
    ax1.set_xlabel("t U / D", color=INK)
    ax1.set_ylabel("v / U at probe", color=INK)
    ax1.set_title("Cross-stream velocity in the wake", color=INK, loc="left")
    ax1.set_xlim(0, t[-1])
    style(ax1)

    st_axis = freqs * D / U
    keep = st_axis <= 0.5
    ax2.plot(st_axis[keep], amp[keep] / amp.max(), color=SERIES, lw=1.5)
    ax2.axvline(ST_REF, color=MUTED, ls="--", lw=1.2)
    ax2.text(ST_REF - 0.008, 0.93, f"experiment\n{ST_REF}", color=MUTED, fontsize=9,
             ha="right", va="top")
    ax2.annotate(f"LBM St = {st_zc:.3f}", (st_fft, 1.0), xytext=(14, -14),
                 textcoords="offset points", color=INK, fontsize=10)
    ax2.set_xlabel("Strouhal number  f D / U", color=INK)
    ax2.set_ylabel("normalized amplitude", color=INK)
    ax2.set_title("Spectrum", color=INK, loc="left")
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 1.1)
    style(ax2)
    fig.savefig(os.path.join(out, f"strouhal_{tag}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
