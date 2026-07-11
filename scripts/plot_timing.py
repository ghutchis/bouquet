#!/usr/bin/env python
"""Plot the timing benchmark (scripts/timing_bench.py output).

Consumes the aggregated certificate CSV (``timing/timing_cert.csv``: per-step
config/name/seed/num_dihedrals/step/wall_s/t_eval/t_gp_fit/t_acq/t_select) and
produces, for each arm (base, cat_pca):

  fig_time_vs_steps.pdf   wall-clock vs BO step, one line per molecular size
                          (sequential blue ramp -- size is ordinal), checkpoints
                          at 50/100/.../300 marked. Small-multiple per arm.
  fig_time_vs_size.pdf    the transpose: wall-clock vs molecular size (x =
                          num_dihedrals), one line per BO-step budget
                          (50/100/.../300). Small-multiple per arm.
  fig_breakdown_abs.pdf   total wall-time at the final checkpoint split into
                          GFN2 (t_eval) / GP (t_gp_fit) / acquisition (t_acq) /
                          other (t_select - gp - acq, ~= certificate), stacked
                          bar per molecular size. Small-multiple per arm.
  fig_breakdown_frac.pdf  the same, normalized to 100% -- shows how the GFN2
                          share grows with molecular size.
  checkpoints.csv         size x step -> median wall_s (both arms), the table
                          behind the line plot.

Colours: molecular size uses a single-hue blue ordinal ramp (light=small ->
dark=large); the four phase buckets use validated categorical hues, with the
"other" (certificate) bucket a recessive grey since it is overhead, not search.

Seeds are combined by median (per size, per step / per bucket).

  python scripts/plot_timing.py --cert timing/timing_cert.csv --out timing/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHECKPOINTS = [50, 100, 150, 200, 250, 300]
ARMS = ["base", "cat_pca"]
ARM_TITLES = {"base": "base (grad reference)", "cat_pca": "cat_pca (production)"}

# Phase buckets -> (column, label, colour). GFN2/GP/acq/collective are validated
# categorical hues (blue/aqua/yellow/violet); "other" is recessive grey (overhead).
# "collective" is the low-mode/category kick+relax time (0 for the pure-BO base arm).
BUCKETS = [
    ("t_eval", "GFN2 (energy + relax)", "#2a78d6"),
    ("t_gp_fit", "GP fit", "#1baf7a"),
    ("t_acq", "acquisition", "#eda100"),
    ("t_collective", "collective (kick + relax)", "#4a3aa7"),
    ("other", "other (certificate)", "#b3b2ab"),
]

# Sequential blue ordinal ramp (dataviz palette.md, steps 250..700): light=small.
SIZE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"


def load(cert_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cert_path)
    for c in ("step", "num_dihedrals", "wall_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Timing columns: coerce and fill blanks with 0 (collective rows leave the GP/acq
    # fields blank; standard rows leave t_collective at 0). t_collective may be absent
    # in certificates written before the logging fix -- default it to 0.
    for c in ("t_eval", "t_gp_fit", "t_acq", "t_select", "t_collective"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    df = df.dropna(subset=["step", "wall_s", "num_dihedrals"])
    df["bo_steps"] = df["step"].astype(int) + 1  # 0-based step -> count of BO steps
    # Residual overhead on standard steps (mostly the certificate's own GP eval).
    df["other"] = (df["t_select"] - df["t_gp_fit"] - df["t_acq"]).clip(lower=0)
    return df


def size_colors(sizes):
    """Map each ordinal size to a ramp colour (interpolate if >len(ramp))."""
    if len(sizes) <= len(SIZE_RAMP):
        idx = np.linspace(0, len(SIZE_RAMP) - 1, len(sizes)).round().astype(int)
        return {s: SIZE_RAMP[i] for s, i in zip(sizes, idx, strict=True)}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", SIZE_RAMP)
    return {s: matplotlib.colors.to_hex(cmap(i / (len(sizes) - 1)))
            for i, s in enumerate(sizes)}


def _style_ax(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def median_wall_by_step(df_arm):
    """size -> DataFrame(bo_steps, wall_s) median over seeds (aligned on step)."""
    out = {}
    for size, g in df_arm.groupby("num_dihedrals"):
        m = g.groupby("bo_steps")["wall_s"].median().reset_index()
        out[int(size)] = m
    return dict(sorted(out.items()))


def plot_time_vs_steps(df, out_dir, arms):
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 5), sharey=True)
    axes = np.atleast_1d(axes)
    sizes_all = sorted(df["num_dihedrals"].astype(int).unique())
    cmap = size_colors(sizes_all)
    for ax, arm in zip(axes, arms, strict=True):
        _style_ax(ax)
        series = median_wall_by_step(df[df["config"] == arm])
        max_step = 0
        for size, m in series.items():
            m = m[m["bo_steps"] <= max(CHECKPOINTS)]
            if m.empty:
                continue
            ax.plot(m["bo_steps"], m["wall_s"] / 60.0, color=cmap[size],
                    lw=2, zorder=3, solid_capstyle="round")
            # Direct label at the right end (identity without a legend box).
            last = m.iloc[-1]
            max_step = max(max_step, last["bo_steps"])
            ax.annotate(f"d{size}", (last["bo_steps"], last["wall_s"] / 60.0),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=cmap[size], fontweight="bold")
        for cp in CHECKPOINTS:
            ax.axvline(cp, color=GRID, lw=0.8, zorder=1)
        ax.set_title(ARM_TITLES[arm], fontsize=11, color=TEXT, fontweight="bold")
        ax.set_xlabel("BO steps", fontsize=10, color=MUTED)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlim(0, max(CHECKPOINTS) + 18)
    axes[0].set_ylabel("cumulative wall-clock (min)", fontsize=10, color=MUTED)
    fig.suptitle("Search wall-clock vs BO steps, by molecular size (polyalanine)",
                 fontsize=13, color=TEXT, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "colour = molecular size (light d11 → dark d35); "
             "median over 3 seeds", fontsize=9, color=MUTED, ha="center")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    p = out_dir / "fig_time_vs_steps.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def wall_at_checkpoints(df_arm):
    """checkpoint -> {size: median wall_s to reach that many BO steps} (over seeds)."""
    series = median_wall_by_step(df_arm)  # size -> DataFrame(bo_steps, wall_s)
    out = {cp: {} for cp in CHECKPOINTS}
    for size, m in series.items():
        lut = dict(zip(m["bo_steps"], m["wall_s"], strict=True))
        for cp in CHECKPOINTS:
            avail = [s for s in lut if s <= cp]
            if avail:
                out[cp][size] = lut[max(avail)]
    return out


def plot_time_vs_size(df, out_dir, arms):
    """x = molecular size (num_dihedrals), one line per BO-step budget."""
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 5), sharey=True)
    axes = np.atleast_1d(axes)
    # Step budget is ordinal -> sequential blue ramp (light=50 -> dark=300).
    cmap = size_colors(CHECKPOINTS)
    for ax, arm in zip(axes, arms, strict=True):
        _style_ax(ax)
        by_cp = wall_at_checkpoints(df[df["config"] == arm])
        for cp in CHECKPOINTS:
            pts = sorted(by_cp[cp].items())
            if not pts:
                continue
            xs = [s for s, _ in pts]
            ys = [w / 60.0 for _, w in pts]
            ax.plot(xs, ys, color=cmap[cp], lw=2, marker="o", ms=4,
                    zorder=3, solid_capstyle="round")
            ax.annotate(f"{cp}", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=cmap[cp], fontweight="bold")
        ax.set_title(ARM_TITLES[arm], fontsize=11, color=TEXT, fontweight="bold")
        ax.set_xlabel("number of dihedrals", fontsize=10, color=MUTED)
        sizes = sorted(df["num_dihedrals"].astype(int).unique())
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=9)
    axes[0].set_ylabel("cumulative wall-clock (min)", fontsize=10, color=MUTED)
    fig.suptitle("Search wall-clock vs molecular size, by BO-step budget (polyalanine)",
                 fontsize=13, color=TEXT, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "colour = BO-step budget (light 50 → dark 300); "
             "median over 3 seeds", fontsize=9, color=MUTED, ha="center")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    p = out_dir / "fig_time_vs_size.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def bucket_totals_at(df_arm, checkpoint):
    """size -> {bucket: median-over-seeds cumulative seconds at `checkpoint` steps}."""
    rows = {}
    for size, g in df_arm.groupby("num_dihedrals"):
        per_seed = []
        for _seed, gs in g.groupby("seed"):
            gs = gs[gs["bo_steps"] <= checkpoint]
            if gs.empty:
                continue
            per_seed.append({b: gs[b].sum() for b, _l, _c in BUCKETS
                             if b != "other"} | {"other": gs["other"].sum()})
        if not per_seed:
            continue
        med = {k: float(np.median([d[k] for d in per_seed]))
               for k in [b for b, _l, _c in BUCKETS]}
        rows[int(size)] = med
    return dict(sorted(rows.items()))


def plot_breakdown(df, out_dir, checkpoint, normalize, fname, title, arms):
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms, strict=True):
        _style_ax(ax)
        totals = bucket_totals_at(df[df["config"] == arm], checkpoint)
        sizes = list(totals.keys())
        x = np.arange(len(sizes))
        bottoms = np.zeros(len(sizes))
        denom = np.array([sum(totals[s].values()) for s in sizes]) if normalize else None
        for b, label, color in BUCKETS:
            vals = np.array([totals[s][b] for s in sizes], dtype=float)
            plotv = (vals / denom * 100.0) if normalize else (vals / 60.0)
            ax.bar(x, plotv, bottom=bottoms, color=color, width=0.72,
                   label=label, zorder=3, edgecolor="white", linewidth=1.2)
            bottoms += plotv
        ax.set_title(ARM_TITLES[arm], fontsize=11, color=TEXT, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"d{s}" for s in sizes], fontsize=9)
        ax.set_xlabel("molecular size", fontsize=10, color=MUTED)
    axes[0].set_ylabel("share of wall-clock (%)" if normalize
                       else f"wall-clock at {checkpoint} steps (min)",
                       fontsize=10, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=9.5,
               ncol=len(BUCKETS), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13, color=TEXT, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    p = out_dir / fname
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def write_checkpoints(df, out_dir, arms):
    recs = []
    for arm in arms:
        series = median_wall_by_step(df[df["config"] == arm])
        for size, m in series.items():
            lut = dict(zip(m["bo_steps"], m["wall_s"], strict=True))
            for cp in CHECKPOINTS:
                # nearest recorded step <= cp (cumulative wall time to reach it)
                avail = [s for s in lut if s <= cp]
                recs.append({"config": arm, "num_dihedrals": size, "steps": cp,
                             "wall_s": round(lut[max(avail)], 2) if avail else ""})
    p = out_dir / "checkpoints.csv"
    pd.DataFrame(recs).to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cert", type=Path, default=Path("timing/timing_cert.csv"))
    ap.add_argument("--out", type=Path, default=Path("timing/"))
    ap.add_argument("--checkpoint", type=int, default=max(CHECKPOINTS),
                    help="Step count for the breakdown bars (default 300).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.cert)
    have = sorted(df["config"].unique())
    arms = [a for a in ARMS if a in have]  # only plot arms with data (partial runs)
    print(f"Loaded {len(df)} cert rows; configs={have}; plotting arms={arms}; "
          f"sizes={sorted(df['num_dihedrals'].astype(int).unique())}")
    outputs = [
        plot_time_vs_steps(df, args.out, arms),
        plot_time_vs_size(df, args.out, arms),
        plot_breakdown(df, args.out, args.checkpoint, False,
                       "fig_breakdown_abs.pdf",
                       f"Where the time goes at {args.checkpoint} steps (polyalanine)",
                       arms),
        plot_breakdown(df, args.out, args.checkpoint, True,
                       "fig_breakdown_frac.pdf",
                       f"Wall-clock composition at {args.checkpoint} steps (share)",
                       arms),
        write_checkpoints(df, args.out, arms),
    ]
    for p in outputs:
        print("wrote", p)


if __name__ == "__main__":
    main()
