"""Turn raw run outputs into the study's tables and figures.

Reads only files under ``results/`` and writes ``results/summary.csv`` plus the figures in
``report/figures/``. Nothing here re-runs a battle, so every figure regenerates from committed
CSVs alone — which is the reproducibility property the study claims.

Usage:  python scripts/analyse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokerl.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

FIG_DIR = PROJECT_ROOT / "report" / "figures"

# Validated categorical palette (light surface #fcfcfb): adjacent CVD dE 9.1, normal-vision
# 19.6. Three slots sit below 3:1 contrast, so every chart here carries direct labels or an
# accompanying table — never colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

OPPONENT_LABEL = {"random": "Random", "max_power": "MaxBasePower", "heuristic": "SimpleHeuristics"}


def _style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _new_fig(width: float = 9.0, height: float = 5.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


# --------------------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------------------


def load_evals() -> pd.DataFrame:
    rows = [pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("*/eval.csv"))]
    if not rows:
        raise SystemExit(f"No eval.csv found under {RESULTS_DIR}. Run training first.")
    df = pd.concat(rows, ignore_index=True)
    # run_id is "<config>_seed<N>"; the config name is the experiment cell.
    df["cell"] = df["run_id"].str.rsplit("_seed", n=1).str[0]
    return df


def load_learning_curve(run_dir: Path) -> pd.DataFrame | None:
    """Concatenate per-env Monitor logs into one timestep-ordered curve."""
    files = sorted(run_dir.glob("monitor/*.monitor.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, skiprows=1))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("t")
    df["timesteps"] = df["l"].cumsum()
    return df[["timesteps", "r"]]


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std win rate across seeds, one row per (cell, opponent)."""
    summary = (
        df.groupby(["cell", "algo", "encoder", "reward", "train_opponent", "opponent"])
        .agg(win_rate_mean=("win_rate", "mean"),
             win_rate_std=("win_rate", "std"),
             n_seeds=("seed", "nunique"))
        .reset_index()
    )
    summary["win_rate_std"] = summary["win_rate_std"].fillna(0.0)
    summary = summary.round(2)
    out = RESULTS_DIR / "summary.csv"
    summary.to_csv(out, index=False)
    print(f"Summary table -> {out}  ({len(summary)} rows)")

    pivot = summary.pivot_table(index="cell", columns="opponent", values="win_rate_mean")
    pivot = pivot.reindex(columns=[c for c in OPPONENT_LABEL if c in pivot.columns])
    print("\nWin rate (%) by cell:")
    print(pivot.to_string())
    return summary


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def fig_win_rates(summary: pd.DataFrame) -> None:
    """Grouped bars: one group per cell, one bar per evaluation opponent."""
    cells = sorted(summary["cell"].unique())
    opponents = [o for o in OPPONENT_LABEL if o in set(summary["opponent"])]

    fig, ax = _new_fig(width=max(9.0, 1.5 * len(cells)), height=5.5)
    x = np.arange(len(cells))
    # A 2px surface gap between adjacent bars keeps the fills from fusing.
    total = 0.8
    width = total / len(opponents)

    for i, opp in enumerate(opponents):
        sub = summary[summary["opponent"] == opp].set_index("cell")
        means = [sub["win_rate_mean"].get(c, np.nan) for c in cells]
        errs = [sub["win_rate_std"].get(c, 0.0) for c in cells]
        pos = x - total / 2 + width * (i + 0.5)
        bars = ax.bar(pos, means, width * 0.92, label=OPPONENT_LABEL[opp],
                      color=SERIES[i], zorder=2, linewidth=0)
        ax.errorbar(pos, means, yerr=errs, fmt="none", ecolor=INK_MUTED,
                    elinewidth=1.0, capsize=2.5, zorder=3)
        # Contrast relief: every bar is labelled, so identity never rests on colour.
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)

    ax.axhline(50, color=AXIS, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(len(cells) - 0.45, 51.5, "coin flip", fontsize=7.5, color=INK_MUTED, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("ppo_masked_", "") for c in cells],
                       rotation=30, ha="right", color=INK, fontsize=9)
    ax.set_ylabel("Win rate (%)", color=INK, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Win rate against each baseline opponent (mean ± std over seeds)",
                 color=INK, fontsize=12, pad=14, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, ncol=len(opponents), loc="upper left",
                       bbox_to_anchor=(0, -0.28))
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out = FIG_DIR / "win_rates.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out}")


def fig_learning_curves(cells: list[str], seed: int = 0, smooth: int = 50) -> None:
    """One line per cell: rolling-mean episode return against environment steps."""
    fig, ax = _new_fig(width=9.0, height=5.0)
    plotted = 0

    for cell in cells:
        run_dir = RESULTS_DIR / f"{cell}_seed{seed}"
        curve = load_learning_curve(run_dir)
        if curve is None or curve.empty:
            continue
        y = curve["r"].rolling(smooth, min_periods=max(smooth // 5, 1)).mean()
        colour = SERIES[plotted % len(SERIES)]
        ax.plot(curve["timesteps"], y, color=colour, linewidth=2.0, zorder=2,
                label=cell.replace("ppo_masked_", ""))
        # Direct label at the line end; the legend is a backup, not the only channel.
        if len(curve) > 0 and not np.isnan(y.iloc[-1]):
            ax.annotate(cell.replace("ppo_masked_", ""),
                        xy=(curve["timesteps"].iloc[-1], y.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No monitor logs found; skipping learning curves.")
        return

    ax.set_xlabel("Environment steps", color=INK, fontsize=10)
    ax.set_ylabel(f"Episode return (rolling mean, {smooth} episodes)", color=INK, fontsize=10)
    ax.set_title(f"Learning curves (seed {seed})", color=INK, fontsize=12, pad=14, loc="left")
    if plotted >= 2:
        legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for text in legend.get_texts():
            text.set_color(INK)
    ax.margins(x=0.18)

    fig.tight_layout()
    out = FIG_DIR / "learning_curves.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out}")


def fig_cross_matrix() -> None:
    """Heatmap of the pairwise win-rate matrix. Sequential single hue, light -> dark."""
    path = RESULTS_DIR / "cross_evaluation.csv"
    if not path.exists():
        print("No cross_evaluation.csv; skipping matrix figure.")
        return

    df = pd.read_csv(path, index_col=0)
    data = df.to_numpy(dtype=float)

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    fig, ax = _new_fig(width=1.1 * len(df) + 3, height=1.0 * len(df) + 2.5)
    ax.grid(False)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=40, ha="right", fontsize=8, color=INK)
    ax.set_yticklabels(df.index, fontsize=8, color=INK)

    # Every cell carries its number: the value is never encoded by colour alone.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if val > 55 else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Row player win rate (%)", color=INK, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title("Cross-evaluation: row player's win rate vs column player",
                 color=INK, fontsize=12, pad=14, loc="left")
    fig.tight_layout()
    out = FIG_DIR / "cross_evaluation.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_evals()
    summary = write_summary(df)
    fig_win_rates(summary)
    fig_learning_curves(sorted(summary["cell"].unique()))
    fig_cross_matrix()
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
