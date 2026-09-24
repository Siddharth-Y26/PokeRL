"""Write the study's results report from the committed CSVs.

Runs unattended at the end of the matrix, so the report exists whether or not anyone is
watching. Reads only ``results/*.csv`` — no battles are re-run — and writes
``report/RESULTS.md``.

Usage:  python scripts/make_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokerl.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

REPORT = PROJECT_ROOT / "report" / "RESULTS.md"
OPP_ORDER = ["random", "max_power", "heuristic"]
OPP_LABEL = {"random": "Random", "max_power": "MaxBasePower", "heuristic": "SimpleHeuristics"}


def _fmt(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "—"
    return f"{mean:.1f} ± {std:.1f}" if n > 1 else f"{mean:.1f}"


def _table(df: pd.DataFrame, cells: list[str], title: str) -> list[str]:
    """One markdown table: rows are cells, columns are evaluation opponents."""
    present = [c for c in cells if c in set(df["cell"])]
    if not present:
        return []
    out = [f"### {title}", ""]
    out.append("| Configuration | " + " | ".join(OPP_LABEL[o] for o in OPP_ORDER) + " | Seeds |")
    out.append("|---" * (len(OPP_ORDER) + 2) + "|")
    for cell in present:
        sub = df[df["cell"] == cell]
        cols = []
        seeds = 0
        for opp in OPP_ORDER:
            row = sub[sub["opponent"] == opp]
            if row.empty:
                cols.append("—")
            else:
                r = row.iloc[0]
                seeds = max(seeds, int(r["n_seeds"]))
                cols.append(_fmt(r["win_rate_mean"], r["win_rate_std"], int(r["n_seeds"])))
        out.append(f"| `{cell}` | " + " | ".join(cols) + f" | {seeds} |")
    out.append("")
    return out


def main() -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} missing — run scripts/analyse.py first.")
    df = pd.read_csv(summary_path)

    lines: list[str] = [
        "# PokeRL — Results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
        "All figures regenerate from `results/*.csv` via `scripts/analyse.py`.",
        "",
        "Format: **win rate % ± std across seeds**, 500 battles per opponent per seed, "
        "deterministic action selection. Battle format `gen9randombattle`.",
        "",
    ]

    ref = "ppo_masked_v0"
    algo_cells = [ref, "ppo_unmasked_v0", "dqn_masked_v0", "a2c_masked_v0", "tabular_q_v0"]
    lines += _table(df, algo_cells, "Axis A — Algorithm (encoder `v0`, shaped reward, vs heuristic)")
    lines += _table(df, [ref, "ppo_masked_v1"], "Axis B — State encoding")
    lines += _table(
        df,
        ["ppo_masked_v0_sparse", ref, "ppo_masked_v0_aggressive"],
        "Axis C — Reward shaping",
    )
    lines += _table(
        df,
        [
            "ppo_masked_v0_vs_random",
            "ppo_masked_v0_vs_maxpower",
            ref,
            "ppo_masked_v0_vs_mixed",
        ],
        "Axis D — Training opponent",
    )

    # Full ranking by the hardest opponent — the discriminating column.
    hard = (
        df[df["opponent"] == "heuristic"]
        .sort_values("win_rate_mean", ascending=False)[
            ["cell", "algo", "encoder", "reward", "train_opponent",
             "win_rate_mean", "win_rate_std", "n_seeds"]
        ]
    )
    lines += ["### Overall ranking (vs SimpleHeuristics, the discriminating opponent)", ""]
    lines.append("| Rank | Configuration | Win rate | Seeds |")
    lines.append("|---|---|---|---|")
    for i, (_, r) in enumerate(hard.iterrows(), start=1):
        lines.append(
            f"| {i} | `{r['cell']}` | "
            f"{_fmt(r['win_rate_mean'], r['win_rate_std'], int(r['n_seeds']))} | "
            f"{int(r['n_seeds'])} |"
        )
    lines.append("")

    cross = RESULTS_DIR / "cross_evaluation.csv"
    if cross.exists():
        cdf = pd.read_csv(cross, index_col=0)
        lines += [
            "### Cross-evaluation",
            "",
            "Row player's win rate (%) against the column player.",
            "",
            "| | " + " | ".join(str(c) for c in cdf.columns) + " |",
            "|---" * (len(cdf.columns) + 1) + "|",
        ]
        for label, row in cdf.iterrows():
            vals = ["—" if pd.isna(v) else f"{v:.0f}" for v in row]
            lines.append(f"| **{label}** | " + " | ".join(vals) + " |")
        lines.append("")

    lines += [
        "### Figures",
        "",
        "- `report/figures/win_rates.png` — win rate by configuration and opponent",
        "- `report/figures/learning_curves.png` — episode return vs environment steps",
        "- `report/figures/cross_evaluation.png` — pairwise win-rate matrix",
        "",
        "### Run inventory",
        "",
    ]
    runs = sorted(p.parent.name for p in RESULTS_DIR.glob("*/eval.csv"))
    lines.append(f"{len(runs)} evaluated runs:")
    lines.append("")
    for r in runs:
        lines.append(f"- `{r}`")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {REPORT}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
