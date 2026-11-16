#!/usr/bin/env python3
"""
Analyze Self vs Other detection rates; McNemar test; bar chart; qualitative examples.

Supports single-file mode and combined multi-provider CSVs.

Sprint: Apart Research AI x Epistemics, 13–15 Nov 2026.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

DIR = Path(__file__).resolve().parent


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    n = b + c
    if n == 0:
        return float("nan"), 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    exact = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided")
    return chi2, float(exact.pvalue)


def analyze_one(
    df: pd.DataFrame,
    *,
    stimuli: dict,
    meta: dict,
    png: Path,
    summary: Path,
    title_suffix: str = "",
) -> dict:
    self_df = df[df["condicion"] == "SELF"].set_index("problema_id")
    other_df = df[df["condicion"] == "OTHER"].set_index("problema_id")
    common = sorted(set(self_df.index) & set(other_df.index))

    self_det = self_df.loc[common, "detecto_error"].astype(int)
    other_det = other_df.loc[common, "detecto_error"].astype(int)

    dr_self = float(self_det.mean()) if common else float("nan")
    dr_other = float(other_det.mean()) if common else float("nan")
    gap = dr_other - dr_self
    frr_self = 1.0 - dr_self
    frr_other = 1.0 - dr_other

    both = int(((self_det == 1) & (other_det == 1)).sum())
    neither = int(((self_det == 0) & (other_det == 0)).sum())
    only_other = int(((self_det == 0) & (other_det == 1)).sum())
    only_self = int(((self_det == 1) & (other_det == 0)).sum())
    chi2, p_exact = mcnemar_exact(only_other, only_self)

    blind_ids = [i for i in common if int(self_det.loc[i]) == 0 and int(other_det.loc[i]) == 1]
    examples = []
    for pid in blind_ids[:3]:
        s = stimuli.get(pid, {})
        examples.append(
            {
                "id": pid,
                "pregunta": s.get("pregunta", ""),
                "error_real": s.get("error_real", ""),
                "self_expl": str(self_df.loc[pid, "explicacion"])[:200],
                "other_expl": str(other_df.loc[pid, "explicacion"])[:200],
            }
        )

    model_label = meta.get("model") or (df["modelo"].iloc[0] if len(df) else "?")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    rates = [dr_self, dr_other]
    bars = ax.bar(["Self-audit", "Other-audit"], rates, color=["#3d5a5b", "#c45c26"], width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate (known-error items)")
    ax.set_title("Self–Other Gap in Epistemic Auditing")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.0%}", ha="center", fontsize=11)
    ax.axhline(0.5, color="#999", ls="--", lw=0.8)

    ax2 = axes[1]
    vals = [both, only_other, only_self, neither]
    ax2.bar(["Both detect", "Only Other", "Only Self", "Neither"], vals, color=["#2f4f4f", "#c45c26", "#6b8e8f", "#b0b0b0"])
    ax2.set_ylabel("Item count (paired)")
    ax2.set_title("McNemar discordant pattern")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.15, str(v), ha="center")
    fig.suptitle(f"Model: {model_label} | live={meta.get('live', '?')}{title_suffix}", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    plt.close(fig)

    live = meta.get("live", False)
    banner = "LIVE API RESULTS" if live else "⚠ SIMULATED / MOCK RESULTS"
    lines = [
        banner,
        f"provider={meta.get('provider')} model={model_label}",
        meta.get("mock_note") or meta.get("note") or "",
        "",
        f"N paired items: {len(common)}",
        f"Detection Rate SELF:  {dr_self:.3f}  (failed-to-flag / FRR={frr_self:.3f})",
        f"Detection Rate OTHER: {dr_other:.3f}  (failed-to-flag / FRR={frr_other:.3f})",
        f"Self–Other Gap (Other − Self): {gap:+.3f}",
        "",
        "McNemar contingency:",
        f"  both detect={both}, only_other={only_other}, only_self={only_self}, neither={neither}",
        f"  Edwards χ² ≈ {chi2:.3f}",
        f"  Exact binomial McNemar two-sided p = {p_exact:.4g}",
        "",
        "Qualitative blind-spot examples (Other detects, Self misses):",
    ]
    if not examples:
        lines.append("  (none in this run)")
    for ex in examples:
        lines.append(f"  [{ex['id']}] Q: {ex['pregunta']}")
        lines.append(f"       true error: {ex['error_real']}")
        lines.append(f"       SELF:  {ex['self_expl']}")
        lines.append(f"       OTHER: {ex['other_expl']}")
        lines.append("")

    summary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote {png} and {summary}")
    return {
        "model": model_label,
        "provider": meta.get("provider"),
        "dr_self": dr_self,
        "dr_other": dr_other,
        "gap": gap,
        "both": both,
        "only_other": only_other,
        "only_self": only_self,
        "neither": neither,
        "p_exact": p_exact,
        "n": len(common),
        "blind_ids": blind_ids,
    }


def plot_combined(stats: list[dict], png: Path) -> None:
    labels = [f"{s.get('provider')}\n{s.get('model')}" for s in stats]
    x = range(len(stats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(stats)), 4.5))
    self_vals = [s["dr_self"] for s in stats]
    other_vals = [s["dr_other"] for s in stats]
    ax.bar([i - width / 2 for i in x], self_vals, width, label="Self", color="#3d5a5b")
    ax.bar([i + width / 2 for i in x], other_vals, width, label="Other", color="#c45c26")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate")
    ax.set_title("Self–Other detection by model (soft_v3, harder stimuli)")
    ax.legend()
    for i, s in enumerate(stats):
        ax.text(i, max(s["dr_self"], s["dr_other"]) + 0.03, f"gap {s['gap']:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    plt.close(fig)
    print(f"Wrote combined figure {png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DIR / "results.csv")
    ap.add_argument("--meta", type=Path, default=DIR / "run_meta.json")
    ap.add_argument("--png", type=Path, default=DIR / "analysis.png")
    ap.add_argument("--summary", type=Path, default=DIR / "stats_summary.txt")
    ap.add_argument(
        "--combined",
        nargs="*",
        help="Optional list of results CSVs for combined multi-model figure/summary",
    )
    args = ap.parse_args()

    stimuli = {s["id"]: s for s in json.loads((DIR / "stimuli.json").read_text(encoding="utf-8"))}

    if args.combined:
        all_stats = []
        lines = ["COMBINED LIVE RESULTS (soft_v3 prompts, v2 harder stimuli)", ""]
        for csv_path in args.combined:
            csv_path = Path(csv_path)
            meta_path = csv_path.with_name(csv_path.stem.replace("results", "run_meta") + ".json")
            if not meta_path.exists():
                # results_groq.csv → run_meta_groq.json
                stem = csv_path.stem
                if stem.startswith("results_"):
                    meta_path = csv_path.with_name("run_meta_" + stem[len("results_") :] + ".json")
                else:
                    meta_path = DIR / "run_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            df = pd.read_csv(csv_path)
            png = csv_path.with_name("analysis_" + (meta.get("provider") or csv_path.stem) + ".png")
            summary = csv_path.with_name("stats_" + (meta.get("provider") or csv_path.stem) + ".txt")
            st = analyze_one(df, stimuli=stimuli, meta=meta, png=png, summary=summary)
            all_stats.append(st)
            lines.append(
                f"- {st['provider']} / {st['model']}: Self={st['dr_self']:.3f} Other={st['dr_other']:.3f} "
                f"gap={st['gap']:+.3f} McNemar p={st['p_exact']:.4g} "
                f"(both={st['both']}, only_other={st['only_other']}, only_self={st['only_self']}, neither={st['neither']})"
            )
            if st["blind_ids"]:
                lines.append(f"  blind spots: {', '.join(st['blind_ids'])}")
            else:
                lines.append("  blind spots: none")
        plot_combined(all_stats, DIR / "analysis.png")
        (DIR / "stats_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta.exists() else {}
    df = pd.read_csv(args.results)
    analyze_one(df, stimuli=stimuli, meta=meta, png=args.png, summary=args.summary)


if __name__ == "__main__":
    main()
