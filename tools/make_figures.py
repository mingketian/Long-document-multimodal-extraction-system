"""Render the figures in results/figures/ from results/tables/.

Figures are labelled by provenance. Anything drawn from ``deployment_*.csv`` carries
a visible note that the numbers are reported outcomes from the Ricoh deployment on
proprietary data, not something this repository measured; anything from
``measured_*.csv`` says which command produced it.

Run with:  python tools/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Categorical slots 1-3 of the validated default palette (light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#c9c8c2"
SURFACE = "#fcfcfb"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#84837c"
GRID = "#e8e7e2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "svg.fonttype": "none",
})

CORNER_PX = 4.0
PROVENANCE_REPORTED = (
    "Reported outcome from the Ricoh USA deployment (Sep–Dec 2025) on ~600 proprietary "
    "enterprise documents.\nNot measured by, and not reproducible from, this repository."
)


def read_table(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _radius(ax, axis: str) -> float:
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    if axis == "x":
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        return CORNER_PX / max(bbox.width, 1.0) * span
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    return CORNER_PX / max(bbox.height, 1.0) * span


def rounded_bar(ax, x: float, height: float, width: float, color: str) -> None:
    """A bar with rounded data-end corners, square on the baseline."""
    rx = min(_radius(ax, "x"), width / 2)
    ry = min(_radius(ax, "y"), abs(height) / 2) if height else 0.0
    x0, x1 = x - width / 2, x + width / 2
    verts = [
        (x0, 0), (x0, height - ry), (x0, height), (x0 + rx, height),
        (x1 - rx, height), (x1, height), (x1, height - ry), (x1, 0), (x0, 0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def headline(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=34)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=INK_MUTED, va="bottom", ha="left")


def style(ax, *, ymax: float = 1.0) -> None:
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9)


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"wrote {FIGURES / stem}.svg / .png")


# ── 1. reported: LoRA fine-tuning ─────────────────────────────────────────
def figure_fine_tuning() -> None:
    rows = read_table("deployment_fine_tuning.csv")
    labels = [row["label"] for row in rows]
    before = [float(row["before_lora"]) for row in rows]
    after = [float(row["after_lora"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style(ax, ymax=1.10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    width = 0.30

    for index, (b, a) in enumerate(zip(before, after, strict=True)):
        rounded_bar(ax, index - width / 2 - 0.02, b, width, NEUTRAL)
        rounded_bar(ax, index + width / 2 + 0.02, a, width, BLUE)
        ax.text(index - width / 2 - 0.02, b + 0.018, f"{b:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK_SOFT)
        ax.text(index + width / 2 + 0.02, a + 0.018, f"{a:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK, fontweight="bold")
        ax.annotate(
            f"+{(a - b) * 100:.1f} pts",
            xy=(index, max(a, b) + 0.055),
            ha="center", fontsize=9, color=BLUE, fontweight="bold",
        )

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.add_patch(Rectangle((0, 0), 0, 0, color=NEUTRAL, label="Base Qwen2.5-VL-7B"))
    ax.add_patch(Rectangle((0, 0), 0, 0, color=BLUE, label="+ LoRA on 2.4K page-group sets"))
    headline(ax, "LoRA fine-tuning on page-group sets",
             "Task-specific structured extraction targets · r=16, vision tower frozen")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=2, handlelength=1.0, handleheight=1.0, borderpad=0)
    fig.text(0.02, -0.20, PROVENANCE_REPORTED, fontsize=8, color=INK_MUTED, va="top")
    save(fig, "reported_fine_tuning")


# ── 2. reported: orchestration harness ────────────────────────────────────
def figure_orchestration() -> None:
    rows = read_table("deployment_orchestration.csv")
    labels = [row["label"] for row in rows]
    before = [float(row["before_harness"]) for row in rows]
    after = [float(row["after_harness"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style(ax, ymax=1.10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    width = 0.30

    for index, (b, a) in enumerate(zip(before, after, strict=True)):
        rounded_bar(ax, index - width / 2 - 0.02, b, width, NEUTRAL)
        rounded_bar(ax, index + width / 2 + 0.02, a, width, AQUA)
        ax.text(index - width / 2 - 0.02, b + 0.018, f"{b:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK_SOFT)
        ax.text(index + width / 2 + 0.02, a + 0.018, f"{a:.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK, fontweight="bold")
        ax.annotate(f"+{(a - b) * 100:.1f} pts", xy=(index, max(a, b) + 0.055),
                    ha="center", fontsize=9, color=AQUA, fontweight="bold")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.add_patch(Rectangle((0, 0), 0, 0, color=NEUTRAL, label="Single-pass prompting"))
    ax.add_patch(
        Rectangle((0, 0), 0, 0, color=AQUA,
                  label="+ relevant-page retrieval and evidence attribution")
    )
    headline(ax, "What the orchestration harness bought",
             "Cross-page fields and citation quality · the two things page-at-a-time prompting loses")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=1, handlelength=1.0, handleheight=1.0, borderpad=0)
    fig.text(0.02, -0.22, PROVENANCE_REPORTED, fontsize=8, color=INK_MUTED, va="top")
    save(fig, "reported_orchestration")


# ── 3. measured: accuracy vs pages read ───────────────────────────────────
def figure_measured_tradeoff() -> None:
    rows = [r for r in read_table("measured_profiles.csv") if r["schema"] == "service_agreement"]
    order = ["accuracy", "balanced", "fast"]
    rows.sort(key=lambda r: order.index(r["profile"]))

    profiles = [r["profile"] for r in rows]
    f1 = [float(r["weighted_f1"]) for r in rows]
    pages = [float(r["pages_read_fraction"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for ax, values, colour, title, note in (
        (axes[0], f1, BLUE, "Weighted F1", "higher is better"),
        (axes[1], pages, ORANGE, "Pages actually read", "lower is cheaper"),
    ):
        style(ax, ymax=1.10)
        ax.set_xlim(-0.6, len(profiles) - 0.4)
        for index, value in enumerate(values):
            rounded_bar(ax, index, value, 0.42, colour)
            ax.text(index, value + 0.02, f"{value:.3f}" if colour == BLUE else f"{value:.1%}",
                    ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels(profiles, fontsize=10.5, color=INK)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left", pad=20)
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color=INK_MUTED)

    fig.suptitle(
        "The accuracy / coverage trade-off, measured in this repository",
        fontsize=13.5, fontweight="bold", color=INK, x=0.005, ha="left", y=1.13,
    )
    fig.text(
        0.005, 1.045,
        "8 synthetic 18–26 page service agreements · rule-based baseline backend · "
        "`throughline sweep examples/corpus_agreements --schema service_agreement`",
        fontsize=9, color=INK_MUTED, ha="left",
    )
    fig.text(
        0.005, -0.10,
        "Early exit stops once every required key is present, schema-valid and evidenced. "
        "On this corpus the balanced profile reads 32% of the pages\nfor 77% of the accuracy "
        "ceiling; the fast profile trades far more. A stronger backend narrows the accuracy "
        "gap - this is the floor, not the ceiling.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="top",
    )
    save(fig, "measured_tradeoff")


# ── 4. measured: per-key F1 on the invoice corpus ─────────────────────────
def figure_measured_invoice() -> None:
    from throughline.config import PROFILES
    from throughline.evaluation.harness import EvaluationConfig, evaluate, load_corpus
    from throughline.schema import registry

    config = PROFILES["accuracy"]
    config.schema = "invoice"
    config.cache.enabled = False
    run = evaluate(
        config.build_pipeline(),
        load_corpus(ROOT / "examples" / "corpus"),
        registry.get("invoice"),
        EvaluationConfig(run_name="invoice-accuracy", log_to_mlflow=False),
    )

    scores = sorted(run.report.per_key.values(), key=lambda s: (-s.f1, s.key))
    labels = [s.key for s in scores]
    values = [s.f1 for s in scores]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    for index, value in enumerate(values):
        colour = BLUE if value >= 0.99 else (ORANGE if value >= 0.5 else NEUTRAL)
        ax.barh(index, value, height=0.5, color=colour, linewidth=0)
        ax.text(value + 0.012, index, f"{value:.3f}", va="center", ha="left",
                fontsize=9.5, color=INK)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9)

    headline(ax, "Per-key F1 on the synthetic invoice corpus",
             "12 documents · 69 pages · rule-based baseline · accuracy profile (reads every page)")
    fig.text(
        0.02, -0.06,
        "`line_items` at 1.000 is the result that matters: the table spans several page groups "
        "and the overlap re-shows boundary rows,\nso a correct row count means cross-page "
        "accumulation and deduplication both worked. `vendor_name` is where a keyword matcher "
        "fails\nand a vision-language model is needed - it reads a legal clause as a party name.",
        fontsize=8.5, color=INK_MUTED, va="top",
    )
    save(fig, "measured_invoice_per_key")


def main() -> None:
    figure_fine_tuning()
    figure_orchestration()
    figure_measured_tradeoff()
    figure_measured_invoice()


if __name__ == "__main__":
    main()
