"""
Shared figure style and output helpers.

Every stylized fact produces (a) one or more per-regime figures and (b) a
normal-vs-stress comparison figure, all with the same visual grammar so the
result directory reads as one document.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

GRID_KW = dict(color="#D8DEE6", linewidth=0.7, alpha=0.9)
REFERENCE_KW = dict(color="#6B7280", linewidth=1.2, linestyle="--", alpha=0.85)
LITERATURE_KW = dict(color="#0E7C66", linewidth=1.4, linestyle=":", alpha=0.95)


def apply_style() -> None:
    """House style: light, high-contrast, print-safe."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4B5563",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#111827",
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID_KW["color"],
        "grid.linewidth": GRID_KW["linewidth"],
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def new_figure(nrows: int = 1, ncols: int = 1, width: float = 7.2,
               height: float = 4.4, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), **kwargs)
    return fig, axes


NOTE_FONTSIZE = 7.8


def _wrap_to_axes(ax, text: str, fontsize: float = NOTE_FONTSIZE) -> str:
    """Hard-wrap a caption to the width of its own axes.

    Matplotlib's ``wrap=True`` does not work for annotations placed outside the
    axes, and an unwrapped one-line caption is wide enough that ``tight_layout``
    shrinks every panel in the figure to make room for it.  Wrapping here is
    what keeps multi-panel figures legible.
    """
    import textwrap

    fig = ax.get_figure()
    width_inches = ax.get_position().width * fig.get_figwidth()
    # DejaVu Sans averages ~0.5 em per character.
    chars_per_inch = 72.0 / (0.52 * fontsize)
    width = max(int(width_inches * chars_per_inch), 24)
    return "\n".join(textwrap.wrap(text, width=width))


def finish(ax, title: str = "", xlabel: str = "", ylabel: str = "",
           legend: bool = False, note: str = "") -> None:
    """Apply the common cosmetic pass to one axis."""
    if title:
        ax.set_title(title, loc="left", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend:
        ax.legend(loc="best", fontsize=8.5, framealpha=0.85, facecolor="white",
                  frameon=True, edgecolor="none")
    # The caption is stashed rather than drawn: it can only be positioned once
    # `layout` has settled the axes, otherwise it lands on top of the x-label.
    ax._sf_note = _wrap_to_axes(ax, note) if note else ""


#: Rendered height of one caption line, in inches, at NOTE_FONTSIZE.
_NOTE_LINE_INCHES = 0.155


def layout(fig, rect=(0, 0, 1, 0.95), hspace: float | None = None) -> None:
    """Lay a multi-panel figure out, then hang each caption below its own panel.

    Captions are placed from each axes' *measured* tight bounding box, so they
    always clear the tick labels and the axis label however tall those turn out
    to be, and vertical space is reserved for them up front so they never fall
    off the bottom of the figure.
    """
    axes = [ax for ax in fig.axes if getattr(ax, "_sf_note", "")]
    max_lines = max((n.count("\n") + 1 for n in (ax._sf_note for ax in axes)),
                    default=0)
    pad_inches = _NOTE_LINE_INCHES * max_lines + 0.06

    fig_height = fig.get_figheight()
    n_rows = 1
    if fig.axes:
        spec = fig.axes[0].get_subplotspec()
        if spec is not None:
            n_rows = spec.get_gridspec().nrows

    bottom = pad_inches / fig_height if max_lines else rect[1]
    fig.tight_layout(rect=(rect[0], bottom, rect[2], rect[3]))

    if hspace is None and max_lines:
        # The gap between rows must clear the caption *and* the next row's
        # title, hence the fixed base term on top of the caption allowance.
        axes_height_inches = fig_height * (rect[3] - bottom) / max(n_rows, 1)
        hspace = 0.42 + 1.4 * pad_inches / max(axes_height_inches, 0.5)
    if hspace is not None and n_rows > 1:
        fig.subplots_adjust(hspace=hspace)

    if not axes:
        return
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax in axes:
        tight = ax.get_tightbbox(renderer).transformed(inv)
        fig.text(ax.get_position().x0, tight.y0 - 0.006, ax._sf_note,
                 fontsize=NOTE_FONTSIZE, color="#4B5563", va="top", ha="left",
                 linespacing=1.35)


def bps_formatter():
    return FuncFormatter(lambda v, _: f"{v:,.3g}")


def save(fig, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
