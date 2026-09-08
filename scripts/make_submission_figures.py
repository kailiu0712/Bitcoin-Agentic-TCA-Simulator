"""Render the submission figures from the versioned evidence artifacts.

The competition PDF and slide deck must not carry hand-drawn or hand-typed
numbers. Every figure here is produced from `outputs/latest_p0p1/` or from a
live paired simulation, in one consistent visual system: white surface, hairline
chrome, one blue for the scheduled algorithm, one orange for the urgency
benchmark, neutral gray for the no-execution control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "outputs" / "latest_p0p1" / "stylized_facts.json"
FIGURES = ROOT / "submission" / "assets" / "figures"

INK = "#111827"
INK_2 = "#4b5563"
INK_3 = "#8a94a3"
LINE = "#e3e7ed"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0f8a4d"
AMBER = "#b7791f"
GRAY = "#8a94a3"

GATE_NAMES = {
    "sign_memory_exponent": "订单符号记忆指数",
    "arrival_clustering": "成交到达聚集 Fano",
    "ofi_slope": "OFI 斜率",
    "ofi_r2": "OFI 解释力 R²",
    "trade_size_tail_p99_to_median": "成交规模 p99/中位数",
    "trade_size_tail_top_1pct_volume_share": "前 1% 成交量占比",
    "response_monotone_fraction": "成交响应单调性",
    "depth_shape_ask": "卖方深度形态",
    "depth_shape_bid": "买方深度形态",
    "spread_one_tick_share": "一档价差占比",
}

def configure() -> None:
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "font.size": 10,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#eef1f5",
        "grid.linewidth": 0.8,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def trim(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def figure_impact_scaling(evidence: dict) -> Path:
    law = evidence["impact_law"]
    curve = law["curve"]
    x = np.array([point["participation"] for point in curve])
    y = np.array([point["impact_bp"] for point in curve])
    err = np.array([point["std_bp"] for point in curve])
    low = np.maximum(y - err, y * 0.25)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    grid = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 1.25), 80)
    ax.plot(grid, law["coefficient"] * 1e4 * grid ** law["delta"], color=ORANGE, lw=1.8, ls="--",
            label=f"幂律拟合  δ = {law['delta']:.3f}  (R² = {law['power_r2']:.6f})", zorder=2)
    ax.vlines(x, low, y + err, color=BLUE, lw=1.4, alpha=0.45, zorder=3)
    ax.plot(x, y, "o", color=BLUE, ms=6, mec="white", mew=1.4, zorder=4,
            label="配对干预实验均值 ±1σ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("母单规模 / 同期成交量")
    ax.set_ylabel("峰值因果冲击（bp）")
    ax.set_title("元订单冲击标度律：市场自发产生的平方根冲击", pad=12)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:g}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:g}")
    ax.legend(loc="upper left")
    trim(ax)
    ax.annotate(f"理论平方根 δ = 0.5\n实测偏差 {law['distance_from_half']:.4f}",
                xy=(0.98, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec=LINE, lw=0.8))
    path = FIGURES / "impact_scaling.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_gate_distances(evidence: dict) -> Path:
    gate = sorted(evidence["gate"], key=lambda row: row["distance"])
    labels = [GATE_NAMES.get(row["metric"], row["metric"]) for row in gate]
    values = [row["distance"] for row in gate]

    fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=200)
    positions = np.arange(len(labels))
    ax.barh(positions, values, height=0.6, color=BLUE, alpha=0.9)
    tolerance = evidence["headline"].get("tolerance", 1.0)
    ax.axvline(tolerance, color=ORANGE, lw=1.5, ls="--")
    ax.text(tolerance + 0.02, len(labels) - 0.4, f"参考容差 {tolerance:.1f}", color=ORANGE, fontsize=9, va="center")
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("与 Kraken L3 实测参考的归一化偏差（越小越接近真实市场）")
    ax.set_title(f"校准偏差：留出交易日与 {evidence['headline']['validation_seeds']} 个独立随机种子", pad=12)
    for position, value in zip(positions, values):
        ax.text(value + 0.015, position, f"{value:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.grid(axis="y", visible=False)
    trim(ax)
    path = FIGURES / "gate_distances.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_architecture(evidence: dict) -> Path:
    """The multi-agent market, and which stylized facts each layer generates.

    This is the load-bearing picture of the report: impact is credible because
    it is produced by named mechanisms, not asserted by a formula.
    """
    layers = evidence.get("architecture", [])
    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=200)
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 100)

    def card(x, y, w, h, face, edge):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2))

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=1.3, shrinkA=0, shrinkB=0), zorder=3)

    def two_lines(items):
        """Keep mechanism lists inside the card by breaking them onto two rows."""
        half = (len(items) + 1) // 2
        return " · ".join(items[:half]) + "\n" + " · ".join(items[half:])

    stack_top, height, gap = 82.0, 25.0, 3.5
    mid = stack_top - (3 * height + 2 * gap) / 2

    card(1, mid - 25, 13, 50, "#f8fafc", LINE)
    ax.text(7.5, mid + 15, "事件驱动内核", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(7.5, mid, "优先队列事件时钟\n统一调度所有智能体\n公共随机数种子", ha="center", va="center",
            fontsize=8.5, color=INK_2, linespacing=1.9)
    arrow(14.5, 19, mid)

    colours = [BLUE, GREEN, ORANGE]
    tints = ["#eaf2fd", "#e9f5ee", "#fdeee7"]
    left, width = 19.5, 43.0
    for index, layer in enumerate(layers[:3]):
        y = stack_top - index * (height + gap) - height
        card(left, y, width, height, tints[index], colours[index])
        ax.text(left + 3, y + height - 6, f"0{index + 1}", fontsize=9, fontweight="bold",
                color=colours[index], va="center")
        ax.text(left + 8, y + height - 6, layer["name"], fontsize=10.5, fontweight="bold",
                color=INK, va="center")
        ax.text(left + 3, y + height - 14, two_lines(layer["mechanisms"]), fontsize=8,
                color=INK_2, va="center", linespacing=1.8)
        ax.text(left + 3, y + 3.5, "生成事实：" + " / ".join(layer["facts"]), fontsize=8,
                color=colours[index], va="center", fontweight="bold")
    arrow(left + width + 1, left + width + 5, mid)

    card(67.5, mid - 25, 14, 50, "#f8fafc", LINE)
    ax.text(74.5, mid + 15, "限价簿撮合", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(74.5, mid, "价格—时间优先\n多档挂单与撤单\n成交与盘口事件流", ha="center", va="center",
            fontsize=8.5, color=INK_2, linespacing=1.9)
    arrow(82, 85.5, mid)

    card(85.5, mid - 25, 13.5, 50, "#eaf2fd", BLUE)
    ax.text(92.2, mid + 15, "配对反事实度量", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(92.2, mid, "同种子无执行对照\n逐点相减\n冲击 / 衰减 / 成本", ha="center", va="center",
            fontsize=8.5, color=INK_2, linespacing=1.9)

    ax.text(0, 97, "多智能体市场架构：价格冲击由机制涌现，而非写入的公式",
            fontsize=12, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0, 90, "三类智能体在同一事件时钟与同一限价簿上共同演化；每一层负责生成一组可被独立检验的微观结构事实。",
            fontsize=9, color=INK_3, ha="left", va="center")
    path = FIGURES / "architecture.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_impact_trajectory() -> Path:
    """A live paired run: the product's own measurement, not a cached picture."""
    from app.main import SimulationRequest, run_execution_simulation

    request = SimulationRequest(side="BUY", parent_quantity=4.5, duration_seconds=600,
                                volatility=2.4, spread=.13, arrival_rate=.58,
                                displayed_depth=.18, seed=2027)
    result = run_execution_simulation(request)
    styles = {"twap": (BLUE, "TWAP", 1.6, "solid"),
              "immediate": (ORANGE, "Immediate", 1.6, "solid")}

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.4, 4.2), dpi=200,
                                 gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.12},
                                 sharex=True)
    for key, (color, label, width, dash) in styles.items():
        path = result["trajectories"].get(key) or []
        points = [(row["elapsed_seconds"], row.get("causal_impact_bp") if row.get("causal_impact_bp") is not None else row.get("impact_bp"))
                  for row in path if (row.get("filled") or 0) == 0]
        points = [(t, v) for t, v in points if v is not None]
        if not points:
            continue
        ax.plot([t for t, _ in points], [v for _, v in points], color=color, lw=width,
                ls=dash, label=label)
    ax.axhline(0, color=LINE, lw=1)
    ax.set_ylabel("因果冲击（bp）", fontsize=10)
    ax.set_title("同一市场路径下的因果冲击轨迹：一次性成交 vs 均匀拆单", pad=10)
    ax.legend(loc="upper right", ncol=2)
    trim(ax)

    children = [row for row in result["execution_paths"]["twap"] if (row.get("filled") or 0) > 0]
    bx.bar([row["elapsed_seconds"] for row in children],
           [row["requested"] for row in children],
           width=result["request"]["duration_seconds"] / max(len(children), 1) * 0.55,
           color=BLUE, alpha=0.55)
    bx.set_ylabel("子单规模\n(BTC)", fontsize=9)
    bx.set_xlabel("执行时间（秒，含结束后 30 秒观察窗）")
    bx.grid(axis="x", visible=False)
    trim(bx)

    peak = result["comparison"]
    bx.text(0.995, -0.62,
            f"峰值冲击  Immediate {peak['immediate']['impact_bp']:.3f} bp  ·  "
            f"TWAP {peak['twap']['impact_bp']:.3f} bp",
            transform=bx.transAxes, ha="right", va="top", fontsize=9, color=INK_2)
    path = FIGURES / "impact_trajectory.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    configure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    paths = [
        figure_impact_scaling(evidence),
        figure_gate_distances(evidence),
        figure_architecture(evidence),
        figure_impact_trajectory(),
    ]
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
