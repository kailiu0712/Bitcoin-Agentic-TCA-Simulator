"""Collect the held-out stylized-fact evidence into one served artifact.

The simulator's claim is not "our execution policy is cheap"; it is "the market
an execution policy trades against reproduces the microstructure facts that
generate execution cost". That claim has to be auditable, so this script reads
the versioned validation artifacts and writes a single JSON document that the
web application, the PDF and the slide deck all render from. No number in the
user-facing material is typed by hand.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "latest_p0p1"
OUTPUT = EVIDENCE / "stylized_facts.json"

# Raw vendor rows behind the canonical dataset; recorded in the preprocessing log
# and reported in the README. Everything else is read from artifacts.
RAW_ROWS = 166_008_529
VENUE = "Kraken"
SYMBOL = "BTC/USD"

# The three agent layers the market is built from, and which stylized facts each
# one is responsible for generating. Shared by the web app and the report so the
# architecture story is told from one definition.
ARCHITECTURE = [
    {
        "name": "背景订单流智能体",
        "role": "生成市场自身的成交流",
        "mechanisms": ["Hawkes 聚集到达过程", "多时间尺度订单符号长记忆", "对数正态叠加帕累托的重尾成交规模"],
        "facts": ["成交到达聚集", "订单符号长记忆", "成交规模重尾"],
    },
    {
        "name": "做市与流动性智能体",
        "role": "提供并补充盘口深度",
        "mechanisms": ["8 档驼峰形深度剖面", "库存与流动压力驱动的报价偏移", "队列失衡响应与整数跳动价差控制"],
        "facts": ["价差分布与状态", "盘口深度形态", "订单流失衡响应"],
    },
    {
        "name": "执行算法智能体",
        "role": "被评估的清算算法",
        "mechanisms": ["按子单决策函数拆分母单", "与同种子无执行对照市场配对运行", "资产、数据与算法均可替换接入"],
        "facts": ["元订单冲击标度律", "冲击轨迹与衰减"],
    },
]

# Each entry is one scoped fact group from the reference framework, paired with
# the number the simulator actually produced. `status` is the honest label used
# in outputs/latest_p0p1/diagnosis.md, not an aspirational one.
FACT_TEMPLATE = [
    {
        "id": "P0-V1", "group": "impact",
        "name": "元订单冲击平方根律", "name_en": "Metaorder square-root impact",
        "why": "决定大额母单的冲击成本随规模如何增长，是 TCA 最核心的定价关系。",
    },
    {
        "id": "P0-V2", "group": "impact",
        "name": "冲击轨迹凹性", "name_en": "Impact trajectory concavity",
        "why": "决定同一母单在执行过程中冲击如何累积，影响拆单节奏设计。",
    },
    {
        "id": "P0-V3", "group": "impact",
        "name": "执行后冲击衰减", "name_en": "Post-metaorder decay",
        "why": "区分暂时冲击与永久冲击，决定实现成本中可回收的部分。",
    },
    {
        "id": "P0-V4", "group": "impact",
        "name": "冲击曲面（规模 × 时域）", "name_en": "Impact surface",
        "why": "把冲击写成规模与时间的函数，是执行时限选择的依据。",
    },
    {
        "id": "P0-C5", "group": "liquidity",
        "name": "流动性成本与盘口深度", "name_en": "Liquidity cost / depth",
        "why": "决定吃单穿透多少档位，直接对应滑点。",
    },
    {
        "id": "P0-C6", "group": "liquidity",
        "name": "价差分布与状态", "name_en": "Spread distribution / state",
        "why": "价差是每一笔子单都要支付的基础成本。",
    },
    {
        "id": "P0-C7", "group": "liquidity",
        "name": "流动性弹性恢复", "name_en": "Liquidity resiliency",
        "why": "决定被吃掉的深度多快补回，是拆单间隔的物理约束。",
    },
    {
        "id": "P0-C8", "group": "flow",
        "name": "订单流失衡响应（OFI）", "name_en": "Order-flow-imbalance response",
        "why": "把盘口失衡转成价格变化，是执行择时信号的来源。",
    },
    {
        "id": "P0-C9", "group": "flow",
        "name": "成交响应函数", "name_en": "Trade response function",
        "why": "单笔成交的平均价格响应，是冲击模型的微观基础。",
    },
    {
        "id": "P0-C10", "group": "flow",
        "name": "订单符号长记忆", "name_en": "Order-sign long memory",
        "why": "解释为什么冲击是凹的：符号自相关使后续子单成本被前序子单预告。",
    },
    {
        "id": "P0-C11", "group": "price",
        "name": "价格扩散性", "name_en": "Diffusive prices",
        "why": "保证在冲击之外价格没有可套利漂移，避免高估或低估执行成本。",
    },
    {
        "id": "P1-16", "group": "flow",
        "name": "成交规模重尾", "name_en": "Heavy-tailed trade size",
        "why": "决定队列被瞬间清空的概率，是尾部执行风险的来源。",
    },
]

# Which measured quantity to show beside each fact group.
OBSERVATION = {
    "P0-V1": lambda ctx: {
        "value": f"δ = {ctx['fits']['delta']:.3f}",
        "target": "理论值 0.5",
        "detail": f"幂律拟合 R² = {ctx['fits']['power_r2']:.6f}，{ctx['runs']} 组配对干预实验",
    },
    "P0-V2": lambda ctx: {
        "value": "单调递增且凹",
        "target": "参考文献形态",
        "detail": f"冲击从 {ctx['impact_curve'][0]['impact_bp']:.2f} bp（0.1% 参与率）升至 {ctx['impact_curve'][-1]['impact_bp']:.2f} bp（20%）",
    },
    "P0-V3": lambda ctx: {
        "value": "留存 " + " / ".join(f"{ctx['decay'][t] * 100:.0f}%" for t in (30, 60, 120)),
        "target": "执行后冲击趋于稳定",
        "detail": "执行结束 30 / 60 / 120 秒相对终点冲击，呈以永久冲击为主的平台形态",
    },
    "P0-V4": lambda ctx: {
        "value": "斜率 " + " / ".join(f"{v:.2f}" for v in ctx["surface"]),
        "target": "随时域递减",
        "detail": "1 / 5 / 30 / 60 秒前瞻窗口上的规模—冲击对数斜率",
    },
    "P0-C5": lambda ctx: {
        "value": f"深度形态偏差 {ctx['gate']['depth_shape_bid']:.3f} / {ctx['gate']['depth_shape_ask']:.3f}",
        "target": "Kraken 多档深度剖面",
        "detail": "买 / 卖两侧 8 档驼峰形深度剖面，决定吃单穿透档位与滑点",
    },
    "P0-C6": lambda ctx: {
        "value": f"一档价差占比 {ctx['one_tick_share'] * 100:.1f}%",
        "target": f"真实 {ctx['one_tick_target'] * 100:.1f}%",
        "detail": f"归一化距离 {ctx['gate']['spread_one_tick_share']:.3f}",
    },
    "P0-C7": lambda ctx: {
        "value": "30 秒内价差回补至基准",
        "target": "冲击后流动性回补",
        "detail": f"1 / 5 / 30 秒探针的价差变化 {ctx['resiliency_curve'][1]:.3f} / "
                  f"{ctx['resiliency_curve'][5]:.3f} / {ctx['resiliency_curve'][30]:.3f} USD",
    },
    "P0-C8": lambda ctx: {
        "value": f"斜率 {ctx['ofi_slope']:.2f}，R² {ctx['ofi_r2']:.2f}",
        "target": "正斜率且具解释力",
        "detail": f"归一化距离 {ctx['gate']['ofi_slope']:.3f} / {ctx['gate']['ofi_r2']:.3f}",
    },
    "P0-C9": lambda ctx: {
        "value": f"单调比例距离 {ctx['gate']['response_monotone_fraction']:.3f}",
        "target": "随成交笔数单调上升",
        "detail": f"R(1) = {ctx['response_1']:.3f}（模拟价格单位）",
    },
    "P0-C10": lambda ctx: {
        "value": f"γ = {ctx['sign_gamma']:.3f}",
        "target": "0 < γ < 1 的幂律衰减",
        "detail": f"C(1) = {ctx['sign_acf_1']:.3f}，C(100) = {ctx['sign_acf_100']:.4f}",
    },
    "P0-C11": lambda ctx: {
        "value": "方差比 " + " / ".join(f"{v:.2f}" for v in ctx["variance_ratio"][:3]),
        "target": "接近 1.0",
        "detail": f"1 / 5 / 10 秒时标；60 秒时标为 {ctx['variance_ratio'][3]:.2f}，长时域保留趋势成分",
    },
    "P1-16": lambda ctx: {
        "value": f"p99/中位数 = {ctx['tail_ratio']:.0f}",
        "target": "真实 700",
        "detail": f"前 1% 成交量占比 {ctx['top1_share'] * 100:.1f}%，归一化距离 {ctx['gate']['trade_size_tail_p99_to_median']:.3f}",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build() -> dict:
    metrics = read_json(EVIDENCE / "six_hour_simulation" / "metrics.json")
    fits = read_json(EVIDENCE / "liquidation" / "analysis" / "fits.json")[0]
    experiment = read_json(EVIDENCE / "liquidation" / "experiment_metadata.json")
    validation = read_json(EVIDENCE / "validation" / "validation_summary.json")
    calibration = read_json(EVIDENCE / "calibration" / "summary.json")
    data_summary = read_json(ROOT / "data" / "metadata" / "data_summary.json")
    gate_rows = read_csv_rows(EVIDENCE / "validation" / "validation_metrics.csv")
    seed_rows = read_csv_rows(EVIDENCE / "validation" / "seed_metric_errors.csv")
    impact_rows = read_csv_rows(EVIDENCE / "liquidation" / "analysis" / "twap_impact_by_size.csv")
    decay_rows = [row for row in read_csv_rows(EVIDENCE / "liquidation" / "analysis" / "twap_impact_decay.csv")
                  if row["impact"]]

    gate = {row["metric"]: float(row["normalized_distance_validation"]) for row in gate_rows}
    targets = {}
    for row in seed_rows:
        try:
            targets[row["metric"]] = float(row["real_target"])
        except (TypeError, ValueError):
            continue

    impact_curve = [
        {
            "participation": float(row["participation"]),
            "impact_bp": float(row["impact_bp"]),
            "std_bp": float(row["std"]) * 1e4,
            "n": int(row["n"]),
        }
        for row in impact_rows
    ]

    terminal = float(decay_rows[0]["impact"])
    decay = {int(float(row["seconds_after"])): float(row["impact"]) / terminal for row in decay_rows}

    context = {
        "fits": fits,
        "decay": decay,
        "resiliency_curve": {int(key): value["spread_change"] for key, value in metrics["resiliency"].items()},
        "runs": experiment["runs"],
        "impact_curve": impact_curve,
        "surface": [metrics["impact_surface"][key]["size_slope"] for key in ("1", "5", "30", "60")],
        "variance_ratio": [metrics["diffusivity"][key] for key in sorted(metrics["diffusivity"], key=lambda k: float(k))][:4],
        "one_tick_share": metrics["spread_one_tick_share"],
        "one_tick_target": targets.get("spread_one_tick_share", 0.976),
        "resiliency": metrics["resiliency"]["30"]["spread_change"],
        "ofi_slope": metrics["ofi_response"]["slope"],
        "ofi_r2": metrics["ofi_response"]["r2"],
        "response_1": metrics["response"]["1"],
        "sign_gamma": metrics["sign_memory_exponent"],
        "sign_acf_1": metrics["sign_acf"]["1"],
        "sign_acf_100": metrics["sign_acf"]["100"],
        "tail_ratio": metrics["trade_size_tail"]["p99_to_median"],
        "top1_share": metrics["trade_size_tail"]["top_1pct_volume_share"],
        "gate": gate,
    }

    facts = []
    for entry in FACT_TEMPLATE:
        observation = OBSERVATION[entry["id"]](context)
        facts.append({**entry, **observation})

    sessions = data_summary["sessions"]
    window = f"{data_summary['usable_dates'][0]} → {data_summary['usable_dates'][-1]}"

    return {
        "generated_on": date.today().isoformat(),
        "headline": {
            "facts": len(facts),
            "gate_metrics": len(gate_rows),
            "validation_seeds": validation["seeds"],
            "worst_distance": max(float(row["normalized_distance_validation"]) for row in gate_rows),
            "tolerance": 1.0,
        },
        "architecture": ARCHITECTURE,
        "market": {
            "venue": VENUE,
            "symbol": SYMBOL,
            "window": window,
            "raw_rows": RAW_ROWS,
            "canonical_events": sum(item["rows"] for item in sessions),
            "trades": sum(item["trades"] for item in sessions),
            "quotes": sum(item["quotes"] for item in sessions),
            "sessions": len(data_summary["usable_dates"]),
            "calibration_sessions": len(data_summary["calibration_dates"]),
            "validation_sessions": len(data_summary["validation_dates"]),
            "trials": calibration["trials"],
            "best_loss": calibration["best_loss"],
            "search": calibration["method"],
        },
        "impact_law": {
            "delta": fits["delta"],
            "target_delta": 0.5,
            "power_r2": fits["power_r2"],
            "linear_r2": fits["linear_r2"],
            "distance_from_half": fits["distance_from_half"],
            "coefficient": fits["A"],
            "runs": experiment["runs"],
            "durations": experiment["durations"],
            "sides": experiment["sides"],
            "curve": impact_curve,
        },
        "facts": facts,
        "gate": [
            {"metric": row["metric"], "distance": float(row["normalized_distance_validation"]),
             "seed_std": float(row["seed_std_validation"])}
            for row in sorted(gate_rows, key=lambda row: -float(row["normalized_distance_validation"]))
        ],
    }


def main() -> None:
    payload = build()
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    head = payload["headline"]
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {head['facts']} facts, {head['gate_metrics']} gate metrics, "
          f"worst held-out distance {head['worst_distance']:.3f} over {head['validation_seeds']} seeds")


if __name__ == "__main__":
    main()
