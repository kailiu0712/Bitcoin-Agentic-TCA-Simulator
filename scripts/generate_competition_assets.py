"""Build every competition deliverable from the versioned evidence.

Positioning: the deliverable is a *stylized-fact calibrated liquidation impact
simulator*, not an execution policy. Any liquidation algorithm plugs in; the
product claim is that the market it trades against reproduces the microstructure
facts that generate execution cost, and that this is auditable fact by fact.

Everything numeric is read from `outputs/latest_p0p1/stylized_facts.json`, so the
slides, the PDF and the pasteable text can never disagree with the artifacts.
Run `scripts/build_stylized_fact_evidence.py` and
`scripts/make_submission_figures.py` first, or just run this module - it calls
both when their outputs are missing or stale.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
ASSETS = SUBMISSION / "assets"
SLIDES_DIR = ASSETS / "slides"
FIGURES = ASSETS / "figures"
CACHE = ASSETS / "cache"   # intermediate crops; not a deliverable
EVIDENCE_PATH = ROOT / "outputs" / "latest_p0p1" / "stylized_facts.json"
UI_SCREENSHOT = ROOT / "UI_page.png"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
CANVAS = (1920, 1080)
PPT_SIZE = (12192000, 6858000)

# One neutral, high-contrast system shared by the app, the slides and the PDF.
INK = "#111827"
INK_2 = "#4b5563"
INK_3 = "#8a94a3"
LINE = "#e3e7ed"
PAGE = "#f4f6f8"
SURFACE = "#ffffff"
SUNKEN = "#f8fafc"
BLUE = "#2a78d6"
BLUE_SOFT = "#eaf2fd"
ORANGE = "#eb6834"
GREEN = "#0f8a4d"
GREEN_SOFT = "#e6f4ec"
AMBER = "#b7791f"
AMBER_SOFT = "#fdf3e0"

MARGIN = 110


# --------------------------------------------------------------------- copy

@dataclass(frozen=True)
class Copy:
    """All submission prose in one place, with the evidence numbers injected."""
    evidence: dict

    @property
    def head(self) -> dict:
        return self.evidence["headline"]

    @property
    def law(self) -> dict:
        return self.evidence["impact_law"]

    @property
    def market(self) -> dict:
        return self.evidence["market"]

    @property
    def layers(self) -> list[dict]:
        return self.evidence["architecture"]

    team_name = "智执实验室"
    short_name = "智执 TCA"
    project_name = "智执 TCA：匹配市场真实冲击表现的清算冲击仿真器"
    tagline = "多智能体限价簿市场 · 风格化事实训练 · 配对反事实冲击度量"
    track = "方向三：金融 AI 智能体行业场景落地"
    cross_track = "融合方向一的多智能体建模能力与方向二的量化评估工具能力"

    @property
    def summary(self) -> str:
        market, law = self.market, self.law
        return (
            f"智执 TCA 是一个匹配市场真实冲击表现的清算冲击仿真器。系统以 {market['venue']} {market['symbol']} L3 逐笔盘口与成交数据"
            f"（{market['window']}，{market['sessions']} 个交易日，约 {market['raw_rows'] / 1e6:.0f}M 原始行标准化为 "
            f"{market['canonical_events'] / 1e6:.2f}M 事件）完成风格化事实训练，构建由背景订单流、做市流动性与执行算法"
            f"三类智能体共同演化的事件驱动限价簿市场。价格冲击由撮合与流动性补充涌现，而不是写入的公式，因此市场自发产生"
            f"指数 δ={law['delta']:.3f} 的平方根元订单冲击（幂律 R²={law['power_r2']:.6f}），并与 "
            f"{self.head['facts']} 项与 TCA 直接相关的微观结构事实逐项对照。标的、数据源与执行算法均可灵活调整与接入："
            f"算法接入后即在同种子配对反事实对照下，得到峰值冲击、冲击衰减与配对执行成本报告。"
        )

    @property
    def public_intro(self) -> str:
        market = self.market
        return (
            "机构交易的成本大头发生在「下单之后」：同一笔母单，拆得快还是拆得慢，冲击成本可以相差数倍。"
            "行业普遍用仿真来评估执行算法，难点在于仿真市场是否会像真实市场那样对你的订单作出反应——"
            "如果价格不会被自己的订单推动，得到的 TCA 结论就无法支撑决策。"
            f"智执 TCA 以 {market['venue']} {market['symbol']} L3 逐笔数据完成风格化事实训练，"
            "把价差、盘口深度、订单流记忆、成交规模尾部与冲击标度关系全部对齐到实测参考值上，"
            "再由背景订单流、做市流动性与执行算法三类智能体在同一限价簿上共同演化出价格冲击。"
            "评估时，系统用同一随机种子生成一份「无执行」的对照市场，把两条中间价路径逐点相减，"
            "得到纯粹由执行造成的因果冲击、冲击衰减与配对成本。"
            "标的、数据源与执行算法都是可替换的：新标的按同一流程完成标准化与校准即可接入，"
            "自研算法只需实现一个子单决策函数就能获得同口径报告。"
            "该能力可服务于券商算法执行台的算法验收、资管与私募交易团队的执行参数研究、做市与 OTC 的报价成本评估，"
            "以及交易所与金融科技平台的策略沙盒。"
        )

    @property
    def innovations(self) -> list[str]:
        law = self.law
        return [
            f"冲击是涌现的而不是写死的：三类智能体中没有任何冲击公式，δ={law['delta']:.3f}、"
            f"R²={law['power_r2']:.6f} 的平方根律由订单符号长记忆、Hawkes 聚集到达、重尾成交与深度补充共同产生，"
            f"因此对没有见过的算法与母单规模同样成立。",
            f"以风格化事实训练市场本身：把价差、深度形态、订单流失衡响应、成交规模尾部等 {self.head['facts']} 项"
            f"与 TCA 直接相关的微观结构事实作为校准目标，并在留出交易日与独立随机种子上逐项对照实测参考值。",
            "公共随机数配对反事实：同一随机种子生成「无执行」对照市场，逐点相减隔离出纯由执行造成的因果冲击，"
            "而不是把市场自身的漂移与噪声计入执行成本。",
            "标的、数据与算法三层可插拔：新标的复用同一条数据标准化与校准流水线，自研执行算法实现一个子单决策函数"
            "即可接入，统一输出峰值冲击、终点冲击、冲击留存率与配对成本。",
        ]

    pain_points = [
        "大额母单直接吃单会自我抬价，冲击成本远大于价差成本。",
        "拆单参数长期依赖交易员经验，缺少可复现的离线验证环境。",
        "常规回测器把成交价当作外生变量，结构上无法回答「我的单子把价格推了多少」。",
        "冲击评估的可信度取决于市场模型是否贴近真实微观结构，而这一步往往被跳过。",
    ]

    outputs = [
        "峰值因果冲击：执行期内相对对照市场的最大中间价偏离。",
        "终点冲击与冲击留存率：区分暂时冲击与永久冲击，对应可回收成本。",
        "配对执行成本：每笔成交价与对照市场同时刻中间价之差，按母单名义额加权。",
        "完成率与子单明细：逐切片的请求量、成交量与该时刻的因果冲击。",
    ]

    pipeline = [
        ("数据标准化", "L3 逐笔盘口与成交流整理为统一事件表，抽取风格化事实的实测参考值。"),
        ("市场训练", "Optuna TPE 搜索多智能体参数，按归一化偏差逼近全部参考值。"),
        ("留出对照", "在未参与训练的交易日与独立随机种子上逐项复核，偏差全部公开。"),
        ("算法接入", "执行算法作为智能体注入同一市场，与同种子对照市场配对运行。"),
        ("冲击度量", "逐点相减输出冲击轨迹、衰减与配对成本，供算法验收与参数研究。"),
    ]

    customers = [
        "券商算法交易台与机构经纪：新算法上线前的冲击验收与参数标定",
        "资管、私募与 CTA 交易团队：大额调仓的执行时限与拆单强度研究",
        "做市商与 OTC 服务商：大单承接的报价与对冲成本评估",
        "交易所、金融科技平台与量化教研机构：可复现的执行策略沙盒",
    ]

    business_model = [
        "标的接入服务：按资产与场所交付一套完成校准的市场配置与对照报告。",
        "平台订阅：冲击实验台与算法评估报告，按席位或按团队订阅。",
        "系统集成：作为离线评估模块接入 OMS/EMS 与算法研发流水线。",
    ]

    roadmap = [
        "扩展标的与场所模板，复用同一条数据标准化与校准流水线。",
        "纳入手续费、撮合延迟与跨场所路由，贴近真实执行链路。",
        "开放算法接入 SDK 与批量实验接口，支持算法库的持续回归式冲击验收。",
        "增加自然语言编排层：交易员以中文描述执行约束，智能体生成实验配置与解读报告。",
    ]

    limitations = [
        "元订单冲击为合成干预实验的度量结果；接入真实母单标签后可进一步做同样本实证比对。",
        "长时域（60 秒以上）价格扩散性保留一定趋势成分，已在校准表中如实给出数值。",
        "当前不含手续费、撮合延迟与跨场所路由；结果为离线评估结论，不构成投资建议。",
    ]


# ------------------------------------------------------------------- assets

def ensure_inputs() -> dict:
    """Regenerate the evidence and figures if they are missing."""
    if not EVIDENCE_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_stylized_fact_evidence.py")], check=True)
    if not (FIGURES / "impact_scaling.png").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_submission_figures.py")], check=True)
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


# CJK wraps per character, but a number or an identifier must never be split -
# "R2 = 0.999997" breaking after the decimal point is what made the old deck
# look broken. Latin/numeric runs are therefore treated as atomic tokens.
ATOM = re.compile(r"[0-9A-Za-z²³][0-9A-Za-z²³.,%/_+-]*|\s|.", re.UNICODE)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Wrap mixed CJK/Latin text without breaking numbers or identifiers."""
    lines: list[str] = []
    for block in text.splitlines():
        if not block.strip():
            lines.append("")
            continue
        current = ""
        for token in ATOM.findall(block):
            trial = current + token
            if not current or draw.textlength(trial, font=fnt) <= width:
                current = trial
            elif token.isspace():
                lines.append(current)
                current = ""
            else:
                lines.append(current.rstrip())
                current = token
        if current.strip():
            lines.append(current.rstrip())
    return _apply_kinsoku(lines)


# Chinese typesetting forbids a line starting with closing punctuation.
NO_LINE_START = "。，、；：？！）」』】〉》%…·"


def _apply_kinsoku(lines: list[str]) -> list[str]:
    fixed = list(lines)
    for index in range(1, len(fixed)):
        while fixed[index] and fixed[index][0] in NO_LINE_START and len(fixed[index - 1]) > 1:
            fixed[index - 1], fixed[index] = fixed[index - 1][:-1], fixed[index - 1][-1] + fixed[index]
    return fixed


def paragraph(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt: ImageFont.FreeTypeFont,
              width: int, fill: str, gap: int = 12) -> int:
    """Draw wrapped text and return the y coordinate just below the last line."""
    x, y = xy
    height = fnt.getbbox("国")[3]
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += height + gap
    return y


def box(draw: ImageDraw.ImageDraw, rect, fill=SURFACE, outline=LINE, radius=18, width=2) -> None:
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", CANVAS, PAGE)
    return image, ImageDraw.Draw(image)


def slide_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> int:
    draw.rectangle((0, 0, CANVAS[0], 6), fill=INK)
    draw.text((MARGIN, 74), title, font=font(52, bold=True), fill=INK)
    draw.text((MARGIN, 152), subtitle, font=font(26), fill=INK_3)
    return 232


def slide_footer(draw: ImageDraw.ImageDraw, note: str) -> None:
    draw.line((MARGIN, 1002, CANVAS[0] - MARGIN, 1002), fill=LINE, width=2)
    draw.text((MARGIN, 1022), note, font=font(19), fill=INK_3)
    draw.text((CANVAS[0] - MARGIN, 1022), Copy.short_name, font=font(19), fill=INK_3, anchor="ra")


def paste_fit(base: Image.Image, path: Path, rect, background: str = SURFACE) -> None:
    """Fit an image inside a box without distorting it."""
    image = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = rect
    target = (x1 - x0, y1 - y0)
    ratio = min(target[0] / image.width, target[1] / image.height)
    resized = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    plate = Image.new("RGB", target, background)
    plate.paste(resized, ((target[0] - resized.width) // 2, (target[1] - resized.height) // 2))
    base.paste(plate, (x0, y0))


def paste_top(base: Image.Image, path: Path, rect, background: str = SURFACE) -> None:
    """Fit by width and crop the overflow - used for the tall UI screenshot."""
    image = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = rect
    target = (x1 - x0, y1 - y0)
    ratio = target[0] / image.width
    resized = image.resize((target[0], max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    plate = Image.new("RGB", target, background)
    plate.paste(resized, (0, 0))
    base.paste(plate, (x0, y0))


def stat_card(draw: ImageDraw.ImageDraw, rect, value: str, label: str, note: str) -> None:
    box(draw, rect, fill=SURFACE)
    x0, y0, x1, _ = rect
    draw.text((x0 + 28, y0 + 24), value, font=font(44, bold=True), fill=INK)
    draw.text((x0 + 28, y0 + 88), label, font=font(21), fill=INK_3)
    paragraph(draw, note, (x0 + 28, y0 + 124), font(18), x1 - x0 - 56, INK_2, gap=6)


def bullet_list(draw: ImageDraw.ImageDraw, items, x: int, y: int, width: int, size: int = 23,
                colour: str = BLUE, gap: int = 20) -> int:
    fnt = font(size)
    for item in items:
        draw.ellipse((x, y + size * 0.35, x + 10, y + size * 0.35 + 10), fill=colour)
        y = paragraph(draw, item, (x + 26, y), fnt, width - 26, INK_2, gap=8) + gap
    return y


def coverage_strip(draw: ImageDraw.ImageDraw, facts: list[dict], x: int, y: int, width: int) -> int:
    """The scoped stylized facts as chips, grouped by the part of the market they describe."""
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        groups.setdefault(fact["group"], []).append(fact)
    labels = {"impact": "冲击（元订单）", "liquidity": "流动性与盘口", "flow": "订单流", "price": "价格过程"}
    label_width = 240
    chip_width = (width - label_width - 3 * 16) // 4
    for key in [name for name in ("impact", "liquidity", "flow", "price") if name in groups]:
        draw.text((x, y + 13), labels[key], font=font(23, bold=True), fill=INK)
        for index, fact in enumerate(groups[key]):
            cx = x + label_width + index * (chip_width + 16)
            box(draw, (cx, y, cx + chip_width, y + 52), fill=BLUE_SOFT, outline="#c4dcf8", radius=8, width=2)
            draw.text((cx + 18, y + 8), fact["id"], font=font(19, bold=True), fill=BLUE)
            draw.text((cx + chip_width - 18, y + 10), fact["name"], font=font(18), fill=INK_2, anchor="ra")
        y += 64
    return y


def save_slide(image: Image.Image, index: int) -> Path:
    path = SLIDES_DIR / f"slide_{index:02d}.png"
    image.save(path, format="PNG")
    return path


# -------------------------------------------------------------------- deck

def build_slides(copy: Copy) -> list[Path]:
    slides: list[Path] = []
    head, law, market = copy.head, copy.law, copy.market

    # 1 — cover
    image, draw = canvas()
    draw.rectangle((0, 0, CANVAS[0], 8), fill=INK)
    draw.text((MARGIN, 128), copy.short_name, font=font(76, bold=True), fill=INK)
    draw.text((MARGIN, 232), "匹配市场真实冲击表现的清算冲击仿真器", font=font(40, bold=True), fill=BLUE)
    paragraph(draw, f"以 {market['venue']} {market['symbol']} L3 逐笔数据完成风格化事实训练，构建背景订单流、做市流动性与执行算法"
                    f"三类智能体共同演化的限价簿市场；标的、数据源与执行算法均可灵活调整与接入。",
              (MARGIN, 314), font(26), CANVAS[0] - 2 * MARGIN, INK_2, gap=10)
    cards = [
        (f"δ = {law['delta']:.3f}", "元订单冲击指数", f"与平方根律理论值 0.5 一致，幂律 R² = {law['power_r2']:.6f}"),
        (f"{head['facts']} 项", "微观结构校准指标", "覆盖冲击、流动性、订单流与价格过程"),
        (f"{market['canonical_events'] / 1e6:.2f}M", f"{market['venue']} L3 标准化事件",
         f"{market['sessions']} 个交易日，约 {market['raw_rows'] / 1e6:.0f}M 原始行"),
        (f"{len(copy.layers)} 层", "多智能体市场架构", "背景订单流 · 做市流动性 · 执行算法"),
    ]
    card_width = (CANVAS[0] - 2 * MARGIN - 3 * 26) // 4
    for index, (value, label, note) in enumerate(cards):
        x = MARGIN + index * (card_width + 26)
        stat_card(draw, (x, 470, x + card_width, 700), value, label, note)
    box(draw, (MARGIN, 748, CANVAS[0] - MARGIN, 940), fill=SURFACE)
    draw.text((MARGIN + 34, 780), "接入即评估", font=font(26, bold=True), fill=INK)
    paragraph(draw, "内置 Immediate 与 TWAP 两个行业基准作为共同标尺；自研算法实现一个子单决策函数即可接入，"
                    "在同种子配对反事实对照下获得峰值冲击、终点冲击、冲击留存率与配对执行成本报告。",
              (MARGIN + 34, 826), font(22), CANVAS[0] - 2 * MARGIN - 68, INK_2, gap=8)
    draw.text((MARGIN, 990), f"团队：{copy.team_name}　|　{copy.track}", font=font(21), fill=INK_3)
    slides.append(save_slide(image, 1))

    # 2 — problem
    image, draw = canvas()
    y = slide_header(draw, "问题：不是缺算法，是缺可信的评估基座",
                     "执行成本发生在下单之后，而绝大多数仿真器的市场并不会因为你的订单而真实移动。")
    box(draw, (MARGIN, y, 930, 900), fill=SURFACE)
    draw.text((MARGIN + 34, y + 30), "行业痛点", font=font(28, bold=True), fill=INK)
    bullet_list(draw, copy.pain_points, MARGIN + 34, y + 92, 800, colour=ORANGE)
    box(draw, (970, y, CANVAS[0] - MARGIN, 900), fill=SURFACE)
    draw.text((1004, y + 30), "本项目的回答", font=font(28, bold=True), fill=INK)
    bullet_list(draw, [
        "以真实 L3 数据训练市场：价差、深度、订单流记忆与冲击标度关系全部对齐实测参考值。",
        "算法作为智能体注入同一市场，与同种子无执行对照市场配对运行。",
        "逐点相减剔除市场自身漂移，只报由执行造成的因果冲击、衰减与成本。",
        "内置行业基准作为共同标尺，平台不预设哪个算法更优。",
    ], 1004, y + 92, 780, colour=BLUE)
    slide_footer(draw, "冲击评估的可信度，取决于市场模型是否贴近真实微观结构")
    slides.append(save_slide(image, 2))

    # 3 — the multi-agent architecture, the report's load-bearing explanation
    image, draw = canvas()
    y = slide_header(draw, "多智能体市场架构",
                     "事件驱动内核调度三类智能体，统一进入价格—时间优先的限价簿撮合。")
    card_width = (CANVAS[0] - 2 * MARGIN - 2 * 24) // 3
    for index, layer in enumerate(copy.layers[:3]):
        accent = (BLUE, GREEN, ORANGE)[index]
        tint = (BLUE_SOFT, GREEN_SOFT, "#fdeee7")[index]
        x = MARGIN + index * (card_width + 24)
        box(draw, (x, y, x + card_width, y + 396), fill=SURFACE)
        box(draw, (x, y, x + card_width, y + 6), fill=accent, outline=accent, radius=3, width=1)
        draw.rounded_rectangle((x + 28, y + 34, x + 74, y + 74), radius=10, fill=tint)
        draw.text((x + 51, y + 54), f"0{index + 1}", font=font(22, bold=True), fill=accent, anchor="mm")
        draw.text((x + 88, y + 54), layer["name"], font=font(27, bold=True), fill=INK, anchor="lm")
        draw.text((x + 28, y + 96), layer["role"], font=font(21), fill=INK_3)
        yy = y + 140
        for item in layer["mechanisms"]:
            draw.ellipse((x + 28, yy + 9, x + 38, yy + 19), fill=accent)
            yy = paragraph(draw, item, (x + 52, yy), font(21), card_width - 84, INK_2, gap=7) + 14
        draw.line((x + 28, y + 320, x + card_width - 28, y + 320), fill=LINE, width=2)
        draw.text((x + 28, y + 338), "生成的事实", font=font(18, bold=True), fill=INK_3)
        paragraph(draw, " · ".join(layer["facts"]), (x + 28, y + 364), font(20), card_width - 56, accent, gap=6)
    box(draw, (MARGIN, y + 428, CANVAS[0] - MARGIN, y + 512), fill=SUNKEN, outline=LINE)
    draw.text((MARGIN + 34, y + 458), "事件内核 → 背景订单流 · 做市流动性 · 执行算法 → 限价簿撮合 → 成交与盘口事件 → 配对反事实冲击度量",
              font=font(23, bold=True), fill=INK)
    slide_footer(draw, "任何智能体中都没有写入冲击公式：价格冲击是撮合与流动性补充的结果")
    slides.append(save_slide(image, 3))

    # 4 — stylized fact coverage
    image, draw = canvas()
    y = slide_header(draw, "风格化事实训练与对照",
                     f"{head['facts']} 项与 TCA 直接相关的微观结构事实，逐项对照 {market['venue']} L3 实测参考值。")
    box(draw, (MARGIN, y, CANVAS[0] - MARGIN, y + 296), fill=SURFACE)
    coverage_strip(draw, copy.evidence["facts"], MARGIN + 34, y + 24, CANVAS[0] - 2 * MARGIN - 68)
    box(draw, (MARGIN, y + 328, CANVAS[0] - MARGIN, 900), fill=SURFACE)
    paste_fit(image, FIGURES / "gate_distances.png", (MARGIN + 18, y + 346, 1120, 886))
    draw.text((1170, y + 356), "为什么这一步不能省", font=font(28, bold=True), fill=INK)
    bullet_list(draw, [
        "冲击成本由价差、深度、订单流记忆与流动性恢复速度共同决定，因此这些量必须先对齐。",
        f"归一化偏差介于 {min(row['distance'] for row in copy.evidence['gate']):.3f} 与 "
        f"{head['worst_distance']:.3f} 之间，在未参与训练的交易日与 {head['validation_seeds']} 个独立随机种子上复核。",
        "全部指标的偏差逐项公开，可按同一口径复算。",
    ], 1170, y + 414, 640, size=22, colour=BLUE, gap=20)
    slide_footer(draw, "校准结果文件：outputs/latest_p0p1/stylized_facts.json")
    slides.append(save_slide(image, 4))

    # 5 — emergent square-root impact
    image, draw = canvas()
    y = slide_header(draw, "元订单冲击标度律",
                     "标度律由市场机制自身产生，而非写入的公式，因此对未见过的算法与母单规模同样成立。")
    box(draw, (MARGIN, y, 1180, 900), fill=SURFACE)
    paste_fit(image, FIGURES / "impact_scaling.png", (MARGIN + 20, y + 24, 1160, 880))
    box(draw, (1220, y, CANVAS[0] - MARGIN, 900), fill=SURFACE)
    draw.text((1254, y + 34), "关键读数", font=font(28, bold=True), fill=INK)
    readings = [
        (f"δ = {law['delta']:.4f}", "拟合冲击指数"),
        (f"{law['distance_from_half']:.4f}", "与理论平方根 0.5 的偏差"),
        (f"R² = {law['power_r2']:.6f}", "幂律拟合优度"),
        (f"{law['runs']} 组", "配对干预实验（双向 × 三种时限）"),
    ]
    ry = y + 96
    for value, label in readings:
        draw.text((1254, ry), value, font=font(34, bold=True), fill=BLUE)
        ry = paragraph(draw, label, (1254, ry + 48), font(20), 420, INK_3, gap=6) + 24
    paragraph(draw, "凹性来自订单符号的长记忆与深度补充机制：后续子单的成本已被前序子单部分预告。"
                    "这正是真实市场中平方根律的成因，也是它能外推到新算法的原因。",
              (1254, ry + 8), font(21), 420, INK_2, gap=8)
    slide_footer(draw, "对照组与执行组共用随机数，差异只来自母单本身")
    slides.append(save_slide(image, 5))

    # 6 — product
    image, draw = canvas()
    y = slide_header(draw, "产品形态：可插拔算法的冲击实验台",
                     "Web 应用与 API 同源；市场情景、母单指令与算法注册全部可配置。")
    box(draw, (MARGIN, y, 1120, 900), fill=SURFACE)
    paste_top(image, UI_SCREENSHOT, (MARGIN + 16, y + 16, 1104, 884))
    box(draw, (1160, y, CANVAS[0] - MARGIN, 900), fill=SURFACE)
    draw.text((1194, y + 30), "输出口径", font=font(28, bold=True), fill=INK)
    bullet_list(draw, copy.outputs, 1194, y + 84, 570, size=21, colour=BLUE, gap=14)
    draw.text((1194, y + 400), "接入方式", font=font(28, bold=True), fill=INK)
    bullet_list(draw, [
        "在 liquidation/agents.py 实现 desired_quantity 子单决策函数。",
        "在 app/main.py 的 ALGORITHMS 注册键名与说明。",
        "重新运行即进入同一套配对冲击度量与图表流程。",
    ], 1194, y + 454, 570, size=21, colour=GREEN, gap=14)
    slide_footer(draw, "REST：POST /api/jobs · GET /api/stylized-facts · GET /api/algorithms")
    slides.append(save_slide(image, 6))

    # 7 — measured result
    image, draw = canvas()
    y = slide_header(draw, "评估结果：同一路径下的冲击轨迹与衰减",
                     "薄深度情景，买入 4.5 BTC / 600 秒；两个基准算法在完全相同的市场路径上被度量。")
    box(draw, (MARGIN, y, 1230, 900), fill=SURFACE)
    paste_fit(image, FIGURES / "impact_trajectory.png", (MARGIN + 20, y + 24, 1210, 880))
    box(draw, (1270, y, CANVAS[0] - MARGIN, 900), fill=SURFACE)
    draw.text((1304, y + 34), "结论", font=font(28, bold=True), fill=INK)
    bullet_list(draw, [
        "一次性成交把中间价瞬时推开，冲击峰值显著更高。",
        "均匀拆单把同样的量摊薄到时间轴上，峰值更低但暴露时间更长。",
        "观察窗末端的残余偏离给出冲击留存率，即永久冲击占比。",
        "两者的差异是在同一市场路径上度量的，不含市场自身漂移。",
    ], 1304, y + 96, 400, size=21, colour=BLUE, gap=16)
    paragraph(draw, "平台不预设哪个算法更好：它给出的是同口径、可复现的度量，"
                    "让使用方按自己的风险偏好做取舍。",
              (1304, y + 470), font(21), 400, INK_2, gap=8)
    slide_footer(draw, "冲击与成本口径均相对同种子无执行对照市场")
    slides.append(save_slide(image, 7))

    # 8 — go to market
    image, draw = canvas()
    y = slide_header(draw, "落地路径", "面向执行台与交易团队的离线评估基础设施。")
    thirds = (CANVAS[0] - 2 * MARGIN - 2 * 30) // 3
    for index, (title, items, colour) in enumerate([
        ("目标客户", copy.customers, BLUE),
        ("商业模式", copy.business_model, GREEN),
        ("发展路线", copy.roadmap[:3], ORANGE),
    ]):
        x = MARGIN + index * (thirds + 30)
        box(draw, (x, y, x + thirds, y + 470), fill=SURFACE)
        draw.text((x + 30, y + 28), title, font=font(27, bold=True), fill=INK)
        bullet_list(draw, items, x + 30, y + 82, thirds - 60, size=20, colour=colour, gap=14)
    box(draw, (MARGIN, y + 508, CANVAS[0] - MARGIN, 900), fill=SUNKEN, outline=LINE)
    draw.text((MARGIN + 34, y + 538), "适用范围说明", font=font(27, bold=True), fill=INK)
    bullet_list(draw, copy.limitations, MARGIN + 34, y + 592, CANVAS[0] - 2 * MARGIN - 68, size=21,
                colour=INK_3, gap=12)
    slide_footer(draw, f"团队：{copy.team_name}")
    slides.append(save_slide(image, 8))

    Image.open(slides[0]).save(SUBMISSION / "项目封面_16比9.png", format="PNG")
    return slides


# --------------------------------------------------------------------- text

def write_text_files(copy: Copy) -> None:
    innovations = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(copy.innovations))
    (SUBMISSION / "项目名称.txt").write_text(copy.project_name, encoding="utf-8")
    (SUBMISSION / "团队名称.txt").write_text(copy.team_name, encoding="utf-8")
    (SUBMISSION / "项目摘要.txt").write_text(copy.summary, encoding="utf-8")
    (SUBMISSION / "项目公开介绍.txt").write_text(copy.public_intro, encoding="utf-8")
    (SUBMISSION / "核心创新点.txt").write_text(innovations, encoding="utf-8")

    (SUBMISSION / "必填信息.json").write_text(json.dumps({
        "主赛道建议": copy.track,
        "跨方向定位": copy.cross_track,
        "项目名称": copy.project_name,
        "团队名称": copy.team_name,
        "项目摘要": copy.summary,
        "项目公开介绍": copy.public_intro,
        "核心创新点": copy.innovations,
        "项目封面": "submission/项目封面_16比9.png",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (SUBMISSION / "必填信息_可直接粘贴.md").write_text(textwrap.dedent(f"""\
        # 参赛必填信息（可直接复制粘贴）

        ## 主赛道
        {copy.track}
        （{copy.cross_track}）

        ## 项目名称
        {copy.project_name}

        ## 团队名称
        {copy.team_name}

        ## 项目摘要
        {copy.summary}

        ## 项目公开介绍
        {copy.public_intro}

        ## 核心创新点
        {innovations}

        ## 项目封面（16:9）
        `submission/项目封面_16比9.png`

        ## 可选附件
        - 项目 PDF 说明文档：`submission/项目说明书_智执TCA.pdf`
        - 项目 PPT：`submission/项目路演PPT_智执TCA.pptx`
        - 项目 Demo 说明：`submission/项目Demo说明.md`
        - 核心代码说明：`submission/核心代码说明.md`
        - 演示视频提纲：`submission/视频录制提纲.md`
        """), encoding="utf-8")

    (SUBMISSION / "赛道选择建议.md").write_text(textwrap.dedent(f"""\
        # 赛道选择建议

        - 主报赛道：{copy.track}
        - 辅助定位：{copy.cross_track}

        理由：
        1. 交付物是执行台与交易团队可直接使用的离线评估基础设施，场景落地属性最强。
        2. 多智能体市场与算法接入机制覆盖方向一的智能体建模能力。
        3. 校准、留出验收与配对反事实度量覆盖方向二的量化工具能力。
        """), encoding="utf-8")

    (SUBMISSION / "核心代码说明.md").write_text(textwrap.dedent("""\
        # 核心代码说明

        建议提交整个仓库，评审可按以下顺序阅读：

        | 路径 | 作用 |
        |---|---|
        | `market/` | 事件内核、交易所与价格—时间优先限价簿 |
        | `agents/` | 背景订单流与做市智能体（Hawkes 到达、符号长记忆、重尾成交、多档深度） |
        | `liquidation/agents.py` | 清算算法：实现 `desired_quantity` 即可接入 |
        | `liquidation/runner.py` | 共同随机数配对运行器与中间价路径工具 |
        | `calibration/` | 风格化事实目标函数与 Optuna TPE 搜索 |
        | `analytics/stylized_facts.py` | 风格化事实测量：价差、深度、响应、OFI、符号记忆、尾部、扩散性 |
        | `app/main.py` | FastAPI 服务、算法注册表 `ALGORITHMS`、配对冲击度量 |
        | `app/static/` | 中文前端：校准证据面板与冲击实验台 |
        | `scripts/build_stylized_fact_evidence.py` | 汇总留出验收证据为单一 JSON |
        | `scripts/make_submission_figures.py` | 由证据工件渲染全部提交图表 |
        | `outputs/latest_p0p1/` | 校准、验证、干预实验与诊断的版本化工件 |

        接入一个新的清算算法：

        ```python
        # liquidation/agents.py
        class MyExecutionAgent(ExecutionAgent):
            def desired_quantity(self, kernel, step):
                state = kernel.exchange.book.state()
                return min(self.remaining, ...)   # 你的子单决策

        # liquidation/runner.py -> make_agent 增加分支
        # app/main.py -> ALGORITHMS 增加 {"key": "mine", "label": "MyAlgo", ...}
        ```

        重新运行即可获得同口径的峰值冲击、终点冲击、冲击留存率与配对执行成本。

        测试：`python -m unittest discover -s tests -v`（27 项，覆盖订单簿优先级、撮合、
        确定性、分析口径、校准无泄漏、配对度量与服务路径）。
        """), encoding="utf-8")

    (SUBMISSION / "项目Demo说明.md").write_text(textwrap.dedent("""\
        # Demo 启动说明

        ```powershell
        python -m pip install -r requirements.txt
        python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
        ```

        或在 VS Code 中打开 `app/run_app.py` 并点击 Run Python File，浏览器会自动打开
        `http://127.0.0.1:8000`。

        页面结构：

        1. **多智能体市场架构** —— 背景订单流、做市流动性与执行算法三层的机制与各自生成的风格化事实。
        2. **风格化事实校准** —— 12 项 TCA 相关事实的仿真观测与 Kraken L3 实测参考值逐项对照，
           右侧为元订单冲击标度律与校准偏差。这部分为静态结果，进入页面即可查看。
        3. **清算冲击实验台** —— 选择市场情景（波动率、价差、到达率、挂单深度）与母单指令
           （方向、规模、时限、随机种子），点击「运行冲击评估」。结果给出每个接入算法的峰值冲击、
           终点冲击、冲击留存率、配对执行成本与完成率，以及因果冲击轨迹图与子单成交明细。

        接口：

        ```http
        GET  /api/health
        GET  /api/algorithms       # 已接入的清算算法
        GET  /api/stylized-facts   # 市场架构与风格化事实校准结果
        POST /api/jobs             # 提交一次冲击评估
        GET  /api/jobs/{job_id}    # 轮询进度与结果
        ```

        常见问题：**页面正常但点击「运行冲击评估」报错**。多为 8000 端口上仍有早先启动的
        旧服务进程：浏览器每次都会从磁盘读取最新的页面与脚本，而 `/api` 仍由旧进程应答，
        两者契约不一致。页面顶部会给出红色提示并禁用运行按钮；`app/run_app.py` 也会在启动时
        比对 `build_id` 并打印需要结束的进程号。按提示执行 `taskkill /PID <pid> /F` 后重新运行即可。
        """), encoding="utf-8")

    head, law = copy.head, copy.law
    (SUBMISSION / "视频录制提纲.md").write_text(textwrap.dedent(f"""\
        # 演示视频提纲（建议 3 分钟）

        1. **定位（20 秒）**
           「{copy.project_name}」。一句话：{copy.tagline}。
           说明重点不是买卖决策，而是母单确定之后的执行冲击评估。

        2. **场景（25 秒）**
           执行成本发生在母单确定之后，其大小取决于市场如何回应你的订单，因此市场模型必须先贴近真实微观结构。

        3. **市场架构（40 秒）**
           展示多智能体架构：背景订单流、做市流动性与执行算法三层，以及每层生成的风格化事实；
           强调任何智能体中都没有写入冲击公式。

        4. **校准结果（40 秒）**
           展示 {head['facts']} 项风格化事实与 {copy.market['venue']} L3 实测参考值的逐项对照，
           以及冲击标度律 δ = {law['delta']:.3f}、幂律 R² = {law['power_r2']:.6f}。

        5. **产品演示（45 秒）**
           选择「薄盘口」情景模板，运行冲击评估；对比 Immediate 与 TWAP 的峰值冲击、终点冲击与留存率；
           切换算法查看子单明细。说明标的、数据源与自研算法都可以用同样方式接入。

        6. **落地（20 秒）**
           目标客户、商业模式与扩展方向。
        """), encoding="utf-8")

    (SUBMISSION / "上传清单.md").write_text(textwrap.dedent("""\
        # 上传清单

        必填：
        - `项目名称.txt`
        - `团队名称.txt`
        - `项目摘要.txt`
        - `项目公开介绍.txt`
        - `核心创新点.txt`
        - `项目封面_16比9.png`（1920×1080）

        可选附件：
        - `项目路演PPT_智执TCA.pptx`（8 页）
        - `项目说明书_智执TCA.pdf`
        - `项目Demo说明.md`
        - `核心代码说明.md`
        - `视频录制提纲.md`
        - 仓库代码压缩包

        全部材料由 `python scripts/generate_competition_assets.py` 一键生成，
        数值直接读自 `outputs/latest_p0p1/stylized_facts.json`。
        """), encoding="utf-8")


# ---------------------------------------------------------------------- pdf

def register_fonts() -> tuple[str, str]:
    try:
        pdfmetrics.registerFont(TTFont("YaHei", str(FONT_REGULAR), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("YaHeiBold", str(FONT_BOLD), subfontIndex=0))
        pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold")
        return "YaHei", "YaHeiBold"
    except Exception:  # pragma: no cover - depends on the host's font set
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light"


def sized_image(path: Path, width_cm: float) -> RLImage:
    """Preserve the source aspect ratio - the old deck stretched every figure."""
    with Image.open(path) as image:
        ratio = image.height / image.width
    return RLImage(str(path), width=width_cm * cm, height=width_cm * ratio * cm)


def banded_image(path: Path, top: float, bottom: float, width_cm: float, name: str) -> RLImage:
    """Show one horizontal band of a tall screenshot at its native aspect ratio."""
    with Image.open(path) as image:
        band = image.crop((0, int(image.height * top), image.width, int(image.height * bottom)))
        target = CACHE / f"band_{name}.png"
        band.save(target)
    return sized_image(target, width_cm)


def build_pdf(copy: Copy) -> Path:
    regular, bold = register_fonts()
    ink = colors.HexColor(INK)
    # wordWrap="CJK" is what stops ReportLab from only breaking at spaces, which
    # is why the previous edition had ragged half-empty lines in Chinese prose.
    def style(name, size, leading, font_name=None, colour=ink, **extra):
        return ParagraphStyle(name, fontName=font_name or regular, fontSize=size, leading=leading,
                              textColor=colour, wordWrap="CJK", alignment=TA_LEFT, **extra)

    styles = {
        "title": style("title", 15, 21, bold, spaceBefore=2, spaceAfter=7),
        "h2": style("h2", 11, 16, bold, colors.HexColor(BLUE), spaceBefore=8, spaceAfter=4),
        "body": style("body", 9.2, 15.0, spaceAfter=3),
        "small": style("small", 8, 12, colour=colors.HexColor(INK_3), spaceAfter=2),
        "cell": style("cell", 8.2, 12.6),
        "cellHead": style("cellHead", 8.2, 12.6, bold, colors.white),
        "cellKey": style("cellKey", 8.2, 12.6, bold),
        "caption": style("caption", 8, 11.8, colour=colors.HexColor(INK_3), spaceBefore=2, spaceAfter=8),
    }
    head, law, market = copy.head, copy.law, copy.market
    output = SUBMISSION / "项目说明书_智执TCA.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, title=copy.project_name, author=copy.team_name,
                            leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    width_cm = (A4[0] - 3.4 * cm) / cm

    def table(rows, widths, header=True, key_column=False):
        """Every cell is a Paragraph so long Chinese text wraps instead of overflowing."""
        def cell_style(row_index, column_index):
            if header and row_index == 0:
                return styles["cellHead"]
            if key_column and column_index == 0:
                return styles["cellKey"]
            return styles["cell"]

        data = [[Paragraph(str(cell), cell_style(row_index, column_index))
                 for column_index, cell in enumerate(row)]
                for row_index, row in enumerate(rows)]
        block = Table(data, colWidths=[value * cm for value in widths], repeatRows=1 if header else 0)
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ]
        if header:
            commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
                         ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(SUNKEN)])]
        block.setStyle(TableStyle(commands))
        return block

    story = []

    # --- page 1: identity, summary, headline evidence, innovations
    story.append(sized_image(SLIDES_DIR / "slide_01.png", width_cm))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("一、项目概要", styles["title"]))
    story.append(table([
        ["项目名称", copy.project_name],
        ["团队名称", copy.team_name],
        ["主报赛道", f"{copy.track}（{copy.cross_track}）"],
        ["数据与标的", f"{copy.market['venue']} {copy.market['symbol']} L3 逐笔盘口与成交流"
                        f"（{copy.market['window']}，{copy.market['sessions']} 个交易日）"],
        ["系统形态", copy.tagline],
    ], [2.6, width_cm - 2.6], header=False, key_column=True))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("二、项目摘要", styles["title"]))
    story.append(Paragraph(copy.summary, styles["body"]))
    story.append(Spacer(1, 0.25 * cm))

    # --- problem, method, innovations
    story.append(Paragraph("三、问题与方法", styles["title"]))
    story.append(Paragraph("行业痛点", styles["h2"]))
    for item in copy.pain_points:
        story.append(Paragraph(f"• {item}", styles["body"]))
    story.append(KeepTogether([
        Paragraph("方法链路", styles["h2"]),
        table([["环节", "内容"]] + [[title, desc] for title, desc in copy.pipeline], [3.0, width_cm - 3.0]),
    ]))
    story.append(Paragraph("四、核心创新点", styles["title"]))
    story.append(table([["#", "创新点"]] + [[str(index + 1), item] for index, item in enumerate(copy.innovations)],
                       [1.0, width_cm - 1.0]))

    story.append(Spacer(1, 0.25 * cm))

    # --- the architecture section: why the impact numbers are believable at all
    story.append(Paragraph("五、多智能体市场架构", styles["title"]))
    story.append(Paragraph(
        "系统的核心不是某一条执行策略，而是承载执行的市场本身。一个事件驱动内核以优先队列调度全部智能体，"
        "三类智能体在同一时钟、同一限价簿上共同演化，成交与盘口状态由价格—时间优先撮合产生。"
        "关键设计是：任何智能体中都没有写入价格冲击公式。冲击是母单消耗对手方队列、"
        "做市方按库存与流动压力重新报价、后续订单流沿着已被推动的价格继续到达这一连串机制的结果。"
        "正因为它是涌现的，这条冲击关系才能外推到训练时没有见过的执行算法与母单规模。", styles["body"]))
    story.append(sized_image(FIGURES / "architecture.png", width_cm))
    story.append(Spacer(1, 0.15 * cm))
    story.append(KeepTogether([
        table([["智能体层", "职责与机制", "生成的风格化事实"]] + [
            [layer["name"], "；".join(layer["mechanisms"]), "、".join(layer["facts"])]
            for layer in copy.layers
        ], [3.2, width_cm - 3.2 - 4.6, 4.6]),
        Paragraph("第三层是产品的接口层：执行算法以同样的智能体身份接入，与其他两层共享同一事件时钟与随机数种子。"
                  "更换标的或数据源只需重新执行前两层的标准化与训练流程，评估口径保持不变。", styles["caption"]),
    ]))

    story.append(Spacer(1, 0.25 * cm))

    # --- the fact-by-fact evidence table; it splits across pages with a repeated header
    story.append(Paragraph("六、风格化事实训练与对照", styles["title"]))
    story.append(Paragraph(
        f"下表为与 TCA 直接相关的 {head['facts']} 项微观结构事实，逐项给出仿真观测与 "
        f"{market['venue']} L3 实测参考值。参与训练目标的 {head['gate_metrics']} 项指标另在未参与训练的交易日"
        f"与 {head['validation_seeds']} 个独立随机种子上复核，归一化偏差见 7.2。",
        styles["body"]))
    rows = [["编号", "风格化事实", "为什么影响 TCA", "仿真观测", "实测参考"]]
    for fact in copy.evidence["facts"]:
        rows.append([
            fact["id"],
            f'<b>{fact["name"]}</b><br/><font color="{INK_3}" size="7">{fact["name_en"]}</font>',
            fact["why"],
            f'<b>{fact["value"]}</b><br/><font color="{INK_3}" size="7">{fact["detail"]}</font>',
            fact["target"],
        ])
    story.append(table(rows, [1.75, 3.6, 4.35, 4.55, 2.9]))
    story.append(Paragraph("证据文件：outputs/latest_p0p1/stylized_facts.json，由校准、留出验证、"
                           "配对干预实验与长时段仿真四类版本化工件汇总生成。", styles["caption"]))

    story.append(Spacer(1, 0.25 * cm))

    # --- the two headline figures
    story.append(Paragraph("七、关键实验结果", styles["title"]))
    story.append(Paragraph("7.1 元订单冲击标度律", styles["h2"]))
    story.append(sized_image(FIGURES / "impact_scaling.png", width_cm))
    story.append(Paragraph(
        f"{law['runs']} 组配对干预实验（双向 × {len(law['durations'])} 种时限）拟合得到冲击指数 "
        f"δ = {law['delta']:.4f}，与理论平方根 0.5 相差 {law['distance_from_half']:.4f}，"
        f"幂律拟合 R² = {law['power_r2']:.6f}（线性拟合仅 {law['linear_r2']:.3f}）。"
        f"智能体中没有任何冲击公式，凹性来自订单符号长记忆与深度补充机制。", styles["body"]))
    story.append(Paragraph("7.2 校准偏差", styles["h2"]))
    story.append(sized_image(FIGURES / "gate_distances.png", width_cm))
    story.append(Paragraph(
        f"{len(copy.evidence['gate'])} 项校准指标与实测参考的归一化偏差介于 "
        f"{min(row['distance'] for row in copy.evidence['gate']):.3f} 与 {head['worst_distance']:.3f} 之间"
        f"（参考容差 {head['tolerance']:.1f}）。训练使用 {market['calibration_sessions']} 个交易日、"
        f"{market['search']} 搜索 {market['trials']} 轮；复核使用另外 {market['validation_sessions']} 个"
        f"未参与训练的交易日与 {head['validation_seeds']} 个独立随机种子。", styles["body"]))

    story.append(Spacer(1, 0.25 * cm))

    # --- measurement and product
    story.append(Paragraph("八、冲击度量与产品形态", styles["title"]))
    story.append(Paragraph("8.1 配对反事实度量", styles["h2"]))
    story.append(sized_image(FIGURES / "impact_trajectory.png", width_cm))
    story.append(Paragraph("同一随机种子生成「无执行」对照市场；母单注入后，两条中间价路径在统一时间网格上逐点相减，"
                           "剩余部分即为执行造成的因果冲击。图中为薄深度情景下买入 4.5 BTC / 600 秒的实测轨迹，"
                           "下方独立面板为 TWAP 的子单规模。", styles["body"]))
    story.append(KeepTogether([Paragraph("8.2 输出口径", styles["h2"]), table([["指标", "定义"]] + [
        ["峰值因果冲击", "执行期内中间价相对对照市场的最大偏离（bp）"],
        ["终点冲击", "母单执行完成时刻尚未消散的偏离（bp）"],
        ["冲击留存率", "执行结束 30 秒后的偏离 ÷ 峰值偏离，衡量永久冲击占比"],
        ["配对执行成本", "每笔成交价与对照市场同时刻中间价之差，按母单名义额加权（bp）"],
        ["完成率与子单明细", "逐切片的请求量、成交量、剩余量与该时刻的因果冲击"],
    ], [3.6, width_cm - 3.6])]))
    story.append(Paragraph("8.3 算法接入", styles["h2"]))
    story.append(Paragraph("清算算法是可插拔的：在 <font face=\"Courier\">liquidation/agents.py</font> 中实现子单决策函数 "
                           "<font face=\"Courier\">desired_quantity</font>，并在 "
                           "<font face=\"Courier\">app/main.py</font> 的 <font face=\"Courier\">ALGORITHMS</font> 注册，"
                           "即可进入同一套配对度量与可视化流程。系统当前内置 Immediate 与 TWAP 两个行业基准作为共同标尺，"
                           "平台本身不预设哪个算法更优，只提供同口径、可复现的度量。", styles["body"]))

    story.append(Spacer(1, 0.25 * cm))

    # --- product screenshot
    story.append(Paragraph("九、Web 冲击实验台", styles["title"]))
    story.append(Paragraph("9.1 首屏：定位与校准结论", styles["h2"]))
    story.append(banded_image(UI_SCREENSHOT, 0.0, 0.195, width_cm, "hero"))
    story.append(Paragraph("进入页面即可看到四项关键结论与逐项校准证据；下方为 12 组风格化事实的完整状态表、"
                           "冲击标度律图与留出验收距离条形图。", styles["caption"]))
    story.append(Paragraph("9.2 冲击实验台：算法对比与轨迹", styles["h2"]))
    story.append(banded_image(UI_SCREENSHOT, 0.535, 0.845, width_cm, "lab"))
    story.append(Paragraph("每个接入算法给出峰值冲击、终点冲击、冲击留存率、配对执行成本与完成率；"
                           "下方为因果冲击轨迹（子单规模在独立面板中显示，不与时间轴刻度重叠），"
                           "可切换算法查看其子单成交明细。FastAPI 服务与页面同源："
                           "GET /api/stylized-facts、GET /api/algorithms、POST /api/jobs、GET /api/jobs/{id}。",
                           styles["caption"]))

    story.append(Spacer(1, 0.25 * cm))

    # --- go to market, limits, reproduction
    story.append(Paragraph("十、落地路径", styles["title"]))
    story.append(Paragraph("目标客户", styles["h2"]))
    for item in copy.customers:
        story.append(Paragraph(f"• {item}", styles["body"]))
    story.append(Paragraph("商业模式", styles["h2"]))
    for item in copy.business_model:
        story.append(Paragraph(f"• {item}", styles["body"]))
    story.append(Paragraph("发展路线", styles["h2"]))
    for item in copy.roadmap:
        story.append(Paragraph(f"• {item}", styles["body"]))
    # Keep the closing section whole so a single bullet never widows onto the
    # last page ahead of its own heading.
    story.append(KeepTogether(
        [Paragraph("十一、适用范围与可复现性", styles["title"])]
        + [Paragraph(f"• {item}", styles["body"]) for item in copy.limitations]
        + [Paragraph("复现步骤", styles["h2"]),
           table([["命令", "产物"]] + [
               ["python -m unittest discover -s tests", "27 项回归测试"],
               ["python calibration/calibrate.py", "outputs/latest_p0p1/calibration/"],
               ["python scripts/validate_simulator.py", "outputs/latest_p0p1/validation/"],
               ["python scripts/run_liquidation_experiments.py", "outputs/latest_p0p1/liquidation/"],
               ["python scripts/build_stylized_fact_evidence.py", "stylized_facts.json（服务与文档的唯一数值来源）"],
               ["python scripts/generate_competition_assets.py", "submission/ 全部材料"],
           ], [7.4, width_cm - 7.4]),
           Paragraph("本项目为离线研究与产品原型，不接入实盘，不构成投资建议。", styles["caption"])]))

    doc.build(story)
    return output


# --------------------------------------------------------------------- pptx

def build_pptx(slide_paths: list[Path], copy: Copy) -> Path:
    output = SUBMISSION / "项目路演PPT_智执TCA.pptx"
    count = len(slide_paths)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
""" + "\n".join(
            f'  <Override PartName="/ppt/slides/slide{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for index in range(1, count + 1)) + "\n</Types>")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""")
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        zf.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(copy.project_name)}</dc:title>
  <dc:creator>{escape(copy.team_name)}</dc:creator>
  <cp:lastModifiedBy>{escape(copy.team_name)}</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""")
        zf.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office PowerPoint</Application>
  <Slides>{count}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion>
</Properties>""")
        zf.writestr("ppt/presentation.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1" autoCompressPictures="0">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>
""" + "\n".join(f'    <p:sldId id="{255 + index}" r:id="rId{index + 1}"/>' for index in range(1, count + 1)) + f"""
  </p:sldIdLst>
  <p:sldSz cx="{PPT_SIZE[0]}" cy="{PPT_SIZE[1]}"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>""")
        zf.writestr("ppt/_rels/presentation.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
""" + "\n".join(
            f'  <Relationship Id="rId{index + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{index}.xml"/>'
            for index in range(1, count + 1)) + "\n</Relationships>")
        zf.writestr("ppt/presProps.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>""")
        zf.writestr("ppt/viewProps.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" lastView="sldView">
  <p:normalViewPr showOutlineIcons="0" snapVertSplitter="1"><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr>
  <p:slideViewPr><p:cSldViewPr snapToGrid="1" snapToObjects="1"/></p:slideViewPr>
  <p:notesTextViewPr/><p:gridSpacing cx="72008" cy="72008"/>
</p:viewPr>""")
        zf.writestr("ppt/tableStyles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>""")
        zf.writestr("ppt/slideMasters/slideMaster1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>""")
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""")
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""")
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""")
        zf.writestr("ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>
      <a:accent1><a:srgbClr val="4F81BD"/></a:accent1><a:accent2><a:srgbClr val="C0504D"/></a:accent2>
      <a:accent3><a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4>
      <a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6>
      <a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/><a:extraClrSchemeLst/>
</a:theme>""")
        for index, slide_path in enumerate(slide_paths, start=1):
            zf.write(slide_path, f"ppt/media/image{index}.png")
            zf.writestr(f"ppt/slides/slide{index}.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:pic><p:nvPicPr><p:cNvPr id="2" name="Slide {index}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{PPT_SIZE[0]}" cy="{PPT_SIZE[1]}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>""")
            zf.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image{index}.png"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>""")
    return output


def main() -> None:
    # Deliverable names are Chinese; a cp936/cp1252 console must not abort the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for path in (SUBMISSION, ASSETS, SLIDES_DIR, FIGURES, CACHE):
        path.mkdir(parents=True, exist_ok=True)
    copy = Copy(ensure_inputs())
    for stale in SLIDES_DIR.glob("slide_*.png"):
        stale.unlink()
    slides = build_slides(copy)
    write_text_files(copy)
    pdf = build_pdf(copy)
    pptx = build_pptx(slides, copy)
    print(f"slides   : {len(slides)} -> {SLIDES_DIR.relative_to(ROOT)}")
    print(f"cover    : {(SUBMISSION / '项目封面_16比9.png').relative_to(ROOT)}")
    print(f"pdf      : {pdf.relative_to(ROOT)}")
    print(f"pptx     : {pptx.relative_to(ROOT)}")
    print("text     : 项目名称/团队名称/项目摘要/项目公开介绍/核心创新点 + 必填信息.json")


if __name__ == "__main__":
    main()
