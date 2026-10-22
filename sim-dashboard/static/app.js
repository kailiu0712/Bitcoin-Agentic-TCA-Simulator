// Liquidation replay: settings -> /api/run -> poll /api/status -> /api/replay -> play.
// All charts are drawn once with the whole run; playing moves a "now" line and a cover
// that hides the future, so a frame costs one cheap relayout per chart.

const C = {
  with: "#eb6834", without: "#2a78d6", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781",
  grid: "#e1e0d9", axis: "#c3c2b7", surface: "#fcfcfb",
};
// bids (negative volume) blue, asks (positive) red, gray midpoint
const BOOK_SCALE = [[0, "#104281"], [0.25, "#5598e7"], [0.5, "#f0efec"], [0.75, "#e66767"], [1, "#a3201f"]];
const TIME_CHARTS = ["chart-price", "chart-impact", "chart-depth", "chart-book-liq", "chart-book-cf"];

const $ = (id) => document.getElementById(id);
const form = $("settings");
const fmt = (x, d = 0) => x == null || Number.isNaN(x) ? "–"
  : x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (x, d = 1) => x == null || Number.isNaN(x) ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const SETTINGS = ["model", "seed_index", "minutes", "side", "quantity", "start_min", "duration_min", "slices"];

let opt = null;                // /api/options: models, defaults, market
let R = null;                  // the replay being shown
let pxd = 2;                   // price decimals: enough for one tick
let now = 0, playing = false, lastTick = 0, lastDraw = 0;

// ---------------------------------------------------------------- settings

async function init() {
  opt = await (await fetch("/api/options")).json();
  $("model").innerHTML = opt.models.map((m) => `<option value="${m.key}">${m.label}</option>`).join("");
  $("seed-range").textContent = `0–${opt.n_seeds - 1}`;
  form.seed_index.max = opt.n_seeds - 1;
  form.minutes.max = opt.max_minutes;
  const query = new URLSearchParams(location.search);       // e.g. ?autorun&model=smart_mm
  for (const k of SETTINGS) form[k].value = query.has(k) ? query.get(k) : opt.defaults[k];
  updateHint();
}

function updateHint() {
  const q = Number(form.quantity.value), mk = opt.market;
  $("qty-hint").textContent = `≈ ${fmt(100 * q / mk.daily_volume, 2)}% of daily volume · ≈ $${fmt(q * mk.price)}`;
}

form.quantity.addEventListener("input", updateHint);

// ---------------------------------------------------------------- running

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {};
  for (const k of SETTINGS) body[k] = ["model", "side"].includes(k) ? form[k].value : Number(form[k].value);
  setStatus("Starting…");
  $("run").disabled = true;
  try {
    const res = await fetch("/api/run", { method: "POST", body: JSON.stringify(body) });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error);
    await waitFor(out.id);
    setStatus("Loading the replay…");
    show(await (await fetch(`/api/replay/${out.id}`)).json());
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("run").disabled = false;
    $("progress").hidden = true;
  }
});

async function waitFor(id) {
  $("progress").hidden = false;
  const started = Date.now();
  for (;;) {
    const s = await (await fetch(`/api/status/${id}`)).json();
    if (s.state === "done") return;
    if (s.state !== "running") throw new Error(s.message || "the run failed");
    $("progress-bar").style.width = `${Math.round(100 * s.progress)}%`;
    setStatus(`Simulating both runs… ${Math.round(100 * s.progress)}% · ${clock((Date.now() - started) / 1000)}`);
    await new Promise((r) => setTimeout(r, 500));
  }
}

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

// ---------------------------------------------------------------- charts

const minutes = (a) => a.map((s) => s / 60);

function baseLayout(yTitle, extra = {}) {
  return {
    margin: { l: 64, r: 16, t: 8, b: 40 },
    paper_bgcolor: C.surface, plot_bgcolor: C.surface,
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: C.ink2 },
    xaxis: { title: { text: "Simulated time (min)" }, range: [0, R.meta.total_s / 60], gridcolor: C.grid,
             linecolor: C.axis, zeroline: false },
    yaxis: { title: { text: yTitle }, gridcolor: C.grid, linecolor: C.axis, zeroline: false },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)" },
    hovermode: "x unified", showlegend: true, ...extra,
  };
}

const line = (x, y, name, color, extra = {}) =>
  ({ type: "scatter", mode: "lines", x, y, name, line: { color, width: 2 }, connectgaps: false, ...extra });

function show(replay) {
  R = replay;
  const m = R.meta, t = minutes(R.t), asset = m.asset;
  pxd = Math.max(2, Math.ceil(-Math.log10(m.tick) - 1e-9));
  const L = R.liquidation, CF = R.counterfactual;
  const hitSide = m.side === "sell" ? "bid" : "ask";
  const model = opt.models.find((x) => x.key === m.model);

  document.body.classList.add("loaded");
  $("subtitle").textContent = `Liquidation replay · ${model ? model.label : m.model} · ${m.venue} ${asset}-USD · `
    + `seed index ${m.seed_index} · ${m.side} ${fmt(m.quantity)} ${asset} by TWAP, minute `
    + `${fmt(m.liq_start_s / 60, 1)} to ${fmt(m.liq_end_s / 60, 1)}`;
  $("badge").hidden = !m.identical_before_liquidation;
  $("badge").textContent = "✓ Identical market before the liquidation";
  $("depth-title").textContent = `${hitSide === "bid" ? "Bid" : "Ask"}-side depth, top 10 levels (${asset}, 30 s average)`;

  const cfg = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  const orders = childOrders(R.fills);
  Plotly.newPlot("chart-price", [
    line(t, CF.mid, "Without liquidation", C.without),
    line(t, L.mid, "With liquidation", C.with),
    { type: "scatter", mode: "markers", x: minutes(orders.t), y: orders.vwap, name: "Liquidation orders (VWAP)",
      marker: { color: C.with, size: 9, symbol: "circle", line: { color: C.surface, width: 2 } },
      customdata: orders.size, hovertemplate: `order %{customdata:,.2f} ${asset} at VWAP %{y:.${pxd}f}<extra></extra>` },
  ], baseLayout("USD"), cfg);

  Plotly.newPlot("chart-impact", [
    line(t, R.impact, "Liquidation impact", C.with, { hovertemplate: "%{y:+.1f} bps<extra></extra>" }),
  ], baseLayout("bps", { showlegend: false, yaxis: { title: { text: "bps" }, gridcolor: C.grid, linecolor: C.axis,
                                                       zeroline: true, zerolinecolor: C.axis } }), cfg);

  const w = Math.max(1, Math.round(30 / m.step_s));
  Plotly.newPlot("chart-depth", [
    line(t, trailingMean(CF[`depth_${hitSide}`], w), "Without liquidation", C.without),
    line(t, trailingMean(L[`depth_${hitSide}`], w), "With liquidation", C.with),
  ], baseLayout(asset), cfg);

  // one colour range for both books: the 98th percentile of |volume|
  const vols = [...L.heatmap.flat(), ...CF.heatmap.flat()].filter((v) => v != null).map(Math.abs).sort((a, b) => a - b);
  const zmax = vols[Math.floor(0.98 * (vols.length - 1))] || 1;
  const yRange = [R.price_grid[0] - m.tick, R.price_grid[R.price_grid.length - 1] + m.tick];
  for (const [id, run, showScale] of [["chart-book-liq", L, false], ["chart-book-cf", CF, true]]) {
    const traces = [
      { type: "heatmap", x: t, y: R.price_grid, z: run.heatmap, zmin: -zmax, zmax, zmid: 0,
        colorscale: BOOK_SCALE, showscale: showScale, hoverongaps: false, name: "Book",
        // the scale stops at the 98th percentile: larger levels take the darkest colour ("599+")
        colorbar: { title: { text: `${asset} resting<br>at the price`, side: "top" }, thickness: 10, len: 0.95,
                    tickvals: [-zmax, -zmax / 2, 0, zmax / 2, zmax],
                    ticktext: [`${fmt(zmax)}+ bid`, `${fmt(zmax / 2)} bid`, "0", `${fmt(zmax / 2)} ask`, `${fmt(zmax)}+ ask`] },
        customdata: run.heatmap.map((r) => r.map((v) => v == null ? null : Math.abs(v))),
        hovertemplate: `%{y:.${pxd}f} USD · %{customdata:,.2f} ${asset}<extra></extra>` },
      ...tradeTraces(run.trades, asset),
    ];
    if (run === L) {
      traces.push({ type: "scatter", mode: "markers", x: minutes(R.fills.t), y: R.fills.px, name: "Liquidation fills",
                    marker: { color: C.with, size: 8, line: { color: C.surface, width: 2 } },
                    hovertemplate: `liquidation fill %{y:.${pxd}f} USD<extra></extra>` });
    }
    Plotly.newPlot(id, traces, baseLayout("USD", { hovermode: "closest", yaxis: { title: { text: "USD" },
      range: yRange, gridcolor: C.grid, linecolor: C.axis, zeroline: false } }), cfg);
  }

  // player
  $("scrub").max = m.total_s;
  $("scrub").step = m.step_s;
  $("scrub").disabled = $("play").disabled = false;
  $("speed").value = [10, 30, 60, 120, 300].find((s) => m.total_s / s <= 45) || 300;  // ~45 s per replay
  const query = new URLSearchParams(location.search);
  now = query.has("t") ? Math.min(Math.max(Number(query.get("t")) || 0, 0), m.total_s) : 0;
  render(true);
  play(!query.has("t"));
}

// one marker per child order: its fills share a time; VWAP and total size
function childOrders(f) {
  const out = { t: [], vwap: [], size: [] };
  f.t.forEach((t, i) => {
    const k = out.t.length - 1;
    if (k >= 0 && out.t[k] === t) {
      out.vwap[k] = (out.vwap[k] * out.size[k] + f.px[i] * f.size[i]) / (out.size[k] + f.size[i]);
      out.size[k] += f.size[i];
    } else {
      out.t.push(t); out.vwap.push(f.px[i]); out.size.push(f.size[i]);
    }
  });
  return out;
}

// mean of the last w frames (only the past, so the replay never shows the future)
function trailingMean(a, w) {
  let sum = 0, n = 0;
  return a.map((v, i) => {
    if (v != null) { sum += v; n++; }
    const old = a[i - w];
    if (i >= w && old != null) { sum -= old; n--; }
    return n ? sum / n : null;
  });
}

function tradeTraces(tr, asset) {
  const pick = (side) => tr.side.map((s, i) => s === side ? i : -1).filter((i) => i >= 0);
  return [["buy", "triangle-up", "Buy trades"], ["sell", "triangle-down", "Sell trades"]].map(([side, symbol, name]) => {
    const idx = pick(side);
    return {
      type: "scatter", mode: "markers", name, x: idx.map((i) => tr.t[i] / 60), y: idx.map((i) => tr.px[i]),
      customdata: idx.map((i) => tr.size[i]),
      marker: { symbol, color: C.ink, opacity: 0.75, line: { color: C.surface, width: 1 },
                size: idx.map((i) => Math.min(14, 5 + 2 * Math.sqrt(tr.size[i] / R.meta.order_size))) },
      hovertemplate: `${side} %{customdata:,.2f} ${asset} at %{y:.${pxd}f}<extra></extra>`,
    };
  });
}

// ---------------------------------------------------------------- playback

function shapes(tNow) {
  const m = R.meta, x = tNow / 60, end = m.total_s / 60;
  return [
    // cover: hides what has not happened yet
    { type: "rect", xref: "x", yref: "paper", x0: x, x1: end, y0: 0, y1: 1, fillcolor: C.surface,
      line: { width: 0 }, layer: "above" },
    // liquidation window, drawn over the cover so the plan is visible ahead of time
    { type: "rect", xref: "x", yref: "paper", x0: m.liq_start_s / 60, x1: m.liq_end_s / 60, y0: 0, y1: 1,
      fillcolor: C.with, opacity: 0.08, line: { width: 0 }, layer: "above" },
    { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: C.ink, width: 1 },
      layer: "above" },
  ];
}

function render(force = false) {
  const m = R.meta;
  const i = Math.min(R.t.length - 1, Math.floor(now / m.step_s));
  $("clock").textContent = clock(now);
  $("scrub").value = now;
  const t = performance.now();
  if (force || t - lastDraw > 60) {          // charts at most ~16 times a second
    lastDraw = t;
    for (const id of TIME_CHARTS) Plotly.relayout(id, { shapes: shapes(now) });
  }

  const asset = m.asset;
  let n = 0;
  while (n < R.fills.t.length && R.fills.t[n] <= now) n++;
  const done = n ? R.fills.cum[n - 1] : 0;
  const d = m.quantity < 100 ? 2 : 0;          // decimals for small orders (e.g. BTC)
  $("kpi-done").textContent = `${fmt(100 * done / m.quantity)}%`;
  $("kpi-done-sub").textContent = now >= m.liq_end_s && m.executed < m.quantity - m.lot_size / 2
    ? `${fmt(m.quantity - m.executed, d)} ${asset} unfilled: the book ran out`
    : `${fmt(done, d)} of ${fmt(m.quantity, d)} ${asset}`;
  $("kpi-impact").textContent = `${signed(R.impact[i])} bps`;
  const ended = now >= m.liq_end_s && m.is_bps != null;
  $("kpi-cost").textContent = ended ? `${fmt(m.is_bps, 2)} bps` : "–";
  $("kpi-cost-sub").textContent = ended
    ? `VWAP ${fmt(m.vwap, pxd)} vs arrival mid ${fmt(m.p0, pxd)} USD`
    : "vs the arrival mid; shown when the liquidation ends";
}

let frame = 0;                 // the pending animation frame: one playback loop at a time

function play(on) {
  playing = on && R != null;
  if (playing && now >= R.meta.total_s) now = 0;     // replay from the start
  $("play").textContent = playing ? "❚❚" : "▶";
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  lastTick = performance.now();
  cancelAnimationFrame(frame);
  if (playing) frame = requestAnimationFrame(tick);
}

function tick(ts) {
  if (!playing) return;
  now = Math.min(R.meta.total_s, now + Math.max(0, ts - lastTick) / 1000 * Number($("speed").value));
  lastTick = ts;
  render(now >= R.meta.total_s);
  if (now >= R.meta.total_s) play(false);
  else frame = requestAnimationFrame(tick);
}

$("play").addEventListener("click", () => play(!playing));
$("scrub").addEventListener("input", () => { now = Number($("scrub").value); render(true); });
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && R && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
    e.preventDefault();
    play(!playing);
  }
});

// ?autorun runs the settings on load (instant if cached); any setting can be given too (&model=smart_mm);
// &t=<seconds> opens the replay paused at that time
init()
  .then(() => { if (new URLSearchParams(location.search).has("autorun")) form.requestSubmit(); })
  .catch((err) => setStatus(`Could not reach the server: ${err.message}`, true));
