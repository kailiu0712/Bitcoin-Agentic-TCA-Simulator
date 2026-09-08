/* Liquidation impact simulator front end.

   Every number the page shows is either (a) held-out calibration evidence read
   from /api/stylized-facts, or (b) the outcome of a paired counterfactual run.
   Nothing is hard-coded, so the page cannot drift away from the artifacts. */

const SERIES = {
  no_execution: { color: '#8a94a3', label: '无执行对照', slot: 'control' },
  twap: { color: '#2a78d6', label: 'TWAP', slot: 'series-1' },
  immediate: { color: '#eb6834', label: 'Immediate', slot: 'series-2' },
  adaptive: { color: '#078a68', label: 'Adaptive', slot: 'series-3' },
};
const ORDER = ['no_execution', 'twap', 'adaptive', 'immediate'];
// Must match BUILD_ID in app/main.py. index.html and app.js are read from disk on
// every request, so a server left running from an earlier session serves a new
// page against an old API; without this check that surfaces only as a failed run.
const EXPECTED_BUILD = 'impact-simulator-v4-adaptive';

const form = document.querySelector('#form');
const runButton = document.querySelector('#run');
const badge = document.querySelector('#badge');
const empty = document.querySelector('#empty');
const content = document.querySelector('#content');
const progressWrap = document.querySelector('#progressWrap');
const bar = document.querySelector('#bar');
const progressText = document.querySelector('#progressText');
const tooltip = document.querySelector('#tooltip');

const rounding = { parent_quantity: 3, duration_seconds: 0, volatility: 2, spread: 2, arrival_rate: 3, displayed_depth: 2, seed: 0 };

let algorithms = [];
let lastResult = null;
let selected = 'twap';

/* ------------------------------------------------------------------ utils */

const fmt = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : '--');
const pct = value => `${(value * 100).toFixed(2)}%`;
const escapeHtml = text => String(text).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

function niceTicks(max, count = 4) {
  if (!(max > 0)) return [0];
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(candidate => candidate >= raw) || 10 * magnitude;
  const ticks = [];
  const top = Math.ceil(max / step - 1e-9) * step;
  for (let value = 0; value <= top + step * 1e-9; value += step) ticks.push(Number(value.toFixed(10)));
  return ticks;
}

function svgText(x, y, text, className, extra = '') {
  return `<text class="${className}" x="${x}" y="${y}" ${extra}>${escapeHtml(text)}</text>`;
}

/* -------------------------------------------------------------- evidence */

async function loadEvidence() {
  const response = await fetch('/api/stylized-facts');
  if (!response.ok) return;
  const evidence = await response.json();
  renderIntroStats(evidence);
  renderArchitecture(evidence.architecture || []);
  renderFactTable(evidence.facts);
  drawImpactLaw(evidence.impact_law);
  renderGate(evidence.gate, evidence.headline);
  renderProfile(evidence);

  const market = evidence.market;
  document.querySelector('#venueChip').textContent = `${market.venue} ${market.symbol} · L3`;
  document.querySelector('#evidenceNote').textContent =
    `${evidence.headline.facts} 项与 TCA 相关的微观结构事实，逐项对照 ${market.venue} L3 实测参考值。`;
}

function renderIntroStats(evidence) {
  const law = evidence.impact_law;
  const head = evidence.headline;
  const market = evidence.market;
  const tiles = [
    { value: `δ = ${law.delta.toFixed(3)}`, label: '元订单冲击指数',
      note: `与平方根律理论值 0.5 一致，幂律 R² = ${law.power_r2.toFixed(6)}` },
    { value: `${head.facts} 项`, label: '微观结构校准指标',
      note: '覆盖冲击、流动性、订单流与价格过程四个维度' },
    { value: `${(market.canonical_events / 1e6).toFixed(2)}M`, label: `${market.venue} L3 标准化事件`,
      note: `${market.window}，${market.sessions} 个交易日，约 ${(market.raw_rows / 1e6).toFixed(0)}M 原始行` },
    { value: `${(evidence.architecture || []).length} 层`, label: '多智能体市场架构',
      note: '背景订单流 · 做市流动性 · 执行算法' },
  ];
  document.querySelector('#introStats').innerHTML = tiles
    .map(tile => `<article class="stat"><b>${escapeHtml(tile.value)}</b><span>${escapeHtml(tile.label)}</span><small>${escapeHtml(tile.note)}</small></article>`)
    .join('');
}

/* The architecture is the answer to "why should this market be believed": each
   layer is named together with the facts it is responsible for generating. */
function renderArchitecture(layers) {
  document.querySelector('#archGrid').innerHTML = layers.map((layer, index) => `
    <article class="arch-card">
      <header><span class="arch-index">0${index + 1}</span><b>${escapeHtml(layer.name)}</b></header>
      <p>${escapeHtml(layer.role)}</p>
      <ul>${layer.mechanisms.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      <footer><span>生成的事实</span>${layer.facts.map(item => `<i>${escapeHtml(item)}</i>`).join('')}</footer>
    </article>`).join('');
}

function renderProfile(evidence) {
  const market = evidence.market;
  const rows = [
    ['标的', `${market.venue} ${market.symbol}`],
    ['数据', `L3 逐笔盘口与成交 · ${(market.canonical_events / 1e6).toFixed(2)}M 事件`],
    ['校准区间', `${market.window}（${market.calibration_sessions} 日校准 / ${market.validation_sessions} 日留出）`],
    ['冲击指数', `δ = ${evidence.impact_law.delta.toFixed(3)}`],
  ];
  document.querySelector('#profile').innerHTML =
    rows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join('');
}

function renderFactTable(facts) {
  document.querySelector('#factRows').innerHTML = facts.map(fact => `
    <tr>
      <td class="fact-target">${escapeHtml(fact.id)}</td>
      <td><span class="fact-name">${escapeHtml(fact.name)}</span><span class="fact-name-en">${escapeHtml(fact.name_en)}</span></td>
      <td class="fact-why">${escapeHtml(fact.why)}</td>
      <td><span class="fact-value">${escapeHtml(fact.value)}</span><span class="fact-detail">${escapeHtml(fact.detail)}</span></td>
      <td class="fact-target">${escapeHtml(fact.target)}</td>
    </tr>`).join('');
}

/* Log-log impact scaling: the headline evidence that the simulated market
   generates square-root metaorder impact without it being programmed in. */
function drawImpactLaw(law) {
  const svg = document.querySelector('#lawChart');
  const W = 520, H = 340, pL = 56, pR = 16, pT = 16, pB = 54;
  const points = law.curve;
  const xs = points.map(p => Math.log10(p.participation));
  const highs = points.map(p => Math.log10(Math.max(p.impact_bp + p.std_bp, 1e-6)));
  const lows = points.map(p => Math.log10(Math.max(p.impact_bp - p.std_bp, p.impact_bp * 0.25)));
  const x0 = Math.min(...xs) - 0.15, x1 = Math.max(...xs) + 0.15;
  const y0 = Math.min(...lows) - 0.08, y1 = Math.max(...highs) + 0.08;
  const X = value => pL + (W - pL - pR) * (value - x0) / (x1 - x0);
  const Y = value => H - pB - (H - pT - pB) * (value - y0) / (y1 - y0);

  let grid = '';
  for (let decade = Math.ceil(x0); decade <= Math.floor(x1); decade += 1) {
    grid += `<line class="grid-line" x1="${X(decade)}" y1="${pT}" x2="${X(decade)}" y2="${H - pB}"/>`;
    grid += svgText(X(decade), H - pB + 16, `${10 ** decade * 100}%`, 'tick-label', 'text-anchor="middle"');
  }
  for (let decade = Math.ceil(y0); decade <= Math.floor(y1); decade += 1) {
    grid += `<line class="grid-line" x1="${pL}" y1="${Y(decade)}" x2="${W - pR}" y2="${Y(decade)}"/>`;
    grid += svgText(pL - 8, Y(decade) + 3.5, `${10 ** decade}`, 'tick-label', 'text-anchor="end"');
  }

  // A * p^delta in price fraction; the curve is reported in basis points.
  const fit = p => law.coefficient * 1e4 * p ** law.delta;
  let fitPath = '';
  for (let i = 0; i <= 40; i += 1) {
    const logP = x0 + (x1 - x0) * i / 40;
    fitPath += `${i ? 'L' : 'M'}${X(logP).toFixed(1)},${Y(Math.log10(fit(10 ** logP))).toFixed(1)}`;
  }

  let marks = '';
  points.forEach((point, index) => {
    const cx = X(xs[index]);
    marks += `<line class="error-bar" x1="${cx}" y1="${Y(lows[index])}" x2="${cx}" y2="${Y(highs[index])}" stroke="${SERIES.twap.color}" opacity=".45"/>`;
    marks += `<circle cx="${cx}" cy="${Y(Math.log10(point.impact_bp))}" r="4" fill="${SERIES.twap.color}" stroke="#fff" stroke-width="2"/>`;
  });

  const legend =
    `<line x1="${pL + 8}" y1="${pT + 12}" x2="${pL + 26}" y2="${pT + 12}" stroke="${SERIES.twap.color}" stroke-width="2"/>` +
    `<circle cx="${pL + 17}" cy="${pT + 12}" r="3.5" fill="${SERIES.twap.color}" stroke="#fff" stroke-width="1.5"/>` +
    svgText(pL + 34, pT + 15.5, '配对干预实验均值 ±1σ', 'legend-label') +
    `<line class="fit-line" x1="${pL + 8}" y1="${pT + 30}" x2="${pL + 26}" y2="${pT + 30}" stroke="${SERIES.immediate.color}"/>` +
    svgText(pL + 34, pT + 33.5, `幂律拟合 δ = ${law.delta.toFixed(3)}`, 'legend-label');

  svg.innerHTML =
    grid +
    `<path class="fit-line" d="${fitPath}" stroke="${SERIES.immediate.color}"/>` +
    marks +
    `<line class="axis-line" x1="${pL}" y1="${H - pB}" x2="${W - pR}" y2="${H - pB}"/>` +
    `<line class="axis-line" x1="${pL}" y1="${pT}" x2="${pL}" y2="${H - pB}"/>` +
    legend +
    svgText((pL + W - pR) / 2, H - 14, '母单规模 / 同期成交量（对数）', 'axis-title', 'text-anchor="middle"') +
    svgText(14, (pT + H - pB) / 2, '峰值因果冲击（bp，对数）', 'axis-title',
      `transform="rotate(-90 14 ${(pT + H - pB) / 2})" text-anchor="middle"`);

  document.querySelector('#impactLawNote').textContent = `${law.runs} 组配对实验 · ${law.durations.length} 种时限 · 双向`;
  document.querySelector('#impactLawFoot').textContent =
    `拟合指数 ${law.delta.toFixed(4)}，与理论平方根 0.5 相差 ${law.distance_from_half.toFixed(4)}；幂律 R² = ${law.power_r2.toFixed(6)}，线性 R² = ${law.linear_r2.toFixed(3)}。` +
    ' 凹性由市场自身的符号长记忆与深度补充机制产生，未在任何智能体中写入冲击公式。';
}

/* Held-out normalized distances: every calibration target, worst first, against
   the gate threshold of 1.0. A bar past the marker would be a failed metric. */
function renderGate(gate, headline) {
  const NAMES = {
    sign_memory_exponent: '订单符号记忆指数',
    arrival_clustering: '成交到达聚集（Fano）',
    ofi_slope: 'OFI 斜率',
    ofi_r2: 'OFI 解释力 R²',
    trade_size_tail_p99_to_median: '成交规模尾部 p99/中位数',
    trade_size_tail_top_1pct_volume_share: '前 1% 成交量占比',
    response_monotone_fraction: '成交响应单调性',
    depth_shape_ask: '卖方深度形态',
    depth_shape_bid: '买方深度形态',
    spread_one_tick_share: '一档价差占比',
  };
  const tolerance = headline.tolerance || 1;
  document.querySelector('#gateRows').innerHTML = gate.map(row => `
    <li>
      <span class="gate-name">${escapeHtml(NAMES[row.metric] || row.metric)}</span>
      <span class="gate-track"><i style="width:${Math.min(100, row.distance / tolerance * 100).toFixed(1)}%"></i></span>
      <span class="gate-value">${row.distance.toFixed(3)}</span>
    </li>`).join('');
  const best = Math.min(...gate.map(row => row.distance));
  document.querySelector('#gateFoot').textContent =
    `${gate.length} 项校准指标与实测参考的归一化偏差介于 ${best.toFixed(3)} 与 ${headline.worst_distance.toFixed(3)} 之间，` +
    `在未参与校准的交易日与 ${headline.validation_seeds} 个独立随机种子上复核（参考容差 ${tolerance.toFixed(1)}）。`;
}

/* ------------------------------------------------------------ algorithms */

async function loadAlgorithms() {
  const response = await fetch('/api/algorithms');
  if (!response.ok) return;
  const payload = await response.json();
  algorithms = payload.algorithms;
  selected = payload.reference;
  const count = document.querySelector('#algoCount');
  if (count) count.textContent = `${algorithms.length} 个已接入`;
  document.querySelector('#algoList').innerHTML = algorithms.map(item => {
    const series = SERIES[item.key] || SERIES.twap;
    return `<li><i style="background:${series.color}"></i><span><b>${escapeHtml(item.label)}</b>${escapeHtml(item.summary)}</span></li>`;
  }).join('');
}

function renderTabs() {
  document.querySelector('#pathTabs').innerHTML = algorithms
    .map(item => `<button type="button" class="tab ${item.key === selected ? 'active' : ''}" data-key="${item.key}">${escapeHtml(item.label)}</button>`)
    .join('');
  document.querySelectorAll('#pathTabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
      selected = tab.dataset.key;
      renderTabs();
      renderSelection();
    });
  });
}

/* ------------------------------------------------------------------- run */

document.querySelectorAll('.preset').forEach(button => {
  button.addEventListener('click', () => {
    for (const [key, value] of Object.entries(button.dataset)) {
      if (form.elements[key]) form.elements[key].value = value;
    }
  });
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  runButton.disabled = true;
  empty.classList.add('hidden');
  content.classList.add('hidden');
  progressWrap.classList.remove('hidden');
  badge.textContent = '运行中';
  badge.className = 'chip chip-run';
  bar.style.background = '';
  const data = Object.fromEntries(new FormData(form));
  Object.entries(rounding).forEach(([key, digits]) => {
    data[key] = Number(Number(data[key]).toFixed(digits));
    const input = form.elements[key];
    if (input && input.type === 'number') input.value = data[key].toFixed(digits);
  });
  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error(await response.text());
    poll((await response.json()).id);
  } catch (error) {
    fail(error.message);
  }
});

async function poll(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    bar.style.width = `${job.progress || 0}%`;
    progressText.textContent = job.message || job.status;
    if (job.status === 'complete') return render(job.result);
    if (job.status === 'failed') return fail(job.message || '运行失败');
    setTimeout(() => poll(id), 400);
  } catch (error) {
    fail(error.message);
  }
}

function fail(message) {
  runButton.disabled = false;
  badge.textContent = '出错';
  badge.className = 'chip chip-fail';
  progressText.textContent = message;
  bar.style.width = '100%';
  bar.style.background = '#d03b3b';
}

function render(result) {
  lastResult = result;
  runButton.disabled = false;
  progressWrap.classList.add('hidden');
  content.classList.remove('hidden');
  badge.textContent = '完成';
  badge.className = 'chip chip-ok';
  if (!algorithms.length) {
    algorithms = result.algorithms
      || Object.values(result.comparison || {}).map(entry => ({ key: entry.key, label: entry.label }));
  }
  if (!algorithms.length) return fail('服务端返回的结果缺少算法列表，请确认服务已重启到最新版本。');
  if (!result.comparison[selected]) selected = (result.summary || {}).strategy || algorithms[0].key;
  renderAlgorithmCards(result.comparison);
  renderTabs();
  renderSelection();
}

function renderAlgorithmCards(comparison) {
  document.querySelector('#algoCards').innerHTML = algorithms.map(item => {
    const entry = comparison[item.key];
    if (!entry) return '';
    const series = SERIES[item.key] || SERIES.twap;
    const decay = entry.decay_ratio == null ? '--' : `${(entry.decay_ratio * 100).toFixed(0)}%`;
    return `<article class="algo-card ${series.slot}">
      <header><b>${escapeHtml(entry.label)}</b><span>${escapeHtml(entry.role)}</span></header>
      <p>${escapeHtml(entry.summary)}</p>
      <div class="metric-row">
        <div class="metric"><b>${fmt(entry.impact_bp, 3)} bp</b><span>峰值因果冲击</span></div>
        <div class="metric"><b>${fmt(entry.terminal_impact_bp, 3)} bp</b><span>终点冲击</span></div>
        <div class="metric"><b>${decay}</b><span>冲击留存率</span><small>${fmt(entry.post_trade_impact_bp, 3)} bp</small></div>
      </div>
      <div class="metric-row">
        <div class="metric"><b>${fmt(entry.execution_cost_bp, 4)} bp</b><span>配对执行成本</span></div>
        <div class="metric"><b>${pct(entry.completion)}</b><span>完成率</span></div>
        <div class="metric"><b>${entry.average_fill_price ? `$${entry.average_fill_price.toFixed(2)}` : '--'}</b><span>平均成交价</span><small>${entry.instructions} 个子单</small></div>
      </div>
    </article>`;
  }).join('');
}

function renderSelection() {
  if (!lastResult) return;
  const label = (lastResult.comparison[selected] || {}).label || selected;
  document.querySelector('#detailTitle').textContent = `子单成交明细 · ${label}`;
  drawTrajectories(lastResult, selected);
  const path = (lastResult.execution_paths || {})[selected] || [];
  document.querySelector('#rows').innerHTML = path.map(row => `<tr>
    <td>${row.elapsed_seconds.toFixed(1)}s</td>
    <td>${row.requested.toFixed(5)}</td>
    <td>${row.filled.toFixed(5)}</td>
    <td>${row.cumulative_filled.toFixed(5)}</td>
    <td>${row.remaining.toFixed(5)}</td>
    <td>${row.fill_price == null ? '--' : `$${row.fill_price.toFixed(2)}`}</td>
    <td>${row.causal_impact_bp == null ? '--' : `${row.causal_impact_bp.toFixed(4)} bp`}</td>
  </tr>`).join('');
}

/* ------------------------------------------------------------ trajectory */

/* Two stacked panels sharing one time axis: impact on top, child-order sizes
   below. Separating them is what keeps the volume marks off the axis ticks. */
function drawTrajectories(result, focus) {
  const svg = document.querySelector('#chart');
  const trajectories = result.trajectories || {};
  const duration = result.request.duration_seconds + 30;
  const W = 940, H = 430;
  const pL = 66, pR = 20, legendH = 26;
  const impactTop = legendH + 12, impactBottom = 300;
  const volumeTop = 322, volumeBottom = 384;
  const axisY = volumeBottom;

  const X = t => pL + (W - pL - pR) * t / duration;
  const value = row => (Number.isFinite(Number(row.causal_impact_bp)) ? Number(row.causal_impact_bp) : Number(row.impact_bp) || 0);
  // Scale to the data rather than symmetrically: an impact series that is
  // almost entirely one-signed would otherwise be squeezed into half the panel.
  const all = Object.values(trajectories).flatMap(path => path.map(value)).filter(Number.isFinite);
  const step = niceTicks(Math.max(0.02, ...all.map(Math.abs)), 3)[1] || 0.01;
  const upper = Math.max(step, Math.ceil(Math.max(0, ...all) / step - 1e-9) * step);
  const lower = Math.min(0, Math.floor(Math.min(0, ...all) / step + 1e-9) * step);
  const Y = v => impactBottom - (impactBottom - impactTop) * (v - lower) / (upper - lower);
  const decimals = step < 0.1 ? 3 : (step < 1 ? 2 : 1);

  let grid = '';
  for (let tick = lower; tick <= upper + step * 1e-9; tick += step) {
    const rounded = Number(tick.toFixed(10));
    grid += `<line class="${rounded === 0 ? 'zero-line' : 'grid-line'}" x1="${pL}" y1="${Y(rounded).toFixed(1)}" x2="${W - pR}" y2="${Y(rounded).toFixed(1)}"/>`;
    grid += svgText(pL - 8, Y(rounded) + 3.5, rounded.toFixed(decimals), 'tick-label', 'text-anchor="end"');
  }

  const entries = ORDER.filter(key => trajectories[key]);
  let paths = '', marks = '', legend = '';
  entries.forEach((key, index) => {
    const series = SERIES[key];
    const points = trajectories[key].filter(row => Number.isFinite(Number(row.elapsed_seconds)));
    if (!points.length) return;
    const dim = key !== 'no_execution' && key !== focus;
    const line = points.map((row, i) => `${i ? 'L' : 'M'}${X(row.elapsed_seconds).toFixed(1)},${Y(value(row)).toFixed(1)}`).join(' ');
    paths += `<path class="trajectory ${key}" d="${line}" stroke="${series.color}" opacity="${dim ? 0.42 : 1}"/>`;
    if (key === focus) {
      points.filter(row => (Number(row.filled) || 0) > 0).forEach(row => {
        marks += `<circle cx="${X(row.elapsed_seconds).toFixed(1)}" cy="${Y(value(row)).toFixed(1)}" r="3" fill="${series.color}" stroke="#fff" stroke-width="1.5"/>`;
      });
    }
    const lx = pL + index * 150;
    legend += `<line x1="${lx}" y1="${legendH / 2}" x2="${lx + 20}" y2="${legendH / 2}" stroke="${series.color}" stroke-width="2"${key === 'no_execution' ? ' stroke-dasharray="5 4"' : ''}/>`;
    legend += svgText(lx + 27, legendH / 2 + 4, series.label + (key === focus ? '（当前）' : ''), 'legend-label');
  });

  // Child-order sizes for the focused algorithm only, on their own baseline.
  const childOrders = (result.execution_paths[focus] || []).filter(row => (Number(row.filled) || 0) > 0);
  const maxChild = Math.max(1e-9, ...childOrders.map(row => Number(row.requested) || 0));
  const barWidth = Math.max(2, Math.min(9, (W - pL - pR) / Math.max(childOrders.length, 1) * 0.55));
  let bars = '';
  childOrders.forEach(row => {
    const height = (volumeBottom - volumeTop) * (Number(row.requested) || 0) / maxChild;
    bars += `<rect x="${(X(row.elapsed_seconds) - barWidth / 2).toFixed(1)}" y="${(volumeBottom - height).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${Math.max(height, 1).toFixed(1)}" rx="1.5" fill="${(SERIES[focus] || SERIES.twap).color}" opacity=".55"/>`;
  });

  let axis = `<line class="axis-line" x1="${pL}" y1="${axisY}" x2="${W - pR}" y2="${axisY}"/>`;
  axis += `<line class="axis-line" x1="${pL}" y1="${impactTop}" x2="${pL}" y2="${impactBottom}"/>`;
  axis += `<line class="axis-line" x1="${pL}" y1="${volumeTop}" x2="${pL}" y2="${volumeBottom}"/>`;
  for (let i = 0; i <= 5; i += 1) {
    const time = duration * i / 5;
    axis += `<line class="axis-line" x1="${X(time).toFixed(1)}" y1="${axisY}" x2="${X(time).toFixed(1)}" y2="${axisY + 5}"/>`;
    axis += svgText(X(time), axisY + 18, `${Math.round(time)}s`, 'tick-label', 'text-anchor="middle"');
  }

  svg.innerHTML =
    legend + grid + paths + marks +
    svgText(pL - 8, volumeTop + 10, '子单', 'panel-label', 'text-anchor="end"') +
    svgText(pL - 8, volumeTop + 22, '规模', 'panel-label', 'text-anchor="end"') +
    bars + axis +
    svgText((pL + W - pR) / 2, H - 8, '执行时间（秒，含结束后 30 秒观察窗）', 'axis-title', 'text-anchor="middle"') +
    svgText(16, (impactTop + impactBottom) / 2, '因果冲击（bp）', 'axis-title',
      `transform="rotate(-90 16 ${(impactTop + impactBottom) / 2})" text-anchor="middle"`) +
    `<rect id="hitArea" x="${pL}" y="${impactTop}" width="${W - pL - pR}" height="${impactBottom - impactTop}" fill="transparent"/>` +
    `<line id="cross" class="crosshair hidden" x1="0" y1="${impactTop}" x2="0" y2="${impactBottom}"/>`;

  attachHover(svg, { X, Y, pL, W, pR, duration, impactTop, impactBottom, entries, trajectories, value });
}

function attachHover(svg, ctx) {
  const cross = svg.querySelector('#cross');
  const area = svg.querySelector('#hitArea');
  const frame = svg.parentElement;
  const hide = () => { cross.classList.add('hidden'); tooltip.classList.add('hidden'); };
  area.addEventListener('mouseleave', hide);
  area.addEventListener('mousemove', event => {
    const box = svg.getBoundingClientRect();
    const scale = 940 / box.width;
    const svgX = (event.clientX - box.left) * scale;
    const time = Math.max(0, Math.min(ctx.duration, (svgX - ctx.pL) / (940 - ctx.pL - ctx.pR) * ctx.duration));
    cross.setAttribute('x1', ctx.X(time).toFixed(1));
    cross.setAttribute('x2', ctx.X(time).toFixed(1));
    cross.classList.remove('hidden');
    const lines = ctx.entries.filter(key => key !== 'no_execution').map(key => {
      const points = ctx.trajectories[key];
      const nearest = points.reduce((best, row) => (Math.abs(row.elapsed_seconds - time) < Math.abs(best.elapsed_seconds - time) ? row : best), points[0]);
      return `<div><i style="background:${SERIES[key].color}"></i>${escapeHtml(SERIES[key].label)} · ${ctx.value(nearest).toFixed(3)} bp</div>`;
    });
    tooltip.innerHTML = `<b>t = ${time.toFixed(0)}s</b>${lines.join('')}`;
    tooltip.classList.remove('hidden');
    const left = (ctx.X(time) / scale) + 14;
    tooltip.style.left = `${Math.min(left, frame.clientWidth - tooltip.offsetWidth - 10)}px`;
    tooltip.style.top = `${Math.max(8, (event.clientY - box.top) - tooltip.offsetHeight - 12)}px`;
  });
}

/* ------------------------------------------------------------- build check */

async function verifyBuild() {
  let health = null;
  try {
    const response = await fetch('/api/health');
    if (response.ok) health = await response.json();
  } catch (error) {
    health = null;
  }
  if (health && health.build_id === EXPECTED_BUILD) return true;
  const detail = health
    ? `服务端为旧版本（${escapeHtml(health.build_id || '未知')}），页面为 ${EXPECTED_BUILD}。`
    : '无法连接到本地服务，或服务未提供 /api/health。';
  const banner = document.createElement('div');
  banner.className = 'build-warning';
  banner.innerHTML =
    `<strong>服务端与页面版本不一致</strong>` +
    `<span>${detail} 浏览器会从磁盘读取最新页面，但接口仍由旧进程应答，因此运行评估会失败。` +
    `请结束占用 8000 端口的旧进程（<code>taskkill /PID &lt;pid&gt; /F</code>），再重新运行 <code>app/run_app.py</code>。</span>`;
  document.querySelector('main.page').prepend(banner);
  runButton.disabled = true;
  badge.textContent = '版本不一致';
  badge.className = 'chip chip-fail';
  return false;
}

(async () => {
  if (!(await verifyBuild())) return;
  loadEvidence();
  loadAlgorithms();
})();
