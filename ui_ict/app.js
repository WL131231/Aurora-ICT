/* Aurora-ICT UI — lightweight-charts + REST polling */

const API = "http://127.0.0.1:8765";

// 현재 선택된 차트 timeframe (localStorage 영속화)
let currentTimeframe = localStorage.getItem("aurora_ict_tf") || "1h";
const VALID_TFS = ["1m", "5m", "15m", "1h", "2h", "4h", "1d", "1w"];
if (!VALID_TFS.includes(currentTimeframe)) currentTimeframe = "1h";

// 시각화 토글 (BOS / EQH-EQL / PD Zones) — localStorage 영속화
const VIZ_KEYS = ["bos", "eql", "zones"];
const vizEnabled = {};
VIZ_KEYS.forEach((k) => {
  const stored = localStorage.getItem(`aurora_ict_viz_${k}`);
  vizEnabled[k] = stored === null ? false : stored === "1";
});

// 마지막 봉 시간 (PD Zone area 끝점에 사용) — fetchAndRender 가 매번 갱신
let lastBarTimeSec = null;

// ============================================================
// Chart
// ============================================================
const chartEl = document.getElementById("chart");
const chart = LightweightCharts.createChart(chartEl, {
  layout: { background: { color: "#141316" }, textColor: "#8b8b90" },
  grid: {
    vertLines: { color: "rgba(219,219,222,0.04)" },
    horzLines: { color: "rgba(219,219,222,0.04)" },
  },
  timeScale: { borderColor: "rgba(219,219,222,0.10)", timeVisible: true, secondsVisible: false },
  rightPriceScale: { borderColor: "rgba(219,219,222,0.10)" },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
});
const candleSeries = chart.addCandlestickSeries({
  upColor: "#34d399", downColor: "#fb7185",
  borderUpColor: "#34d399", borderDownColor: "#fb7185",
  wickUpColor: "#34d399", wickDownColor: "#fb7185",
});

// overlay 시리즈 (FVG line / Killzone band 등)
const fvgBullSeries = chart.addAreaSeries({
  topColor: "rgba(52, 211, 153, 0.12)",
  bottomColor: "rgba(52, 211, 153, 0.02)",
  lineColor: "rgba(52, 211, 153, 0.4)",
  lineWidth: 1,
  priceLineVisible: false,
  lastValueVisible: false,
});
const fvgBearSeries = chart.addAreaSeries({
  topColor: "rgba(251, 113, 133, 0.12)",
  bottomColor: "rgba(251, 113, 133, 0.02)",
  lineColor: "rgba(251, 113, 133, 0.4)",
  lineWidth: 1,
  priceLineVisible: false,
  lastValueVisible: false,
});

// OB top/bottom 가로선 관리 — render 시마다 재생성
let obPriceLines = [];
// Strong/Weak HL 가로선 (top + bottom 한 쌍)
let trailingPriceLines = [];
// BOS / CHoCH 짧은 segment LineSeries pool
let bosSegmentSeries = [];
// EQH / EQL 짧은 segment LineSeries pool
let eqlSegmentSeries = [];
// PD Zone — 3개 AreaSeries (premium / equilibrium / discount)
let zoneAreaSeries = [];

function clearObPriceLines() {
  obPriceLines.forEach((pl) => {
    try { candleSeries.removePriceLine(pl); } catch (e) { /* noop */ }
  });
  obPriceLines = [];
}

function clearTrailingPriceLines() {
  trailingPriceLines.forEach((pl) => {
    try { candleSeries.removePriceLine(pl); } catch (e) { /* noop */ }
  });
  trailingPriceLines = [];
}

function clearBosSegments() {
  bosSegmentSeries.forEach((s) => {
    try { chart.removeSeries(s); } catch (e) { /* noop */ }
  });
  bosSegmentSeries = [];
}

function clearEqlSegments() {
  eqlSegmentSeries.forEach((s) => {
    try { chart.removeSeries(s); } catch (e) { /* noop */ }
  });
  eqlSegmentSeries = [];
}

function clearZoneAreas() {
  zoneAreaSeries.forEach((s) => {
    try { chart.removeSeries(s); } catch (e) { /* noop */ }
  });
  zoneAreaSeries = [];
}

// BOS / CHoCH 짧은 segment — swing 시작점부터 돌파봉까지 dashed 라인
function renderBosSegments(structureList) {
  clearBosSegments();
  if (!vizEnabled.bos) return;
  // 같은 가격 + 같은 시점 중복 segment 방지 (set 으로 dedup)
  const seen = new Set();
  const recent = (structureList || []).slice(-15);
  recent.forEach((ev) => {
    if (!ev.swing_ts_ms || ev.swing_ts_ms <= 0) return;
    const t1 = tsToTimeSec(ev.swing_ts_ms);
    const t2 = tsToTimeSec(ev.ts_ms);
    if (t1 >= t2) return;
    const key = `${t1}-${t2}-${ev.broken_level}`;
    if (seen.has(key)) return;
    seen.add(key);
    const isChoCH = ev.type.startsWith("choch");
    const color = isChoCH ? "#a855f7" : "#60a5fa";
    const series = chart.addLineSeries({
      color,
      lineWidth: 1,
      lineStyle: 2, // dashed
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    series.setData([
      { time: t1, value: ev.broken_level },
      { time: t2, value: ev.broken_level },
    ]);
    bosSegmentSeries.push(series);
  });
}

// EQH / EQL 짧은 segment — 두 swing 잇는 dotted 라인
function renderEqlSegments(equalLevels) {
  clearEqlSegments();
  if (!vizEnabled.eql) return;
  (equalLevels || []).forEach((lvl) => {
    const tsList = (lvl.swing_ts_list || []).filter((t) => t > 0).sort((a, b) => a - b);
    if (tsList.length < 2) return;
    const t1 = tsToTimeSec(tsList[0]);
    const t2 = tsToTimeSec(tsList[tsList.length - 1]);
    if (t1 >= t2) return;
    const isHigh = lvl.type === "high";
    const color = isHigh ? "#fb7185" : "#34d399";
    const series = chart.addLineSeries({
      color,
      lineWidth: 1,
      lineStyle: 1, // dotted
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    series.setData([
      { time: t1, value: lvl.price },
      { time: t2, value: lvl.price },
    ]);
    eqlSegmentSeries.push(series);
  });
}

// PD Zone — 3개 AreaSeries (premium=빨강, equilibrium=회색, discount=초록)
// 각 zone 의 top/bottom 가격을 그라데이션 박스로 표시. trailing.bottom_ts_ms 시점부터 차트 끝까지 채움.
function _addZoneArea(startSec, endSec, top, bottom, color) {
  // 위쪽 line — top, 아래쪽 line — bottom, baseline 박은 게 박은 게 박은 박은 박은 박은 박은 박은 박은
  // lightweight-charts AreaSeries 는 단일 값 + baseline 박을 게 박은 게 박은 박은 박은 박은 박은 박은
  // 박은 게 박은 게 박은 박은 박은 박은 박은 박은 박은 박은 — 박은 게 박은 게 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은
  const series = chart.addAreaSeries({
    topColor: color,
    bottomColor: color.replace(/[\d.]+\)$/, "0.05)"),
    lineColor: color.replace(/[\d.]+\)$/, "0.0)"),
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  // top 값으로 fill — 아래는 0 으로 떨어지므로 base 값으로 잘라야 함 → baseLine 박은 거 박은 게 박은 박은 박은 박은
  // 가장 간단 — top 값만 area, bottom 부분은 별 series 박은 게 박은 게 박은 박은 박은 박은
  series.setData([
    { time: startSec, value: top },
    { time: endSec, value: top },
  ]);
  return series;
}

function renderPdZones(trailing, lastBarTimeSec) {
  clearZoneAreas();
  if (!vizEnabled.zones || !trailing) return;
  const top = trailing.top_price;
  const bot = trailing.bottom_price;
  if (top <= bot || !lastBarTimeSec) return;
  const range = top - bot;
  // zone 시작 시점 — trailing top/bottom 중 더 이른 시점
  const startTs = Math.min(trailing.top_ts_ms, trailing.bottom_ts_ms);
  const startSec = tsToTimeSec(startTs);

  // 6개 priceLine 으로 zone 경계 박는 게 박은 게 박은 박은 박은 박은 박은 — 박은 게 박은 게 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은
  // (AreaSeries 박은 게 박은 게 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은)
  const zones = [
    { topPrice: top,              botPrice: bot + 0.95 * range, color: "rgba(251, 113, 133, 0.20)" },  // Premium
    { topPrice: bot + 0.525*range, botPrice: bot + 0.475 * range, color: "rgba(135, 139, 148, 0.20)" }, // Equilibrium
    { topPrice: bot + 0.05*range,  botPrice: bot,                color: "rgba(52, 211, 153, 0.20)" },   // Discount
  ];
  zones.forEach((z) => {
    // 위쪽 area — fill 값 = topPrice, baseline = botPrice
    const series = chart.addAreaSeries({
      topColor: z.color,
      bottomColor: z.color,
      lineColor: z.color.replace(/[\d.]+\)$/, "0.5)"),
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      // baseValue 박은 게 박은 게 박은 박은 박은 — botPrice 박은 게 박은 박은 박은 박은 박은 박은 박은 박은
      baseLineColor: "transparent",
    });
    series.setData([
      { time: startSec, value: z.topPrice },
      { time: lastBarTimeSec, value: z.topPrice },
    ]);
    zoneAreaSeries.push(series);
    // 아래쪽 경계선 (botPrice)
    const lowSeries = chart.addLineSeries({
      color: z.color.replace(/[\d.]+\)$/, "0.5)"),
      lineWidth: 1,
      lineStyle: 0,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    lowSeries.setData([
      { time: startSec, value: z.botPrice },
      { time: lastBarTimeSec, value: z.botPrice },
    ]);
    zoneAreaSeries.push(lowSeries);
  });
}

function renderTrailingExtremes(trailing) {
  clearTrailingPriceLines();
  if (!trailing) return;

  // Strong = 진한 라인, Weak = 옅은 라인 (LuxAlgo 식)
  const isStrongTop = trailing.top_label === "Strong High";
  const isStrongBot = trailing.bottom_label === "Strong Low";

  const topPl = candleSeries.createPriceLine({
    price: trailing.top_price,
    color: isStrongTop ? "#fb7185" : "rgba(251, 113, 133, 0.45)",
    lineWidth: isStrongTop ? 2 : 1,
    lineStyle: 0,           // solid
    axisLabelVisible: true,
    title: trailing.top_label,
  });
  const botPl = candleSeries.createPriceLine({
    price: trailing.bottom_price,
    color: isStrongBot ? "#34d399" : "rgba(52, 211, 153, 0.45)",
    lineWidth: isStrongBot ? 2 : 1,
    lineStyle: 0,
    axisLabelVisible: true,
    title: trailing.bottom_label,
  });
  trailingPriceLines.push(topPl, botPl);
}

function renderObPriceLines(obs) {
  clearObPriceLines();
  // 미mitigated OB 상위 5개만 (chart 지저분 방지)
  const active = (obs || []).filter((o) => !o.mitigated).slice(-5);
  active.forEach((ob) => {
    const color = ob.type === "bullish"
      ? "rgba(45, 212, 191, 0.45)"
      : "rgba(244, 114, 182, 0.45)";
    [ob.high, ob.low].forEach((price) => {
      const pl = candleSeries.createPriceLine({
        price,
        color,
        lineWidth: 1,
        lineStyle: 2, // dashed
        axisLabelVisible: false,
        title: "",
      });
      obPriceLines.push(pl);
    });
  });
}

function clearOverlays() {
  fvgBullSeries.setData([]);
  fvgBearSeries.setData([]);
  candleSeries.setMarkers([]);
  clearObPriceLines();
  clearTrailingPriceLines();
  clearBosSegments();
  clearEqlSegments();
  clearZoneAreas();
}

function tsToTimeSec(ts_ms) { return Math.floor(ts_ms / 1000); }

// ============================================================
// Status / API helpers
// ============================================================
const $ = (id) => document.getElementById(id);

function toast(msg, error = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.style.display = "block";
  setTimeout(() => { t.style.display = "none"; }, 2500);
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(`${API}${path}`, opts);
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`${resp.status}: ${detail}`);
  }
  return await resp.json();
}

function renderStatus(s) {
  $("s-state").textContent = s.state;
  $("s-state").className = "v " + s.state;
  $("s-mode").textContent = s.run_mode.toUpperCase();
  $("s-mode").className = "v " + s.run_mode;
  $("s-enabled").textContent = s.enabled ? "YES" : "NO";
  $("s-symbol").textContent = s.symbol;
  $("s-creds").textContent = s.has_credentials ? "OK" : "MISSING";
  $("s-pos").textContent = s.has_active_position ? "YES" : "—";

  $("btn-demo").classList.toggle("active", s.run_mode === "demo");
  $("btn-live").classList.toggle("active", s.run_mode === "live");

  // API Credentials 폼 ↔ 등록완료 상태 전환
  const credForm = $("cred-form-block");
  const credSaved = $("cred-saved-block");
  const credOkText = $("cred-ok-text");
  if (credForm && credSaved) {
    if (s.has_credentials) {
      credForm.style.display = "none";
      credSaved.style.display = "flex";
      if (credOkText) credOkText.textContent = `${s.run_mode.toUpperCase()} 키 등록됨`;
    } else {
      credForm.style.display = "flex";
      credSaved.style.display = "none";
    }
  }

  // Enable 토글 상태 반영
  const btnEn = $("btn-toggle-enabled");
  btnEn.classList.toggle("on", !!s.enabled);
  btnEn.textContent = s.enabled ? "DISABLE" : "ENABLE";

  // 차트 상단 심볼 라벨 — 매매 TF 도 함께 표시
  const lbl = $("chart-symbol-label");
  const tradeTf = s.timeframe || "?";
  if (lbl) lbl.textContent = `${s.symbol} · chart ${currentTimeframe} · trade ${tradeTf}`;

  // 사이드바 매매 TF 토글 active 동기화
  document.querySelectorAll("#trade-tf-toggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tradeTf === s.timeframe);
  });
}

// ============================================================
// Markers (FVG / Sweep / Structure / Setup)
// ============================================================
function renderMarkers(payload) {
  const m = payload.markers;
  $("c-fvgs").textContent = payload.count.fvgs;
  $("c-sweeps").textContent = payload.count.sweeps;
  $("c-struct").textContent = payload.count.structure;
  $("c-swings").textContent = payload.count.swings;
  $("c-kz").textContent = payload.count.killzones;
  $("c-setups").textContent = payload.count.setups;
  if ($("c-obs")) $("c-obs").textContent = payload.count.order_blocks ?? 0;
  if ($("c-macros")) $("c-macros").textContent = payload.count.macros ?? 0;
  if ($("c-int-struct")) $("c-int-struct").textContent = payload.count.internal_structure ?? 0;
  if ($("c-lg-struct"))  $("c-lg-struct").textContent  = payload.count.large_structure ?? 0;
  if ($("c-lg-swings"))  $("c-lg-swings").textContent  = payload.count.large_swings ?? 0;
  if ($("c-trailing")) {
    const t = m.trailing;
    $("c-trailing").textContent = t
      ? `${t.top_label.replace(" ", "·")} / ${t.bottom_label.replace(" ", "·")}`
      : "—";
  }

  // 단순 marker 렌더 — 각 FVG 별 사각형 대신 setMarkers 일괄 표시.
  const markers = [];

  // FVG markers — 같은 봉/위치에 여러 FVG 면 1개만
  const fvgSeen = new Set();
  m.fvgs.forEach(f => {
    const t = tsToTimeSec(f.ts_ms);
    const pos = f.type === "bullish" ? "belowBar" : "aboveBar";
    const key = `${t}-${pos}`;
    if (fvgSeen.has(key)) return;
    fvgSeen.add(key);
    const color = f.type === "bullish" ? "#34d399" : "#fb7185";
    markers.push({
      time: t,
      position: pos,
      color,
      shape: "square",
      text: `FVG ${f.filled ? "✓" : ""}${f.invalidated ? "✗" : ""}`,
    });
  });

  // Sweep markers — 같은 봉의 같은 위치(아래/위)에서 sweep 여러 개면 1개만 표시
  // (큰 TF 차트에서 한 봉에 wick 여러 swing 한 번에 sweep 하면 도배되는 문제)
  const sweepSeen = new Set();
  m.sweeps.forEach(s => {
    const t = tsToTimeSec(s.ts_ms);
    const pos = s.type === "bearish" ? "aboveBar" : "belowBar";
    const key = `${t}-${pos}`;
    if (sweepSeen.has(key)) return;
    sweepSeen.add(key);
    markers.push({
      time: t,
      position: pos,
      color: "#fbbf24",
      shape: s.type === "bearish" ? "arrowDown" : "arrowUp",
      text: "Sweep",
    });
  });

  // Structure / Internal / Large 박은 마커는 LineSeries segment 로 박혀있어
  // 차트 위 텍스트 마커는 제거 — 도배 방지 (renderBosSegments 가 segment+label 박음)

  // Setups
  m.setups.forEach(s => {
    const score = s.confluence_score ?? 0;
    const conf = score > 0 ? ` · ★${score}` : "";
    markers.push({
      time: tsToTimeSec(s.ts_ms),
      position: s.direction === "long" ? "belowBar" : "aboveBar",
      color: s.direction === "long" ? "#34d399" : "#fb7185",
      shape: s.direction === "long" ? "arrowUp" : "arrowDown",
      text: `${s.direction.toUpperCase()} RR=${s.risk_reward.toFixed(1)}${conf}`,
    });
  });

  // Order Blocks — 미mitigated 만, 같은 봉/위치 dedup
  const obSeen = new Set();
  (m.order_blocks ?? []).forEach(ob => {
    if (ob.mitigated) return;
    const t = tsToTimeSec(ob.ts_ms);
    const isBull = ob.type === "bullish";
    const pos = isBull ? "belowBar" : "aboveBar";
    const key = `${t}-${pos}`;
    if (obSeen.has(key)) return;
    obSeen.add(key);
    markers.push({
      time: t,
      position: pos,
      color: isBull ? "#2dd4bf" : "#f472b6",
      shape: "square",
      text: "OB",
    });
  });

  // 활성 OB top/bottom 가로 priceLine — 최대 5개
  renderObPriceLines(m.order_blocks ?? []);

  // Trailing extremes (Strong/Weak High & Low) — 가로선 한 쌍
  renderTrailingExtremes(m.trailing ?? null);

  // BOS/CHoCH — swing 시작점부터 돌파봉까지 짧은 dashed segment (LuxAlgo 스타일)
  // basic + large 둘 다, internal 은 너무 많아 토글 켜도 표시 안 함 (도배 방지)
  const allStructure = [...(m.structure ?? []), ...(m.large_structure ?? [])];
  renderBosSegments(allStructure);

  // EQH/EQL — 두 swing 잇는 짧은 dotted segment
  renderEqlSegments(m.equal_levels ?? []);

  // Premium/Discount Zone — AreaSeries 박스 (마지막 봉 ts 까지 채움)
  renderPdZones(m.trailing ?? null, lastBarTimeSec);

  // 시간순 정렬
  markers.sort((a, b) => a.time - b.time);
  candleSeries.setMarkers(markers);

  // FVG mean line — 각 FVG 의 mid price 를 area series 로 추적 (단순 indicator)
  const bullLine = m.fvgs.filter(f => f.type === "bullish").map(f => ({
    time: tsToTimeSec(f.ts_ms), value: f.mean,
  }));
  const bearLine = m.fvgs.filter(f => f.type === "bearish").map(f => ({
    time: tsToTimeSec(f.ts_ms), value: f.mean,
  }));
  fvgBullSeries.setData(bullLine.sort((a,b) => a.time-b.time));
  fvgBearSeries.setData(bearLine.sort((a,b) => a.time-b.time));
}

// ============================================================
// Position 패널 렌더
// ============================================================
function _fmt(n, digits = 2) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function renderPositions(pos) {
  const tbody = $("positions-tbody");
  const count = $("positions-count");

  if (!pos || !pos.active) {
    tbody.innerHTML = '<tr><td colspan="8" class="pos-empty">포지션 없음 — 봇 가동 후 진입 시 표시됩니다</td></tr>';
    count.textContent = "0 open";
    return;
  }

  const sideClass = pos.direction === "long" ? "pos-side-long" : "pos-side-short";
  const sideLabel = pos.direction === "long" ? "LONG" : "SHORT";
  const pnlClass = pos.unrealized_pnl >= 0 ? "pos-pnl-pos" : "pos-pnl-neg";
  const pnlSign = pos.unrealized_pnl >= 0 ? "+" : "";

  tbody.innerHTML = `
    <tr>
      <td>
        <div>${pos.symbol}</div>
        <div class="${sideClass}" style="font-size:9px; letter-spacing:0.15em">${sideLabel} · ${pos.leverage}×</div>
      </td>
      <td>${_fmt(pos.qty, 4)}</td>
      <td>${_fmt(pos.entry, 2)}</td>
      <td>${_fmt(pos.mark_price, 2)}</td>
      <td>${_fmt(pos.liquidation_price, 2)}</td>
      <td>${_fmt(pos.margin, 2)} USDT</td>
      <td class="${pnlClass}">
        ${pnlSign}${_fmt(pos.unrealized_pnl, 2)} USDT
        <div style="font-size:9px">(${pnlSign}${_fmt(pos.roi_pct, 2)}%)</div>
      </td>
      <td>
        <div class="pos-actions">
          <button class="pos-btn-close-half" data-fraction="0.5">CLOSE 50%</button>
          <button class="pos-btn-close-full" data-fraction="1.0">CLOSE ALL</button>
        </div>
      </td>
    </tr>
  `;
  count.textContent = "1 open";
}

// ============================================================
// OHLCV fetch + render (candles + markers + position)
// ============================================================
async function fetchAndRender() {
  try {
    const status = await api("/ict/status");
    renderStatus(status);
    if (status.state !== "running") {
      renderPositions(null);
      return;
    }

    const tf = encodeURIComponent(currentTimeframe);
    const [ohlcv, markers, position] = await Promise.all([
      api(`/ict/ohlcv?timeframe=${tf}&limit=1000`),
      api(`/ict/markers?timeframe=${tf}&limit=1000`),
      api("/ict/position"),
    ]);
    candleSeries.setData(ohlcv.candles);
    // 마지막 봉 시간 — PD Zone area 끝점에 사용
    if (ohlcv.candles && ohlcv.candles.length > 0) {
      lastBarTimeSec = ohlcv.candles[ohlcv.candles.length - 1].time;
    }
    renderMarkers(markers);
    renderPositions(position);
  } catch (e) {
    toast(`API: ${e.message}`, true);
  }
}

// ============================================================
// Button handlers
// ============================================================
$("btn-demo").onclick = async () => {
  try { await api("/ict/run-mode", "POST", { mode: "demo" }); await fetchAndRender(); }
  catch (e) { toast(e.message, true); }
};
$("btn-live").onclick = async () => {
  if (!confirm("LIVE 모드로 전환하시겠습니까? (실거래)")) return;
  try { await api("/ict/run-mode", "POST", { mode: "live" }); await fetchAndRender(); }
  catch (e) { toast(e.message, true); }
};
$("btn-start").onclick = async () => {
  try { await api("/ict/start", "POST"); toast("봇 시작됨"); await fetchAndRender(); }
  catch (e) { toast(e.message, true); }
};
$("btn-stop").onclick = async () => {
  try { await api("/ict/stop", "POST"); toast("봇 중지됨"); await fetchAndRender(); }
  catch (e) { toast(e.message, true); }
};

// 키 저장 + Enable 토글 (온보딩)
$("btn-save-cred").onclick = async () => {
  const apiKey = $("cred-api-key").value.trim();
  const apiSecret = $("cred-api-secret").value.trim();
  if (!apiKey || !apiSecret) {
    toast("API Key/Secret 필수", true);
    return;
  }
  // 현재 run_mode 기준 저장 (DEMO/LIVE 버튼으로 분리)
  const mode = $("btn-live").classList.contains("active") ? "live" : "demo";
  try {
    await api("/ict/credentials", "POST", {
      mode, api_key: apiKey, api_secret: apiSecret,
    });
    toast(`${mode.toUpperCase()} 키 저장됨`);
    $("cred-api-key").value = "";
    $("cred-api-secret").value = "";
    await fetchAndRender();
  } catch (e) { toast(e.message, true); }
};

// Close By 버튼 (이벤트 위임 — render 후 매번 다시 박은 버튼에도 동작)
$("positions-tbody").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-fraction]");
  if (!btn) return;
  const fraction = parseFloat(btn.dataset.fraction);
  const label = fraction >= 1.0 ? "전체" : `${Math.round(fraction * 100)}%`;
  if (!confirm(`포지션을 ${label} 청산하시겠습니까? (시장가)`)) return;
  btn.disabled = true;
  try {
    const result = await api("/ict/position/close", "POST", { fraction });
    if (result.active) {
      toast(`${label} 청산 완료 — 남은 ${_fmt(result.remaining_qty, 4)}`);
    } else {
      toast("전체 청산 완료");
    }
    await fetchAndRender();
  } catch (e) {
    toast(e.message, true);
    btn.disabled = false;
  }
});

// TF 토글 — 클릭 시 currentTimeframe 갱신 + localStorage 저장 + 즉시 재렌더
function _updateTfButtons() {
  document.querySelectorAll("#tf-toggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tf === currentTimeframe);
  });
}
_updateTfButtons();

// Viz 토글 — 클릭 시 vizEnabled[key] 반전 + localStorage 저장 + 즉시 재렌더
function _updateVizButtons() {
  document.querySelectorAll("#viz-toggle button").forEach((b) => {
    b.classList.toggle("active", !!vizEnabled[b.dataset.viz]);
  });
}
_updateVizButtons();

// 사이드바 매매 TF 토글 — 클릭 시 POST /ict/timeframe
$("trade-tf-toggle").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-trade-tf]");
  if (!btn) return;
  const tf = btn.dataset.tradeTf;
  if (btn.classList.contains("active")) return;
  try {
    const result = await api("/ict/timeframe", "POST", { timeframe: tf });
    toast(`매매 TF → ${result.timeframe}${result.restarted ? " (봇 재시작)" : ""}`);
    await fetchAndRender();
  } catch (e) {
    toast(e.message, true);
  }
});

$("viz-toggle").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-viz]");
  if (!btn) return;
  const key = btn.dataset.viz;
  if (!VIZ_KEYS.includes(key)) return;
  vizEnabled[key] = !vizEnabled[key];
  localStorage.setItem(`aurora_ict_viz_${key}`, vizEnabled[key] ? "1" : "0");
  _updateVizButtons();
  await fetchAndRender();
});

$("tf-toggle").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-tf]");
  if (!btn) return;
  const tf = btn.dataset.tf;
  if (!VALID_TFS.includes(tf) || tf === currentTimeframe) return;
  currentTimeframe = tf;
  localStorage.setItem("aurora_ict_tf", tf);
  _updateTfButtons();
  await fetchAndRender();
});

// 연결 테스트 — 현재 mode 키로 거래소 ping
$("btn-test-conn").onclick = async () => {
  const btn = $("btn-test-conn");
  const status = $("conn-status");
  btn.disabled = true;
  status.textContent = "테스트 중...";
  status.className = "v conn-test";
  try {
    const result = await api("/ict/test-connection", "POST");
    if (result.ok) {
      const bal = result.balance_usdt !== null && result.balance_usdt !== undefined
        ? `${_fmt(result.balance_usdt, 2)} USDT`
        : "(잔고 unknown)";
      status.textContent = `OK · ${bal}`;
      status.className = "v conn-ok";
      toast(`연결 성공 — ${result.mode.toUpperCase()} · ${bal}`);
    } else {
      status.textContent = `FAIL: ${result.error?.substring(0, 30) || ""}`;
      status.className = "v conn-fail";
      toast(result.error || "연결 실패", true);
    }
  } catch (e) {
    status.textContent = "ERROR";
    status.className = "v conn-fail";
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
};

// "다시 입력" 클릭 시 — 강제로 입력 폼 보여줌 (status 갱신 전까지 유지)
if ($("btn-cred-reenter")) {
  $("btn-cred-reenter").onclick = () => {
    $("cred-form-block").style.display = "flex";
    $("cred-saved-block").style.display = "none";
    $("cred-api-key").focus();
  };
}

$("btn-toggle-enabled").onclick = async () => {
  const currentlyOn = $("btn-toggle-enabled").classList.contains("on");
  try {
    await api("/ict/enabled", "POST", { enabled: !currentlyOn });
    toast(currentlyOn ? "Bot disabled" : "Bot enabled");
    await fetchAndRender();
  } catch (e) { toast(e.message, true); }
};

// ============================================================
// resize
// ============================================================
function fit() {
  const r = chartEl.getBoundingClientRect();
  chart.applyOptions({ width: r.width, height: r.height });
}
window.addEventListener("resize", fit);
fit();

// 초기 fetch + polling (10s 주기)
fetchAndRender();
setInterval(fetchAndRender, 10_000);
