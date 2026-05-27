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
  rightPriceScale: {
    borderColor: "rgba(219,219,222,0.10)",
    autoScale: true,
    mode: LightweightCharts.PriceScaleMode.Normal,
  },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  // 가격 축 드래그로 상하 확대/축소 활성화 + 휠 스크롤로 시간축 확대
  handleScale: {
    mouseWheel: true,
    pinch: true,
    axisPressedMouseMove: { time: true, price: true },
    axisDoubleClickReset: true,
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: true,
  },
});
const candleSeries = chart.addCandlestickSeries({
  upColor: "#34d399", downColor: "#fb7185",
  borderUpColor: "#34d399", borderDownColor: "#fb7185",
  wickUpColor: "#34d399", wickDownColor: "#fb7185",
});

// 지정가 미체결 라인 — bot.pending_entry 있으면 candleSeries.createPriceLine 으로 표시.
// 매 polling 마다 비교해 추가/갱신/제거 (Bybit 차트의 limit 주문 라인 모사).
let pendingPriceLine = null;
let pendingLineKey = null;  // "${price}|${qty}|${direction}" — 같은 키면 재생성 skip

// OB 박스 — BaselineSeries 풀 (각 활성 OB 마다 1개)
let obBoxSeries = [];
// FVG 박스 — BaselineSeries 풀
let fvgBoxSeries = [];
// IFVG (Inversion FVG) 박스 — BaselineSeries 풀 (대시 라인 + 더 진한 fill)
let ifvgBoxSeries = [];
// Breaker Block 박스 — BaselineSeries 풀 (보라 계열로 OB/IFVG 와 차별)
let breakerBoxSeries = [];
// Sweep zone 박스 — BaselineSeries 풀 (sweep wick 영역, retest 대기)
let sweepZoneBoxSeries = [];
// Strong/Weak HL 가로선 (top + bottom 한 쌍)
let trailingPriceLines = [];
// BOS / CHoCH 짧은 segment LineSeries pool
let bosSegmentSeries = [];
// EQH / EQL 짧은 segment LineSeries pool
let eqlSegmentSeries = [];
// PD Zone — BaselineSeries 3개 (premium / equilibrium / discount)
let zoneBoxSeries = [];

function clearObBoxes() {
  obBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  obBoxSeries = [];
}

function clearFvgBoxes() {
  fvgBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  fvgBoxSeries = [];
}

function clearIfvgBoxes() {
  ifvgBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  ifvgBoxSeries = [];
}

function renderIfvgBoxes(ifvgs) {
  clearIfvgBoxes();
  if (!lastBarTimeSec) return;
  // IFVG = 반전 zone — 진한 색으로 차별화
  const active = (ifvgs || []).slice(-6);
  active.forEach((ifvg) => {
    const isBull = ifvg.type === "bullish";
    const fill = isBull ? "rgba(34, 197, 94, 0.28)" : "rgba(239, 68, 68, 0.28)";
    const line = isBull ? "rgba(34, 197, 94, 0.75)" : "rgba(239, 68, 68, 0.75)";
    const startSec = tsToTimeSec(ifvg.ts_ms);
    if (startSec >= lastBarTimeSec) return;
    ifvgBoxSeries.push(_addBoxSeries(startSec, lastBarTimeSec, ifvg.high, ifvg.low, fill, line));
  });
}

function clearBreakerBoxes() {
  breakerBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  breakerBoxSeries = [];
}

function clearSweepZones() {
  sweepZoneBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  sweepZoneBoxSeries = [];
}

function renderSweepZones(sweeps) {
  clearSweepZones();
  if (!lastBarTimeSec) return;
  // 최근 6개 sweep 의 wick zone (swept_price ~ wick_price) 박스 표시
  // retested=true → 옅게 (이미 활용), false → 진하게 (대기 중)
  const recent = (sweeps || []).slice(-6);
  recent.forEach((sw) => {
    if (!sw.zone_top || !sw.zone_bottom) return;
    const isBull = sw.type === "bullish";
    const alpha = sw.retested ? 0.10 : 0.22;
    const lineAlpha = sw.retested ? 0.40 : 0.65;
    const fill = isBull
      ? `rgba(245, 158, 11, ${alpha})`
      : `rgba(245, 158, 11, ${alpha})`;
    const line = isBull
      ? `rgba(245, 158, 11, ${lineAlpha})`
      : `rgba(245, 158, 11, ${lineAlpha})`;
    const startSec = tsToTimeSec(sw.ts_ms);
    if (startSec >= lastBarTimeSec) return;
    sweepZoneBoxSeries.push(_addBoxSeries(startSec, lastBarTimeSec, sw.zone_top, sw.zone_bottom, fill, line));
  });
}

function renderBreakerBoxes(breakers) {
  clearBreakerBoxes();
  if (!lastBarTimeSec) return;
  // Breaker = OB close break 후 반전 zone — 보라 계열로 OB/IFVG 와 차별
  const active = (breakers || []).slice(-6);
  active.forEach((bb) => {
    const isBull = bb.type === "bullish";
    // bullish breaker = 자주색-청록 계열, bearish = 자주색-주황 계열
    const fill = isBull ? "rgba(168, 85, 247, 0.22)" : "rgba(217, 70, 239, 0.22)";
    const line = isBull ? "rgba(168, 85, 247, 0.70)" : "rgba(217, 70, 239, 0.70)";
    const startSec = tsToTimeSec(bb.ts_ms);
    if (startSec >= lastBarTimeSec) return;
    breakerBoxSeries.push(_addBoxSeries(startSec, lastBarTimeSec, bb.high, bb.low, fill, line));
  });
}

function clearTrailingPriceLines() {
  trailingPriceLines.forEach((pl) => {
    try { candleSeries.removePriceLine(pl); } catch (e) { /* noop */ }
  });
  trailingPriceLines = [];
}

// 공통 헬퍼 — BaselineSeries 박스 1개 (시작 ts ~ 끝 ts, top → baseline=bottom fill)
function _addBoxSeries(startSec, endSec, top, bottom, fillColor, lineColor) {
  const series = chart.addBaselineSeries({
    baseValue: { type: "price", price: bottom },
    topFillColor1: fillColor,
    topFillColor2: fillColor,
    topLineColor: lineColor,
    bottomFillColor1: "rgba(0,0,0,0)",
    bottomFillColor2: "rgba(0,0,0,0)",
    bottomLineColor: "rgba(0,0,0,0)",
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  series.setData([
    { time: startSec, value: top },
    { time: endSec, value: top },
  ]);
  return series;
}

// 활성 OB 박스 — bullish=teal / bearish=pink, OB 봉 ~ 차트 끝
function renderObBoxes(obs) {
  clearObBoxes();
  if (!lastBarTimeSec) return;
  // LuxAlgo OB Detector 표준 — bullish/bearish 각 3개 표시
  const active = (obs || []).filter((o) => !o.mitigated).slice(-3);
  active.forEach((ob) => {
    const isBull = ob.type === "bullish";
    const fill = isBull ? "rgba(45, 212, 191, 0.18)" : "rgba(244, 114, 182, 0.18)";
    const line = isBull ? "rgba(45, 212, 191, 0.55)" : "rgba(244, 114, 182, 0.55)";
    const startSec = tsToTimeSec(ob.ts_ms);
    if (startSec >= lastBarTimeSec) return;
    obBoxSeries.push(_addBoxSeries(startSec, lastBarTimeSec, ob.high, ob.low, fill, line));
  });
}

// 활성 FVG 박스 — bullish=green / bearish=red, FVG 봉 ~ 차트 끝
function renderFvgBoxes(fvgs) {
  clearFvgBoxes();
  if (!lastBarTimeSec) return;
  const active = (fvgs || []).filter((f) => !f.filled && !f.invalidated).slice(-10);
  active.forEach((fvg) => {
    const isBull = fvg.type === "bullish";
    const fill = isBull ? "rgba(52, 211, 153, 0.15)" : "rgba(251, 113, 133, 0.15)";
    const line = isBull ? "rgba(52, 211, 153, 0.45)" : "rgba(251, 113, 133, 0.45)";
    const startSec = tsToTimeSec(fvg.ts_ms);
    if (startSec >= lastBarTimeSec) return;
    fvgBoxSeries.push(_addBoxSeries(startSec, lastBarTimeSec, fvg.high, fvg.low, fill, line));
  });
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
    const isBull = ev.type.endsWith("bullish");
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
    // 텍스트 라벨 "BOS" / "CHoCH" — segment 끝점 (돌파 봉) 위/아래
    series.setMarkers([{
      time: t2,
      position: isBull ? "aboveBar" : "belowBar",
      color,
      shape: "circle",
      text: isChoCH ? "CHoCH" : "BOS",
      size: 0,
    }]);
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
    // 텍스트 라벨 "EQH" / "EQL" — 마지막 swing 끝점
    series.setMarkers([{
      time: t2,
      position: isHigh ? "aboveBar" : "belowBar",
      color,
      shape: "circle",
      text: isHigh ? "EQH" : "EQL",
      size: 0,
    }]);
    eqlSegmentSeries.push(series);
  });
}

// PD Zone — BaselineSeries 박스 3개 (Premium=빨강 / Equilibrium=회색 / Discount=초록)
// 각 zone 은 trailing 시작 ts 부터 차트 끝까지의 시간 한정 박스
function renderPdZones(trailing) {
  clearZoneBoxes();
  if (!vizEnabled.zones || !trailing || !lastBarTimeSec) return;
  const top = trailing.top_price;
  const bot = trailing.bottom_price;
  if (top <= bot) return;
  const range = top - bot;
  const startTs = Math.min(trailing.top_ts_ms, trailing.bottom_ts_ms);
  const startSec = tsToTimeSec(startTs);
  if (startSec >= lastBarTimeSec) return;

  const zones = [
    {
      topPrice: top,
      botPrice: bot + 0.95 * range,
      fill: "rgba(251, 113, 133, 0.18)",
      line: "rgba(251, 113, 133, 0.50)",
    },
    {
      topPrice: bot + 0.525 * range,
      botPrice: bot + 0.475 * range,
      fill: "rgba(135, 139, 148, 0.18)",
      line: "rgba(135, 139, 148, 0.50)",
    },
    {
      topPrice: bot + 0.05 * range,
      botPrice: bot,
      fill: "rgba(52, 211, 153, 0.18)",
      line: "rgba(52, 211, 153, 0.50)",
    },
  ];
  zones.forEach((z) => {
    zoneBoxSeries.push(
      _addBoxSeries(startSec, lastBarTimeSec, z.topPrice, z.botPrice, z.fill, z.line),
    );
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

function clearOverlays() {
  candleSeries.setMarkers([]);
  clearObBoxes();
  clearFvgBoxes();
  clearTrailingPriceLines();
  clearBosSegments();
  clearEqlSegments();
  clearZoneBoxes();
}

function clearZoneBoxes() {
  zoneBoxSeries.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* noop */ } });
  zoneBoxSeries = [];
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

  // 봇 가동 상태 → START / STOP 시각 효과 (running 시 START 녹색 glow)
  $("btn-start").classList.toggle("active", s.state === "running");
  $("btn-stop").classList.toggle("active", s.state === "stopped" && s.enabled);

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

  // 2026-05-27: Bot Enable 버튼 제거됨 — Start/Stop 으로만 제어.

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

  // FVG 박스 (BaselineSeries) 가 시각 자체이므로 텍스트 마커는 제거 (renderFvgBoxes)

  // Sweep markers — large swing (50봉) 기준만 표시 (LuxAlgo 동일).
  // 작은 swing(left=1) sweep 은 도배라 표시 안 함. + 같은 봉/위치 dedup.
  const sweepSeen = new Set();
  (m.large_sweeps ?? []).forEach(s => {
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

  // Setups — v0.4.69: RR 텍스트 제거 (도배 방지). confluence ★N 만 유지 (정보성).
  m.setups.forEach(s => {
    const score = s.confluence_score ?? 0;
    const conf = score > 0 ? `★${score}` : "";
    markers.push({
      time: tsToTimeSec(s.ts_ms),
      position: s.direction === "long" ? "belowBar" : "aboveBar",
      color: s.direction === "long" ? "#34d399" : "#fb7185",
      shape: s.direction === "long" ? "arrowUp" : "arrowDown",
      text: conf,
    });
  });

  // OB 박스 (BaselineSeries) 가 시각 자체이므로 텍스트 마커는 제거 (renderObBoxes)

  // 활성 OB 박스 — BaselineSeries (시작 봉 ~ 차트 끝)
  // chart TF 가 1d / 1w 같은 큰 단위면 large_order_blocks (50봉 swing 기반) 사용,
  // 그 외엔 order_blocks (internal 5봉 swing).
  const isLargeChartTf = currentTimeframe === "1d" || currentTimeframe === "1w";
  renderObBoxes(isLargeChartTf ? (m.large_order_blocks ?? []) : (m.order_blocks ?? []));
  // 활성 FVG 박스 — BaselineSeries
  renderFvgBoxes(m.fvgs ?? []);
  renderIfvgBoxes(m.ifvgs ?? []);
  renderBreakerBoxes(m.breakers ?? []);
  // Sweep zones — large_sweeps 만 표시 (basic sweep 은 너무 많음)
  renderSweepZones(m.large_sweeps ?? []);

  // Trailing extremes (Strong/Weak High & Low) — 가로선 한 쌍
  renderTrailingExtremes(m.trailing ?? null);

  // BOS/CHoCH — large swing (50봉) 기준만 표시 (LuxAlgo 동일, 도배 방지)
  renderBosSegments(m.large_structure ?? []);

  // EQH/EQL — 두 swing 잇는 짧은 dotted segment
  renderEqlSegments(m.equal_levels ?? []);

  // Premium/Discount Zone — AreaSeries 박스 (마지막 봉 ts 까지 채움)
  renderPdZones(m.trailing ?? null);

  // 시간순 정렬
  markers.sort((a, b) => a.time - b.time);
  candleSeries.setMarkers(markers);

  // FVG 는 renderFvgBoxes 의 BaselineSeries 박스로 표시 (mean line 제거)
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
    const [ohlcv, markers, position, pnl] = await Promise.all([
      api(`/ict/ohlcv?timeframe=${tf}&limit=1000`),
      api(`/ict/markers?timeframe=${tf}&limit=1000`),
      api("/ict/position"),
      api("/ict/closed_pnl?limit=20"),
    ]);
    candleSeries.setData(ohlcv.candles);
    // 2026-05-27: 차트 좌측 공백 제거 — 모든 봉 보이게 맞춤.
    try { chart.timeScale().fitContent(); } catch (e) { /* noop */ }
    // 마지막 봉 시간 — PD Zone area 끝점에 사용
    if (ohlcv.candles && ohlcv.candles.length > 0) {
      lastBarTimeSec = ohlcv.candles[ohlcv.candles.length - 1].time;
    }
    renderMarkers(markers);
    renderPositions(position);
    renderPendingLimit(position);
    renderPnL(pnl);
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

// ============================================================
// 우측 P&L 패널 + 지정가 라인 — 사용자 결정 2026-05-27 추가
// ============================================================

/** 거래소 closed-pnl history 를 우측 P&L 패널 테이블에 렌더. */
function renderPnL(data) {
  const tbody = $("pnl-tbody");
  const count = $("pnl-count");
  const trades = (data && Array.isArray(data.trades)) ? data.trades : [];
  count.textContent = trades.length;
  if (trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="pnl-empty">청산 거래 없음 — 첫 종료 시 표시됩니다</td></tr>';
    return;
  }
  const rows = trades.map((t) => {
    const sym = (t.symbol || "").replace(":USDT", "").replace("/USDT", "USDT");
    const dirCls = t.direction === "long" ? "pnl-side-long" : "pnl-side-short";
    const pnl = (typeof t.pnl_usd === "number") ? t.pnl_usd : 0;
    const pnlCls = pnl >= 0 ? "pnl-pos" : "pnl-neg";
    const pnlStr = (pnl >= 0 ? "+" : "") + pnl.toFixed(2);
    const entry = (typeof t.entry_price === "number") ? t.entry_price.toFixed(2) : "—";
    const qty = (typeof t.qty === "number") ? t.qty.toFixed(3) : "—";
    const ts = (typeof t.closed_at_ts === "number") ? new Date(t.closed_at_ts) : null;
    const tstr = ts ? `${String(ts.getMonth()+1).padStart(2,"0")}-${String(ts.getDate()).padStart(2,"0")} ${String(ts.getHours()).padStart(2,"0")}:${String(ts.getMinutes()).padStart(2,"0")}` : "—";
    return `<tr>
      <td><span class="${dirCls}">${sym}</span></td>
      <td class="num">${entry}</td>
      <td class="num">${qty}</td>
      <td class="num ${pnlCls}">${pnlStr}</td>
      <td class="pnl-time">${tstr}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = rows;
}

/** 봇이 지정가 미체결 대기 중이면 차트에 horizontal price line + qty 라벨 표시.
 *  position.pending 없으면 line 제거 (체결됨/취소됨). */
function renderPendingLimit(position) {
  const pe = position && position.pending;
  const key = pe ? `${pe.entry}|${pe.qty}|${pe.direction}` : null;
  if (pendingLineKey === key) return;  // 동일 — skip
  // 기존 라인 제거
  if (pendingPriceLine !== null) {
    try { candleSeries.removePriceLine(pendingPriceLine); } catch (e) { /* noop */ }
    pendingPriceLine = null;
  }
  pendingLineKey = key;
  if (!pe) return;
  const isLong = pe.direction === "long";
  pendingPriceLine = candleSeries.createPriceLine({
    price: pe.entry,
    color: isLong ? "#34d399" : "#fb7185",
    lineWidth: 1,
    lineStyle: 2,   // dashed
    axisLabelVisible: true,
    title: `Limit ${pe.entry.toFixed(2)}  ${pe.qty.toFixed(3)}`,
  });
}

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

// 2026-05-27: btn-test-conn / btn-toggle-enabled 제거됨. 실시간 잔고 + START/STOP 으로 대체.

// "다시 입력" 클릭 시 — 강제로 입력 폼 보여줌 (status 갱신 전까지 유지)
if ($("btn-cred-reenter")) {
  $("btn-cred-reenter").onclick = () => {
    $("cred-form-block").style.display = "flex";
    $("cred-saved-block").style.display = "none";
    $("cred-api-key").focus();
  };
}

// 2026-05-27: ENABLE/DISABLE 토글 제거. Start/Stop 만 사용.

// ============================================================
// #SAFETY-1 Daily Loss Limit — 입력 + 상태 폴링
// ============================================================
async function refreshDailyLossLimit() {
  try {
    const s = await api("/ict/daily_loss_limit", "GET");
    const inp = $("dll-input");
    // input 빈 상태일 때만 server 값 반영 (사용자 타이핑 중이면 덮어쓰지 않음)
    if (document.activeElement !== inp) {
      inp.value = (s.limit_pct > 0) ? s.limit_pct.toFixed(2) : "";
    }
    const pnl = s.today_pnl_usdt;
    const pct = s.today_pct;
    const sign = pnl > 0 ? "+" : (pnl < 0 ? "" : "±");
    $("dll-today").textContent =
      (s.start_equity > 0)
        ? `${sign}${pnl.toFixed(2)} USDT (${sign}${pct.toFixed(2)}%)`
        : "—";
    if (s.limit_pct <= 0) {
      $("dll-status").textContent = "OFF";
      $("dll-status").style.color = "#888";
    } else if (s.hit) {
      $("dll-status").textContent = "HIT — 새 진입 차단";
      $("dll-status").style.color = "#e74c3c";
    } else {
      $("dll-status").textContent = `Active @ ${s.limit_pct.toFixed(2)}%`;
      $("dll-status").style.color = "#2ecc71";
    }
  } catch (e) {
    // 봇 미가동 등 — 조용히 무시
  }
}

$("btn-dll-set").onclick = async () => {
  const v = parseFloat($("dll-input").value);
  const pct = isNaN(v) ? 0 : Math.max(0, Math.min(50, v));
  try {
    await api("/ict/daily_loss_limit", "POST", { pct });
    toast(pct > 0 ? `Daily loss limit ${pct}% 적용` : "Daily loss limit OFF");
    await refreshDailyLossLimit();
  } catch (e) { toast(e.message, true); }
};

setInterval(refreshDailyLossLimit, 5000);
refreshDailyLossLimit();

// ============================================================
// 실시간 잔고 + 세션 상태 (좌사이드바 + 좌상단 배지) — 2026-05-27 추가
// ============================================================
async function refreshEquityAndSession() {
  try {
    const r = await api("/ict/equity", "GET");
    // 잔고
    const valEl = $("equity-value");
    const subEl = $("equity-sub");
    if (r.active && typeof r.equity === "number") {
      valEl.textContent = `${r.equity.toFixed(2)} USDT`;
      subEl.textContent = "실시간 (5초)";
    } else {
      valEl.textContent = "—";
      subEl.textContent = "봇 시작 후 실시간 표시";
    }
    // 세션 상태 — 좌상단 배지
    const badge = $("session-badge");
    const ss = r.session_status || { kind: "none", label: "None" };
    if (badge) {
      badge.textContent = ss.label;
      badge.className = `session-badge session-${ss.kind}`;
    }
  } catch (e) {
    // 조용히 무시
  }
}
setInterval(refreshEquityAndSession, 5000);
refreshEquityAndSession();

// ============================================================
// 봇 판단 시각화 (좌하단 패널) — 2026-05-27 추가
// 약어 + 한글 해석 병기 — 모르는 사람도 이해 가능
// ============================================================
async function refreshJudgment() {
  try {
    const j = await api("/ict/judgment", "GET");
    // 방향
    const longPct = Number(j.direction?.long_pct ?? 50);
    const shortPct = Number(j.direction?.short_pct ?? 50);
    $("jp-dir-label").textContent = j.direction?.label || "—";
    $("jp-dir-long").style.width = `${longPct}%`;
    $("jp-dir-short").style.width = `${shortPct}%`;
    $("jp-dir-pct").innerHTML =
      `<span style="color:#34d399">롱 ${longPct.toFixed(1)}%</span>` +
      `<span style="color:#fb7185">숏 ${shortPct.toFixed(1)}%</span>`;
    // Reasons
    const rs = Array.isArray(j.reasons) ? j.reasons : [];
    $("jp-reasons").innerHTML = rs.length === 0
      ? '<div class="jp-reason-interp">분석 자료 누적 중…</div>'
      : rs.map((r) => `
        <div class="jp-reason">
          <div class="jp-reason-top">
            <span class="jp-reason-dot ${r.color || 'yellow'}"></span>
            <span class="jp-reason-term">${escapeHtml(r.term || '')}</span>
            <span class="jp-reason-range">${escapeHtml(r.range || '')}</span>
          </div>
          <div class="jp-reason-interp">해석: ${escapeHtml(r.interpretation || '')}</div>
        </div>
      `).join("");
    // Entry Condition
    const ec = j.entry_condition || {};
    $("jp-entry").innerHTML =
      `<div>${escapeHtml(ec.title || '—')}</div>` +
      (ec.detail ? `<div class="jp-entry-detail">${escapeHtml(ec.detail)}</div>` : "");
  } catch (e) {
    // 조용히 무시
  }
}
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
setInterval(refreshJudgment, 5000);
refreshJudgment();

// 햄버거 토글 — 우측 P&L 패널 접기/펴기
// 2026-05-27: 파트너 피드백 ("저 pnl 창 말한거양") — 좌측 사이드바가 아니라 우측 P&L 토글
const _btnSidebarToggle = $("btn-sidebar-toggle");
if (_btnSidebarToggle) {
  _btnSidebarToggle.onclick = () => {
    document.body.classList.toggle("pnl-collapsed");
  };
}

// ============================================================
// resize
// ============================================================
function fit() {
  const r = chartEl.getBoundingClientRect();
  chart.applyOptions({ width: r.width, height: r.height });
}
window.addEventListener("resize", fit);
fit();

// 부팅 시 version 1회 fetch (config endpoint)
async function fetchVersion() {
  try {
    const cfg = await api("/ict/config");
    const el = document.getElementById("app-version");
    if (el && cfg && cfg.version) el.textContent = `v${cfg.version}`;
  } catch (e) { /* noop */ }
}
fetchVersion();

// 초기 fetch + polling (10s 주기)
fetchAndRender();
setInterval(fetchAndRender, 10_000);
