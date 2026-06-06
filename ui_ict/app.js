/* Aurora-ICT UI — lightweight-charts + REST polling

   2026-05-28: SaaS 전환 — API base 를 same-origin 으로 변경.
   - single-user (.exe / pywebview):  http://127.0.0.1:8765/ui/ 에서 로드 → same origin
   - SaaS (Docker / 클라우드):       https://<도메인>/ui/ 에서 로드 → same origin
   상대 경로 ("") 사용하면 어떤 호스트에서 서빙되든 cookie session 자동 동작. */

const API = "";  // same-origin — fetch("/auth/status") 처럼 동작

// 현재 선택된 차트 timeframe (localStorage 영속화)
// 2026-05-28: 파트너 결정 — default 1h → 5m (settings.timeframe 정합)
let currentTimeframe = localStorage.getItem("aurora_ict_tf") || "5m";
const VALID_TFS = ["1m", "5m", "15m", "1h", "2h", "4h", "1d", "1w"];
if (!VALID_TFS.includes(currentTimeframe)) currentTimeframe = "5m";

// 현재 보고 있는 차트 페어(ccxt symbol) — 좌측 TRADING PAIR(매매)와 별개로
// "어느 페어 차트를 볼지"만 정한다. 차트 데이터(ohlcv/markers)에만 영향,
// 매매/상태(position/equity 등)는 그대로. (localStorage 영속화)
let currentChartSymbol =
  localStorage.getItem("aurora_ict_chart_symbol") || "BTC/USDT:USDT";

// 차트 데이터 클라이언트 캐시 — key `${symbol}|${tf}` → {candles, markers}.
// TF/페어 전환 시 캐시 hit 이면 즉시 표시(체감 즉답), fresh 는 백그라운드 갱신.
// 페이지 로드 후 _prefetchAllCharts 가 모든 (심볼×TF) 조합을 미리 채운다.
const _chartCache = new Map();

// TF 별 fetch 봉 한도 — 큰 TF 는 history 전부, 작은 TF 는 메모리/시간 상한.
// 2026-05-27 파트너 요청: 거래소 시작부터 현재까지 가능한 만큼.
const CANDLE_LIMIT = {
  "1m": 5000, "5m": 20000, "15m": 50000,
  "1h": 60000, "2h": 30000, "4h": 15000,
  "1d": 5000, "1w": 1500,
};
const MARKER_LIMIT = 2000;

// 첫 로드 또는 TF 변경 시만 차트 zoom 강제. 이후 refresh 는 사용자 view 유지.
let _chartViewInitialized = false;
// 기본 줌 — 최근 N봉만 보이게(봉 크게). autoScale 가 보이는 봉 기준으로 가격축
// 자동 맞춤 → 최근 가격대에 집중된 뷰. (값 ↓ = 더 확대)
const CHART_INIT_VISIBLE_BARS = 90;
// 마지막 setData 봉 수 — 우측 가격축 클릭 리셋 + 봉 추적(auto-scroll) 판정용.
let _lastCandleCount = 0;

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
// 2026-05-30 i18n: 차트 시간축을 사용자 timezone 으로 변환.
// lightweight-charts 의 timeScale.tickMarkFormatter / localization.timeFormatter
// 둘 다 사용 — tick mark (x축 라벨) + crosshair tooltip.
function _userTzTickFormatter(time) {
  const ms = (typeof time === "number") ? time * 1000 : time;
  const tz = (window.AuroraI18n && window.AuroraI18n.getTz())
    || "Asia/Seoul";
  try {
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: tz,
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(new Date(ms));
  } catch (e) {
    return new Date(ms).toISOString().slice(11, 16);
  }
}
function _userTzCrosshairFormatter(time) {
  const ms = (typeof time === "number") ? time * 1000 : time;
  const tz = (window.AuroraI18n && window.AuroraI18n.getTz())
    || "Asia/Seoul";
  try {
    return new Intl.DateTimeFormat("sv-SE", {
      timeZone: tz,
      month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(new Date(ms));
  } catch (e) {
    return new Date(ms).toISOString().slice(5, 16).replace("T", " ");
  }
}
const chart = LightweightCharts.createChart(chartEl, {
  layout: { background: { color: "#141316" }, textColor: "#8b8b90" },
  grid: {
    vertLines: { color: "rgba(219,219,222,0.04)" },
    horzLines: { color: "rgba(219,219,222,0.04)" },
  },
  timeScale: {
    borderColor: "rgba(219,219,222,0.10)",
    timeVisible: true,
    secondsVisible: false,
    tickMarkFormatter: _userTzTickFormatter,
  },
  localization: {
    timeFormatter: _userTzCrosshairFormatter,
  },
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

// 차트를 기본 뷰로 복귀 — 최근 N봉 + 가격축 autoScale (우측 끝 정렬).
// 사용자가 줌/스크롤로 흩뜨린 뒤 우측 가격축을 누르면 원위치.
function resetChartView() {
  if (_lastCandleCount <= 0) return;
  const total = _lastCandleCount;
  const from = Math.max(0, total - CHART_INIT_VISIBLE_BARS);
  try {
    chart.timeScale().setVisibleLogicalRange({ from, to: total });
    chart.priceScale("right").applyOptions({ autoScale: true });
  } catch (e) { /* noop */ }
}

// 우측 가격축 영역 클릭 → 기본 뷰 복귀. lightweight-charts 는 price scale
// 클릭 이벤트를 직접 안 주므로 컨테이너 클릭 x좌표가 가격축 폭 안인지로 판정.
chartEl.addEventListener("click", (ev) => {
  try {
    const rect = chartEl.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const psW = chart.priceScale("right").width() || 0;
    if (psW > 0 && x >= rect.width - psW) resetChartView();
  } catch (e) { /* noop */ }
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
  // credentials: "include" — SaaS 세션 cookie (aurora_ict_session) 자동 포함.
  // single-user (.exe) 흐름에서도 cookie 가 없으면 server 가 무시하므로 안전.
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
    credentials: "include",
  };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(`${API}${path}`, opts);
  if (!resp.ok) {
    const detail = await resp.text();
    // 401 — SaaS 세션 만료 시 게이트 자동 표시 (있을 때만 — single-user 흐름에선 게이트 자체가 hidden).
    if (resp.status === 401 && typeof showAuthGate === "function") {
      try { await showAuthGate(); } catch (_) { /* noop */ }
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return await resp.json();
}

function renderStatus(s) {
  // 2026-05-29 파트너 결정: 상단 STATUS 박스 제거 — 6개 row 요소가 DOM 에
  // 없을 수 있어 null-safe 로 보호. 정보는 다른 섹션 (MODE 토글, PAIR 토글,
  // CREDENTIALS, START/STOP, positions-panel) 에 분산 표시됨.
  const _eS = $("s-state");
  if (_eS) { _eS.textContent = s.state; _eS.className = "v " + s.state; }
  const _eM = $("s-mode");
  if (_eM) { _eM.textContent = s.run_mode.toUpperCase(); _eM.className = "v " + s.run_mode; }
  const _eE = $("s-enabled");
  if (_eE) _eE.textContent = s.enabled ? "YES" : "NO";
  const _eY = $("s-symbol");
  if (_eY) _eY.textContent = s.symbol;
  const _eC = $("s-creds");
  if (_eC) _eC.textContent = s.has_credentials ? "OK" : "MISSING";
  const _eP = $("s-pos");
  if (_eP) _eP.textContent = s.has_active_position ? "YES" : "—";

  $("btn-demo").classList.toggle("active", s.run_mode === "demo");
  $("btn-live").classList.toggle("active", s.run_mode === "live");

  // 봇 가동 상태 → START / STOP 시각 효과
  // 2026-05-27 파트너 피드백 — "stop 누르면 빨간 활성화 안돼"
  // 이전 조건은 (stopped && enabled) 였는데 STOP 클릭 시 enabled=false 도 같이
  // 되어서 active 안 박혔음. state 만 보고 둘 중 하나 켜지게 단순화.
  // 처음 진입 (running 한 번도 안 됨, stopped) 에도 STOP active — 일관성 위해.
  //
  // 2026-05-29: state="resuming" 추가 처리. fly machine 재시작 직후 auto_resume
  // 가 슬롯을 못 살린 상태에서 사용자가 새로고침하면 stopped 로 표시되며 STOP 에
  // 빨간 불 들어오던 버그 (파트너 보고). resuming 일 때는 START 측 active 유지
  // 해 사용자에게 봇이 가동 의도 상태임을 보여줌.
  const isRunning = s.state === "running" || s.state === "resuming";
  $("btn-start").classList.toggle("active", isRunning);
  $("btn-stop").classList.toggle("active", s.state === "stopped");

  // 2026-05-29 v3 파트너 결정: 상태 텍스트 ("등록됨"/"미등록") 제거.
  // 배경색 (.has-key / .no-key) + 버튼 라벨 ("재등록" / "등록") 만으로 시각화.
  const credForm = $("cred-form-block");
  const credSaved = $("cred-saved-block");
  if (credForm && credSaved) {
    const hasDemo = !!s.has_demo_credentials;
    const hasLive = !!s.has_live_credentials;
    const anyKey = hasDemo || hasLive;
    credSaved.style.display = "flex";
    credForm.style.display = anyKey ? "none" : "flex";
    const slotDemo = $("cred-slot-demo");
    const slotLive = $("cred-slot-live");
    if (slotDemo) {
      slotDemo.classList.toggle("has-key", hasDemo);
      slotDemo.classList.toggle("no-key", !hasDemo);
    }
    if (slotLive) {
      slotLive.classList.toggle("has-key", hasLive);
      slotLive.classList.toggle("no-key", !hasLive);
    }
    // 2026-05-30 i18n: 버튼 라벨 사용자 언어. has-key 면 '재등록', no-key 면 '등록'.
    const btnDemo = $("btn-cred-demo");
    const btnLive = $("btn-cred-live");
    const tReg = window.AuroraI18n ? window.AuroraI18n.t("btn.register") : "등록";
    const tReReg = window.AuroraI18n ? window.AuroraI18n.t("btn.reregister") : "재등록";
    if (btnDemo) btnDemo.textContent = hasDemo ? tReReg : tReg;
    if (btnLive) btnLive.textContent = hasLive ? tReReg : tReg;
    // 삭제 버튼은 has-key 일 때만 의미 (no-key 면 어차피 비어있음) — 라벨 갱신.
    const tDel = window.AuroraI18n ? window.AuroraI18n.t("btn.delete") : "삭제";
    const btnDelDemo = $("btn-cred-delete-demo");
    const btnDelLive = $("btn-cred-delete-live");
    if (btnDelDemo) btnDelDemo.textContent = tDel;
    if (btnDelLive) btnDelLive.textContent = tDel;
  }

  // 2026-05-27: Bot Enable 버튼 제거됨 — Start/Stop 으로만 제어.

  // 차트 상단 심볼 라벨 — 보고 있는 차트 페어 + 매매 TF 표시.
  // symbol 은 currentChartSymbol(차트 선택) 기준 — 매매 상태(s.symbol)와 별개.
  const lbl = $("chart-symbol-label");
  const tradeTf = s.timeframe || "?";
  if (lbl) lbl.textContent = `${currentChartSymbol} · chart ${currentTimeframe} · trade ${tradeTf}`;

  // 사이드바 매매 TF 토글 active 동기화
  document.querySelectorAll("#trade-tf-toggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tradeTf === s.timeframe);
  });

  // 2026-05-29 PR C: 페어 토글 active = status.running_symbols 기준.
  // PR B 에서 status 응답에 running_symbols 필드 추가. fetchAndRender 가
  // status 갱신할 때마다 페어 토글도 즉시 동기화 → 다른 탭 / fly 재시작 후에도
  // 정확한 가동 상태 표시.
  if (Array.isArray(s.running_symbols)) {
    _updatePairButtonsFromRunning(new Set(s.running_symbols));
  }
}

// ============================================================
// Markers (FVG / Sweep / Structure / Setup)
// ============================================================
/** OHLCV first ts 이전 markers 일괄 필터링 (mutation in place).
 *
 *  2026-05-28 파트너 피드백: 차트 봉 안 그려진 옛 영역에 BOS/CHoCH/EQH/EQL
 *  라벨만 떠있는 버그 — OHLCV cache miss 시 fast path (200봉) + markers 는
 *  cache 전체 (수천 봉) 반환 불일치.
 *
 *  fix: payload.markers 의 각 list 를 first ts (ms) 이상만 남김.
 *  killzones / macros 는 end_ms 기준, equal_levels 는 swing_ts_list 첫 ts.
 */
function _filterMarkersByFirstBar(payload, firstTsSec) {
  if (!payload || !payload.markers || firstTsSec == null || firstTsSec <= 0) return;
  const firstMs = firstTsSec * 1000;
  const m = payload.markers;
  const byTs = (list) => (list || []).filter((it) => (it.ts_ms || 0) >= firstMs);
  m.fvgs = byTs(m.fvgs);
  m.ifvgs = byTs(m.ifvgs);
  m.breakers = byTs(m.breakers);
  m.dols = byTs(m.dols);
  m.sweeps = byTs(m.sweeps);
  m.structure = byTs(m.structure);
  m.swings = byTs(m.swings);
  m.setups = byTs(m.setups);
  m.order_blocks = byTs(m.order_blocks);
  m.internal_swings = byTs(m.internal_swings);
  m.internal_structure = byTs(m.internal_structure);
  m.large_swings = byTs(m.large_swings);
  m.large_structure = byTs(m.large_structure);
  m.large_sweeps = byTs(m.large_sweeps);
  m.large_order_blocks = byTs(m.large_order_blocks);
  // killzones / macros 는 [start_ms, end_ms] 구간 — 끝점이 first 이후면 그림.
  m.killzones = (m.killzones || []).filter((k) => (k.end_ms || 0) >= firstMs);
  m.macros = (m.macros || []).filter((k) => (k.end_ms || 0) >= firstMs);
  // equal_levels 는 swing_ts_list 의 첫 ts (가장 옛날 swing) 기준.
  m.equal_levels = (m.equal_levels || []).filter((e) => {
    const tsList = e.swing_ts_list || [];
    return tsList.length === 0 || tsList[0] >= firstMs;
  });
}

function renderMarkers(payload) {
  const m = payload.markers;
  // 2026-05-29 v2: Marker Counts 섹션 제거 — null-safe 가드. DOM 없으면 skip.
  // (차트 위 시각화로 충분, 사이드바 군더더기 ↓.)
  const setText = (id, val) => {
    const el = $(id);
    if (el) el.textContent = val;
  };
  setText("c-fvgs", payload.count.fvgs);
  setText("c-sweeps", payload.count.sweeps);
  setText("c-struct", payload.count.structure);
  setText("c-swings", payload.count.swings);
  setText("c-kz", payload.count.killzones);
  setText("c-setups", payload.count.setups);
  setText("c-obs", payload.count.order_blocks ?? 0);
  setText("c-macros", payload.count.macros ?? 0);
  setText("c-int-struct", payload.count.internal_structure ?? 0);
  setText("c-lg-struct", payload.count.large_structure ?? 0);
  setText("c-lg-swings", payload.count.large_swings ?? 0);
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
  // 2026-05-28: 모바일 가로 positions fab badge / 강조 — 활성 포지션 유무 동기화.
  _syncPositionsFab(!!(pos && pos.active));

  // 2026-05-30 파트너 요청: 대기 포지션 (limit 미체결) 도 표시.
  // active=False 라도 pending 있으면 그것만 row 1개로 표시.
  if (!pos || (!pos.active && !pos.pending)) {
    const emptyMsg = window.AuroraI18n
      ? window.AuroraI18n.t("positions.empty")
      : "포지션 없음 — 봇 가동 후 진입 시 표시됩니다";
    tbody.innerHTML = '<tr><td colspan="8" class="pos-empty">' +
      emptyMsg + '</td></tr>';
    count.textContent = "0 open";
    return;
  }

  // 대기 포지션만 (active 없음) — pending row 1개.
  if (!pos.active && pos.pending) {
    const pe = pos.pending;
    const sideClass = pe.direction === "long" ? "pos-side-long" : "pos-side-short";
    const sideLabel = pe.direction === "long" ? "LONG" : "SHORT";
    const tPending = window.AuroraI18n ? window.AuroraI18n.t("positions.pending") : "대기 중";
    const placedMs = pe.placed_ts_ms || 0;
    const tz = (window.AuroraI18n && window.AuroraI18n.getTz()) || "Asia/Seoul";
    let placedStr = "—";
    if (placedMs) {
      try {
        placedStr = new Intl.DateTimeFormat("sv-SE", {
          timeZone: tz, month: "2-digit", day: "2-digit",
          hour: "2-digit", minute: "2-digit", hour12: false,
        }).format(new Date(placedMs));
      } catch (e) {}
    }
    tbody.innerHTML = `
      <tr class="pos-pending">
        <td>
          <div>${pos.symbol || "—"}</div>
          <div class="${sideClass}" style="font-size:9px; letter-spacing:0.15em">${sideLabel} · LIMIT · ${tPending}</div>
        </td>
        <td>${_fmt(pe.qty, 4)}</td>
        <td>${_fmt(pe.entry, 2)}</td>
        <td>—</td>
        <td>—</td>
        <td>SL ${_fmt(pe.stop_loss, 2)}<br/>TP ${_fmt(pe.take_profit, 2)}</td>
        <td style="color:var(--text-3)">${placedStr}</td>
        <td>
          <div class="pos-actions">
            <button class="pos-btn-close-full" id="btn-cancel-pending">CANCEL</button>
          </div>
        </td>
      </tr>`;
    count.textContent = "1 pending";
    // Cancel 버튼 핸들러 — 2026-05-30: 신규 /ict/pending/cancel endpoint.
    const cancelBtn = $("btn-cancel-pending");
    if (cancelBtn) {
      cancelBtn.onclick = async () => {
        try {
          await api("/ict/pending/cancel", "POST");
          toast("Pending order cancelled");
          await fetchAndRender();
        } catch (e) { toast(e.message, true); }
      };
    }
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

// 차트 데이터(candles + markers)를 차트에 적용 — setData + 뷰(첫 로드 줌 /
// auto-scroll) + 마커. fetchAndRender(fresh)와 TF/페어 전환 시 캐시 즉시 표시가
// 공용으로 호출한다.
function _applyChartData(candles, markers) {
  candleSeries.setData(candles || []);
  const _newCount = (candles && candles.length) || 0;
  // 첫 로드/전환 시만 최근 N봉 zoom — 이후 refresh 는 사용자 view 유지.
  if (!_chartViewInitialized && _newCount > 0) {
    _chartViewInitialized = true;
    _lastCandleCount = _newCount;
    resetChartView();
  } else if (_newCount > 0) {
    // 봉 움직임 따라 차트 이동(auto-scroll) — 우측 끝 근처(여백 3봉)일 때만 추적.
    try {
      const range = chart.timeScale().getVisibleLogicalRange();
      if (range && range.to >= _lastCandleCount - 3) {
        const width = range.to - range.from;
        chart.timeScale().setVisibleLogicalRange({
          from: Math.max(0, _newCount - width), to: _newCount,
        });
      }
    } catch (e) { /* noop */ }
    _lastCandleCount = _newCount;
  }
  if (candles && candles.length > 0) {
    lastBarTimeSec = candles[candles.length - 1].time;
    // markers 가 차트 첫 봉보다 옛 영역에 뜨는 것 방지 — first ts 이전 필터.
    _filterMarkersByFirstBar(markers, candles[0].time);
  }
  renderMarkers(markers);
}

// 전환 시 캐시된 차트가 있으면 즉시 표시(체감 즉답). fresh 는 직후 fetchAndRender 가 갱신.
function _showCachedChartIfAny() {
  const c = _chartCache.get(`${currentChartSymbol}|${currentTimeframe}`);
  if (c) _applyChartData(c.candles, c.markers);
}

let _prefetchStarted = false;

// 페이지 로드 후 백그라운드로 모든 (심볼 × TF) 조합 차트를 미리 받아 캐시 →
// TF/페어 전환이 거의 항상 즉답. 현재 보는 심볼·TF 부터(우선순위), 순차로
// (서버/네트워크 부담 분산). 실패는 무시(다음 전환 때 fetch). 1회만 실행.
async function _prefetchAllCharts() {
  if (_prefetchStarted) return;
  _prefetchStarted = true;
  // #PAIR-EXPAND: 페어가 30종목으로 늘어 전종목×TF prefetch 는 폭발(240 요청).
  // 현재 보는 차트 심볼의 TF 들만 미리 받는다. 다른 페어는 전환 시 개별 fetch.
  const symbols = [currentChartSymbol];
  const tfs = [
    currentTimeframe,
    ...VALID_TFS.filter((t) => t !== currentTimeframe),
  ];
  for (const sym of symbols) {
    for (const tf of tfs) {
      const key = `${sym}|${tf}`;
      if (_chartCache.has(key)) continue;
      try {
        const cl = CANDLE_LIMIT[tf] || 5000;
        const [ohlcv, markers] = await Promise.all([
          api(`/ict/ohlcv?timeframe=${tf}&symbol=${encodeURIComponent(sym)}&limit=${cl}`),
          api(`/ict/markers?timeframe=${tf}&symbol=${encodeURIComponent(sym)}&limit=${MARKER_LIMIT}`),
        ]);
        _chartCache.set(key, { candles: ohlcv.candles, markers });
      } catch (e) { /* prefetch 실패는 무시 — 전환 때 개별 fetch */ }
    }
  }
}

async function fetchAndRender() {
  try {
    const status = await api("/ict/status");
    renderStatus(status);
    if (status.state !== "running") {
      renderPositions(null);
      return;
    }

    const tf = encodeURIComponent(currentTimeframe);
    const candleLimit = CANDLE_LIMIT[currentTimeframe] || 5000;
    const cacheKey = `${currentChartSymbol}|${currentTimeframe}`;
    // 2026-05-28: P&L 누적 그래프 (전체기간) 용으로 limit=200 (백엔드 cap).
    const [ohlcv, markers, position, pnl] = await Promise.all([
      api(`/ict/ohlcv?timeframe=${tf}&symbol=${encodeURIComponent(currentChartSymbol)}&limit=${candleLimit}`),
      api(`/ict/markers?timeframe=${tf}&symbol=${encodeURIComponent(currentChartSymbol)}&limit=${MARKER_LIMIT}`),
      api("/ict/position"),
      api("/ict/closed_pnl?limit=200"),
    ]);
    // fresh 데이터 캐시 갱신 — TF/페어 전환·prefetch 가 재사용해 즉시 표시.
    _chartCache.set(cacheKey, { candles: ohlcv.candles, markers });
    _applyChartData(ohlcv.candles, markers);
    renderPositions(position);
    renderPendingLimit(position);
    renderPnL(pnl);
    // 첫 정상 렌더 후 백그라운드 prefetch 1회 시작 — 모든 심볼×TF 미리 캐시해
    // 전환 즉답. 초기 렌더 안정화 후 시작(1.5s 지연).
    if (!_prefetchStarted) setTimeout(_prefetchAllCharts, 1500);
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
  // 2026-05-29: Live 전환 — confirm 1단계만 (LIVE 타이핑 prompt 제거, 파트너 요청).
  // 서버 단 has_api_keys("live") 가드가 있어 키 미등록 시 자동 차단됨.
  const warn = (
    "⚠ LIVE (실거래) 모드로 전환합니다.\n\n" +
    "• 실제 자금이 즉시 거래에 사용됩니다.\n" +
    "• Bybit Live API 키가 등록돼 있어야 합니다 (Demo 키와 별도).\n" +
    "• 봇이 가동 중이면 자동 재시작됩니다.\n\n" +
    "계속하려면 OK 를 누르세요."
  );
  if (!confirm(warn)) return;
  try {
    await api("/ict/run-mode", "POST", { mode: "live" });
    toast("LIVE 모드 전환 완료");
    await fetchAndRender();
  } catch (e) {
    toast(e.message, true);
  }
};
$("btn-start").onclick = async () => {
  // 2026-05-27: 즉시 시각 응답 — fetchAndRender 까지 기다리지 않고 클릭 직후 active.
  $("btn-start").classList.add("active");
  $("btn-stop").classList.remove("active");
  // 2026-06-06: START = 선호 페어(마지막 가동했던 페어들) 모두 복원 가동.
  // /ict/start 를 symbol 없이 호출하면 백엔드가 last_active_pairs 를 복원한다.
  try { await api("/ict/start", "POST"); toast("봇 시작됨"); await fetchAndRender(); await refreshRunningPairs(); }
  catch (e) { toast(e.message, true); }
};
$("btn-stop").onclick = async () => {
  // 2026-05-27: STOP 클릭 즉시 빨간 glow 표시. /ict/stop-all 가 pending 미체결 자동 취소.
  $("btn-stop").classList.add("active");
  $("btn-start").classList.remove("active");
  // 2026-06-06: 전체 STOP — /ict/stop-all 이 모든 페어 정지하되 선호(last_active)는
  // 유지 → START 시 복원. (개별 페어 정지는 페어 칩 클릭 = /ict/stop?symbol=)
  try {
    await api("/ict/stop-all", "POST");
    toast("봇 중지됨");
    await fetchAndRender();
    await refreshRunningPairs();
  }
  catch (e) { toast(e.message, true); }
};

// 2026-05-29: 사이드바 cred-mode-toggle (DEMO/LIVE 슬롯 선택) 핸들러.
// 클릭 시 active 클래스 토글 + hint 문구 갱신. 저장은 btn-save-cred 가 active 모드로.
(function _bindCredModeToggle() {
  const root = document.getElementById("cred-mode-toggle");
  if (!root) return;
  const hint = document.getElementById("cred-form-hint");
  root.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-cred-mode]");
    if (!btn) return;
    root.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    if (hint) {
      const mode = btn.dataset.credMode;
      hint.innerHTML = mode === "live"
        ? '<b style="color:#fb7185">LIVE</b> 슬롯 — 실거래 키 (api.bybit.com). 신중히 등록.'
        : '<b>DEMO</b> 슬롯 — Bybit Demo Trading 키';
    }
  });
})();

// 키 저장 — cred-mode-toggle 의 active 모드 슬롯에 저장 (SaaS /auth/api-keys).
// 단일 사용자 (.exe) 흐름은 /ict/credentials 도 받지만 SaaS 가 주 사용 경로라
// /auth/api-keys 로 통일 호출. 단일 사용자 흐름은 401 받으면 /ict/credentials 로 fallback.
$("btn-save-cred").onclick = async () => {
  const apiKey = $("cred-api-key").value.trim();
  const apiSecret = $("cred-api-secret").value.trim();
  if (!apiKey || !apiSecret) {
    toast("API Key/Secret 필수", true);
    return;
  }
  const activeBtn = document.querySelector(
    "#cred-mode-toggle button[data-cred-mode].active",
  );
  const mode = activeBtn ? activeBtn.dataset.credMode : "demo";
  // LIVE 슬롯 등록 시 한 번 더 확인 — 실수로 LIVE 슬롯에 데모 키 넣는 사고 방지.
  if (mode === "live") {
    const ok = confirm(
      "LIVE 슬롯에 키를 저장합니다.\n\n" +
      "Bybit 실거래 (api.bybit.com) API 키여야 합니다.\n" +
      "Demo 키를 LIVE 슬롯에 저장하면 모드 전환 시 인증 실패가 발생합니다.\n\n" +
      "계속하시겠습니까?",
    );
    if (!ok) return;
  }
  try {
    // SaaS 경로 시도 — multi-user 인증 토큰이 있으면 200, 단일 사용자면 404/401.
    let saved = false;
    try {
      await api("/auth/api-keys", "POST", {
        mode, api_key: apiKey, api_secret: apiSecret,
      });
      saved = true;
    } catch (e1) {
      // 단일 사용자 fallback — /ict/credentials.
      await api("/ict/credentials", "POST", {
        mode, api_key: apiKey, api_secret: apiSecret,
      });
      saved = true;
    }
    if (saved) {
      toast(`${mode.toUpperCase()} 키 저장됨`);
      $("cred-api-key").value = "";
      $("cred-api-secret").value = "";
      await fetchAndRender();
    }
  } catch (e) { toast(e.message, true); }
};

// 2026-05-29: 슬롯별 "재등록" 버튼 — 클릭 시 form 다시 노출 + 해당 모드로 toggle.
document.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-cred-reenter]");
  if (!btn) return;
  const mode = btn.dataset.credReenter;
  const form = document.getElementById("cred-form-block");
  const saved = document.getElementById("cred-saved-block");
  if (form) form.style.display = "flex";
  if (saved) saved.style.display = "none";
  // toggle 그쪽 모드 active 로.
  document
    .querySelectorAll("#cred-mode-toggle button[data-cred-mode]")
    .forEach((b) => b.classList.toggle("active", b.dataset.credMode === mode));
  const hint = document.getElementById("cred-form-hint");
  if (hint) {
    hint.innerHTML = mode === "live"
      ? '<b style="color:#fb7185">LIVE</b> 슬롯 재등록 — 실거래 키 (api.bybit.com)'
      : '<b>DEMO</b> 슬롯 재등록 — Bybit Demo Trading 키';
  }
  const inp = document.getElementById("cred-api-key");
  if (inp) inp.focus();
});

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

// 최근 fetch 한 trades 전체 (그래프용) — renderPnL 가 갱신.
// 토글이 매 fetch 마다 server 호출 없이 클라이언트에서 재필터.
let _lastTrades = [];

/** 거래소 closed-pnl history 를 우측 P&L 패널 테이블 + 누적 그래프에 렌더. */
function renderPnL(data) {
  const tbody = $("pnl-tbody");
  const count = $("pnl-count");
  const trades = (data && Array.isArray(data.trades)) ? data.trades : [];
  _lastTrades = trades;
  count.textContent = trades.length;
  if (trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="pnl-empty">청산 거래 없음 — 첫 종료 시 표시됩니다</td></tr>';
    renderPnLChart(trades);
    return;
  }
  // 리스트는 최근 50건만 표시 (도배 방지). 그래프는 전체 활용.
  const listTrades = trades.slice(0, 50);
  const rows = listTrades.map((t) => {
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
  renderPnLChart(trades);
}

// ============================================================
// (3) 누적 P&L 그래프 — lightweight-charts LineSeries (별도 instance).
// 토글 (전체기간/90D/60D/30D/7D/24h) localStorage 영속.
// 2026-05-28 파트너 요청 — 24h 추가 (가장 짧은 단위, data-range="1" 1일).
// ============================================================
const VALID_PNL_RANGES = ["all", "90", "60", "30", "7", "1"];
let pnlRange = localStorage.getItem("aurora_ict_pnl_range") || "30";
if (!VALID_PNL_RANGES.includes(pnlRange)) pnlRange = "30";

let _pnlChart = null;
let _pnlSeries = null;

function _ensurePnLChart() {
  if (_pnlChart) return;
  const el = document.getElementById("pnl-chart");
  if (!el || typeof LightweightCharts === "undefined") return;
  _pnlChart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#201F1F" }, textColor: "#888892" },
    grid: {
      vertLines: { color: "rgba(219,219,222,0.04)" },
      horzLines: { color: "rgba(219,219,222,0.04)" },
    },
    timeScale: {
      borderColor: "rgba(219,219,222,0.10)",
      timeVisible: true, secondsVisible: false,
      fixLeftEdge: true, fixRightEdge: true,
    },
    rightPriceScale: {
      borderColor: "rgba(219,219,222,0.10)",
      autoScale: true,
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScale: false,
    handleScroll: false,
    width: el.clientWidth,
    height: el.clientHeight || 160,
  });
  _pnlSeries = _pnlChart.addLineSeries({
    color: "#34d399",
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: true,
  });
}

/** trades 전체를 기간 토글로 필터 → 누적 P&L 라인 갱신. */
function renderPnLChart(trades) {
  _ensurePnLChart();
  if (!_pnlChart || !_pnlSeries) return;

  // 기간 필터 — 전체 / N일.
  const nowMs = Date.now();
  let filtered = (trades || []).filter(
    (t) => typeof t.closed_at_ts === "number"
      && typeof t.pnl_usd === "number",
  );
  if (pnlRange !== "all") {
    const days = parseInt(pnlRange, 10);
    if (!isNaN(days) && days > 0) {
      const cutoffMs = nowMs - days * 24 * 60 * 60 * 1000;
      filtered = filtered.filter((t) => t.closed_at_ts >= cutoffMs);
    }
  }
  // 시간 오름차순 (서버는 신→구).
  filtered.sort((a, b) => a.closed_at_ts - b.closed_at_ts);

  // 누적 PnL 계산 — {time(sec), value}.
  // 같은 시각 거래 dedup (lightweight-charts 가 time 중복 거부).
  let cum = 0;
  const points = [];
  const seenTs = new Set();
  filtered.forEach((t) => {
    cum += Number(t.pnl_usd);
    let ts = Math.floor(t.closed_at_ts / 1000);
    while (seenTs.has(ts)) ts += 1;  // 중복 시 1초씩 밀어서 회피.
    seenTs.add(ts);
    points.push({ time: ts, value: Number(cum.toFixed(2)) });
  });

  // 색 — 마지막 누적이 양수면 green, 음수면 red.
  const finalCum = points.length ? points[points.length - 1].value : 0;
  const lineColor = finalCum >= 0 ? "#34d399" : "#fb7185";
  _pnlSeries.applyOptions({ color: lineColor });
  _pnlSeries.setData(points);
  try { _pnlChart.timeScale().fitContent(); } catch (e) { /* noop */ }

  // 헤더 우측 누적 표시.
  const cumEl = $("pnl-chart-cum");
  if (cumEl) {
    if (points.length === 0) {
      cumEl.textContent = "—";
      cumEl.className = "pnl-chart-cum";
    } else {
      const sign = finalCum >= 0 ? "+" : "";
      cumEl.textContent = `${sign}${finalCum.toFixed(2)} USDT`;
      cumEl.className = "pnl-chart-cum " + (finalCum >= 0 ? "pos" : "neg");
    }
  }
}

function _updatePnLRangeButtons() {
  document.querySelectorAll("#pnl-range-toggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.range === pnlRange);
  });
}

// 토글 버튼 — 클릭 시 pnlRange 변경 + 그래프 재렌더 (server 호출 X).
const _pnlRangeToggleEl = document.getElementById("pnl-range-toggle");
if (_pnlRangeToggleEl) {
  _pnlRangeToggleEl.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-range]");
    if (!btn) return;
    const r = btn.dataset.range;
    if (!VALID_PNL_RANGES.includes(r) || r === pnlRange) return;
    pnlRange = r;
    localStorage.setItem("aurora_ict_pnl_range", r);
    _updatePnLRangeButtons();
    renderPnLChart(_lastTrades);
  });
}
_updatePnLRangeButtons();

// ============================================================
// (2) 라이선스 카드 렌더 — /auth/status 응답 활용
// ============================================================
const LICENSE_KO = {
  referral: "레퍼럴",
  sub_30d: "구독 30일",
  sub_90d: "구독 90일",
  sub_365d: "구독 365일",
};

function _fmtDateYmd(iso) {
  // ISO 8601 → "YYYY-MM-DD" (KST 변환은 표시 정합성을 위해 local time).
  if (!iso || typeof iso !== "string") return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}

/** /auth/status 응답을 받아 라이선스 카드 갱신. status 가 null 이면 placeholder. */
function renderLicenseCard(status) {
  const typeEl = $("lc-type");
  const startEl = $("lc-start");
  const endEl = $("lc-end");
  const daysEl = $("lc-days-left");
  if (!typeEl || !startEl || !endEl || !daysEl) return;
  if (!status || !status.authenticated) {
    typeEl.textContent = "—";
    startEl.textContent = "—";
    endEl.textContent = "—";
    daysEl.textContent = "—";
    daysEl.className = "lc-v";
    return;
  }
  const lt = status.license_type || "referral";
  // 2026-05-30 i18n: 라이선스 type 라벨 사용자 언어.
  const ltLabel = window.AuroraI18n
    ? window.AuroraI18n.t("license." + lt)
    : (LICENSE_KO[lt] || lt);
  typeEl.textContent = ltLabel;
  startEl.textContent = _fmtDateYmd(status.created_at);

  // "남음" 라벨도 i18n.
  const tDaysLeft = window.AuroraI18n
    ? window.AuroraI18n.t("license.daysLeft")
    : "남음";
  const tUnlimited = window.AuroraI18n
    ? window.AuroraI18n.t("license.unlimited")
    : "무기한";

  if (!status.expires_at) {
    // referral — 무기한.
    endEl.textContent = tUnlimited;
    daysEl.textContent = "—";
    daysEl.className = "lc-v";
  } else {
    endEl.textContent = _fmtDateYmd(status.expires_at);
    // 잔여일 계산 — 만료까지 (음수면 0).
    const now = Date.now();
    const exp = new Date(status.expires_at).getTime();
    if (isNaN(exp)) {
      daysEl.textContent = "—";
      daysEl.className = "lc-v";
    } else {
      const days = Math.max(0, Math.ceil((exp - now) / (24 * 60 * 60 * 1000)));
      daysEl.textContent = `${days} ${tDaysLeft}`;
      // 7일 미만 critical, 30일 미만 warn.
      daysEl.className = "lc-v";
      if (days < 7) daysEl.classList.add("lc-days-crit");
      else if (days < 30) daysEl.classList.add("lc-days-warn");
    }
  }
}

// 2026-05-30 i18n: 언어 변경 시 라이선스 카드도 다시 그리기 (캐시된 status 사용).
let _lastAuthStatus = null;
window.addEventListener("aurora-i18n-changed", () => {
  if (_lastAuthStatus) renderLicenseCard(_lastAuthStatus);
});

// ============================================================
// UI 자동 버전 갱신 — /auth/status.app_version 비교 (2026-05-28)
// ============================================================
// 부팅 시점에 한 번 캡쳐, 이후 polling 응답마다 비교 — 다르면 location.reload.
// SaaS 재배포 후 사용자가 새로고침 안 해도 다음 polling 사이클 (≤ 5분) 안에
// 새 UI 코드가 자동 로드. single-user (.exe) 모드는 /auth/status 가 404 →
// app_version 응답 X → noop (이 함수 자체가 호출되지 않음).
let _bootedAppVersion = null;
// 같은 reload 가 중복 발사되지 않도록 가드 (polling 두 군데에서 동시 감지 케이스).
let _versionReloadScheduled = false;

/** /auth/status 응답에서 app_version 을 비교해 drift 있으면 자동 reload.
 *
 *  - 첫 호출: 부팅 버전 캡쳐만 (reload X).
 *  - 이후 호출: 부팅 버전과 다르면 1.5s 뒤 location.reload.
 *
 *  reload URL 에 cache-bust 쿼리가 박혀있어 새 코드가 즉시 적용됨.
 */
function _checkVersionDrift(statusResponse) {
  const v = statusResponse && statusResponse.app_version;
  if (!v) return;  // single-user / 구버전 SaaS — 필드 없으면 skip
  if (_bootedAppVersion === null) {
    _bootedAppVersion = v;
    return;
  }
  if (_bootedAppVersion === v) return;
  if (_versionReloadScheduled) return;
  _versionReloadScheduled = true;
  try {
    toast(`새 버전 적용 — 자동 새로고침 (${_bootedAppVersion} → ${v})`, false);
  } catch (_) { /* noop */ }
  setTimeout(() => location.reload(), 1500);
}

// 라이선스 카드 갱신 — 5분마다 /auth/status 재폴링 (만료일 잔여 표시 동기화).
async function refreshLicenseCard() {
  try {
    const opts = { credentials: "include" };
    const resp = await fetch(`${API}/auth/status`, opts);
    if (!resp.ok) return;
    const s = await resp.json();
    _lastAuthStatus = s;  // i18n 변경 시 재렌더용.
    renderLicenseCard(s);
    // 버전 drift 감지 — SaaS 재배포 직후 사용자 새로고침 없이 자동 reload.
    _checkVersionDrift(s);
  } catch (e) { /* noop */ }
}
setInterval(refreshLicenseCard, 5 * 60 * 1000);

// ============================================================
// 공지 banner — 우상단, dismissible
// 2026-05-28: 관리자가 Telegram bot 또는 /admin/notice 로 등록 → 모든 사용자에게 표시.
// localStorage 에 dismissed id 저장 — 같은 공지 다시 안 보임.
// 새 공지 (다른 id) 는 다시 표시.
// ============================================================
const _NOTICE_DISMISSED_KEY = "aurora_ict_dismissed_notices";

function _getDismissedNoticeIds() {
  try {
    const raw = localStorage.getItem(_NOTICE_DISMISSED_KEY);
    if (!raw) return new Set();
    return new Set(JSON.parse(raw));
  } catch (e) { return new Set(); }
}

function _markNoticeDismissed(id) {
  const set = _getDismissedNoticeIds();
  set.add(id);
  try { localStorage.setItem(_NOTICE_DISMISSED_KEY, JSON.stringify([...set])); }
  catch (e) { /* noop */ }
}

async function refreshNotice() {
  const banner = document.getElementById("notice-banner");
  if (!banner) return;
  try {
    const data = await api("/ict/notice");
    if (!data || !data.active || _getDismissedNoticeIds().has(data.id)) {
      banner.style.display = "none";
      return;
    }
    const msgEl = document.getElementById("notice-banner-msg");
    if (msgEl) msgEl.textContent = data.message || "";
    banner.className = "notice-banner sev-" + (data.severity || "info");
    banner.style.display = "flex";
    banner.dataset.noticeId = String(data.id);
  } catch (e) { /* noop — 미인증 시 401 정상 */ }
}

// 닫기 버튼 (이벤트 위임 — banner 가 동적으로 보였다 안 보였다 해도 작동)
document.addEventListener("click", (ev) => {
  const btn = ev.target && ev.target.closest && ev.target.closest("#notice-banner-close");
  if (!btn) return;
  const banner = document.getElementById("notice-banner");
  if (!banner) return;
  const id = parseInt(banner.dataset.noticeId || "0", 10);
  if (id > 0) _markNoticeDismissed(id);
  banner.style.display = "none";
});

// 30초 주기 polling — 새 공지 빠르게 감지 (등록 직후 max 30초 안에 표시).
// 2026-05-28 파트너 피드백: "새로고침해야 뜨네" → 5분 → 30초.
// 부담: GET /ict/notice 매 30초, 사용자별 1번 SQL 쿼리 → 무시 가능.
setInterval(refreshNotice, 30 * 1000);

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
// 2026-05-28: 모바일 가로에서 차트 toolbar viz-toggle 숨김 → 사이드바 viz-toggle-side
// 도 같이 표시. 두 토글 묶음의 active 상태를 vizEnabled 기준으로 동기화.
function _updateVizButtons() {
  document.querySelectorAll(
    "#viz-toggle button[data-viz], #viz-toggle-side button[data-viz]",
  ).forEach((b) => {
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

// 2026-05-28: viz-toggle 클릭 핸들러 — 차트 toolbar / 사이드바 양쪽에서 공유.
async function _handleVizClick(ev) {
  const btn = ev.target.closest("button[data-viz]");
  if (!btn) return;
  const key = btn.dataset.viz;
  if (!VIZ_KEYS.includes(key)) return;
  vizEnabled[key] = !vizEnabled[key];
  localStorage.setItem(`aurora_ict_viz_${key}`, vizEnabled[key] ? "1" : "0");
  _updateVizButtons();
  await fetchAndRender();
}
$("viz-toggle").addEventListener("click", _handleVizClick);
// 사이드바 viz-toggle-side — 모바일 가로에서 차트 toolbar viz 숨겼을 때 대체 진입점.
// 데스크탑에서도 노출되어 양쪽 어느 곳에서나 토글 가능.
// (2026-05-29 v2: 사이드바에선 페어 선택으로 대체. viz-toggle-side DOM 자체
// 제거됐지만 null-safe 라 동작. 차트 toolbar 의 viz-toggle 만 살아있음.)
const _vizToggleSide = $("viz-toggle-side");
if (_vizToggleSide) _vizToggleSide.addEventListener("click", _handleVizClick);

// 2026-05-29 PR C: 사이드바 페어 토글 — 즉시 가동/정지.
//
// 동작:
//   - active 표시는 백엔드의 running_symbols 가 진실값. localStorage 의 cache 만.
//   - 클릭 → 가동 중이면 /ict/stop?symbol=... 으로 정지, 미가동이면 /ict/start?symbol=...
//   - 가동 실패 (LIVE 키 미등록 등) 시 active 안 박힘 (백엔드 status 따라감).
//   - 페어 토글로 모든 페어 끌 수 있음 — 봇 자체 정지와 동일 효과 (마지막 1개 보호 X).
//
// 2026-06-06 페어 선택기 — Bybit 스타일 모달(가격/24H%/거래량 + 검색).
// 매매 페어: '+ 페어 추가' 버튼 → 모달 선택 → 가동. 켠 페어는 칩(클릭=정지).
// 차트 페어: 현재 심볼 버튼 → 모달 선택 → 차트 전환(매매와 별개).
let _tradablePairs = ["BTC/USDT:USDT", "ETH/USDT:USDT"];  // markets 실패 시 폴백
let _marketTickers = [];   // [{symbol,last,pct24h,volume}] — 거래대금 정렬
let _maxPairs = 5;
let _runningSymbols = new Set();

/** ccxt symbol → 표시명. 예: "BTC/USDT:USDT" → "BTC". */
function _symLabel(sym) {
  return (sym && sym.split("/")[0]) || sym || "?";
}

function _fmtPrice(v) {
  if (v == null || isNaN(v)) return "-";
  if (v >= 1000) return Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });
  if (v >= 1) return Number(v).toFixed(3);
  return Number(v).toFixed(5);
}
function _fmtPct(v) {
  if (v == null || isNaN(v)) return { text: "-", cls: "" };
  return { text: (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "%", cls: v >= 0 ? "up" : "down" };
}
function _fmtVol(v) {
  if (!v || isNaN(v)) return "-";
  if (v >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(0) + "K";
  return String(Math.round(v));
}

/** /ict/markets 로 거래 가능 페어 + 시세 + 상한 로드. */
async function loadTradablePairs() {
  try {
    const r = await api("/ict/markets");
    if (Array.isArray(r.pairs) && r.pairs.length) _tradablePairs = r.pairs;
    if (Array.isArray(r.tickers)) _marketTickers = r.tickers;
    if (r.max_pairs) _maxPairs = r.max_pairs;
  } catch (e) { /* 폴백(BTC/ETH) 유지 */ }
  _updateChartPairBtnLabel();
}

function _updateChartPairBtnLabel() {
  const btn = $("chart-pair-btn");
  if (btn) btn.textContent = `${_symLabel(currentChartSymbol)} ▾`;
}

function _renderRunningChips() {
  const box = $("running-pair-chips");
  if (!box) return;
  box.innerHTML = "";
  for (const sym of _runningSymbols) {
    const chip = document.createElement("span");
    chip.className = "pair-chip";
    chip.dataset.symbol = sym;
    chip.title = `${_symLabel(sym)} 정지`;
    chip.innerHTML = `${_symLabel(sym)} <span class="x">×</span>`;
    box.appendChild(chip);
  }
  const addBtn = $("pair-add-btn");
  if (addBtn) {
    const atLimit = _runningSymbols.size >= _maxPairs;
    addBtn.disabled = atLimit;
    addBtn.textContent = atLimit ? `최대 ${_maxPairs}개 가동 중` : "+ 페어 추가";
  }
}

/** 622 renderStatus 호환 — running set 갱신 + 칩 재렌더. */
function _updatePairButtonsFromRunning(runningSet) {
  _runningSymbols = runningSet;
  _renderRunningChips();
}

/** 백엔드 running-pairs 로 매매 칩 동기화. */
async function refreshRunningPairs() {
  try {
    const r = await api("/ict/running-pairs");
    _runningSymbols = new Set(r.running_symbols || []);
  } catch (e) { /* 인증 게이트 등 — 기존 상태 유지 */ }
  _renderRunningChips();
  return _runningSymbols;
}

async function _startPair(symbol) {
  try {
    await api(`/ict/start?symbol=${encodeURIComponent(symbol)}`, "POST");
    toast(`${_symLabel(symbol)} 가동`);
  } catch (e) {
    toast(`${_symLabel(symbol)} 가동 실패: ${e.message}`, true);
  }
  await refreshRunningPairs();
  await fetchAndRender();
}

async function _stopPair(symbol) {
  try {
    await api(`/ict/stop?symbol=${encodeURIComponent(symbol)}`, "POST");
    toast(`${_symLabel(symbol)} 정지`);
  } catch (e) {
    toast(`${_symLabel(symbol)} 정지 실패: ${e.message}`, true);
  }
  await refreshRunningPairs();
  await fetchAndRender();
}

// ── 페어 선택기 모달 ──────────────────────────────────────
let _pickerOnSelect = null;
let _pickerExclude = new Set();

function _renderPickerRows(filter) {
  const list = $("pair-picker-list");
  if (!list) return;
  const q = (filter || "").trim().toUpperCase();
  // 시세(tickers) 우선, 없으면 심볼 목록만(가격 -).
  let rows = _marketTickers.length
    ? _marketTickers
    : _tradablePairs.map((s) => ({ symbol: s, last: null, pct24h: null, volume: null }));
  rows = rows.filter((r) => !_pickerExclude.has(r.symbol));
  if (q) rows = rows.filter((r) => _symLabel(r.symbol).toUpperCase().includes(q));
  list.innerHTML = "";
  if (!rows.length) {
    list.innerHTML = '<div class="pair-picker-empty">결과 없음</div>';
    return;
  }
  for (const r of rows) {
    const pct = _fmtPct(r.pct24h);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "pair-picker-row";
    row.dataset.symbol = r.symbol;
    row.innerHTML =
      `<span class="ppr-sym">${_symLabel(r.symbol)}</span>` +
      `<span class="ppr-price">${_fmtPrice(r.last)}</span>` +
      `<span class="ppr-pct ${pct.cls}">${pct.text}</span>` +
      `<span class="ppr-vol">${_fmtVol(r.volume)}</span>`;
    list.appendChild(row);
  }
}

function openPairPicker(onSelect, excludeSet) {
  _pickerOnSelect = onSelect;
  _pickerExclude = excludeSet || new Set();
  const ov = $("pair-picker-overlay");
  const search = $("pair-picker-search");
  if (search) search.value = "";
  _renderPickerRows("");
  if (ov) ov.hidden = false;
  if (search) search.focus();
}

function closePairPicker() {
  const ov = $("pair-picker-overlay");
  if (ov) ov.hidden = true;
  _pickerOnSelect = null;
}

const _pickerOverlay = $("pair-picker-overlay");
if (_pickerOverlay) {
  _pickerOverlay.addEventListener("click", (e) => {
    if (e.target === _pickerOverlay) closePairPicker();  // 배경 클릭 닫기
  });
}
const _pickerClose = $("pair-picker-close");
if (_pickerClose) _pickerClose.addEventListener("click", closePairPicker);
const _pickerSearch = $("pair-picker-search");
if (_pickerSearch) {
  _pickerSearch.addEventListener("input", (e) => _renderPickerRows(e.target.value));
  _pickerSearch.addEventListener("keydown", (e) => { if (e.key === "Escape") closePairPicker(); });
}
const _pickerList = $("pair-picker-list");
if (_pickerList) {
  _pickerList.addEventListener("click", async (e) => {
    const row = e.target.closest(".pair-picker-row");
    if (!row || !row.dataset.symbol) return;
    const sym = row.dataset.symbol;
    const cb = _pickerOnSelect;
    closePairPicker();
    if (cb) await cb(sym);
  });
}

// 매매 페어 추가 버튼 → 모달(켠 페어 제외).
const _pairAddBtn = $("pair-add-btn");
if (_pairAddBtn) {
  _pairAddBtn.addEventListener("click", () => {
    if (_runningSymbols.size >= _maxPairs) {
      toast(`최대 ${_maxPairs}개까지 가동 가능`, true);
      return;
    }
    openPairPicker(_startPair, _runningSymbols);
  });
}

// 켠 페어 칩 — 클릭 시 정지.
const _runningChipsBox = $("running-pair-chips");
if (_runningChipsBox) {
  _runningChipsBox.addEventListener("click", async (e) => {
    const chip = e.target.closest(".pair-chip");
    if (!chip || !chip.dataset.symbol) return;
    await _stopPair(chip.dataset.symbol);
  });
}

// 차트 페어 버튼 → 모달(전체). 선택 시 차트 심볼만 전환.
const _chartPairBtn = $("chart-pair-btn");
if (_chartPairBtn) {
  _chartPairBtn.addEventListener("click", () => {
    openPairPicker(async (sym) => {
      if (sym === currentChartSymbol) return;
      currentChartSymbol = sym;
      localStorage.setItem("aurora_ict_chart_symbol", sym);
      _updateChartPairBtnLabel();
      _chartViewInitialized = false;  // 페어 바뀌면 최근 N봉 zoom 재적용
      _showCachedChartIfAny();        // 캐시 있으면 즉시 표시
      await fetchAndRender();         // fresh 갱신
    }, new Set());
  });
}

// bootstrap — 페어 목록/시세 로드 후 running 동기화.
_updateChartPairBtnLabel();
loadTradablePairs().then(() => refreshRunningPairs());

$("tf-toggle").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-tf]");
  if (!btn) return;
  const tf = btn.dataset.tf;
  if (!VALID_TFS.includes(tf) || tf === currentTimeframe) return;
  currentTimeframe = tf;
  localStorage.setItem("aurora_ict_tf", tf);
  _updateTfButtons();
  _chartViewInitialized = false;  // TF 바뀌면 다시 최근 N봉 zoom
  _showCachedChartIfAny();  // 캐시 있으면 즉시 표시(체감 즉답)
  await fetchAndRender();   // fresh 갱신
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

// 2026-05-30 파트너 요청: 등록/재등록 (cred-action) 클릭 → 입력 폼 표시 (mode 자동).
document.querySelectorAll("[data-cred-reenter]").forEach((btn) => {
  btn.onclick = () => {
    const mode = btn.getAttribute("data-cred-reenter");
    $("cred-form-block").style.display = "flex";
    $("cred-saved-block").style.display = "none";
    // mode 토글도 그쪽으로.
    document.querySelectorAll("[data-cred-mode]").forEach((b) => {
      b.classList.toggle("active", b.dataset.credMode === mode);
    });
    $("cred-api-key").focus();
  };
});

// 2026-05-30 파트너 요청: 삭제 버튼 → DELETE /auth/api-keys?mode=demo|live.
document.querySelectorAll("[data-cred-delete]").forEach((btn) => {
  btn.onclick = async () => {
    const mode = btn.getAttribute("data-cred-delete");
    const modeLabel = mode.toUpperCase();
    const tConfirm = window.AuroraI18n
      ? window.AuroraI18n.t("confirm.deleteKey").replace("{mode}", modeLabel)
      : `${modeLabel} API 키를 삭제할까요?`;
    if (!confirm(tConfirm)) return;
    try {
      await api(`/auth/api-keys?mode=${mode}`, "DELETE");
      toast(`${modeLabel} ${tConfirm.includes("Delete") ? "deleted" : "삭제됨"}`);
      await fetchAndRender();
    } catch (e) {
      toast(e.message, true);
    }
  };
});

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
// 2026-05-30 파트너 요청: 4개 Killzone 가로 배지 활성 갱신.
// NY local 시간 기준 — backend session_status 무관하게 client 가 직접 계산
// (zoneinfo 없어도 Intl API 로 NY tz 변환 OK).
//
// Killzone 시간 (NY EST/EDT, DST 자동):
//   Asian:  19:00 ~ 24:00
//   London: 02:00 ~ 05:00
//   NY AM:  07:00 ~ 10:00
//   NY PM:  13:30 ~ 16:00
//
// London Close (10:00~12:00) 는 UI 4-배지 정책 (파트너 결정) 으로 표시 X.
// ============================================================
function _getActiveKillzoneNY() {
  const now = new Date();
  // NY local "HH:mm" 추출 (DST 자동 처리).
  const nyStr = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(now);
  // "24:30" → "00:30" 정규화 (Intl 가 가끔 24 반환).
  const [hRaw, mRaw] = nyStr.split(":").map(Number);
  const h = hRaw === 24 ? 0 : hRaw;
  const mins = h * 60 + mRaw;
  if (mins >= 19 * 60) return "asian";          // 19:00 ~ 24:00
  if (mins >= 2 * 60 && mins < 5 * 60) return "london";   // 02:00 ~ 05:00
  if (mins >= 7 * 60 && mins < 10 * 60) return "ny_am";   // 07:00 ~ 10:00
  if (mins >= 13 * 60 + 30 && mins < 16 * 60) return "pm"; // 13:30 ~ 16:00
  return null;
}

// NY 요일이 토/일이면 주말(NYSE 휴장) — 봇은 평일만 매매.
function _isNYWeekend() {
  const wd = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", weekday: "short",
  }).format(new Date());
  return wd === "Sat" || wd === "Sun";
}

function _updateKillzoneBar() {
  const weekend = _isNYWeekend();
  // 주말이면 어떤 killzone 도 활성 아님 + 주말 배지 표시.
  const active = weekend ? null : _getActiveKillzoneNY();
  const wkEl = document.getElementById("weekend-badge");
  if (wkEl) {
    wkEl.hidden = !weekend;
    wkEl.classList.toggle("active", weekend);
  }
  document.querySelectorAll("#killzone-bar .kz-badge:not(.weekend-badge)").forEach((el) => {
    el.classList.toggle("active", el.dataset.kz === active);
    // 2026-05-30 i18n: data-ny-start / data-ny-end (NY local "HH:mm") 를
    // window.AuroraI18n.nyTimeToUserTz 로 사용자 timezone 으로 변환.
    if (window.AuroraI18n && window.AuroraI18n.nyTimeToUserTz) {
      const tEl = el.querySelector(".kz-time");
      if (!tEl) return;
      const nyStart = el.dataset.nyStart || "";
      const nyEnd = el.dataset.nyEnd || "";
      try {
        const [sH, sM] = nyStart.split(":").map(Number);
        const [eH, eM] = nyEnd.split(":").map(Number);
        const startUser = window.AuroraI18n.nyTimeToUserTz(sH, sM);
        const endUser = window.AuroraI18n.nyTimeToUserTz(eH, eM);
        tEl.textContent = `${startUser}-${endUser}`;
      } catch (e) {
        tEl.textContent = `${nyStart}-${nyEnd}`;
      }
    }
  });
}
setInterval(_updateKillzoneBar, 30_000);
_updateKillzoneBar();

// 2026-05-30 i18n: 언어/시간대 변경 시 killzone 배지 즉시 갱신.
window.addEventListener("aurora-i18n-changed", () => {
  if (window.AuroraI18n) window.AuroraI18n.applyI18nToDOM();
  _updateKillzoneBar();
  // 2026-05-30: 봇 판단 패널 즉시 재렌더 (롱/숏/해석 라벨 변경).
  try { refreshJudgment(); } catch (e) {}
});
window.addEventListener("aurora-tz-changed", () => {
  _updateKillzoneBar();
  // 2026-05-30: 차트 시간축도 새 tz 로 갱신. applyOptions 로 옵션 갱신 후
  // timeScale 재그리기 강제.
  try {
    chart.applyOptions({
      timeScale: { tickMarkFormatter: _userTzTickFormatter },
      localization: { timeFormatter: _userTzCrosshairFormatter },
    });
  } catch (e) {}
});

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
    // 2026-05-30 i18n: 롱/숏 라벨 + 해석 prefix 사용자 언어.
    const tLong = window.AuroraI18n ? window.AuroraI18n.t("judgment.long") : "롱";
    const tShort = window.AuroraI18n ? window.AuroraI18n.t("judgment.short") : "숏";
    const tInterp = window.AuroraI18n ? window.AuroraI18n.t("judgment.interpretation") : "해석";
    const tAccum = window.AuroraI18n ? window.AuroraI18n.t("judgment.accumulating") : "분석 자료 누적 중…";
    $("jp-dir-pct").innerHTML =
      `<span style="color:#34d399">${tLong} ${longPct.toFixed(1)}%</span>` +
      `<span style="color:#fb7185">${tShort} ${shortPct.toFixed(1)}%</span>`;
    // Reasons
    const rs = Array.isArray(j.reasons) ? j.reasons : [];
    $("jp-reasons").innerHTML = rs.length === 0
      ? `<div class="jp-reason-interp">${tAccum}</div>`
      : rs.map((r) => `
        <div class="jp-reason">
          <div class="jp-reason-top">
            <span class="jp-reason-dot ${r.color || 'yellow'}"></span>
            <span class="jp-reason-term">${escapeHtml(r.term || '')}</span>
            <span class="jp-reason-range">${escapeHtml(r.range || '')}</span>
          </div>
          <div class="jp-reason-interp">${tInterp}: ${escapeHtml(r.interpretation || '')}</div>
        </div>
      `).join("");
    // 2026-05-29: 진단 인지 보강 — hint + recent setups + diagnostics.
    // 파트너 의문 "long 비율 우세인데 short 만 진입" 직접 해소.
    const hint = j.direction?.hint;
    const recent = j.direction?.recent_setups || {};
    let hintHtml = "";
    if (hint) {
      hintHtml += `<div class="jp-dir-hint">${escapeHtml(hint)}</div>`;
    }
    if (recent.long != null || recent.short != null) {
      const l = recent.long || 0, s = recent.short || 0;
      const tRecentLast = window.AuroraI18n
        ? window.AuroraI18n.t("judgment.recentLast")
        : "최근";
      const tRecentTitle = window.AuroraI18n
        ? window.AuroraI18n.t("judgment.recentEntries")
        : "최근 진입 분포";
      const lastDir = recent.last ? ` · ${tRecentLast}=${recent.last}` : "";
      hintHtml += `<div class="jp-dir-recent">` +
        `${tRecentTitle} — <span style="color:#34d399">${tLong} ${l}</span> / ` +
        `<span style="color:#fb7185">${tShort} ${s}</span>${escapeHtml(lastDir)}` +
        `</div>`;
    }
    const dirHintEl = $("jp-dir-hint-area");
    if (dirHintEl) dirHintEl.innerHTML = hintHtml;
    // Entry Condition
    const ec = j.entry_condition || {};
    $("jp-entry").innerHTML =
      `<div>${escapeHtml(ec.title || '—')}</div>` +
      (ec.detail ? `<div class="jp-entry-detail">${escapeHtml(ec.detail)}</div>` : "");
    // Diagnostics — silent failure 가시화. 정상이면 표시 안 함, 비정상이면 강조.
    const d = j.diagnostics || {};
    const alerts = [];
    if (d.recovery_failed) {
      alerts.push("⚠ 거래소 포지션 복원 실패 — 신규 진입 차단 중");
    }
    if ((d.sync_failure_streak || 0) >= 5) {
      alerts.push(`⚠ fetch_position 연속 ${d.sync_failure_streak}회 실패 — API/네트워크 점검`);
    }
    if ((d.order_failure_count || 0) > 0) {
      alerts.push(`⚠ 진입 주문 실패 누적 ${d.order_failure_count}건`);
    }
    if ((d.tpsl_failure_streak || 0) >= 3) {
      alerts.push(`⚠ SL/TP 설정 연속 ${d.tpsl_failure_streak}회 실패`);
    }
    const diagEl = $("jp-diagnostics");
    if (diagEl) {
      diagEl.innerHTML = alerts.length === 0
        ? ""
        : alerts.map((a) => `<div class="jp-diag-alert">${escapeHtml(a)}</div>`).join("");
      diagEl.style.display = alerts.length === 0 ? "none" : "block";
    }
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

// 햄버거 토글 — 우측 P&L 패널 / 좌측 사이드바 / 봇 판단 패널 (모바일 가로)
// 2026-05-27: 파트너 피드백 ("저 pnl 창 말한거양") — 좌측 사이드바가 아니라 우측 P&L 토글
// 2026-05-28: 모바일 가로 모드 재설계 — 셋 다 default hidden + overlay 형식.
//   · 좌상단 햄버거 (≡)  → 좌측 사이드바 (.sidebar-open)
//   · 우상단 햄버거 (≡)  → 우측 P&L 패널 (.pnl-open)
//   · 우하단 fab (ⓘ)     → 좌하단 봇 판단 패널 (.judgment-open)
//   · 셋은 서로 배타적 (하나 열면 나머지 자동 닫힘)
//   · overlay 클릭 시 모두 닫힘
// 데스크탑에서는 기존처럼 우측 P&L 만 'pnl-collapsed' 토글.
function _isMobileLandscape() {
  return window.matchMedia(
    "(orientation: landscape) and (max-width: 1024px)," +
    "(orientation: landscape) and (max-height: 600px)",
  ).matches;
}

// 2026-05-28 v2 — 모바일/태블릿 세로(portrait) 모드도 풀스크린 레이아웃으로 작동.
// CSS 미디어 쿼리 (orientation: portrait) and (max-width: 1024px) 와 매칭.
function _isMobilePortrait() {
  return window.matchMedia(
    "(orientation: portrait) and (max-width: 1024px)",
  ).matches;
}

// 가로 OR 세로 — 모바일/태블릿 (overlay 토글 UI 가 활성화되는 모든 케이스).
// P&L 햄버거 분기 / resize 시 잔존 클래스 정리 등에서 사용.
function _isMobile() {
  return _isMobileLandscape() || _isMobilePortrait();
}

/** 모바일 가로 — 지정한 패널만 열고 나머지 자동 닫음.
 *  panel: "sidebar" | "pnl" | "judgment" | "positions" | null. */
function _setMobilePanel(panel) {
  const body = document.body;
  body.classList.toggle("sidebar-open",   panel === "sidebar");
  body.classList.toggle("pnl-open",       panel === "pnl");
  body.classList.toggle("judgment-open",  panel === "judgment");
  body.classList.toggle("positions-open", panel === "positions");
  // fab active 시각 동기화 — 해당 패널 열렸을 때만 fab 강조.
  const fab = document.getElementById("btn-judgment-toggle");
  if (fab) fab.classList.toggle("active", panel === "judgment");
  const pfab = document.getElementById("btn-positions-toggle");
  if (pfab) pfab.classList.toggle("active", panel === "positions");
}

/** POSITIONS fab badge / 강조 갱신 — renderPositions 가 호출.
 *  active 포지션 있으면 fab 에 녹색 glow + badge 표시 (모바일 가로에서만 시각). */
function _syncPositionsFab(hasPosition) {
  const pfab = document.getElementById("btn-positions-toggle");
  if (!pfab) return;
  pfab.classList.toggle("has-position", !!hasPosition);
  const badge = document.getElementById("positions-fab-badge");
  if (badge) {
    badge.style.display = hasPosition ? "block" : "none";
    badge.textContent = hasPosition ? "1" : "0";
  }
}

/** 현재 어떤 패널이 열려있는지 — 가로 모드 토글 핸들러용. */
function _currentMobilePanel() {
  const c = document.body.classList;
  if (c.contains("sidebar-open"))   return "sidebar";
  if (c.contains("pnl-open"))       return "pnl";
  if (c.contains("judgment-open"))  return "judgment";
  if (c.contains("positions-open")) return "positions";
  return null;
}

const _btnSidebarToggle = $("btn-sidebar-toggle");
if (_btnSidebarToggle) {
  _btnSidebarToggle.onclick = () => {
    // 2026-05-28 v2 — 가로 OR 세로 모바일 모두 overlay 토글, 데스크탑만 접기/펴기.
    if (_isMobile()) {
      // 모바일 (가로/세로) — P&L overlay 열기/닫기 + 서로 배타.
      _setMobilePanel(_currentMobilePanel() === "pnl" ? null : "pnl");
    } else {
      // 데스크탑 — 기존 접기/펴기.
      // 2026-05-30 파트너 보고: 두 번째 토글 시 사이드바 압축 + 차트 깨짐 버그.
      // 원인: chart 가 옛 큰 width 그대로 유지 → chart-wrap 이 viewport 초과
      // → flex layout 사이드바 압축. 단발성 raf+fit 만으론 timing 부족.
      // Fix: chart 를 즉시 0 으로 압축 → flex 가 자유롭게 재계산 → 두 번
      // raf 후 chart-wrap 새 폭으로 정확히 적용. ResizeObserver 가 추가로
      // 모든 layout 변화 자동 캐치 (chart-wrap 부착 — 아래 별도 등록).
      document.body.classList.toggle("pnl-collapsed");
      try { chart.applyOptions({ width: 0 }); } catch (e) {}
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { try { fit(); } catch (e) {} });
      });
    }
  };
}

// 2026-05-30: chart-wrap 의 모든 layout 변화 자동 감지 → fit() 호출.
// 햄버거 토글 외 다른 경로 (창 크기, font 로딩 등) 에서도 chart 가 정확히
// 따라옴. ResizeObserver fallback 처리 — 옛 브라우저는 window resize 만 의존.
if (typeof ResizeObserver !== "undefined") {
  const _chartWrap = document.querySelector(".chart-wrap");
  if (_chartWrap) {
    const _ro = new ResizeObserver(() => {
      // 동기 호출하면 ResizeObserver loop 경고 → next frame.
      requestAnimationFrame(() => { try { fit(); } catch (e) {} });
    });
    _ro.observe(_chartWrap);
  }
}

// 좌상단 모바일 사이드바 햄버거 — 가로 모드에서만 활성 (CSS).
const _btnMobileSidebar = $("btn-mobile-sidebar");
if (_btnMobileSidebar) {
  _btnMobileSidebar.onclick = () => {
    _setMobilePanel(_currentMobilePanel() === "sidebar" ? null : "sidebar");
  };
}

// 우하단 봇 판단 패널 fab — 모바일 가로에서만 표시 (CSS).
// 데스크탑에서는 좌하단 판단 패널이 항상 보임 → fab 클릭 자체가 발생 X.
const _btnJudgmentToggle = $("btn-judgment-toggle");
if (_btnJudgmentToggle) {
  _btnJudgmentToggle.onclick = () => {
    _setMobilePanel(_currentMobilePanel() === "judgment" ? null : "judgment");
  };
}

// 2026-05-28: 우하단 POSITIONS fab — 모바일 가로에서만 표시 (CSS).
// 데스크탑에서는 positions-panel 이 차트 아래 항상 보임 → fab 자체 hidden.
const _btnPositionsToggle = $("btn-positions-toggle");
if (_btnPositionsToggle) {
  _btnPositionsToggle.onclick = () => {
    _setMobilePanel(_currentMobilePanel() === "positions" ? null : "positions");
  };
}

// Overlay 클릭 — 열린 패널 모두 닫음 (외부 탭 close UX).
const _mobileOverlay = $("mobile-overlay");
if (_mobileOverlay) {
  _mobileOverlay.onclick = () => _setMobilePanel(null);
}

// 가로↔세로 회전 / 데스크탑↔모바일 viewport 전환 시 잔존 클래스 정리.
// (데스크탑으로 넘어왔는데 sidebar-open 클래스 남아 있으면 사이드바 absolute 자리 X →
//  display 는 정상이지만 시각 일관성 위해 클래스 제거.)
// 2026-05-28 v2 — 모바일 세로도 overlay UI 쓰므로, "모바일 자체가 아닐 때" 에만 reset.
//   가로↔세로 전환은 어차피 같은 클래스 셋 → reset 불필요 (열린 패널 그대로 유지).
function _resetMobilePanelsIfDesktop() {
  if (!_isMobile()) {
    document.body.classList.remove(
      "sidebar-open", "pnl-open", "judgment-open", "positions-open",
    );
    const fab = document.getElementById("btn-judgment-toggle");
    if (fab) fab.classList.remove("active");
    const pfab = document.getElementById("btn-positions-toggle");
    if (pfab) pfab.classList.remove("active");
  }
}
window.addEventListener("resize", _resetMobilePanelsIfDesktop);
window.addEventListener("orientationchange", () => setTimeout(_resetMobilePanelsIfDesktop, 100));

// ============================================================
// resize
// ============================================================
function fit() {
  const r = chartEl.getBoundingClientRect();
  chart.applyOptions({ width: r.width, height: r.height });
  // P&L 차트도 같이 — 우측 패널 너비 변화 / 모바일 회전 시 폭 재계산.
  if (_pnlChart) {
    const pe = document.getElementById("pnl-chart");
    if (pe) {
      const w = pe.clientWidth;
      const h = pe.clientHeight || 160;
      try { _pnlChart.applyOptions({ width: w, height: h }); } catch (e) { /* noop */ }
    }
  }
}
window.addEventListener("resize", fit);
window.addEventListener("orientationchange", () => setTimeout(fit, 100));
fit();

// 부팅 시 version 1회 fetch (config endpoint)
async function fetchVersion() {
  try {
    const cfg = await api("/ict/config");
    const el = document.getElementById("app-version");
    if (el && cfg && cfg.version) el.textContent = `v${cfg.version}`;
  } catch (e) { /* noop */ }
}

// 초기 fetch + polling (10s 주기) — bootstrap 이 시작 결정.
setInterval(fetchAndRender, 10_000);

// ============================================================
// SaaS 인증 게이트 — bootstrap()
// 2026-05-28: 다중 사용자 모드 (FastAPI /auth/*) 가 켜져 있으면 게이트 표시.
// single-user (.exe / pywebview) 모드에서는 /auth/status 가 404 → 게이트 hide.
// ============================================================
const AUTH = {
  state: "boot",         // boot | setup | login | keys | main
  userCode: null,
  multiUser: false,      // /auth/status 가 200 응답하면 true
};

function _showOnly(...ids) {
  ["auth-setup", "auth-login", "auth-keys"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = ids.includes(id) ? "block" : "none";
  });
}

function _hideAuthError(prefix) {
  const e = document.getElementById(`${prefix}-error`);
  if (e) { e.style.display = "none"; e.textContent = ""; }
}

function _showAuthError(prefix, msg) {
  const e = document.getElementById(`${prefix}-error`);
  if (!e) return;
  e.textContent = msg;
  e.style.display = "block";
}

function showAuthGate(panel) {
  // 2026-05-28 파트너 피드백 — PIN 설정 입력 중에 polling 401 핸들러가
  // 자동으로 login 패널로 전환시키는 버그. 가드:
  // panel 인자 명시 안 됐고 이미 인증 게이트 표시 중이면 noop (현재 패널 유지).
  const gate = document.getElementById("auth-gate");
  if (panel === undefined && gate && gate.style.display !== "none") {
    return;
  }
  panel = panel || "login";
  const main = document.getElementById("main-screen");
  if (gate) gate.style.display = "flex";
  if (main) main.style.display = "none";
  if (panel === "setup") _showOnly("auth-setup");
  else if (panel === "keys") _showOnly("auth-keys");
  else _showOnly("auth-login");
  AUTH.state = panel;
}

function showMainScreen() {
  const gate = document.getElementById("auth-gate");
  const main = document.getElementById("main-screen");
  if (gate) gate.style.display = "none";
  if (main) main.style.display = "flex";
  AUTH.state = "main";
  // 2026-05-28 fix — auth-gate 가 화면 가린 동안 차트 영역 0x0 박혀있어서
  // 그 size 로 차트 초기화됨. F12 (DevTools) 열면 viewport 변화로 resize → 봉
  // 보이게 된 게 그 증상. main-screen 표시 직후 layout 끝나면 명시 resize.
  requestAnimationFrame(() => {
    try { fit(); } catch (e) { /* noop */ }
    try { chart.timeScale().fitContent(); } catch (e) { /* noop */ }
    if (typeof _pnlChart !== "undefined" && _pnlChart) {
      try { _pnlChart.timeScale().fitContent(); } catch (e) { /* noop */ }
    }
  });
}

// PIN 강도 평가 — server-side validate_pin_strength 와 가벼운 일관 (UX hint).
// 실제 검증은 백엔드가 단일 진실. 여기선 UI feedback 용.
function _evalPinStrength(pin) {
  if (!pin) return { score: 0, label: "" };
  let score = 0;
  if (pin.length >= 8) score++;
  if (/[A-Za-z]/.test(pin)) score++;
  if (/\d/.test(pin)) score++;
  if (/[^A-Za-z0-9]/.test(pin)) score++;
  const labels = ["너무 짧음", "약함", "보통", "양호", "강함"];
  return { score, label: labels[score] || "" };
}

function _bindPinStrengthMeter() {
  const input = document.getElementById("setup-pin");
  const meter = document.getElementById("setup-pin-strength");
  const lbl = document.getElementById("setup-pin-strength-label");
  if (!input || !meter || !lbl) return;
  input.addEventListener("input", () => {
    const r = _evalPinStrength(input.value);
    meter.className = "pin-strength " +
      (r.score >= 4 ? "s-strong"
        : r.score === 3 ? "s-good"
        : r.score === 2 ? "s-fair"
        : r.score >= 1 ? "s-weak" : "");
    lbl.textContent = input.value ? `강도: ${r.label}` : "8자 이상, 영문/숫자/특수문자 혼합";
  });
}

async function _checkAuthStatus() {
  // SaaS 모드면 /auth/status 가 200 + {authenticated, ...} 반환.
  // single-user 모드면 404 — fetch error 던짐 → multiUser=false 로 판단.
  try {
    const opts = { credentials: "include" };
    const resp = await fetch(`${API}/auth/status`, opts);
    if (!resp.ok) {
      if (resp.status === 404) return { multiUser: false };
      throw new Error(`status ${resp.status}`);
    }
    const s = await resp.json();
    return { multiUser: true, status: s };
  } catch (e) {
    // 네트워크 오류 — single-user 가정 + 게이트 hide.
    return { multiUser: false };
  }
}

async function bootstrap() {
  const r = await _checkAuthStatus();
  AUTH.multiUser = r.multiUser;
  if (!r.multiUser) {
    // single-user (.exe / pywebview) — 게이트 한 번도 표시 X, 기존 흐름 그대로.
    // 라이선스 카드는 SaaS 다중 사용자용 — single-user 면 의미 X, 숨김.
    const lc = document.getElementById("license-card");
    if (lc) lc.style.display = "none";
    showMainScreen();
    fetchVersion();
    fetchAndRender();
    return;
  }
  // SaaS 모드 — 로그아웃 행 표시 + 상태별 패널 분기.
  const lr = document.getElementById("logout-row");
  if (lr) lr.classList.add("show");

  const s = r.status || {};
  // 부팅 버전 캡쳐 — 이후 polling (refreshLicenseCard) 이 drift 감지에 사용.
  _checkVersionDrift(s);
  if (!s.authenticated) {
    showAuthGate(s.needs_pin_setup ? "setup" : "login");
    return;
  }
  AUTH.userCode = s.code || null;
  const codeEl = document.getElementById("user-code-display");
  if (codeEl && AUTH.userCode) codeEl.textContent = AUTH.userCode;

  if (!s.has_api_keys) {
    showAuthGate("keys");
    return;
  }
  // 인증 + 키 등록 완료 — 메인 봇 UI.
  showMainScreen();
  // 라이선스 카드 초기 렌더 (status 응답 그대로 활용 — /auth/status 재호출 생략).
  renderLicenseCard(s);
  fetchVersion();
  fetchAndRender();
}

// PIN 설정 버튼 / 로그인 / 키 저장 / 로그아웃 핸들러 등록.
// 2026-05-28 fix: cache-bust 인라인 script 가 app.js 를 동적 append 하면
// DOMContentLoaded 이벤트는 이미 발생 후 → addEventListener 가 잡지 못함.
// readyState 가드로 양쪽 케이스 모두 호환.
function _bindAuthHandlers() {
  _bindPinStrengthMeter();

  const btnSetup = document.getElementById("btn-setup-pin");
  if (btnSetup) btnSetup.onclick = async () => {
    _hideAuthError("setup");
    const code = (document.getElementById("setup-code").value || "").trim();
    const pin = document.getElementById("setup-pin").value || "";
    const pin2 = document.getElementById("setup-pin2").value || "";
    if (!code) return _showAuthError("setup", "라이선스 코드를 입력하세요.");
    if (!pin || !pin2) return _showAuthError("setup", "PIN 과 PIN 확인을 모두 입력하세요.");
    if (pin !== pin2) return _showAuthError("setup", "PIN 과 PIN 확인이 일치하지 않습니다.");
    btnSetup.disabled = true;
    try {
      await api("/auth/setup-pin", "POST", { code, pin, pin_confirm: pin2 });
      // 자동 로그인 됨 — bootstrap 다시 돌려서 키 등록 화면으로 진입.
      await bootstrap();
    } catch (e) {
      _showAuthError("setup", `등록 실패 — ${e.message}`);
    } finally {
      btnSetup.disabled = false;
    }
  };

  const btnLogin = document.getElementById("btn-login");
  if (btnLogin) btnLogin.onclick = async () => {
    _hideAuthError("login");
    const code = (document.getElementById("login-code").value || "").trim();
    const pin = document.getElementById("login-pin").value || "";
    if (!code || !pin) return _showAuthError("login", "코드와 PIN 을 입력하세요.");
    btnLogin.disabled = true;
    try {
      await api("/auth/login", "POST", { code, pin });
      await bootstrap();
    } catch (e) {
      _showAuthError("login", `로그인 실패 — ${e.message}`);
    } finally {
      btnLogin.disabled = false;
    }
  };

  // 2026-05-29: 인증 게이트 키 등록 — auth-mode-toggle (DEMO/LIVE) 핸들러.
  const authModeRoot = document.getElementById("auth-mode-toggle");
  if (authModeRoot) {
    const hint = document.getElementById("auth-mode-hint");
    authModeRoot.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-auth-mode]");
      if (!btn) return;
      authModeRoot.querySelectorAll("button").forEach((b) =>
        b.classList.remove("active"),
      );
      btn.classList.add("active");
      if (hint) {
        const m = btn.dataset.authMode;
        hint.innerHTML = m === "live"
          ? '⚠ <b style="color:#fb7185">LIVE</b> 슬롯 — 실거래 키. Demo 키와 혼동 주의.'
          : 'Bybit <b>Demo Trading</b> 키 (api-demo.bybit.com). 처음 사용자는 데모로 시작 권장.';
      }
    });
  }

  const btnKeys = document.getElementById("btn-save-keys");
  if (btnKeys) btnKeys.onclick = async () => {
    _hideAuthError("keys");
    const api_key = (document.getElementById("keys-api-key").value || "").trim();
    const api_secret = (document.getElementById("keys-api-secret").value || "").trim();
    if (!api_key || !api_secret) {
      return _showAuthError("keys", "API Key 와 Secret 을 모두 입력하세요.");
    }
    const activeBtn = document.querySelector(
      "#auth-mode-toggle button[data-auth-mode].active",
    );
    const mode = activeBtn ? activeBtn.dataset.authMode : "demo";
    if (mode === "live") {
      // 인증 게이트에서도 LIVE 슬롯 등록은 한 번 더 확인 — 첫 사용자가 실수로 LIVE 키
      // 박는 사고 방지. (UI 흐름상 신규 가입자는 데모로 시작 권장.)
      const ok = confirm(
        "LIVE 슬롯에 키를 저장합니다.\n\n" +
        "실거래 (api.bybit.com) 키여야 합니다. Demo 키와 다릅니다.\n" +
        "계속하시겠습니까?",
      );
      if (!ok) return;
    }
    btnKeys.disabled = true;
    try {
      await api("/auth/api-keys", "POST", { api_key, api_secret, mode });
      await bootstrap();
    } catch (e) {
      _showAuthError("keys", `저장 실패 — ${e.message}`);
    } finally {
      btnKeys.disabled = false;
    }
  };

  const swLogin = document.getElementById("switch-to-login");
  if (swLogin) swLogin.onclick = () => showAuthGate("login");
  const swSetup = document.getElementById("switch-to-setup");
  if (swSetup) swSetup.onclick = () => showAuthGate("setup");

  const btnLogout = document.getElementById("btn-logout");
  if (btnLogout) btnLogout.onclick = async () => {
    try { await api("/auth/logout", "POST"); } catch (_) { /* noop */ }
    location.reload();
  };
  const keysLogout = document.getElementById("keys-logout");
  if (keysLogout) keysLogout.onclick = async () => {
    try { await api("/auth/logout", "POST"); } catch (_) { /* noop */ }
    location.reload();
  };
}
// DOM 이미 준비됐으면 즉시, 아니면 DOMContentLoaded 대기.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _bindAuthHandlers);
} else {
  _bindAuthHandlers();
}

// 부팅 — DOM 준비 즉시 게이트/메인 결정.
bootstrap();

// 공지 banner — 인증 후 1회 즉시 fetch (이후는 setInterval 5분 주기).
setTimeout(() => { refreshNotice(); }, 2000);
