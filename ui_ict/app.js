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

  // 봇 가동 상태 → START / STOP 시각 효과
  // 2026-05-27 파트너 피드백 — "stop 누르면 빨간 활성화 안돼"
  // 이전 조건은 (stopped && enabled) 였는데 STOP 클릭 시 enabled=false 도 같이
  // 되어서 active 안 박혔음. state 만 보고 둘 중 하나 켜지게 단순화.
  // 처음 진입 (running 한 번도 안 됨, stopped) 에도 STOP active — 일관성 위해.
  $("btn-start").classList.toggle("active", s.state === "running");
  $("btn-stop").classList.toggle("active", s.state === "stopped");

  // 2026-05-29: API Credentials — 양쪽 슬롯 (DEMO/LIVE) 상태 분리 표시.
  // 둘 중 하나라도 등록돼 있으면 saved-block 표시 (재등록 버튼 박스),
  // 둘 다 없으면 form-block 표시 (입력 폼). 부분 등록 상태도 saved-block 에서
  // "누락" 표시로 안내 → 사용자가 누락 슬롯 채우러 가도록.
  const credForm = $("cred-form-block");
  const credSaved = $("cred-saved-block");
  if (credForm && credSaved) {
    const hasDemo = !!s.has_demo_credentials;
    const hasLive = !!s.has_live_credentials;
    const anyKey = hasDemo || hasLive;
    if (anyKey) {
      credForm.style.display = "none";
      credSaved.style.display = "flex";
      const demoStatus = $("cred-demo-status");
      const liveStatus = $("cred-live-status");
      if (demoStatus) {
        demoStatus.textContent = hasDemo ? "✓ 등록됨" : "미등록";
        demoStatus.className = "cred-slot-status " + (hasDemo ? "ok" : "missing");
      }
      if (liveStatus) {
        liveStatus.textContent = hasLive ? "✓ 등록됨" : "미등록";
        liveStatus.className = "cred-slot-status " + (hasLive ? "ok" : "missing");
      }
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
  // 2026-05-28: 모바일 가로 positions fab badge / 강조 — 활성 포지션 유무 동기화.
  _syncPositionsFab(!!(pos && pos.active));

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
    // 2026-05-27 파트너 요청 — "거래소 시작(2020-03) 부터 지금까지 봉 전부".
    // TF 별 한도 — 큰 TF (1h~1w) 는 Bybit history 다 받기 가능, 작은 TF
    // (1m/5m) 는 메모리·시간 부담으로 합리적 max. ccxt fetch_ohlcv 가 history
    // 끝나면 자동 stop (max_pages=100, max 100,000봉).
    const CANDLE_LIMIT = {
      "1m": 5000, "5m": 20000, "15m": 50000,
      "1h": 60000, "2h": 30000, "4h": 15000,
      "1d": 5000, "1w": 1500,
    };
    // 마커 계산은 봉 수에 비례 → 5초 polling 부담 방지 위해 최근 2000봉만.
    // (옛 봉에는 마커 표시 X — 가격 캔들만 보임. trade-off 안내됨)
    const candleLimit = CANDLE_LIMIT[currentTimeframe] || 5000;
    const markerLimit = 2000;
    // 2026-05-28: P&L 누적 그래프 (전체기간) 용으로 limit=200 (백엔드 cap).
    // 리스트 화면은 상위 50건만 표시, 그래프는 전체를 활용.
    const [ohlcv, markers, position, pnl] = await Promise.all([
      api(`/ict/ohlcv?timeframe=${tf}&limit=${candleLimit}`),
      api(`/ict/markers?timeframe=${tf}&limit=${markerLimit}`),
      api("/ict/position"),
      api("/ict/closed_pnl?limit=200"),
    ]);
    candleSeries.setData(ohlcv.candles);
    // 2026-05-27: 차트 좌측 공백 제거 — 모든 봉 보이게 맞춤.
    try { chart.timeScale().fitContent(); } catch (e) { /* noop */ }
    // 마지막 봉 시간 — PD Zone area 끝점에 사용
    if (ohlcv.candles && ohlcv.candles.length > 0) {
      lastBarTimeSec = ohlcv.candles[ohlcv.candles.length - 1].time;
    }
    // 2026-05-28 fix — OHLCV cache miss 시 200봉 sync + background prefetch.
    // markers 는 전체 cache 봉 영역 계산해 반환 → 차트 봉보다 옛 영역에 마커만 떠
    // "봉 없는 라벨" 보임. ohlcv 의 first ts 이전 markers 전부 필터링.
    if (ohlcv.candles && ohlcv.candles.length > 0) {
      _filterMarkersByFirstBar(markers, ohlcv.candles[0].time);
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
  // 2026-05-29: Live 전환 강화된 확인. 1단계 안내 + 2단계 "LIVE" 타이핑 검증.
  // 서버 단도 has_api_keys("live") 가드 있지만 사용자 인지 향상 위해 한 번 더.
  const warn = (
    "⚠ LIVE (실거래) 모드로 전환합니다.\n\n" +
    "• 실제 자금이 즉시 거래에 사용됩니다.\n" +
    "• Bybit Live API 키가 등록돼 있어야 합니다 (Demo 키와 별도).\n" +
    "• 봇이 가동 중이면 자동 재시작됩니다.\n\n" +
    "계속하려면 OK 를 누르세요."
  );
  if (!confirm(warn)) return;
  const typed = prompt('확인을 위해 대문자로 "LIVE" 라고 입력하세요:');
  if (typed !== "LIVE") {
    toast("LIVE 입력 확인 실패 — 전환 취소", true);
    return;
  }
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
  try { await api("/ict/start", "POST"); toast("봇 시작됨"); await fetchAndRender(); }
  catch (e) { toast(e.message, true); }
};
$("btn-stop").onclick = async () => {
  // 2026-05-27: STOP 클릭 즉시 빨간 glow 표시. /ict/stop 가 pending 미체결 자동 취소.
  $("btn-stop").classList.add("active");
  $("btn-start").classList.remove("active");
  try { await api("/ict/stop", "POST"); toast("봇 중지됨"); await fetchAndRender(); }
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
  typeEl.textContent = LICENSE_KO[lt] || lt;
  startEl.textContent = _fmtDateYmd(status.created_at);

  if (!status.expires_at) {
    // referral — 무기한.
    endEl.textContent = "무기한";
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
      daysEl.textContent = `${days}일 남음`;
      // 7일 미만 critical, 30일 미만 warn.
      daysEl.className = "lc-v";
      if (days < 7) daysEl.classList.add("lc-days-crit");
      else if (days < 30) daysEl.classList.add("lc-days-warn");
    }
  }
}

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
const _vizToggleSide = $("viz-toggle-side");
if (_vizToggleSide) _vizToggleSide.addEventListener("click", _handleVizClick);

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
      document.body.classList.toggle("pnl-collapsed");
    }
  };
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
