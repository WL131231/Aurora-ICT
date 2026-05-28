/* Aurora-ICT UI — lightweight-charts + REST polling

   2026-05-28: SaaS 전환 — API base 를 same-origin 으로 변경.
   - single-user (.exe / pywebview):  http://127.0.0.1:8765/ui/ 에서 로드 → same origin
   - SaaS (Docker / 클라우드):       https://<도메인>/ui/ 에서 로드 → same origin
   상대 경로 ("") 사용하면 어떤 호스트에서 서빙되든 cookie session 자동 동작. */

const API = "";  // same-origin — fetch("/auth/status") 처럼 동작

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
    const [ohlcv, markers, position, pnl] = await Promise.all([
      api(`/ict/ohlcv?timeframe=${tf}&limit=${candleLimit}`),
      api(`/ict/markers?timeframe=${tf}&limit=${markerLimit}`),
      api("/ict/position"),
      api("/ict/closed_pnl?limit=50"),
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

function showAuthGate(panel = "login") {
  const gate = document.getElementById("auth-gate");
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
    showMainScreen();
    fetchVersion();
    fetchAndRender();
    return;
  }
  // SaaS 모드 — 로그아웃 행 표시 + 상태별 패널 분기.
  const lr = document.getElementById("logout-row");
  if (lr) lr.classList.add("show");

  const s = r.status || {};
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
  fetchVersion();
  fetchAndRender();
}

// PIN 설정 버튼
document.addEventListener("DOMContentLoaded", () => {
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

  const btnKeys = document.getElementById("btn-save-keys");
  if (btnKeys) btnKeys.onclick = async () => {
    _hideAuthError("keys");
    const api_key = (document.getElementById("keys-api-key").value || "").trim();
    const api_secret = (document.getElementById("keys-api-secret").value || "").trim();
    if (!api_key || !api_secret) {
      return _showAuthError("keys", "API Key 와 Secret 을 모두 입력하세요.");
    }
    btnKeys.disabled = true;
    try {
      await api("/auth/api-keys", "POST", { api_key, api_secret });
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
});

// 부팅 — DOM 준비 즉시 게이트/메인 결정.
bootstrap();
