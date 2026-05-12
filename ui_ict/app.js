/* Aurora-ICT UI — lightweight-charts + REST polling */

const API = "http://127.0.0.1:8765";

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

function clearOverlays() {
  fvgBullSeries.setData([]);
  fvgBearSeries.setData([]);
  candleSeries.setMarkers([]);
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

  // Enable 토글 상태 반영
  const btnEn = $("btn-toggle-enabled");
  btnEn.classList.toggle("on", !!s.enabled);
  btnEn.textContent = s.enabled ? "DISABLE" : "ENABLE";
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

  // 단순 marker 렌더 — 각 FVG 별 사각형 대신 setMarkers 일괄 표시.
  const markers = [];

  // FVG markers
  m.fvgs.forEach(f => {
    const color = f.type === "bullish" ? "#34d399" : "#fb7185";
    markers.push({
      time: tsToTimeSec(f.ts_ms),
      position: f.type === "bullish" ? "belowBar" : "aboveBar",
      color,
      shape: "square",
      text: `FVG ${f.filled ? "✓" : ""}${f.invalidated ? "✗" : ""}`,
    });
  });

  // Sweep markers (wick-only sweeps)
  m.sweeps.forEach(s => {
    markers.push({
      time: tsToTimeSec(s.ts_ms),
      position: s.type === "bearish" ? "aboveBar" : "belowBar",
      color: "#fbbf24",
      shape: s.type === "bearish" ? "arrowDown" : "arrowUp",
      text: "Sweep",
    });
  });

  // Structure (BOS/CHoCH)
  m.structure.forEach(ev => {
    const isChoCH = ev.type.startsWith("choch");
    const isBull = ev.type.endsWith("bullish");
    markers.push({
      time: tsToTimeSec(ev.ts_ms),
      position: isBull ? "belowBar" : "aboveBar",
      color: isChoCH ? "#a855f7" : "#60a5fa",
      shape: "circle",
      text: isChoCH ? "MSS" : "BOS",
    });
  });

  // Setups
  m.setups.forEach(s => {
    markers.push({
      time: tsToTimeSec(s.ts_ms),
      position: s.direction === "long" ? "belowBar" : "aboveBar",
      color: s.direction === "long" ? "#34d399" : "#fb7185",
      shape: s.direction === "long" ? "arrowUp" : "arrowDown",
      text: `${s.direction.toUpperCase()} RR=${s.risk_reward.toFixed(1)}`,
    });
  });

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
// OHLCV fetch + render (candles + markers)
// ============================================================
async function fetchAndRender() {
  try {
    const status = await api("/ict/status");
    renderStatus(status);
    if (status.state !== "running") return;

    const [ohlcv, markers] = await Promise.all([
      api("/ict/ohlcv?limit=200"),
      api("/ict/markers?limit=200"),
    ]);
    candleSeries.setData(ohlcv.candles);
    renderMarkers(markers);
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
