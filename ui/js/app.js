// Aurora GUI 진입점 — 라우팅 + 데이터 바인딩 + UI 인터랙션.
//
// vanilla JS — Pywebview 환경에서 가벼움 우선.
//
// 담당: 정용우

const Api = window.AuroraApi;

// ============================================================
// 0. 부팅 스플래시 — AURORA 즉시 가시 → backend ready / timeout 시 fade-out
// ============================================================
//
// v0.1.108 (ChoYoon 권장 (a) 외부 사용자 친화):
// - inline critical CSS 측 splash 즉시 박힘 (WebView2 init 측 까만 화면 차단)
// - status 텍스트 측 ("SYSTEM 시작 중" → "백엔드 연결 중" → "준비 완료")
//   사용자 측 진행 상황 명확
// - fade-out 트리거: (1) /status 응답 200 OK 박힘 OR (2) 5초 timeout
//
// 타이밍:
//   0.0s        splash 즉시 보임 (inline CSS, html 박히자마자 표시)
//   ~1.0s       /status fetch 시도 시작
//   ready/5s    fade-out (0.6s) → DOM 제거

(function bootSplash() {
    const splash = document.getElementById("boot-splash");
    if (!splash) return;
    const status = document.getElementById("boot-status-text");
    let faded = false;

    function fadeOut(reason) {
        if (faded) return;
        faded = true;
        if (status) status.textContent = "준비 완료";
        // 짧은 딜레이 후 fade-out (사용자 측 status 변화 인지)
        setTimeout(() => {
            splash.classList.add("fade-out");
            setTimeout(() => {
                splash.remove();
                document.body.classList.remove("splash-active");
            }, 600);
        }, 200);
        if (reason && console && console.info) {
            console.info("[boot-splash] fade-out:", reason);
        }
    }

    // 1초 후 status 변경 + backend ready 확인 시작
    setTimeout(() => {
        if (status) status.textContent = "백엔드 연결 중";
    }, 600);

    // backend ready 확인 — 첫 /status 200 OK 시 fade-out
    let attempts = 0;
    const tryReady = () => {
        if (faded) return;
        attempts += 1;
        fetch("http://127.0.0.1:8765/status", { cache: "no-cache" })
            .then((r) => {
                if (r.ok) fadeOut(`backend ready (attempt ${attempts})`);
                else setTimeout(tryReady, 500);
            })
            .catch(() => {
                setTimeout(tryReady, 500);
            });
    };
    setTimeout(tryReady, 800);

    // hard timeout — backend 측 5초 안 응답 X 면 fade-out (사용자 측 frontend
    // 측 disconnect 표시 측 박힘 — boot-splash 영원 박혀있는 본질 X)
    setTimeout(() => fadeOut("5s timeout (backend 측 응답 X — UI 측 자체 진행)"), 5000);
})();

// ============================================================
// 1. 라우팅 (사이드바 네비 → view 토글)
// ============================================================

function switchView(viewName) {
    document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.view === viewName);
    });
    document.querySelectorAll("[data-view-content]").forEach((sec) => {
        sec.classList.toggle("view-active", sec.dataset.viewContent === viewName);
    });
}

document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        switchView(btn.dataset.view);
        closeSidebar(); // 모바일에서 메뉴 선택 후 사이드바 자동 닫기
    });
});

// ============================================================
// 모바일 사이드바 토글
// ============================================================

function openSidebar() {
    document.getElementById("sidebar").classList.add("sidebar-open");
    document.getElementById("sidebar-overlay").classList.add("sidebar-open");
}

function closeSidebar() {
    document.getElementById("sidebar").classList.remove("sidebar-open");
    document.getElementById("sidebar-overlay").classList.remove("sidebar-open");
}

document.getElementById("sidebar-toggle")?.addEventListener("click", openSidebar);
document.getElementById("sidebar-overlay")?.addEventListener("click", closeSidebar);

// v0.3.4: 데스크탑 사이드바 collapse / expand (localStorage 박힘)
const SIDEBAR_COLLAPSED_KEY = "aurora.sidebar.collapsed";

function applySidebarCollapsed(collapsed) {
    const sb = document.getElementById("sidebar");
    if (!sb) return;
    sb.classList.toggle("sidebar-collapsed", collapsed);
    document.body.classList.toggle("sidebar-is-collapsed", collapsed);
}

function collapseSidebar() {
    applySidebarCollapsed(true);
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "1"); } catch (_) { /* ignore */ }
}

function expandSidebar() {
    applySidebarCollapsed(false);
    try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "0"); } catch (_) { /* ignore */ }
}

document.getElementById("sidebar-collapse")?.addEventListener("click", collapseSidebar);
document.getElementById("sidebar-expand")?.addEventListener("click", expandSidebar);

// 초기 상태 — localStorage 박힌 상태 복원
try {
    if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1") {
        applySidebarCollapsed(true);
    }
} catch (_) { /* ignore */ }

// ============================================================
// 2. 슬라이더 라이브 업데이트 (값 표시)
// ============================================================

function bindSlider(inputId, valueId, formatter = (v) => v) {
    const input = document.getElementById(inputId);
    const value = document.getElementById(valueId);
    if (!input || !value) return;
    const update = () => {
        value.textContent = formatter(input.value);
    };
    input.addEventListener("input", update);
    update();
}

bindSlider("vol-period", "vol-period-val", (v) => v);
bindSlider("vol-mult", "vol-mult-val", (v) => `${parseFloat(v).toFixed(1)}×`);
bindSlider("vol-boost", "vol-boost-val", (v) => `${parseFloat(v).toFixed(1)}×`);

bindSlider("tp1", "tp1-val", (v) => `${v}%`);
bindSlider("tp2", "tp2-val", (v) => `${v}%`);
bindSlider("tp3", "tp3-val", (v) => `${v}%`);
bindSlider("tp4", "tp4-val", (v) => `${v}%`);

bindSlider("min-seed", "min-seed-val", (v) => `${v}%`);
bindSlider("risk-pct", "risk-pct-val", (v) => `${parseFloat(v).toFixed(1)}%`);

// ============================================================
// 3. 레버리지 슬라이더 → 자동 SL/TP 계산 (risk.py 룰 동등)
// ============================================================

function slPctForLeverage(L) {
    // src/aurora/core/risk.py 의 sl_pct_for_leverage 와 동일 공식.
    if (L <= 37) return 2.0 + (L - 10) / 27.0;
    return 0.08 * L;
}

function tpRangeForLeverage(L) {
    const sl = slPctForLeverage(L);
    if (L <= 37) return [sl + 0.8, sl + 1.8];
    return [sl + 2.0, sl + 3.0];
}

const levInput = document.getElementById("lev");
const levVal = document.getElementById("lev-val");
const autoSlEl = document.getElementById("auto-sl");
const autoTpEl = document.getElementById("auto-tp");

function updateLevDisplay() {
    if (!levInput) return;
    const L = parseInt(levInput.value, 10);
    levVal.textContent = `${L}×`;
    autoSlEl.textContent = `${slPctForLeverage(L).toFixed(2)}%`;
    const [tpMin, tpMax] = tpRangeForLeverage(L);
    autoTpEl.textContent = `${tpMin.toFixed(2)} ~ ${tpMax.toFixed(2)}%`;
}

if (levInput) {
    levInput.addEventListener("input", updateLevDisplay);
    updateLevDisplay();
}

// ============================================================
// 4. TP 4단계 분할 합계 검증 (합 100 권장)
// ============================================================

const tpInputs = ["tp1", "tp2", "tp3", "tp4"].map((id) => document.getElementById(id));
const tpSumEl = document.getElementById("tp-sum");

function updateTpSum() {
    if (!tpSumEl || tpInputs.some((i) => !i)) return;
    const sum = tpInputs.reduce((acc, i) => acc + parseInt(i.value, 10), 0);
    const color = sum === 100 ? "var(--aurora-purple)" : "#fbbf24";
    tpSumEl.innerHTML = `합계: <span style="color: ${color}">${sum}%</span>`;
}

tpInputs.forEach((i) => i?.addEventListener("input", updateTpSum));
updateTpSum();

// ============================================================
// 5. 페어 카드 토글 (선택 / 해제)
// ============================================================

document.querySelectorAll(".pair-card").forEach((card) => {
    card.addEventListener("click", () => {
        card.classList.toggle("selected");
        const meta = card.querySelector(".pair-meta");
        if (meta) {
            meta.textContent = card.classList.contains("selected") ? "SELECTED" : "—";
        }
    });
});

// ============================================================
// 6. 연결 상태 + 대시보드 메트릭 폴링
// ============================================================

// UTC ISO → KST 표시 (Aurora 정책)
function toKstString(isoStr) {
    return new Date(isoStr).toLocaleString("ko-KR", {
        timeZone: "Asia/Seoul", hour12: false,
        year: "numeric", month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit",
    });
}

// v0.1.29 — 봇 활동 indicator. running 시 펄스 dot + 최근 평가 시각, 10초 이상
// stale 면 노란 경고 (정체). 사용자 "봇이 자리 보고 있는지 / 작동 안하는지" 즉시 확인용.
const _ACTIVITY_STALE_THRESHOLD_MS = 10_000;

function _updateBotActivity(running, lastStepTs) {
    const wrap = document.getElementById("m-activity");
    const txt = document.getElementById("m-activity-text");
    if (!wrap || !txt) return;
    if (!running || !lastStepTs) {
        wrap.style.display = "none";
        return;
    }
    wrap.style.display = "flex";
    const ago = Date.now() - lastStepTs;
    const stale = ago > _ACTIVITY_STALE_THRESHOLD_MS;
    wrap.classList.toggle("stale", stale);
    if (stale) {
        const sec = Math.floor(ago / 1000);
        txt.textContent = `정체 ${sec}초`;
    } else {
        const sec = Math.max(0, Math.floor(ago / 1000));
        txt.textContent = sec === 0 ? "평가 중" : `${sec}초 전 평가`;
    }
}

function _setStatusBadge(el, running, backendDown) {
    el.className = "status-badge " +
        (backendDown ? "badge-error" : running ? "badge-running" : "badge-stopped");
    el.textContent = backendDown ? "연결 끊김" : running ? "실행 중" : "중지";
}

function _setButtons(btnStart, btnStop, running, backendDown) {
    if (!btnStart || !btnStop) return;
    const btnRestart = document.getElementById("btn-restart");
    if (backendDown) {
        btnStart.disabled = true;
        btnStop.disabled = true;
        if (btnRestart) btnRestart.disabled = true;
        return;
    }
    btnStart.disabled = running;
    btnStop.disabled = !running;
    // 재시작은 백엔드 살아있으면 항상 가능 (stop+start 묶음 — running 상관 X)
    if (btnRestart) btnRestart.disabled = false;
}

async function refreshDashboard() {
    const connDot   = document.getElementById("conn-dot");
    const connLabel = document.getElementById("conn-label");
    const modeLabel = document.getElementById("mode-label");
    const btnStart  = document.getElementById("btn-start");
    const btnStop   = document.getElementById("btn-stop");
    const mStatus   = document.getElementById("m-status");
    const versionLabel = document.getElementById("version-label");

    // v0.1.114: per-endpoint independent — /status fail 측 다른 refresh 측 영향 X.
    // 이전 (v0.1.113) 측 outer try 박혀있어 /status throw 시 entire chain 측 skip.
    // 사용자 보고 (2026-05-09) "연결 OK 인데 데이터 안 뜸" 본질 fix.
    let s = null;
    try {
        s = await Api.status();
    } catch (err) {
        if (typeof console !== "undefined" && console.warn) {
            console.warn("[refreshDashboard] /status fail (skip status update):", err);
        }
    }

    if (s !== null) {
        connDot.style.background = "#22d3ee";
        connDot.style.boxShadow  = "0 0 8px #22d3ee";
        connLabel.textContent = "CONNECTED";

        const mode = (s.mode || "").toUpperCase();
        modeLabel.textContent = mode;
        document.getElementById("m-mode").textContent = mode;

        // 사이드바 footer version — / 엔드포인트 1회 호출, 시작 후 변경 X 라 캐싱
        if (versionLabel && versionLabel.textContent === "v—") {
            try {
                const root = await Api.health();  // {status, version, mode}
                versionLabel.textContent = "v" + root.version;
            } catch (_) {
                // 백엔드 응답 실패 시 그대로 v— 유지
            }
        }

        _setStatusBadge(mStatus, s.running, false);

        // v0.1.29: 봇 활동 indicator — running 시 펄스 + 최근 평가 시각, stale 시 정체 경고
        _updateBotActivity(s.running, s.last_step_ts);

        document.getElementById("m-positions").textContent = String(s.open_positions ?? 0);
        document.getElementById("m-equity").textContent =
            s.equity_usd == null ? "—"
                : s.equity_usd.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        // equity 실 값 들어오면 stub 메시지 숨김
        const stub = document.getElementById("m-equity-stub");
        if (stub) stub.style.display = (s.equity_usd == null) ? "" : "none";

        // 외부 포지션 알림 — bot.external_position_detected = true 일 때만 표시
        const extAlert = document.getElementById("external-position-alert");
        if (extAlert) extAlert.style.display = s.external_position ? "" : "none";

        // 지표 트리거 상태 패널 (v0.1.14 + v0.1.29 redesign) — 카드 색만으로 4-state 표시.
        // "대기" 텍스트 제거 (sr-only). 활성·롱/숏 시 카드 글로우 + 색 변화. 비활성 점선.
        const indStatus = s.indicator_status || {};
        document.querySelectorAll(".indicator-pill").forEach((pill) => {
            const cat = pill.dataset.cat;
            const state = indStatus[cat];  // "long" | "short" | "neutral" | "disabled"
            pill.classList.remove("dir-long", "dir-short", "dir-neutral", "dir-disabled");
            const stEl = pill.querySelector(".ind-state");  // sr-only (CSS clip)
            if (state === "long") {
                pill.classList.add("dir-long");
                if (stEl) stEl.textContent = "활성·롱";
            } else if (state === "short") {
                pill.classList.add("dir-short");
                if (stEl) stEl.textContent = "활성·숏";
            } else if (state === "disabled") {
                pill.classList.add("dir-disabled");
                if (stEl) stEl.textContent = "비활성";
            } else {
                pill.classList.add("dir-neutral");
                if (stEl) stEl.textContent = "대기";
            }
        });

        const lu = document.getElementById("m-last-update");
        if (lu) lu.textContent = toKstString(new Date().toISOString()) + " KST";

        _setButtons(btnStart, btnStop, s.running, false);
    }

    // v0.1.114: 각 refresh 측 독립 박음 — 한 곳 fail 측 다른 곳 영향 X.
    // 각 refresh 함수 측 자체 try/catch 박혀있어 silent fail 박힘.
    await refreshPositions();
    await refreshTrades();
    await refreshStats();
    await refreshReleaseAlert();
    await refreshMarketTrend();
    await refreshBotChart();
    await refreshDashboardFlow();
    await refreshDashboardRatios();
    await refreshDashboardSeries();
}

// v0.1.113: connection 상태 측 /health 별도 polling 박음 (3초 주기).
// 3번 연속 fail 측 DISCONNECTED, 1번 success 측 CONNECTED. data refresh 측 영향 X.
let _healthFailCount = 0;
const _HEALTH_FAIL_THRESHOLD = 3;

async function _checkHealth() {
    const connDot   = document.getElementById("conn-dot");
    const connLabel = document.getElementById("conn-label");
    if (!connDot || !connLabel) return;

    try {
        const h = await Api.health();
        if (!(h && h.status === "ok")) throw new Error("health not ok");
        // CONNECTED — 정상 응답
        _healthFailCount = 0;
        connDot.style.background = "#22d3ee";
        connDot.style.boxShadow  = "0 0 8px #22d3ee";
        connLabel.textContent = "CONNECTED";
    } catch (err) {
        _healthFailCount++;
        if (typeof console !== "undefined" && console.warn) {
            console.warn(`[_checkHealth] fail #${_healthFailCount}:`, err);
        }
        if (_healthFailCount >= _HEALTH_FAIL_THRESHOLD) {
            // DISCONNECTED — backend 측 진짜 죽음 (3번 연속 = ~9초 응답 X)
            connDot.style.background = "#fb7185";
            connDot.style.boxShadow  = "0 0 8px #fb7185";
            connLabel.textContent = `DISCONNECTED (재연결 시도 ${_healthFailCount}회)`;
        }
        // 1~2번 측 silent — UI 변화 X
    }
}

// 열린 포지션 표 갱신 — /positions API 호출 + tbody 행 렌더링
async function refreshPositions() {
    const tbody = document.getElementById("pos-tbody");
    if (!tbody) return;
    let positions = [];
    try {
        positions = await Api.getPositions();
    } catch (_) {
        // 백엔드 끊기면 status 분기에서 이미 처리, 여기는 빈 표 유지
        return;
    }
    if (!Array.isArray(positions) || positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="pos-empty">열린 포지션 없음</td></tr>';
        return;
    }
    const fmtPrice = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    // 미실현 손익 표시 — ROI% (왼쪽) + USDT (오른쪽). ROI = pnl / initial_margin × 100
    // initial_margin = (entry × qty) / leverage (cross margin 가정)
    const fmtPnl = (p) => {
        const n = Number(p.unrealized_pnl_usd);
        const sign = n >= 0 ? "+" : "";
        const color = n >= 0 ? "#34d399" : "#fb7185";
        const margin = (Number(p.entry_price) * Number(p.quantity)) / Math.max(Number(p.leverage) || 1, 1);
        const roi = margin > 0 ? (n / margin) * 100 : 0;
        const roiSign = roi >= 0 ? "+" : "";
        return `<span style="color:${color}">${roiSign}${roi.toFixed(2)}% &nbsp;&nbsp;${sign}${n.toFixed(2)} USDT</span>`;
    };
    // 진입 트리거 — 봇 자기 진입한 포지션만 값 있음. 외부 포지션은 빈 list → "—"
    const fmtTrigger = (arr) => {
        if (!Array.isArray(arr) || arr.length === 0) {
            return '<span style="color:var(--text-3)">—</span>';
        }
        return arr.map(t => `<span class="trigger-tag">${t}</span>`).join(" ");
    };
    tbody.innerHTML = positions.map(p => `
        <tr>
            <td class="mono">${p.symbol}</td>
            <td>${fmtTrigger(p.triggered_by)}</td>
            <td>${p.direction === "long" ? "롱" : "숏"}</td>
            <td class="mono">${fmtPrice(p.entry_price)}</td>
            <td class="mono">${Number(p.quantity).toFixed(4)}</td>
            <td class="mono">${p.leverage}×</td>
            <td class="mono">${fmtPnl(p)}</td>
        </tr>
    `).join("");
}

// 거래내역 (P&L) 표 갱신 — Bybit 스타일 (v0.1.20 + v0.1.23 기간 필터 + v0.1.30 source/pair).
// /trades?days=N&source=X 호출 + tbody 행 렌더. days/source 는 사용자 토글, pair 는 클라이언트 필터.
let _tradesPeriodDays = 7;
let _tradesSource = "all";          // "all" | "bot" | "external"
let _tradesPairFilter = null;       // null = 전체 페어, or "BTC/USDT:USDT" 등

// 마지막 fetch 결과 캐시 — pair 필터 변경 시 재조회 X (클라이언트 필터)
let _tradesCache = [];

async function refreshTrades() {
    const tbody = document.getElementById("trades-tbody");
    if (!tbody) return;
    try {
        // backend 호출 — source 는 backend 분기, pair 는 클라이언트 필터
        const sourceParam = (_tradesSource === "external") ? "exchange" : _tradesSource;
        _tradesCache = await Api.getTrades(200, _tradesPeriodDays, sourceParam);
    } catch (_) {
        return;
    }
    _renderTradesFiltered();
}

// 클라이언트 측 페어 필터 적용 + 행 렌더 + 페어 chip 갱신
function _renderTradesFiltered() {
    const tbody = document.getElementById("trades-tbody");
    if (!tbody) return;
    const all = Array.isArray(_tradesCache) ? _tradesCache : [];

    // 페어 chip 동적 생성 (cache 기준 unique symbols)
    _renderPairChips(all);

    // 페어 필터 적용
    const trades = _tradesPairFilter
        ? all.filter((t) => t.symbol === _tradesPairFilter)
        : all;

    // 누적 PnL 차트 갱신 (필터 적용된 trades 기준)
    _renderPnlChart(trades);

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="trades-empty">거래내역 없음</td></tr>';
        return;
    }
    const fmtPrice = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const fmtTime = (ts) => {
        const d = new Date(ts);
        const yy = d.getFullYear();
        const mm = String(d.getMonth() + 1).padStart(2, "0");
        const dd = String(d.getDate()).padStart(2, "0");
        const hh = String(d.getHours()).padStart(2, "0");
        const mi = String(d.getMinutes()).padStart(2, "0");
        const ss = String(d.getSeconds()).padStart(2, "0");
        return `${yy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
    };
    const fmtPnl = (n) => {
        const v = Number(n);
        const sign = v >= 0 ? "+" : "";
        const color = v >= 0 ? "#34d399" : "#fb7185";
        return `<span style="color:${color}">${sign}${v.toFixed(4)}</span>`;
    };
    // Bybit 패턴 — qty 색: short = 빨강 (sell), long = 초록 (buy 진입 → reduce sell 청산)
    // Aurora 는 Bybit P&L 화면처럼 "방향 = 진입 방향" 기준 표시.
    const fmtQty = (qty, dir) => {
        const color = dir === "short" ? "#fb7185" : "#34d399";
        return `<span style="color:${color}">${Number(qty).toFixed(4)}</span>`;
    };
    // Symbol 표시 — "BTC/USDT:USDT" → "BTCUSDT Perp"
    const fmtSymbol = (s) => {
        const base = s.split("/")[0] || s;
        const quote = (s.split("/")[1] || "").split(":")[0] || "USDT";
        return `${base}${quote} <span class="trade-perp">Perp</span>`;
    };
    // v0.1.27 — reason 별 Trade Type 표시. Aurora 자기 vs 외부 시각 구분.
    //   external  → "외부" (사용자 직접 / 봇 외부 거래)
    //   tp_full   → "TP" (전량 익절)
    //   tp_partial→ "TP 부분"
    //   sl        → "SL"
    //   reverse   → "REV" (REVERSE 신호 청산)
    //   manual    → "Manual"
    //   기타/누락  → "Trade" (fallback)
    const fmtTradeType = (reason) => {
        if (reason === "external") return '<span class="trade-type-external">외부</span>';
        if (reason === "tp_full") return '<span class="trade-type-tp">TP</span>';
        if (reason === "tp_partial") return '<span class="trade-type-tp">TP 부분</span>';
        if (reason === "sl") return '<span class="trade-type-sl">SL</span>';
        if (reason === "reverse") return '<span class="trade-type-rev">REV</span>';
        if (reason === "manual") return 'Manual';
        return 'Trade';
    };
    // v0.1.65: 수수료 표시 — 0 이면 "—" / 양수면 빨강 (-) 톤 (사용자 비용)
    const fmtFee = (fee) => {
        if (fee == null || fee <= 0) return '<span class="trade-fee-empty">—</span>';
        return `<span class="trade-fee">-${Number(fee).toFixed(4)}</span>`;
    };
    tbody.innerHTML = trades.map((t, idx) => `
        <tr class="trade-row" data-trade-idx="${idx}">
            <td>${fmtSymbol(t.symbol)}</td>
            <td class="mono">${t.instrument}</td>
            <td class="mono trade-model">${t.model || "—"}</td>
            <td class="mono">${fmtPrice(t.entry_price)}</td>
            <td class="mono">${fmtPrice(t.exit_price)}</td>
            <td class="mono">${fmtQty(t.qty, t.direction)}</td>
            <td>${fmtTradeType(t.reason)}</td>
            <td class="mono">${fmtPnl(t.pnl_usd)}</td>
            <td class="mono">${fmtFee(t.fee_usd)}</td>
            <td class="mono">${fmtTime(t.closed_at_ts)}</td>
        </tr>
    `).join("");
    // v0.1.21 — 행 클릭 시 PnL 공유 카드 모달 오픈
    tbody.querySelectorAll("tr.trade-row").forEach((row) => {
        row.addEventListener("click", () => {
            const idx = parseInt(row.dataset.tradeIdx, 10);
            const t = trades[idx];
            if (t) openPnlCard(t);
        });
    });
}

// 거래내역 기간 토글 (v0.1.23) — 7D / 30D / 180D
(() => {
    const toggle = document.getElementById("trades-period-toggle");
    if (!toggle) return;
    toggle.querySelectorAll("button[data-days]").forEach((btn) => {
        btn.addEventListener("click", () => {
            _tradesPeriodDays = parseInt(btn.dataset.days, 10);
            toggle.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            // 봇 자기 거래 + 거래소 history 합쳐 즉시 다시 조회
            refreshTrades();
            // v0.1.24 — 통계 카드도 같은 기간으로 즉시 갱신
            refreshStats();
        });
    });
})();

// 거래내역 source 토글 (v0.1.30) — 전체 / Aurora / 외부
(() => {
    const toggle = document.getElementById("trades-source-toggle");
    if (!toggle) return;
    toggle.querySelectorAll("button[data-source]").forEach((btn) => {
        btn.addEventListener("click", () => {
            _tradesSource = btn.dataset.source;
            toggle.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            // pair 필터 reset (다른 source 면 페어 list 다를 수 있음)
            _tradesPairFilter = null;
            refreshTrades();
        });
    });
})();

// 누적 PnL 차트 (v0.1.30) — chart.js, 필터 적용된 trades 기준 line chart.
// trades 는 백엔드 응답 그대로 (신→구 정렬). 차트는 시간 순 (구→신) 으로 누적.
let _pnlChartInstance = null;

function _renderPnlChart(trades) {
    const canvas = document.getElementById("pnl-chart");
    const empty = document.getElementById("pnl-chart-empty");
    if (!canvas || typeof Chart !== "function") return;

    if (!trades || trades.length === 0) {
        if (_pnlChartInstance) {
            _pnlChartInstance.destroy();
            _pnlChartInstance = null;
        }
        if (empty) empty.style.display = "flex";
        return;
    }
    if (empty) empty.style.display = "none";

    // 시간 순 정렬 (closed_at_ts 오름차순) + 누적 PnL 계산
    const sorted = [...trades].sort((a, b) => a.closed_at_ts - b.closed_at_ts);
    let cum = 0;
    const points = sorted.map((t) => {
        cum += Number(t.pnl_usd) || 0;
        return { x: t.closed_at_ts, y: cum };
    });

    // 양수/음수 색 — 마지막 누적값 기준 (양=초록, 음=빨강)
    const positive = cum >= 0;
    const lineColor = positive ? "#34d399" : "#fb7185";
    const fillColor = positive ? "rgba(52,211,153,0.12)" : "rgba(251,113,133,0.12)";

    // 라벨 (간단한 시간 텍스트) — chart.js category scale 사용 (linear 도 가능)
    const fmtShort = (ts) => {
        const d = new Date(ts);
        return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    };
    const labels = points.map((p) => fmtShort(p.x));
    const data = points.map((p) => p.y);

    if (_pnlChartInstance) {
        _pnlChartInstance.data.labels = labels;
        _pnlChartInstance.data.datasets[0].data = data;
        _pnlChartInstance.data.datasets[0].borderColor = lineColor;
        _pnlChartInstance.data.datasets[0].backgroundColor = fillColor;
        _pnlChartInstance.update("none");
        return;
    }

    _pnlChartInstance = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "누적 PnL (USDT)",
                data,
                borderColor: lineColor,
                backgroundColor: fillColor,
                borderWidth: 1.5,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.15,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(13, 14, 22, 0.95)",
                    borderColor: "rgba(96, 81, 155, 0.4)",
                    borderWidth: 1,
                    titleColor: "#bfc0d1",
                    bodyColor: "#bfc0d1",
                    padding: 10,
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.parsed.y;
                            return `${v >= 0 ? "+" : ""}${v.toFixed(2)} USDT`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#666770", font: { size: 9 }, maxRotation: 0, autoSkipPadding: 32 },
                    grid: { color: "rgba(255,255,255,0.03)" },
                },
                y: {
                    ticks: {
                        color: "#666770",
                        font: { size: 10 },
                        callback: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}`,
                    },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
            },
        },
    });
}

// 페어 chip 동적 생성 (v0.1.30) — 거래내역 cache 기준 unique symbols
function _renderPairChips(trades) {
    const wrap = document.getElementById("trades-pair-filter");
    if (!wrap) return;
    const symbols = [...new Set(trades.map((t) => t.symbol))].sort();
    if (symbols.length <= 1) {
        wrap.innerHTML = "";  // 페어 1개 이하면 chip 표시 의미 X
        return;
    }
    // "전체" + 각 페어
    const fmt = (s) => {
        const base = s.split("/")[0] || s;
        return base;  // "BTC", "ETH" 등 짧게
    };
    const chips = [
        `<button data-pair="" class="${_tradesPairFilter === null ? "active" : ""}">전체</button>`,
        ...symbols.map((s) =>
            `<button data-pair="${s}" class="${_tradesPairFilter === s ? "active" : ""}">${fmt(s)}</button>`,
        ),
    ];
    wrap.innerHTML = chips.join("");
    wrap.querySelectorAll("button[data-pair]").forEach((btn) => {
        btn.addEventListener("click", () => {
            _tradesPairFilter = btn.dataset.pair || null;
            _renderTradesFiltered();  // 클라이언트 필터 — 재조회 X
        });
    });
}

// 결과 통계 6 카드 갱신 (v0.1.24) — /stats?days=N 호출 + 카드 값 채우기.
async function refreshStats() {
    let s;
    try {
        s = await Api.getStats(_tradesPeriodDays);
    } catch (_) {
        return;
    }
    const $ = (id) => document.getElementById(id);

    // 데이터 없으면 모두 "—" (n=0 → 의미 없는 0/0 보다 깔끔)
    if (!s || s.total_trades === 0) {
        ["stat-total", "stat-winrate", "stat-cumulative", "stat-dd", "stat-sharpe", "stat-hold"]
            .forEach((id) => { const el = $(id); if (el) el.textContent = "—"; });
        const wl = $("stat-winloss");
        if (wl) wl.textContent = "—";
        return;
    }

    $("stat-total").textContent = String(s.total_trades);
    $("stat-winloss").textContent = `${s.win_count}W / ${s.loss_count}L`;
    $("stat-winrate").textContent = `${s.win_rate_pct.toFixed(1)}%`;

    // 누적 수익률 — USDT + 평균 ROI (양수=초록, 음수=빨강)
    const pnl = s.cumulative_pnl_usd;
    const sign = pnl >= 0 ? "+" : "";
    const color = pnl >= 0 ? "#34d399" : "#fb7185";
    const cum = $("stat-cumulative");
    cum.innerHTML =
        `<span style="color:${color}">${sign}${pnl.toFixed(2)} USDT</span>` +
        `<span class="stat-aux mono">${s.avg_roi_pct >= 0 ? "+" : ""}${s.avg_roi_pct.toFixed(2)}% avg</span>`;

    $("stat-dd").textContent = `${s.max_drawdown_pct.toFixed(2)}%`;
    $("stat-sharpe").textContent = s.sharpe_ratio.toFixed(2);

    // 평균 보유 — 분 → 시:분 또는 분 단위
    const m = s.avg_hold_minutes;
    const hold = m >= 60 ? `${Math.floor(m / 60)}h ${Math.round(m % 60)}m` : `${m.toFixed(1)}m`;
    $("stat-hold").textContent = hold;
}

// ============================================================
// 6c. 업데이트 알림 (v0.1.25) — 5분 주기 폴링 → 우상단 팝업
// ============================================================
//
// 흐름:
//   - 백엔드 release_check 가 5분 주기로 GitHub Releases /latest 체크
//   - dashboard 폴링 (15초) 이 같이 /release/latest 호출
//   - has_update + dismissed 안 한 tag 면 우상단 알림 표시
//   - 사용자 × 클릭 시 localStorage 에 dismissed_<tag> 저장 (해당 tag 다시 안 뜸)
//   - "자세히 보기" 클릭 시 release html_url 새 창
//   - 새 release 가 또 올라오면 다른 tag → 다시 표시 (dismissed 는 tag 별)

function _isReleaseDismissed(tag) {
    if (!tag) return false;
    try {
        return localStorage.getItem(`aurora_release_dismissed_${tag}`) === "1";
    } catch (_) {
        return false;
    }
}

function _dismissRelease(tag) {
    if (!tag) return;
    try {
        localStorage.setItem(`aurora_release_dismissed_${tag}`, "1");
    } catch (_) { /* private mode 등 — 무시 */ }
}

async function refreshReleaseAlert() {
    const alert = document.getElementById("release-alert");
    if (!alert) return;
    let info;
    try {
        info = await Api.getReleaseLatest();
    } catch (_) {
        // 첫 폴링 결과 도착 전에는 알림 X
        alert.style.display = "none";
        return;
    }
    if (!info || !info.has_update || !info.tag || _isReleaseDismissed(info.tag)) {
        alert.style.display = "none";
        return;
    }
    document.getElementById("release-alert-tag").textContent = info.tag;
    document.getElementById("release-alert-current").textContent = "v" + (info.current_version || "");
    // dataset.tag 에 현재 표시중 tag 저장 — × 클릭 시 dismiss 키 사용
    alert.dataset.tag = info.tag;
    alert.dataset.url = info.html_url || "";
    alert.style.display = "flex";

    // Android: APK 다운로드 진행 표시 — apk-status 폴링으로 진행률 갱신
    if (window.Android) {
        try {
            const apkR = await fetch("/update/apk-status");
            const apk = await apkR.json();
            const dlRow = document.getElementById("release-alert-dl");
            const dlBar = document.getElementById("release-alert-dl-bar");
            const dlText = document.getElementById("release-alert-dl-text");
            // v0.2.27: restart button 측 제거 박혀 측 — restart 측 갱신 흐름 측 X
            if (dlRow) {
                if (apk.status === "downloading") {
                    const pct = apk.download_pct ?? 0;
                    dlRow.style.display = "flex";
                    if (dlBar) dlBar.style.width = pct + "%";
                    if (dlText) dlText.textContent = pct + "%";
                } else {
                    dlRow.style.display = "none";
                }
            }
        } catch (_) { /* 네트워크 오류 — 진행바 숨김 유지 */ }
    }
}

// ============================================================
// 6c. Market Data (Coinalyze 추세, v0.1.54) — 5분 cache 활용
// ============================================================
//
// /market-trend 폴링 (dashboard 사이클과 같이) → BTC/ETH 카드 갱신.
// 백엔드 cache 5분 박혀있어 UI 폴링 빈도와 API 한도 무관.
// enabled=False (COINALYZE_API_KEY 미설정) 면 카드 자체 숨김.

function _fmtPct(now, prev) {
    if (now == null || prev == null || prev === 0) return "—";
    const pct = (now - prev) / Math.abs(prev) * 100;
    const sign = pct >= 0 ? "+" : "";
    return `${sign}${pct.toFixed(1)}%`;
}

function _fmtBigNum(value) {
    if (value == null) return "—";
    const abs = Math.abs(value);
    const sign = value > 0 ? "+" : (value < 0 ? "" : "");
    if (abs >= 1e9) return `${sign}${(value / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${sign}${(value / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${sign}${(value / 1e3).toFixed(2)}K`;
    return `${sign}${value.toFixed(2)}`;
}

function _fmtPrice(value) {
    if (value == null) return "—";
    if (value >= 1000) return `$${value.toLocaleString("en-US", {minimumFractionDigits: 1, maximumFractionDigits: 2})}`;
    return `$${value.toFixed(4)}`;
}

function _trendBadgeClass(score, direction) {
    if (direction === "neutral") return "trend-neutral";
    const strong = Math.abs(score) >= 2;
    if (direction === "long") return strong ? "trend-long-strong" : "trend-long";
    return strong ? "trend-short-strong" : "trend-short";
}

function _trendBadgeText(score, direction) {
    if (direction === "neutral") return "중립";
    const strong = Math.abs(score) >= 2;
    if (direction === "long") return strong ? "강한 롱" : "롱";
    return strong ? "강한 숏" : "숏";
}

function _renderMarketCard(card, t) {
    if (!card) return;

    // v0.2.25 (사용자 보고 2026-05-11 — ETH 측 모든 field None 본질):
    // backend Coinalyze rate limit (429) 측 측 — fetch_trend 측 success 박아도
    // 모든 field None 박는 케이스. 측 — 측 — 측 placeholder 박아 사용자 측 본질
    // 인지 + 다음 fetch (5분 cache 측 측 회복) 측 자동 박힐 거 박는 안내.
    const allNone = (
        t.price == null && t.price_24h == null && t.oi == null &&
        t.cvd_spot == null && t.cvd_futures == null && t.funding_rate == null
    );
    if (allNone) {
        card.querySelector(".md-coin-price").textContent = "—";
        const badge = card.querySelector(".md-trend-badge");
        badge.className = "md-trend-badge trend-loading";
        badge.textContent = "데이터 측 박힘 중";
        // multi-tf badges 측 "—"
        for (const sel of [".md-trend-short", ".md-trend-mid-short", ".md-trend-mid"]) {
            const el = card.querySelector(sel);
            if (el) {
                el.textContent = "—";
                el.className = sel.slice(1) + " md-trend-badge trend-loading";
            }
        }
        // 측 가격 / 24h / OI / CVD / Funding 측 "—"
        for (const sel of [".md-24h", ".md-oi", ".md-cvd-spot", ".md-cvd-futures", ".md-funding"]) {
            const el = card.querySelector(sel);
            if (el) {
                el.textContent = "—";
                el.className = sel.slice(1) + " mono";
            }
        }
        // 갱신 시각 측 — Coinalyze 측 한도 회복 측 안내
        const ts = card.querySelector(".md-updated-ts");
        if (ts) ts.textContent = "Coinalyze 한도 — 잠시 후";
        return;
    }

    card.querySelector(".md-coin-price").textContent = _fmtPrice(t.price);

    const badge = card.querySelector(".md-trend-badge");
    badge.className = "md-trend-badge " + _trendBadgeClass(t.score, t.direction);
    badge.textContent = _trendBadgeText(t.score, t.direction);

    // v0.1.84: multi-tf trend (단기 15m / 중단기 4h / 중기 1D)
    const tfMap = [
        { sel: ".md-trend-short",     score: t.score_short,     dir: t.direction_short },
        { sel: ".md-trend-mid-short", score: t.score_mid_short, dir: t.direction_mid_short },
        { sel: ".md-trend-mid",       score: t.score_mid,       dir: t.direction_mid },
    ];
    for (const { sel, score, dir } of tfMap) {
        const el = card.querySelector(sel);
        if (!el) continue;
        const text = _trendBadgeText(score ?? 0, dir ?? "neutral");
        const cls = _trendBadgeClass(score ?? 0, dir ?? "neutral");
        el.textContent = text;
        el.className = sel.slice(1) + " md-trend-badge " + cls;
    }

    // 24h 변화율 (가격)
    const c24h = card.querySelector(".md-24h");
    const pctText = _fmtPct(t.price, t.price_24h);
    c24h.textContent = pctText;
    c24h.className = "md-24h mono " + (pctText.startsWith("+") ? "pos" : (pctText.startsWith("-") ? "neg" : ""));

    // OI (값 + 24h ▲/▼)
    const cOi = card.querySelector(".md-oi");
    if (t.oi != null) {
        const oiPct = _fmtPct(t.oi, t.oi_24h);
        cOi.textContent = `${_fmtBigNum(t.oi).replace("+","")} ${oiPct}`;
        cOi.className = "md-oi mono " + (oiPct.startsWith("+") ? "pos" : (oiPct.startsWith("-") ? "neg" : ""));
    } else {
        cOi.textContent = "—";
        cOi.className = "md-oi mono";
    }

    // CVD Spot
    const cSpot = card.querySelector(".md-cvd-spot");
    cSpot.textContent = _fmtBigNum(t.cvd_spot);
    cSpot.className = "md-cvd-spot mono " + (t.cvd_spot > 0 ? "pos" : (t.cvd_spot < 0 ? "neg" : ""));

    // CVD Futures
    const cFut = card.querySelector(".md-cvd-futures");
    cFut.textContent = _fmtBigNum(t.cvd_futures);
    cFut.className = "md-cvd-futures mono " + (t.cvd_futures > 0 ? "pos" : (t.cvd_futures < 0 ? "neg" : ""));

    // Funding Rate
    const cFr = card.querySelector(".md-funding");
    if (t.funding_rate != null) {
        const pct = (t.funding_rate * 100).toFixed(4);
        cFr.textContent = `${pct}%`;
        cFr.className = "md-funding mono " + (t.funding_rate >= 0 ? "pos" : "neg");
    } else {
        cFr.textContent = "—";
        cFr.className = "md-funding mono";
    }

    // 갱신 시각 (KST HH:MM)
    if (t.fetched_at_ms) {
        const d = new Date(t.fetched_at_ms);
        const hh = String(d.getHours()).padStart(2, "0");
        const mm = String(d.getMinutes()).padStart(2, "0");
        card.querySelector(".md-updated-ts").textContent = `${hh}:${mm}`;
    }
}

async function refreshMarketTrend() {
    const section = document.getElementById("market-data-section");
    if (!section) return;
    let data;
    try {
        data = await Api.getMarketTrend();
    } catch (_) {
        section.style.display = "none";
        return;
    }
    if (!data || !data.enabled) {
        section.style.display = "none";
        return;
    }
    section.style.display = "block";
    for (const t of data.trends || []) {
        const card = section.querySelector(`.md-card[data-coin="${t.coin}"]`);
        _renderMarketCard(card, t);
    }
}

// ============================================================
// v0.1.86: 봇 시점 차트 (lightweight-charts) — OHLCV + 지표 라인 + 진입 마커
// ============================================================
const Chart = {
    chart: null,
    candleSeries: null,
    lineSeries: {},      // { ema_fast, ema_slow, bb_upper, bb_middle, bb_lower, st_fast, st_slow, ichi_a, ichi_b }
    // v0.3.0 (사용자 요청 2026-05-11): RSI subchart + Ichimoku line 추가
    rsiChart: null,
    rsiSeries: null,
    timeframe: "1H",
    overlays: { ema: true, bb: true, st: true, rsi: true, ichi: true },
    _resizeObserver: null,
};

function _initBotChart() {
    if (Chart.chart) return;
    if (typeof LightweightCharts === "undefined") {
        // CDN 로드 실패 — 빈 placeholder 유지
        return;
    }
    const container = document.getElementById("bot-chart");
    if (!container) return;
    Chart.chart = LightweightCharts.createChart(container, {
        layout: {
            background: { type: "solid", color: "transparent" },
            textColor: "#a1a1aa",
            fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
            vertLines: { color: "rgba(96, 81, 155, 0.08)" },
            horzLines: { color: "rgba(96, 81, 155, 0.08)" },
        },
        crosshair: { mode: 1 },
        rightPriceScale: { borderColor: "rgba(96, 81, 155, 0.18)" },
        timeScale: {
            borderColor: "rgba(96, 81, 155, 0.18)",
            timeVisible: true,
            secondsVisible: false,
        },
        autoSize: true,
    });
    Chart.candleSeries = Chart.chart.addCandlestickSeries({
        upColor: "#34d399",
        downColor: "#fb7185",
        borderUpColor: "#34d399",
        borderDownColor: "#fb7185",
        wickUpColor: "#34d399",
        wickDownColor: "#fb7185",
    });
    // 지표 라인 — 색 / 두께 차별
    const lineDefs = {
        ema_fast:  { color: "#a78bfa", lineWidth: 1, title: "EMA20" },
        ema_slow:  { color: "#7c3aed", lineWidth: 1, title: "EMA50" },
        bb_upper:  { color: "rgba(96, 165, 250, 0.6)", lineWidth: 1, title: "BB↑" },
        bb_middle: { color: "rgba(96, 165, 250, 0.35)", lineWidth: 1, lineStyle: 2, title: "BB·" },
        bb_lower:  { color: "rgba(96, 165, 250, 0.6)", lineWidth: 1, title: "BB↓" },
        st_fast:   { color: "#fbbf24", lineWidth: 1, title: "ST(2)" },
        st_slow:   { color: "#f97316", lineWidth: 1, title: "ST(3)" },
        // v0.3.0 (사용자 요청): Ichimoku Span A/B 측 main chart 측 line + 두 line 사이 cloud tint
        ichi_a:    { color: "rgba(52, 211, 153, 0.55)", lineWidth: 1, title: "Ichimoku Span A" },
        ichi_b:    { color: "rgba(251, 113, 133, 0.55)", lineWidth: 1, title: "Ichimoku Span B" },
    };
    for (const [key, opts] of Object.entries(lineDefs)) {
        Chart.lineSeries[key] = Chart.chart.addLineSeries({
            ...opts,
            lastValueVisible: false,
            priceLineVisible: false,
        });
    }

    // v0.3.0: RSI subchart 측 별도 — 0~100 axis + 30/70 horizontal lines
    const rsiContainer = document.getElementById("bot-chart-rsi");
    if (rsiContainer && !Chart.rsiChart) {
        Chart.rsiChart = LightweightCharts.createChart(rsiContainer, {
            layout: {
                background: { type: "solid", color: "transparent" },
                textColor: "#a1a1aa",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
            },
            grid: {
                vertLines: { color: "rgba(96, 81, 155, 0.06)" },
                horzLines: { color: "rgba(96, 81, 155, 0.06)" },
            },
            rightPriceScale: {
                borderColor: "rgba(96, 81, 155, 0.18)",
                autoScale: false,
                scaleMargins: { top: 0.05, bottom: 0.05 },
            },
            timeScale: {
                borderColor: "rgba(96, 81, 155, 0.18)",
                timeVisible: false,
                secondsVisible: false,
                visible: false,
            },
            crosshair: { mode: 1 },
            autoSize: true,
        });
        Chart.rsiSeries = Chart.rsiChart.addLineSeries({
            color: "#a78bfa",
            lineWidth: 2,
            title: "RSI(14)",
            lastValueVisible: true,
            priceLineVisible: false,
        });
        // 30 / 70 horizontal lines
        Chart.rsiSeries.createPriceLine({
            price: 70, color: "rgba(251, 113, 133, 0.5)",
            lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "70",
        });
        Chart.rsiSeries.createPriceLine({
            price: 30, color: "rgba(52, 211, 153, 0.5)",
            lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "30",
        });
        Chart.rsiSeries.createPriceLine({
            price: 50, color: "rgba(161, 161, 170, 0.3)",
            lineWidth: 1, lineStyle: 3, axisLabelVisible: false, title: "",
        });
        // main chart + rsi chart 측 timescale 측 sync (사용자 measure pan/zoom 측 동기)
        Chart.chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
            if (Chart.rsiChart && range) {
                Chart.rsiChart.timeScale().setVisibleRange(range);
            }
        });
    }
}

function _applyChartOverlays() {
    // 토글 상태에 따라 라인 표시/숨김
    const map = {
        ema: ["ema_fast", "ema_slow"],
        bb:  ["bb_upper", "bb_middle", "bb_lower"],
        st:  ["st_fast", "st_slow"],
        ichi: ["ichi_a", "ichi_b"],  // v0.3.0
    };
    for (const [group, keys] of Object.entries(map)) {
        const visible = !!Chart.overlays[group];
        for (const k of keys) {
            const s = Chart.lineSeries[k];
            if (s) s.applyOptions({ visible });
        }
    }
    // v0.3.0: RSI subchart 측 toggle — container 측 visibility 박음
    const rsiWrap = document.querySelector(".chart-wrap-rsi");
    if (rsiWrap) {
        rsiWrap.style.display = Chart.overlays.rsi ? "block" : "none";
    }
}

async function refreshBotChart() {
    const section = document.getElementById("bot-chart-section");
    const empty = document.getElementById("chart-empty");
    const status = document.getElementById("chart-status");
    const symEl = document.getElementById("chart-symbol");
    if (!section) return;

    let data;
    try {
        // v0.2.26 (사용자 요청): 100봉 → 500봉 (backend max). 사용자 시점 차트 측 더 많은 history.
        data = await Api.getChart(Chart.timeframe, 500);
    } catch (_) {
        if (status) status.textContent = "데이터 로드 실패";
        return;
    }

    // 2026-08-18: 봇이 STOP 이어도 차트/마커는 보여준다(파트너 요청).
    // 백엔드 /ict/ohlcv 는 ensure_bot_ready 로 START 전에도 캔들을 준다 —
    // 여기서 enabled 로 막고 있어서 안 보였던 것뿐이다.
    if (!data || !(data.candles || []).length) {
        if (empty) {
            empty.classList.remove("hidden");
            empty.textContent = "차트 데이터를 불러오지 못했습니다";
        }
        if (status) status.textContent = "데이터 없음";
        if (symEl) symEl.textContent = "—";
        return;
    }
    _initBotChart();
    if (!Chart.chart) {
        if (status) status.textContent = "차트 라이브러리 로드 실패 (네트워크?)";
        return;
    }

    // 캔들 + 지표 박기
    Chart.candleSeries.setData(data.candles || []);
    const lineKeys = ["ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower", "st_fast", "st_slow"];
    for (const k of lineKeys) {
        const s = Chart.lineSeries[k];
        if (s) s.setData(data[k] || []);
    }
    // v0.3.0: Ichimoku Span A/B 측 main chart 측 line + RSI 측 subchart
    if (Chart.lineSeries.ichi_a) Chart.lineSeries.ichi_a.setData(data.ichimoku_span_a || []);
    if (Chart.lineSeries.ichi_b) Chart.lineSeries.ichi_b.setData(data.ichimoku_span_b || []);
    if (Chart.rsiSeries) {
        Chart.rsiSeries.setData(data.rsi || []);
    }
    _applyChartOverlays();

    // 마커 — 진입(▲/▼) + 청산(○) + 지지/저항 터치(★) — 캔들 시리즈 위에 박힘
    // 2026-08-18: lightweight-charts 는 time 오름차순 배열을 요구한다. 매매 마커와
    // 터치 별을 그냥 이어붙이면 순서가 섞여 마커가 통째로 안 나오므로 합친 뒤 정렬.
    const touches = data.cycle_touches || [];
    const marks = (data.markers || []).concat(touches);
    marks.sort((a, b) => a.time - b.time);
    Chart.candleSeries.setMarkers(marks);

    // 메타 갱신
    if (symEl) symEl.textContent = `${data.symbol || "—"} · ${data.timeframe}`;
    if (status) {
        const n = (data.candles || []).length;
        const m = (data.markers || []).length;
        const t = touches.length;
        const base = `${n} 봉 · 마커 ${m}${t ? ` · 지지저항 ★${t}` : ""}`;
        status.textContent = data.enabled === false ? `${base} · 봇 미가동` : base;
    }
    if (empty) empty.classList.add("hidden");
}

// ============================================================
// v0.1.87: Phase 3 Dashboard Flow — 5 거래소 합본 (Market Flow + L/S Ratio)
// ============================================================
const Dflow = {
    coin: "BTC",
};

function _fmtUsd(v) {
    if (v == null) return "—";
    const abs = Math.abs(v);
    if (abs >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (abs >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (abs >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return v.toFixed(0);
}
// v0.2.19 (사용자 보고 fix): 이전 _fmtPct(v, digits) 측 line 851 _fmtPct(now, prev)
// 측 같은 이름 박혀있어 JS hoisting last-wins → Market Data 카드 측 _fmtPct(price, price_24h)
// 호출 측 (price).toFixed(price_24h) 박힘 = price_24h ~80000 박혀 RangeError throw →
// _renderMarketCard 24h/OI/CVD/Funding 측 모두 fail. v0.2.19 측 _fmtPercent 측 rename.
function _fmtPercent(v, digits = 2) {
    if (v == null) return "—";
    return v.toFixed(digits) + "%";
}
function _fmtRatio(v) {
    if (v == null) return "—";
    return v.toFixed(2);
}
function _fmtFunding(v) {
    if (v == null) return "—";
    return (v * 100).toFixed(4) + "%";
}
function _signClass(v) {
    if (v == null) return "";
    return v > 0 ? "pos" : v < 0 ? "neg" : "";
}

async function refreshDashboardFlow() {
    const tbody = document.getElementById("dflow-tbody");
    if (!tbody) return;

    let data;
    try {
        data = await Api.getDashboardFlow(Dflow.coin);
    } catch (_) {
        tbody.innerHTML = `<tr><td colspan="8" class="dflow-empty">로드 실패 (네트워크?)</td></tr>`;
        return;
    }

    // 합계 카드
    const set = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
    };
    set("dflow-total-oi", data.total_oi_usd != null ? "$" + _fmtUsd(data.total_oi_usd) : "—");
    set("dflow-total-vol", data.total_volume_24h_usd != null ? "$" + _fmtUsd(data.total_volume_24h_usd) : "—");
    set("dflow-avg-funding", _fmtFunding(data.avg_funding_rate));
    set("dflow-avg-ls", _fmtRatio(data.avg_ls_ratio_global));
    // v0.1.90: Whale Notional 합 카드
    set("dflow-whale-buy", data.total_whale_buy_5m_usd != null ? "$" + _fmtUsd(data.total_whale_buy_5m_usd) : "—");
    set("dflow-whale-sell", data.total_whale_sell_5m_usd != null ? "$" + _fmtUsd(data.total_whale_sell_5m_usd) : "—");
    set("dflow-whale-count", data.total_whale_count_5m != null ? data.total_whale_count_5m + "건" : "—");

    // 거래소별 row — 등록된 exchanges 순서대로 (snapshot 미존재 시 placeholder)
    const snapsByEx = {};
    for (const s of (data.snapshots || [])) {
        snapsByEx[s.exchange] = s;
    }
    const rows = (data.exchanges || []).map(ex => {
        const s = snapsByEx[ex];
        if (!s) {
            return `<tr><td class="dflow-exchange">${ex}</td>` +
                   `<td colspan="8" class="dflow-empty">데이터 없음</td></tr>`;
        }
        const chgCls = _signClass(s.price_24h_change_pct);
        const fundingCls = _signClass(s.funding_rate);
        // v0.1.90: Whale 셀 — 매수↑ / 매도↓ + 거래 수
        let whaleCell = "—";
        if (s.whale_count_5m != null) {
            const buy = s.whale_buy_5m_usd ?? 0;
            const sell = s.whale_sell_5m_usd ?? 0;
            const cnt = s.whale_count_5m;
            whaleCell = cnt === 0
                ? "0건"
                : `${cnt}건 <span class="pos">↑$${_fmtUsd(buy)}</span> / <span class="neg">↓$${_fmtUsd(sell)}</span>`;
        }
        return `<tr>
            <td class="dflow-exchange">${ex}</td>
            <td>${s.price != null ? "$" + s.price.toLocaleString(undefined, {maximumFractionDigits: 2}) : "—"}</td>
            <td class="${chgCls}">${_fmtPercent(s.price_24h_change_pct)}</td>
            <td>${s.oi_usd != null ? "$" + _fmtUsd(s.oi_usd) : "—"}</td>
            <td class="${fundingCls}">${_fmtFunding(s.funding_rate)}</td>
            <td>${_fmtRatio(s.ls_ratio_global)}</td>
            <td>${_fmtRatio(s.ls_ratio_top_position)}</td>
            <td>${_fmtRatio(s.ls_ratio_top_account)}</td>
            <td>${whaleCell}</td>
        </tr>`;
    }).join("");

    tbody.innerHTML = rows || `<tr><td colspan="9" class="dflow-empty">등록된 거래소 없음</td></tr>`;
}

function _wireDflowToggles() {
    const box = document.getElementById("dflow-coin-toggle");
    if (!box) return;
    for (const btn of box.querySelectorAll("button")) {
        btn.addEventListener("click", () => {
            box.querySelectorAll("button").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            Dflow.coin = btn.dataset.coin;
            refreshDashboardFlow();
            refreshDashboardRatios();
            refreshDashboardSeries();
        });
    }
}
_wireDflowToggles();

// ============================================================
// v0.1.115: 5단 L/S Ratio segments + 14D 시계열 차트 + L/S timeline
// ============================================================

const DflowRatios = {
    // label → CSS class 안내용 (현 미사용, 시각 정합 fallback)
    _labels: ["WHALE NOTIONAL", "WHALE ACCOUNTS", "TOP NOTIONAL", "TOP ACCOUNTS", "GLOBAL ACCOUNTS"],
};

async function refreshDashboardRatios() {
    const list = document.getElementById("dflow-ratios-list");
    const meta = document.getElementById("dflow-ratios-meta");
    if (!list) return;

    let data;
    try {
        data = await Api.getDashboardRatios(Dflow.coin);
    } catch (_) {
        list.innerHTML = `<div class="dflow-empty">로드 실패</div>`;
        return;
    }

    if (!data || !Array.isArray(data.segments) || data.segments.length === 0) {
        list.innerHTML = `<div class="dflow-empty">데이터 없음</div>`;
        return;
    }

    if (meta) {
        const ageMs = Date.now() - (data.fetched_at_ms || Date.now());
        meta.textContent = `5분 snapshot · ${Math.max(0, Math.floor(ageMs / 1000))}초 전`;
    }

    // v0.3.4 (사용자 요청 2026-05-11): row 우측에 L/S ratio 숫자 (참고자료 양식).
    // long_pct / short_pct → ratio = long_pct / short_pct.
    const rowsHtml = data.segments.map(seg => {
        if (seg.long_pct == null || seg.short_pct == null) {
            return `
                <div class="dflow-ratio-row">
                    <span class="dflow-ratio-label">${seg.label}</span>
                    <div class="dflow-ratio-bar">
                        <span class="dflow-ratio-pct empty">— 데이터 없음 —</span>
                    </div>
                    <span class="dflow-ratio-ls empty">—</span>
                </div>`;
        }
        const longPct = seg.long_pct * 100;
        const shortPct = seg.short_pct * 100;
        const ratio = seg.short_pct > 0 ? seg.long_pct / seg.short_pct : null;
        const ratioStr = ratio != null ? ratio.toFixed(2) : "—";
        const ratioCls = ratio == null ? "empty" : (ratio >= 1 ? "long" : "short");
        const sample = seg.sample_size != null ? ` · ${seg.sample_size}건` : "";
        return `
            <div class="dflow-ratio-row" title="${seg.label}${sample}">
                <span class="dflow-ratio-label">${seg.label}</span>
                <div class="dflow-ratio-bar">
                    <div class="dflow-ratio-bar-long" style="width:${longPct.toFixed(1)}%"></div>
                    <div class="dflow-ratio-bar-short" style="width:${shortPct.toFixed(1)}%"></div>
                    <span class="dflow-ratio-pct">
                        <span>${longPct.toFixed(1)}%</span>
                        <span>${shortPct.toFixed(1)}%</span>
                    </span>
                </div>
                <span class="dflow-ratio-ls ${ratioCls}">L/S ${ratioStr}</span>
            </div>`;
    }).join("");

    // 거래소 표기 — KR translate map (사용자 친화)
    const exchangeKR = {
        binance: "바이낸스",
        bybit: "바이빗",
        okx: "오케이엑스",
        bitget: "비트겟",
        hyperliquid: "하이퍼리퀴드",
    };
    const sourcesHtml = data.segments.map(seg => {
        const sources = (seg.source_exchanges || []);
        const sourcesKR = sources.length > 0
            ? sources.map(s => exchangeKR[s] || s.toUpperCase()).join(", ")
            : "데이터 X";
        return `
            <div class="dflow-source-row">
                <span class="dflow-source-label">${seg.label}</span>
                <span class="dflow-source-list">${sourcesKR}</span>
            </div>`;
    }).join("");

    list.innerHTML = rowsHtml + `
        <div class="dflow-sources-section">
            <div class="dflow-sources-title">📊 데이터 소스</div>
            ${sourcesHtml}
        </div>`;
}

// ─── 14D 시계열 차트 ──────────────────────────────

const DflowSeries = {
    priceChart: null,
    flowChart: null,
    priceSeries: null,    // 가격 line (좌 axis)
    oiSeries: null,       // OI line (우 axis)
    cvdSeries: null,      // Perp CVD line (우 axis 별도)
    fundingSeries: null,  // Funding bars (histogram)
    takerSeries: null,    // Taker Δvol histogram
};

function _initDflowCharts() {
    if (typeof LightweightCharts === "undefined") return;

    // 1) 메인 차트: Price + OI + CVD
    const priceContainer = document.getElementById("dflow-chart-price");
    if (priceContainer && !DflowSeries.priceChart) {
        DflowSeries.priceChart = LightweightCharts.createChart(priceContainer, {
            layout: {
                background: { type: "solid", color: "transparent" },
                textColor: "#a1a1aa",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
            },
            grid: {
                vertLines: { color: "rgba(96, 81, 155, 0.06)" },
                horzLines: { color: "rgba(96, 81, 155, 0.06)" },
            },
            rightPriceScale: { borderColor: "rgba(96, 81, 155, 0.18)", visible: true },
            leftPriceScale: { borderColor: "rgba(96, 81, 155, 0.18)", visible: true },
            timeScale: {
                borderColor: "rgba(96, 81, 155, 0.18)",
                timeVisible: false,
                secondsVisible: false,
            },
            crosshair: { mode: 1 },
            handleScroll: false,
            handleScale: false,
            autoSize: true,
        });
        DflowSeries.priceSeries = DflowSeries.priceChart.addLineSeries({
            color: "#f4f4f5",
            lineWidth: 2,
            priceScaleId: "right",
            lastValueVisible: false,
            priceLineVisible: false,
        });
        DflowSeries.oiSeries = DflowSeries.priceChart.addLineSeries({
            color: "#a78bfa",
            lineWidth: 1,
            priceScaleId: "left",
            lastValueVisible: false,
            priceLineVisible: false,
        });
        DflowSeries.cvdSeries = DflowSeries.priceChart.addLineSeries({
            color: "#fbbf24",
            lineWidth: 1,
            priceScaleId: "cvd",
            lastValueVisible: false,
            priceLineVisible: false,
        });
        DflowSeries.priceChart.priceScale("cvd").applyOptions({
            scaleMargins: { top: 0.7, bottom: 0.0 },
            visible: false,
        });
    }

    // 2) 보조 차트: Funding bars + Taker Δvol histogram
    const flowContainer = document.getElementById("dflow-chart-flow");
    if (flowContainer && !DflowSeries.flowChart) {
        DflowSeries.flowChart = LightweightCharts.createChart(flowContainer, {
            layout: {
                background: { type: "solid", color: "transparent" },
                textColor: "#a1a1aa",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 9,
            },
            grid: {
                vertLines: { color: "rgba(96, 81, 155, 0.06)" },
                horzLines: { color: "rgba(96, 81, 155, 0.06)" },
            },
            rightPriceScale: { borderColor: "rgba(96, 81, 155, 0.18)" },
            timeScale: {
                borderColor: "rgba(96, 81, 155, 0.18)",
                timeVisible: false,
                secondsVisible: false,
            },
            handleScroll: false,
            handleScale: false,
            autoSize: true,
        });
        DflowSeries.takerSeries = DflowSeries.flowChart.addHistogramSeries({
            color: "rgba(52, 211, 153, 0.75)",
            priceScaleId: "right",
            lastValueVisible: false,
            priceLineVisible: false,
        });
        DflowSeries.fundingSeries = DflowSeries.flowChart.addHistogramSeries({
            color: "rgba(96, 165, 250, 0.85)",
            priceScaleId: "left",
            lastValueVisible: false,
            priceLineVisible: false,
        });
        DflowSeries.flowChart.priceScale("left").applyOptions({
            visible: false,
            scaleMargins: { top: 0.0, bottom: 0.5 },
        });
    }
}

function _seriesPointsToData(points, transform) {
    // SeriesPoint[] → lightweight-charts {time, value}[] (None 제외)
    const out = [];
    for (const p of points || []) {
        if (p.value == null || !isFinite(p.value)) continue;
        out.push({
            time: Math.floor(p.ts_ms / 1000),
            value: transform ? transform(p.value) : p.value,
        });
    }
    return out;
}

function _seriesPointsToHistogram(points, posColor, negColor) {
    const out = [];
    for (const p of points || []) {
        if (p.value == null || !isFinite(p.value)) continue;
        out.push({
            time: Math.floor(p.ts_ms / 1000),
            value: p.value,
            color: p.value >= 0 ? posColor : negColor,
        });
    }
    return out;
}

async function refreshDashboardSeries() {
    _initDflowCharts();
    const meta = document.getElementById("dflow-series-meta");

    let data;
    try {
        data = await Api.getDashboardSeries(Dflow.coin, 14);
    } catch (_) {
        if (meta) meta.textContent = "로드 실패 (네트워크)";
        return;
    }

    if (meta) {
        const ageMs = Date.now() - (data.fetched_at_ms || Date.now());
        const exCount = (data.exchanges || []).length;
        meta.textContent = `${exCount} 거래소 합본 · ${Math.max(0, Math.floor(ageMs / 60000))}분 전`;
    }

    // 차트 박힘 — lightweight-charts 미로드 시 silent skip
    if (DflowSeries.priceChart) {
        DflowSeries.priceSeries.setData(_seriesPointsToData(data.price_close));
        DflowSeries.oiSeries.setData(_seriesPointsToData(data.oi_usd, v => v / 1e9));
        DflowSeries.cvdSeries.setData(_seriesPointsToData(data.perp_cvd, v => v / 1e9));
        // v0.2.28 (사용자 요청): chart 측 최대 일자 측 박힘 — fitContent 박음
        DflowSeries.priceChart.timeScale().fitContent();
    }
    if (DflowSeries.flowChart) {
        DflowSeries.fundingSeries.setData(_seriesPointsToHistogram(
            data.funding_rate, "rgba(96, 165, 250, 0.85)", "rgba(251, 113, 133, 0.85)",
        ));
        DflowSeries.takerSeries.setData(_seriesPointsToHistogram(
            data.taker_delta_usd, "rgba(52, 211, 153, 0.75)", "rgba(251, 113, 133, 0.75)",
        ));
        DflowSeries.flowChart.timeScale().fitContent();
    }
    // v0.2.28: L/S Timeline 측 삭제 박힘 (사용자 요청)
}

// 차트 토글 — TF / 오버레이 (DOM ready 후 박힘)
function _wireChartToggles() {
    const tfBox = document.getElementById("chart-tf-toggle");
    if (tfBox) {
        for (const btn of tfBox.querySelectorAll("button")) {
            btn.addEventListener("click", () => {
                tfBox.querySelectorAll("button").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                Chart.timeframe = btn.dataset.tf;
                refreshBotChart();
            });
        }
    }
    const ovBox = document.getElementById("chart-overlay-toggle");
    if (ovBox) {
        for (const btn of ovBox.querySelectorAll("button")) {
            btn.addEventListener("click", () => {
                btn.classList.toggle("active");
                Chart.overlays[btn.dataset.ind] = btn.classList.contains("active");
                _applyChartOverlays();
            });
        }
    }
}
_wireChartToggles();

// 알림 X (닫기) — 해당 tag 만 영구 dismiss
document.getElementById("release-alert-close")?.addEventListener("click", () => {
    const alert = document.getElementById("release-alert");
    if (!alert) return;
    _dismissRelease(alert.dataset.tag);
    alert.style.display = "none";
});

// "자세히 보기" — release html_url 새 창. pywebview 환경: window.open 이 외부 브라우저
document.getElementById("release-alert-open")?.addEventListener("click", () => {
    const alert = document.getElementById("release-alert");
    if (!alert) return;
    const url = alert.dataset.url;
    if (url) window.open(url, "_blank", "noopener");
});

// v0.2.27 (사용자 보고 2026-05-11): "재시작하기" 버튼 측 fail 케이스 박혀 제거.
// 사용자 측 launcher 측 직접 종료 + 재시작 박으면 v0.2.21+ launcher 측 자동 update +
// 본체 spawn 박힘. 단순 안내 박음. POST /relaunch 흐름 측 측 backend 측 그대로 둠
// (Android 측 측 — 또는 향후 직접 사용 case).

// ============================================================
// 6b. PnL 공유 카드 (v0.1.21) — 모달 + html2canvas PNG 다운로드
// ============================================================

// 트레이드 객체 → 카드 채움 + 모달 열기.
//   trade = TradeDTO (api.py) — symbol, direction, leverage, entry_price, exit_price,
//                                pnl_usd, roi_pct, opened_at_ts, closed_at_ts, ...
function openPnlCard(trade) {
    const modal = document.getElementById("pnl-modal");
    if (!modal) return;

    // 심볼 — "BTC/USDT:USDT" → "BTCUSDT Perp"
    const symRaw = trade.symbol || "";
    const base = symRaw.split("/")[0] || symRaw;
    const quote = (symRaw.split("/")[1] || "").split(":")[0] || "USDT";
    document.getElementById("pnl-symbol").textContent = `${base}${quote} Perp`;

    // 방향 + 레버리지
    const sideEl = document.getElementById("pnl-side");
    sideEl.textContent = trade.direction === "short" ? "SHORT" : "LONG";
    sideEl.className = `pnl-card-side ${trade.direction === "short" ? "short" : "long"}`;
    document.getElementById("pnl-lev").textContent = `${trade.leverage}×`;

    // ROI (큼지막) + PnL USDT
    const roi = Number(trade.roi_pct || 0);
    const pnl = Number(trade.pnl_usd || 0);
    const roiEl = document.getElementById("pnl-roi");
    const roiSign = roi >= 0 ? "+" : "";
    roiEl.textContent = `${roiSign}${roi.toFixed(2)}%`;
    roiEl.className = `pnl-card-roi ${roi >= 0 ? "positive" : "negative"}`;
    const pnlSign = pnl >= 0 ? "+" : "";
    document.getElementById("pnl-pnl-usd").textContent = `${pnlSign}${pnl.toFixed(4)} USDT`;

    // 진입가 / 청산가
    const fmtP = (v) => Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    document.getElementById("pnl-entry").textContent = fmtP(trade.entry_price);
    document.getElementById("pnl-exit").textContent = fmtP(trade.exit_price);

    // 청산 시간 (KST)
    const d = new Date(trade.closed_at_ts);
    const yy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    const hh = String(d.getHours()).padStart(2, "0");
    const mi = String(d.getMinutes()).padStart(2, "0");
    document.getElementById("pnl-time").textContent = `${yy}-${mm}-${dd} ${hh}:${mi} KST`;

    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
}

function closePnlCard() {
    const modal = document.getElementById("pnl-modal");
    if (!modal) return;
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
}

// PnL 카드 → PNG 다운로드.
// html2canvas 로 #pnl-card 노드를 캡처. backgroundColor: null = 카드 자체 배경 사용.
async function downloadPnlCard() {
    const card = document.getElementById("pnl-card");
    const btn = document.getElementById("pnl-download");
    if (!card || typeof html2canvas !== "function") return;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "캡처 중...";
    try {
        const canvas = await html2canvas(card, {
            backgroundColor: null,
            scale: 2,            // 고해상도 (2x = 960x1200)
            useCORS: true,
            logging: false,
        });
        // 파일명 — symbol_KSTtime.png
        const sym = (document.getElementById("pnl-symbol").textContent || "trade").replace(/[^A-Za-z0-9]/g, "");
        const ts = (document.getElementById("pnl-time").textContent || "")
                       .replace(/[^0-9]/g, "").slice(0, 12);
        const fname = `Aurora_${sym}_${ts}.png`;
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = fname;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, "image/png");
    } catch (e) {
        console.error("PnL 카드 캡처 실패:", e);
        alert(`캡처 실패: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = orig;
    }
}

// 모달 이벤트 — 백드롭 클릭 / X 버튼 / 닫기 버튼 / ESC 키 / 다운로드 버튼
(() => {
    const modal = document.getElementById("pnl-modal");
    if (!modal) return;
    // 백드롭 클릭 (카드 외부) — 카드 자체 클릭은 stopPropagation 으로 무시
    modal.addEventListener("click", (e) => {
        if (e.target === modal) closePnlCard();
    });
    document.getElementById("pnl-close")?.addEventListener("click", closePnlCard);
    document.getElementById("pnl-close-btn")?.addEventListener("click", closePnlCard);
    document.getElementById("pnl-download")?.addEventListener("click", downloadPnlCard);
    // ESC 키
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.classList.contains("open")) closePnlCard();
    });
})();

// 제어 버튼 인라인 피드백 — success: 3초, error: 5초 + × 닫기
function showCtrlMsg(text, ok) {
    const el = document.getElementById("ctrl-msg");
    if (!el) return;
    clearTimeout(el._t);
    const delay = ok ? 3000 : 5000;
    el.innerHTML =
        `<span style="color:${ok ? "#22d3ee" : "#fb7185"}">${text}</span>` +
        (ok ? "" : ` <span class="ctrl-msg-close" onclick="this.parentElement.innerHTML=''">×</span>`);
    el._t = setTimeout(() => { el.innerHTML = ""; }, delay);
}

// ============================================================
// 7. 전략 / 지표 토글 (use_* 4개)
// ============================================================

// data-config 속성을 가진 입력 type 별 적용/수집 (checkbox / radio / range / number / text/select).
// v0.1.28: risk_pct 단위 변환 — UI 슬라이더 % 표시 (1.0 = "1.0%") ↔️ 백엔드 비율
// (0.01 = 1%) 미스매치 fix. UI 저장 시 / 100, 로드 시 × 100.
function _applyConfigValue(input, val) {
    if (val === undefined || val === null) return;
    if (input.type === "checkbox") {
        input.checked = !!val;
    } else if (input.type === "radio") {
        // 같은 name 라디오 그룹 — value 매칭되는 것만 checked
        input.checked = (input.value === String(val));
    } else if (input.type === "range" || input.type === "number") {
        // risk_pct: 백엔드 비율 (0.01) → UI 슬라이더 % (1.0) 변환
        const uiVal = (input.dataset.config === "risk_pct") ? Number(val) * 100 : val;
        input.value = String(uiVal);
        // 슬라이더 표시값 (lev-val 등) 갱신 트리거
        input.dispatchEvent(new Event("input"));
    } else {
        input.value = String(val);
    }
}

function _collectConfigValue(input) {
    if (input.type === "checkbox") return !!input.checked;
    if (input.type === "radio") return input.checked ? input.value : undefined;
    if (input.type === "range" || input.type === "number") {
        const v = parseFloat(input.value);
        // risk_pct: UI % (1.0) → 백엔드 비율 (0.01) 변환
        if (input.dataset.config === "risk_pct") return v / 100;
        return v;
    }
    return input.value;
}

async function loadConfigToToggles() {
    try {
        const cfg = await Api.getConfig();
        // 1. data-config 속성 모든 입력 — checkbox / radio / range / number 포괄
        document.querySelectorAll("[data-config]").forEach((input) => {
            const key = input.dataset.config;
            if (key in cfg) _applyConfigValue(input, cfg[key]);
        });
        // 2. 페어 카드 — primary_symbol 와 매칭 (예: "BTC/USDT:USDT" → "BTC/USDT")
        const primary = cfg.primary_symbol;
        if (primary) {
            const pairKey = primary.split(":")[0];  // ":USDT" suffix 제거
            document.querySelectorAll(".pair-card").forEach((card) => {
                const isSelected = card.dataset.pair === pairKey;
                card.classList.toggle("selected", isSelected);
                const meta = card.querySelector(".pair-meta");
                if (meta) meta.textContent = isSelected ? "SELECTED" : "—";
            });
        }
        // v0.1.38 — tp_allocations 4 슬라이더 복원 + 단일 모드 자동 감지
        if (Array.isArray(cfg.tp_allocations) && cfg.tp_allocations.length === 4) {
            const allocs = cfg.tp_allocations;
            // [100, 0, 0, 0] = 단일 모드 — 토글 자동 set
            const isSingle = (allocs[0] === 100 && allocs[1] === 0 && allocs[2] === 0 && allocs[3] === 0);
            const singleRadio = document.getElementById("tp-single");
            const splitRadio = document.getElementById("tp-split");
            if (isSingle && singleRadio) {
                singleRadio.checked = true;
            } else if (splitRadio) {
                splitRadio.checked = true;
                ["tp1", "tp2", "tp3", "tp4"].forEach((id, i) => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.value = String(allocs[i]);
                        el.dispatchEvent(new Event("input"));
                    }
                });
            }
            if (typeof _applyTpSplitModeUI === "function") _applyTpSplitModeUI();
        }
        // v0.1.38 — manual_tp_pcts 4 입력 복원
        if (Array.isArray(cfg.manual_tp_pcts) && cfg.manual_tp_pcts.length === 4) {
            ["manual-tp1", "manual-tp2", "manual-tp3", "manual-tp4"].forEach((id, i) => {
                const el = document.getElementById(id);
                if (el) el.value = String(cfg.manual_tp_pcts[i]);
            });
        }
        // v0.1.38 — Manual % 모드 block show/hide (cfg.tpsl_mode 따라)
        if (typeof _applyTpslModeUI === "function") _applyTpslModeUI();
    } catch (_) {
        /* 미연결 시 default 유지 */
    }
}

// ===== 외부 사용자 alias 등록 =====
// (API Key + Secret + Nickname) → config_store.user_aliases 에 저장 + bybit_alias 자동 set
// → 다음 진입부턴 alias input 에 nickname 만 입력하면 됨.
document.getElementById("btn-register-alias")?.addEventListener("click", async () => {
    const msg = document.getElementById("register-msg");
    const apiKey = document.getElementById("reg-api-key")?.value.trim();
    const apiSecret = document.getElementById("reg-api-secret")?.value.trim();
    const nickname = document.getElementById("reg-nickname")?.value.trim();

    // 입력 검증 — 셋 중 하나라도 비어있으면 거부
    if (!apiKey || !apiSecret || !nickname) {
        msg.textContent = "API Key / Secret / Nickname 모두 입력 필요";
        msg.style.color = "#fb7185";
        setTimeout(() => { msg.textContent = ""; }, 3000);
        return;
    }

    try {
        // 현재 config 가져와서 user_aliases 에 추가 + bybit_alias 자동 set
        const current = await Api.getConfig();
        const userAliases = current.user_aliases || {};
        userAliases[nickname] = { api_key: apiKey, api_secret: apiSecret };
        const merged = {
            ...current,
            user_aliases: userAliases,
            bybit_alias: nickname,    // 등록 즉시 활성 alias 로 set
        };
        await Api.updateConfig(merged);

        // Android: API 키를 Keystore(TEE/HSM) 에 암호화 저장 — 평문 .env 불필요
        if (window.AndroidKeystore) {
            try { window.AndroidKeystore.saveApiKeys("bybit", apiKey, apiSecret); } catch (_) {}
        }

        // 상단 alias input 에 nickname 자동 채움
        const aliasInput = document.getElementById("bybit-alias");
        if (aliasInput) aliasInput.value = nickname;

        // 등록 폼 초기화 — 키/시크릿 노출 시간 최소화
        document.getElementById("reg-api-key").value = "";
        document.getElementById("reg-api-secret").value = "";
        document.getElementById("reg-nickname").value = "";

        msg.textContent = `✓ '${nickname}' 등록 완료 — 이후 alias 입력만으로 매매 OK`;
        msg.style.color = "#22d3ee";
    } catch (e) {
        msg.textContent = `등록 실패: ${e.message}`;
        msg.style.color = "#fb7185";
    }
    setTimeout(() => { msg.textContent = ""; }, 5000);
});

// ============================================================
// 7a-2. Android Keystore UI (v0.1.59) — API 키 입력 + 저장 상태 배지
// ============================================================
//
// window.AndroidKeystore 있으면 (= Android WebView) 섹션 표시.
// 데스크탑에서는 섹션이 hidden 유지 — 조건 분기 없음.

function _ksRefreshBadges() {
    for (const ex of ["bybit", "okx", "binance"]) {
        const badge = document.getElementById(`ks-badge-${ex}`);
        if (!badge) continue;
        const has = window.AndroidKeystore.hasApiKeys(ex);
        badge.textContent = ex.charAt(0).toUpperCase() + ex.slice(1) + (has ? " ✓" : " ✕");
        badge.className = "ks-badge " + (has ? "ks-badge-ok" : "ks-badge-empty");
    }
}

if (window.AndroidKeystore) {
    const section = document.getElementById("android-keystore-section");
    if (section) section.style.display = "";

    _ksRefreshBadges();

    document.getElementById("ks-btn-save")?.addEventListener("click", () => {
        const exchange = document.querySelector('input[name="ks-exchange"]:checked')?.value;
        const apiKey = document.getElementById("ks-api-key")?.value.trim();
        const apiSecret = document.getElementById("ks-api-secret")?.value.trim();
        const msg = document.getElementById("ks-msg");
        if (!apiKey || !apiSecret) {
            msg.textContent = "API Key 와 Secret 모두 입력 필요";
            msg.style.color = "#fb7185";
            setTimeout(() => { msg.textContent = ""; }, 3000);
            return;
        }
        try {
            window.AndroidKeystore.saveApiKeys(exchange, apiKey, apiSecret);
            document.getElementById("ks-api-key").value = "";
            document.getElementById("ks-api-secret").value = "";
            _ksRefreshBadges();
            msg.textContent = `✓ ${exchange} 저장 완료`;
            msg.style.color = "#22d3ee";
        } catch (e) {
            msg.textContent = `저장 실패: ${e.message}`;
            msg.style.color = "#fb7185";
        }
        setTimeout(() => { msg.textContent = ""; }, 3000);
    });

    document.getElementById("ks-btn-clear")?.addEventListener("click", () => {
        const exchange = document.querySelector('input[name="ks-exchange"]:checked')?.value;
        const msg = document.getElementById("ks-msg");
        try {
            window.AndroidKeystore.clearApiKeys(exchange);
            _ksRefreshBadges();
            msg.textContent = `✓ ${exchange} 삭제 완료`;
            msg.style.color = "#a1a1aa";
        } catch (e) {
            msg.textContent = `삭제 실패: ${e.message}`;
            msg.style.color = "#fb7185";
        }
        setTimeout(() => { msg.textContent = ""; }, 3000);
    });
}

// ============================================================
// 7b. Live config apply (v0.1.28) — UI 변경 즉시 백엔드 + 봇 메모리 반영
// ============================================================
//
// 흐름:
//   1. 사용자가 토글/슬라이더/페어 카드 변경
//   2. debounce 500ms 후 saveLiveConfig() — POST /config 호출
//   3. 백엔드: config_store 저장 + bot.running 이면 apply_live_config (hot reload)
//   4. ▼ 설정 저장 버튼은 fallback 으로 유지 (수동 트리거)

function _debounce(fn, delay = 500) {
    let timer = null;
    return (...args) => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

async function _collectAndSaveConfig() {
    const msg = document.getElementById("config-msg");
    let cfg = {};
    try {
        cfg = { ...(await Api.getConfig()) };
    } catch (_) { /* 미연결 — 빈 dict */ }

    document.querySelectorAll("[data-config]").forEach((input) => {
        const val = _collectConfigValue(input);
        if (val !== undefined) cfg[input.dataset.config] = val;
    });
    const firstSelected = document.querySelector(".pair-card.selected");
    if (firstSelected) {
        cfg.primary_symbol = `${firstSelected.dataset.pair}:USDT`;
    }

    // v0.1.38 — tp_allocations 4 슬라이더 합쳐서 list 로 보냄.
    // TP 분할 모드 = "단일" 시 [100, 0, 0, 0] 강제 (UI 슬라이더 값 무시).
    const splitMode = document.querySelector('input[name="tp-split-mode"]:checked')?.value;
    if (splitMode === "single") {
        cfg.tp_allocations = [100.0, 0.0, 0.0, 0.0];
    } else {
        const tp1 = parseFloat(document.getElementById("tp1")?.value || "25");
        const tp2 = parseFloat(document.getElementById("tp2")?.value || "25");
        const tp3 = parseFloat(document.getElementById("tp3")?.value || "25");
        const tp4 = parseFloat(document.getElementById("tp4")?.value || "25");
        cfg.tp_allocations = [tp1, tp2, tp3, tp4];
    }

    // v0.1.38 — manual_tp_pcts 4 입력 합쳐서 list 로 (Manual % 모드 시 사용)
    const mtp1 = parseFloat(document.getElementById("manual-tp1")?.value || "0.5");
    const mtp2 = parseFloat(document.getElementById("manual-tp2")?.value || "1.0");
    const mtp3 = parseFloat(document.getElementById("manual-tp3")?.value || "1.5");
    const mtp4 = parseFloat(document.getElementById("manual-tp4")?.value || "2.0");
    cfg.manual_tp_pcts = [mtp1, mtp2, mtp3, mtp4];
    try {
        await Api.updateConfig(cfg);
        if (msg) {
            msg.textContent = "✓ 자동 저장됨";
            msg.style.color = "#22d3ee";
            setTimeout(() => { msg.textContent = ""; }, 2000);
        }
    } catch (e) {
        if (msg) {
            msg.textContent = `자동 저장 실패: ${e.message}`;
            msg.style.color = "#fb7185";
        }
    }
}

const saveLiveConfig = _debounce(_collectAndSaveConfig, 500);

// 모든 data-config 입력에 변경 이벤트 연결
document.querySelectorAll("[data-config]").forEach((input) => {
    input.addEventListener("change", saveLiveConfig);
    if (input.type === "range") {
        input.addEventListener("input", saveLiveConfig);
    }
});

// 페어 카드 click 도 라이브 저장 트리거
document.querySelectorAll(".pair-card").forEach((card) => {
    card.addEventListener("click", saveLiveConfig);
});

// v0.1.38 — TP 분할 모드 토글 (분할 4단계 vs 단일 TP1)
function _applyTpSplitModeUI() {
    const mode = document.querySelector('input[name="tp-split-mode"]:checked')?.value;
    const splitBlock = document.getElementById("tp-split-block");
    const splitTitle = document.getElementById("tp-split-title");
    if (!splitBlock) return;
    if (mode === "single") {
        // 단일 TP — 슬라이더 숨김 + 안내 텍스트
        splitBlock.style.display = "none";
        if (splitTitle) splitTitle.style.display = "none";
    } else {
        splitBlock.style.display = "";
        if (splitTitle) splitTitle.style.display = "";
    }
}
document.querySelectorAll('input[name="tp-split-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
        _applyTpSplitModeUI();
        saveLiveConfig();
    });
});
_applyTpSplitModeUI();  // 초기 상태 반영

// v0.1.38 — TP/SL 모드 토글 (Manual % 시 직접 입력 block show/hide)
function _applyTpslModeUI() {
    const mode = document.querySelector('input[name="tpsl-mode"]:checked')?.value;
    const manualBlock = document.getElementById("manual-tpsl-block");
    if (!manualBlock) return;
    manualBlock.style.display = (mode === "manual") ? "" : "none";
}
document.querySelectorAll('input[name="tpsl-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
        _applyTpslModeUI();
        // saveLiveConfig 는 data-config="tpsl_mode" change 이벤트로 자동 호출됨
    });
});
_applyTpslModeUI();  // 초기 상태 반영

// Manual % 입력 변경 시 saveLiveConfig 트리거 (data-config 없는 input 이라 수동 연결)
["manual-tp1", "manual-tp2", "manual-tp3", "manual-tp4"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", saveLiveConfig);
});

// 분할 익절 슬라이더 (data-config 없음, tp_allocations 로 묶어 전송)
["tp1", "tp2", "tp3", "tp4"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", saveLiveConfig);
});

document.getElementById("btn-save-config")?.addEventListener("click", async () => {
    const msg = document.getElementById("config-msg");

    // Why: 기존 config 를 base 로 merge — data-config 없는 필드 (user_aliases 등) 보존.
    // 단순히 cfg = {} 로 시작하면 dict 필드가 빈 default 로 덮어써져 등록 데이터 사라짐.
    let cfg = {};
    try {
        cfg = { ...(await Api.getConfig()) };
    } catch (_) {
        /* 첫 호출 또는 미연결 — 빈 dict 에서 시작 */
    }

    // 1. data-config 속성 모든 입력 수집 (override)
    document.querySelectorAll("[data-config]").forEach((input) => {
        const val = _collectConfigValue(input);
        if (val !== undefined) cfg[input.dataset.config] = val;
    });

    // 2. 페어 카드 — 첫 selected 를 primary_symbol 로 (Phase 1 = 단일 페어 매매)
    const firstSelected = document.querySelector(".pair-card.selected");
    if (firstSelected) {
        // ccxt linear perpetual 표준: "BTC/USDT" → "BTC/USDT:USDT"
        cfg.primary_symbol = `${firstSelected.dataset.pair}:USDT`;
    }

    try {
        await Api.updateConfig(cfg);
        msg.textContent = "✓ 저장됨";
        msg.style.color = "#22d3ee";
    } catch (e) {
        msg.textContent = `저장 실패: ${e.message}`;
        msg.style.color = "#fb7185";
    }
    setTimeout(() => {
        msg.textContent = "";
    }, 3000);
});

// ============================================================
// 8. 봇 시작 / 중지 버튼
// ============================================================

document.getElementById("btn-start")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-start");
    const stop = document.getElementById("btn-stop");
    const orig = btn.textContent;
    btn.disabled = true; stop.disabled = true;
    btn.textContent = "시작 중...";
    try {
        const r = await Api.startBot();
        showCtrlMsg(r.success ? "▶ 봇 시작됨" : `시작 실패: ${r.message}`, r.success);
    } catch (e) {
        showCtrlMsg(`API 오류: ${e.message}`, false);
    } finally {
        btn.textContent = orig;
        refreshDashboard();
    }
});

document.getElementById("btn-stop")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-stop");
    const start = document.getElementById("btn-start");
    const orig = btn.textContent;
    btn.disabled = true; start.disabled = true;
    btn.textContent = "중지 중...";
    try {
        const r = await Api.stopBot();
        showCtrlMsg(r.success ? "■ 봇 중지됨" : `중지 실패: ${r.message}`, r.success);
    } catch (e) {
        showCtrlMsg(`API 오류: ${e.message}`, false);
    } finally {
        btn.textContent = orig;
        refreshDashboard();
    }
});

document.getElementById("btn-restart")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-restart");
    const start = document.getElementById("btn-start");
    const stop = document.getElementById("btn-stop");
    const orig = btn.textContent;
    btn.disabled = true;
    if (start) start.disabled = true;
    if (stop) stop.disabled = true;
    btn.textContent = "재시작 중...";
    try {
        const r = await Api.restartBot();
        showCtrlMsg(r.success ? "↻ 봇 재시작됨" : `재시작 실패: ${r.message}`, r.success);
    } catch (e) {
        showCtrlMsg(`API 오류: ${e.message}`, false);
    } finally {
        btn.textContent = orig;
        refreshDashboard();
    }
});

// ============================================================
// 9. 거래소 연결 테스트 (stub)
// ============================================================

document.getElementById("btn-test-conn")?.addEventListener("click", async () => {
    const msg = document.getElementById("conn-test-msg");
    msg.textContent = "테스트 중...";
    msg.style.color = "var(--aurora-purple)";
    try {
        const h = await Api.health();
        msg.textContent = `✓ API 연결됨 (mode=${h.mode})`;
        msg.style.color = "#4ade80";
    } catch (e) {
        msg.textContent = `✗ ${e.message}`;
        msg.style.color = "#fb7185";
    }
});

// ============================================================
// 10. 백테스트 실행 (stub — 진행바 데모만)
// ============================================================

document.getElementById("btn-run-bt")?.addEventListener("click", () => {
    const bar = document.getElementById("bt-progress");
    const label = document.getElementById("bt-progress-label");
    if (!bar || !label) return;
    bar.style.width = "0%";
    label.textContent = "실행 중... (stub demo)";
    let p = 0;
    const interval = setInterval(() => {
        p += 5;
        bar.style.width = `${p}%`;
        if (p >= 100) {
            clearInterval(interval);
            label.textContent = "완료 (stub) — backtest 엔진 본 구현 후 실 결과 표시";
        }
    }, 100);
});

// ============================================================
// 11. Logs view — WebSocket 실시간 + fallback 폴링 + 필터 / 검색 / 다운로드
// ============================================================

const Logs = (() => {
    // PR-G: 200줄 상한 (기존 1000 → DOM/메모리 절감 + 실용 충분).
    const MAX_LINES = 200;
    const MAX_WS_RETRIES = 5;   // 5회 실패 → 폴링 fallback 고정
    const POLL_INTERVAL = 5000; // fallback 폴링 주기 (ms)

    const buffer = [];
    let liveConn = null;
    let wsRetries = 0;          // WS 재연결 시도 횟수
    let pollTimer = null;       // fallback 폴링 interval ID
    let initialized = false;

    const $ = (id) => document.getElementById(id);
    const $box = () => $("log-box");
    const $empty = () => $("log-empty");
    const $count = () => $("log-count");
    const $status = () => $("log-status");
    const $autoStream = () => $("log-autostream");
    const $search = () => $("log-search");
    const $scrollBtn = () => $("log-scroll-btn");

    // 사용자가 켠 레벨 set
    function enabledLevels() {
        const set = new Set();
        document.querySelectorAll("[data-log-level]").forEach((el) => {
            if (el.checked) set.add(el.dataset.logLevel);
        });
        return set;
    }

    // 모듈 필터 — 버튼에 active-* 클래스가 있으면 켜진 것
    function enabledModules() {
        const set = new Set();
        document.querySelectorAll("[data-log-module]").forEach((el) => {
            const m = el.dataset.logModule;
            if (el.classList.contains(`active-${m}`)) set.add(m);
        });
        return set;
    }

    function levelClass(level) {
        if (level === "ERROR" || level === "CRITICAL") return "log-level-error";
        if (level === "WARNING" || level === "WARN") return "log-level-warn";
        return "log-level-info";
    }

    // logger 이름 → aurora.<module> 추출 (없으면 null)
    function loggerModule(logger) {
        if (!logger) return null;
        const m = logger.match(/^aurora\.(core|exchange|interfaces|backtest)/);
        return m ? m[1] : null;
    }

    // 필터·검색·모듈 매칭
    function isVisible(record, levels, query, modules) {
        const lvl = (record.level === "WARN") ? "WARNING"
                  : (record.level === "CRITICAL") ? "ERROR"
                  : record.level;
        if (!levels.has(lvl)) return false;
        if (query) {
            const hay = `${record.message || ""} ${record.logger || ""}`.toLowerCase();
            if (!hay.includes(query.toLowerCase())) return false;
        }
        // 모듈 필터: aurora.* 로거이고 해당 모듈이 꺼져 있으면 숨김.
        // aurora.* 아닌 로거(root 등)는 모듈 필터 대상 외 → 항상 표시.
        const mod = loggerModule(record.logger);
        if (mod && !modules.has(mod)) return false;
        return true;
    }

    function makeLineEl(record) {
        const el = document.createElement("div");
        el.className = `log-line ${levelClass(record.level)}`;
        el.dataset.level = record.level;
        const ts = document.createElement("span");
        ts.className = "log-line-ts";
        ts.textContent = toKstString(record.ts);
        const lvl = document.createElement("span");
        lvl.className = "log-line-level";
        lvl.textContent = (record.level || "").padEnd(7).slice(0, 7);
        const msg = document.createElement("span");
        msg.className = "log-line-msg";
        if (record.logger) {
            const prefix = document.createElement("span");
            const mod = loggerModule(record.logger);
            prefix.className = mod ? `log-prefix-${mod}` : "";
            prefix.textContent = record.logger + ": ";
            const text = document.createElement("span");
            text.textContent = record.message || "";
            msg.append(prefix, text);
        } else {
            msg.textContent = record.message || "";
        }
        el.append(ts, lvl, msg);
        return el;
    }

    function updateScrollBtn() {
        const box = $box();
        const btn = $scrollBtn();
        if (!box || !btn) return;
        const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
        btn.classList.toggle("visible", !atBottom);
    }

    function appendLine(record) {
        const box = $box();
        if (!box) return;
        const empty = $empty();
        if (empty) empty.style.display = "none";
        const levels = enabledLevels();
        const query = ($search()?.value) || "";
        const modules = enabledModules();
        const wasAtBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
        const el = makeLineEl(record);
        if (!isVisible(record, levels, query, modules)) el.style.display = "none";
        box.appendChild(el);
        while (box.querySelectorAll(".log-line").length > MAX_LINES) {
            box.querySelector(".log-line")?.remove();
        }
        if (wasAtBottom) box.scrollTop = box.scrollHeight;
        updateScrollBtn();
    }

    function rerenderAll() {
        const box = $box();
        if (!box) return;
        Array.from(box.querySelectorAll(".log-line")).forEach((el) => el.remove());
        const levels = enabledLevels();
        const query = ($search()?.value) || "";
        const modules = enabledModules();
        let visibleCount = 0;
        for (const r of buffer) {
            const el = makeLineEl(r);
            if (!isVisible(r, levels, query, modules)) el.style.display = "none";
            else visibleCount++;
            box.appendChild(el);
        }
        const empty = $empty();
        if (empty) empty.style.display = visibleCount === 0 ? "block" : "none";
        box.scrollTop = box.scrollHeight;
        updateCount(visibleCount);
        updateScrollBtn();
    }

    function updateCount(visible) {
        const c = $count();
        if (!c) return;
        if (visible !== undefined && visible !== buffer.length) {
            c.textContent = `(필터 ${visible} / 전체 ${buffer.length})`;
        } else {
            c.textContent = `(${buffer.length} 줄)`;
        }
    }

    function setStatus(text, ok) {
        const el = $status();
        if (!el) return;
        el.textContent = text;
        el.style.color = ok ? "#22d3ee" : "#fb7185";
    }

    function pushRecord(record) {
        buffer.push(record);
        if (buffer.length > MAX_LINES) buffer.shift();
        appendLine(record);
        updateCount();
    }

    async function pollOnce(limit = 100) {
        try {
            const data = await Api.getLogs(limit);
            const lines = (data && data.lines) || [];
            const seen = new Set(buffer.map((r) => `${r.ts}|${r.message}`));
            for (const r of lines) {
                const k = `${r.ts}|${r.message}`;
                if (!seen.has(k)) {
                    buffer.push(r);
                    seen.add(k);
                }
            }
            while (buffer.length > MAX_LINES) buffer.shift();
            rerenderAll();
            setStatus(`폴링 (${lines.length}줄)`, true);
        } catch (e) {
            setStatus(`폴링 실패: ${e.message}`, false);
        }
    }

    function startPolling() {
        if (pollTimer) return;
        pollOnce(100);
        pollTimer = setInterval(() => pollOnce(100), POLL_INTERVAL);
        setStatus("폴링 모드 (WS 불가)", false);
    }

    function stopPolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    function startLive() {
        if (liveConn) return;
        setStatus("실시간 연결 중...", true);
        liveConn = Api.connectLiveLog({
            onOpen: () => setStatus("LIVE (대기)", true),
            onMessage: (record) => {
                pushRecord(record);
                setStatus("LIVE", true);
            },
            onError: () => {
                stopLive();
                if (!$autoStream()?.checked) return;
                wsRetries++;
                if (wsRetries <= MAX_WS_RETRIES) {
                    setStatus(`LIVE 끊김 (${wsRetries}/${MAX_WS_RETRIES}) — 재연결 중...`, false);
                    setTimeout(() => { if ($autoStream()?.checked) startLive(); }, 5000);
                } else {
                    // 5회 초과 → polling fallback 고정 (토글 켜진 동안 유지)
                    startPolling();
                }
            },
        });
    }

    function stopLive() {
        if (liveConn) {
            liveConn.close();
            liveConn = null;
        }
    }

    function init() {
        if (initialized) return;
        const box = $box();
        if (!box) return;
        initialized = true;

        // 레벨 체크박스
        document.querySelectorAll("[data-log-level]").forEach((cb) => {
            cb.addEventListener("change", rerenderAll);
        });

        // 모듈 필터 토글 버튼
        document.querySelectorAll("[data-log-module]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const m = btn.dataset.logModule;
                btn.classList.toggle(`active-${m}`);
                rerenderAll();
            });
        });

        // 검색
        $search()?.addEventListener("input", rerenderAll);

        // 실시간 토글 — 켜면 retry 초기화 후 WS 시도, 끄면 WS+폴링 모두 중단
        $autoStream()?.addEventListener("change", () => {
            if ($autoStream().checked) {
                wsRetries = 0;
                stopPolling();
                startLive();
            } else {
                stopLive();
                stopPolling();
                setStatus("실시간 OFF", true);
            }
        });

        // 퀵 필터 버튼 — ERROR만 / 전체
        $("log-filter-error-only")?.addEventListener("click", () => {
            document.querySelectorAll("[data-log-level]").forEach((el) => {
                el.checked = el.dataset.logLevel === "ERROR";
            });
            rerenderAll();
        });
        $("log-filter-all-levels")?.addEventListener("click", () => {
            document.querySelectorAll("[data-log-level]").forEach((el) => {
                el.checked = true;
            });
            rerenderAll();
        });

        // 버튼들
        $("log-refresh")?.addEventListener("click", () => pollOnce(200));
        $("log-clear")?.addEventListener("click", () => {
            Array.from(box.querySelectorAll(".log-line")).forEach((el) => el.remove());
            const empty = $empty();
            if (empty) empty.style.display = "block";
            setStatus(`화면 비움 (버퍼 ${buffer.length}줄 유지)`, true);
        });

        // ↓ 최신으로 버튼 — 사용자가 위로 스크롤 중일 때 나타남
        box.addEventListener("scroll", updateScrollBtn);
        $scrollBtn()?.addEventListener("click", () => {
            box.scrollTop = box.scrollHeight;
            updateScrollBtn();
        });

        // v0.1.35 — 로그 복사/저장
        function _formatLogText() {
            const lines = Array.from(box.querySelectorAll(".log-line"))
                .filter((el) => el.style.display !== "none");
            return lines.map((el) => {
                const ts = el.querySelector(".log-line-ts")?.textContent || "";
                const lvl = (el.querySelector(".log-line-level")?.textContent || "").trim();
                const msg = el.querySelector(".log-line-msg")?.textContent || "";
                return `${ts}\t${lvl}\t${msg}`;
            }).join("\n");
        }

        $("log-copy")?.addEventListener("click", async () => {
            const text = _formatLogText();
            if (!text) { setStatus("복사 X — 표시된 로그 없음", false); return; }
            try {
                await navigator.clipboard.writeText(text);
                setStatus(`✓ 클립보드 복사 (${text.split("\n").length}줄)`, true);
            } catch (e) {
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.left = "-9999px";
                document.body.appendChild(ta);
                ta.select();
                let ok = false;
                try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
                document.body.removeChild(ta);
                if (ok) {
                    setStatus(`✓ 클립보드 복사 (${text.split("\n").length}줄)`, true);
                } else {
                    setStatus(`복사 실패: ${e.message}`, false);
                }
            }
        });

        $("log-download")?.addEventListener("click", () => {
            const text = _formatLogText();
            if (!text) { setStatus("저장 X — 표시된 로그 없음", false); return; }
            const ts = new Date();
            const fname = `aurora_log_${ts.getFullYear()}${String(ts.getMonth()+1).padStart(2,"0")}${String(ts.getDate()).padStart(2,"0")}_${String(ts.getHours()).padStart(2,"0")}${String(ts.getMinutes()).padStart(2,"0")}.txt`;
            const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = fname;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            setStatus(`✓ 저장됨 (${fname})`, true);
        });

        // 초기 catch-up + 자동 스트리밍 시작
        pollOnce(100).then(() => {
            if ($autoStream()?.checked) startLive();
        });
    }

    return { init };
})();

Logs.init();

// ============================================================
// 12. 초기 로드 + 폴링
// ============================================================

refreshDashboard();
loadConfigToToggles();

// 15초 주기 대시보드 폴링 (봇 상태 자주 안 바뀜, /start /stop 직후엔 즉시 fetch)
setInterval(refreshDashboard, 15000);

// v0.1.113: connection 상태 별도 polling (3초 주기) — data refresh 측 transient
// slow 측 connection 깜빡임 본질 차단. /health 측 ccxt 호출 X 라 즉시 응답.
_checkHealth();
setInterval(_checkHealth, 3000);
