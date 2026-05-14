// Aurora Launcher GUI v0.1.16 — Start 단일 흐름.
// 사용자가 START 클릭 → 자동 update check → has_update 면 download+swap → 본체 실행.

let Api = null;

const versionInfo = document.getElementById("version-info");
const statusLine = document.getElementById("status-line");
const logList = document.getElementById("log-list");
const btnStart = document.getElementById("btn-start");

function log(msg) {
    const li = document.createElement("li");
    li.textContent = msg;
    logList.appendChild(li);
    logList.parentElement.scrollTop = logList.parentElement.scrollHeight;
}

function setStatus(text, color) {
    statusLine.textContent = text;
    statusLine.style.color = color || "var(--text-3)";
}

async function loadVersionInfo() {
    if (!Api) return;
    try {
        const local = await Api.get_local_version();
        const launcherV = await Api.get_launcher_version();
        const localStr = (local && local !== "unknown") ? `v${local}` : "—";
        versionInfo.textContent = `Launcher v${launcherV} · 본체 ${localStr}`;
    } catch (e) {
        log(`정보 조회 실패: ${e.message}`);
    }
}

// START 버튼 — 단일 흐름: check → (필요시) download+swap → launch
async function startFlow() {
    if (!Api) return;
    btnStart.disabled = true;

    // 1. 업데이트 체크
    setStatus("업데이트 확인 중...", "var(--text-2)");
    log("업데이트 확인 시작");
    let checkResult;
    try {
        checkResult = await Api.check_update();
    } catch (e) {
        setStatus(`✗ 체크 실패: ${e.message} — 본체만 실행`, "#fb7185");
        log(`체크 실패: ${e.message}`);
        await launchOnly();
        return;
    }

    if (checkResult.error) {
        // 네트워크 실패 — 본체 그대로 실행
        setStatus(`⚠ ${checkResult.error} — 본체 그대로 실행`, "#fbbf24");
        log(checkResult.error);
        await launchOnly();
        return;
    }

    // 2. has_update = true 면 다운 + swap
    if (checkResult.has_update && checkResult.url) {
        setStatus(`최신 ${checkResult.latest} 다운로드 중...`, "var(--text-2)");
        log(`업데이트 ${checkResult.latest} 발견`);
        try {
            const swap = await Api.download_and_swap(checkResult.url);
            if (!swap.success) {
                setStatus(`✗ ${swap.message} — 기존 본체 실행`, "#fb7185");
                log(`업데이트 실패: ${swap.message}`);
            } else {
                log("업데이트 적용 완료");
                await loadVersionInfo();
            }
        } catch (e) {
            setStatus(`✗ ${e.message}`, "#fb7185");
            log(`업데이트 실패: ${e.message}`);
        }
    } else {
        log(`최신 버전 (${checkResult.latest})`);
    }

    // 3. 본체 실행
    await launchOnly();
}

async function launchOnly() {
    // v0.1.116 (ChoYoon #133): 이전 흐름 측 launch() 측 즉시 "✓ Aurora 시작됨"
    // 박힘 → 사용자 측 까만 화면 (본체 startup ~37초). 신규 흐름 측 launch()
    // 측 backend readiness polling thread 측 status 측 매 5초 박음 →
    //     "본체 시작 중... (10초)" → "본체 시작 중... (35초)" → "✓ Aurora 시작됨"
    // → 0.3초 후 launcher hide. JS 측 init 만 박고 backend 측 갱신 박음.
    setStatus("Aurora 시작 중...", "var(--text-2)");
    try {
        const r = await Api.launch();
        if (r.success) {
            // v0.1.80: launcher 항상 살아있음 + 본체 ready 박힐 때 hide.
            // 본체 종료 시 자동 등장 (LauncherApi 측 polling thread).
            log("Aurora 시작 — readiness polling 박힘 (본체 /health 200 대기)");
            // status 측 backend _start_readiness_polling 측 박음 — JS overwrite X
        } else {
            setStatus(`✗ ${r.message}`, "#fb7185");
            log(`시작 실패: ${r.message}`);
            btnStart.disabled = false;
        }
    } catch (e) {
        setStatus(`✗ ${e.message}`, "#fb7185");
        log(`시작 실패: ${e.message}`);
        btnStart.disabled = false;
    }
}

// v0.1.80: launcher webview show 시점 (polling 측 evaluate_js) 호출 가능하도록
// setStatus 를 window 측에 박음.
window.setStatus = setStatus;

// v0.2.24 (ChoYoon #133 P1 ③): launcher progress bar — backend readiness polling 측
// 매 2.5초 측 setProgress(percent, text) 박음. 사용자 측 멈춤 의심 회피 + 시각
// 자료. 100% + ✓ 박은 후 launcher hide 측 자연 흐름 박음.
window.setProgress = (percent, text) => {
    const container = document.getElementById("progress-container");
    const bar = document.getElementById("progress-bar");
    const label = document.getElementById("progress-text");
    if (!container || !bar || !label) return;
    container.style.display = "block";
    bar.style.width = Math.max(0, Math.min(100, percent)) + "%";
    label.textContent = text || "";
};

window.hideProgress = () => {
    const container = document.getElementById("progress-container");
    if (container) container.style.display = "none";
};

btnStart.addEventListener("click", startFlow);

window.addEventListener("pywebviewready", async () => {
    Api = window.pywebview.api;
    await loadVersionInfo();
    log("Launcher 시작");

    // v0.1.43: 본체 /relaunch 흐름 — auto-start 모드면 START 자동 클릭.
    // 사용자가 본체 UI 의 업데이트 팝업 "재시작하기" 클릭 → 본체가 launcher
    // spawn (env AURORA_LAUNCHER_AUTO_START=1) → launcher 가 자동 START.
    try {
        const auto = await Api.is_auto_start();
        if (auto) {
            log("자동 재시작 모드 — START 자동 트리거");
            // 약간의 지연으로 UI 렌더링 + 사용자가 "뭐 일어나는지" 확인 가능
            setTimeout(() => btnStart.click(), 600);
        }
    } catch (_) { /* 구버전 launcher 호환 — is_auto_start 없으면 silent skip */ }
});
