// Aurora Launcher GUI v0.1.16 — Start 단일 흐름.
// 사용자가 START 클릭 → 자동 update check → has_update 면 download+swap → 본체 실행.
// v0.4.66 (G-2b): 시작 시 라이선스 게이트 추가 — license.json 없거나 verify fail 시
// 코드 입력 화면 표시, 통과 후 메인 화면.

let Api = null;

const versionInfo = document.getElementById("version-info");
const statusLine = document.getElementById("status-line");
const logList = document.getElementById("log-list");
const btnStart = document.getElementById("btn-start");

// 라이선스 게이트 요소 (v0.4.66)
const licenseGate = document.getElementById("license-gate");
const mainScreen = document.getElementById("main-screen");
const licenseInput = document.getElementById("license-code-input");
const licenseSubmit = document.getElementById("license-submit-btn");
const licenseMessage = document.getElementById("license-message");
const licenseBadge = document.getElementById("license-badge");

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

// ========================================================================
// v0.4.66 (G-2b) — 라이선스 게이트
// ========================================================================

function setLicenseMessage(text, kind) {
    licenseMessage.textContent = text || "";
    licenseMessage.className = "license-message" + (kind ? " " + kind : "");
}

function showLicenseGate() {
    licenseGate.style.display = "flex";
    mainScreen.style.display = "none";
    setTimeout(() => licenseInput.focus(), 100);
}

function showMainScreen(status) {
    licenseGate.style.display = "none";
    mainScreen.style.display = "";

    // 라이선스 배지 (좌상단) — type + 만료일 + D-3/D-1 경고 + grace
    if (status && status.has_license) {
        const typeLabel = status.type === "referral" ? "레퍼럴" :
                          status.type === "sub_30d" ? "30일 구독" :
                          status.type === "sub_90d" ? "90일 구독" :
                          status.type === "sub_365d" ? "365일 구독" : status.type;
        let text = typeLabel;
        if (status.expires_at) {
            const d = new Date(status.expires_at);
            text += ` · 만료 ${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
        }

        // v0.4.67 (G-3a): D-3/D-1/today 알림 — 배지에 남은 일수 + 색상
        const days = status.days_until_expiry;
        const level = status.expiry_warning_level || "none";
        licenseBadge.classList.remove("grace", "warn-d3", "warn-d1", "warn-today");

        if (level === "today") {
            text += ` · ⚠ 오늘 만료`;
            licenseBadge.classList.add("warn-today");
        } else if (level === "d1") {
            text += ` · ⚠ 1일 남음`;
            licenseBadge.classList.add("warn-d1");
        } else if (level === "d3" && days !== null && days !== undefined) {
            text += ` · ⚠ ${days}일 남음`;
            licenseBadge.classList.add("warn-d3");
        }

        if (status.verify_ok === false && status.grace_ok && level === "none") {
            text += " · 오프라인 (grace)";
            licenseBadge.classList.add("grace");
        }

        licenseBadge.textContent = text;
        licenseBadge.style.display = "";

        // d1/today 면 status-line 에도 한 번 알림 (시작 시 한정, 사용자 주의 환기)
        if (level === "d1" || level === "today") {
            const msg = level === "today" ? "라이선스가 오늘 만료됩니다 — 갱신 필요"
                                          : "라이선스 만료 1일 남음 — 갱신 필요";
            setStatus(msg, "#fbbf24");
        }
    } else {
        licenseBadge.style.display = "none";
    }
}

// 코드 입력 자동 포맷 — AICT- prefix 자동 + 4글자마다 - 삽입
function formatCodeInput(raw) {
    let cleaned = raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
    // AICT prefix 정리 — 앞 4글자가 AICT 가 아니면 prepend X (사용자가 직접)
    const groups = cleaned.match(/.{1,4}/g) || [];
    return groups.slice(0, 4).join("-");
}

licenseInput.addEventListener("input", (e) => {
    const formatted = formatCodeInput(e.target.value);
    if (formatted !== e.target.value) {
        e.target.value = formatted;
    }
});

licenseInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        licenseSubmit.click();
    }
});

licenseSubmit.addEventListener("click", async () => {
    if (!Api) return;
    const code = licenseInput.value.trim();
    if (!code) {
        setLicenseMessage("코드를 입력해주세요", "error");
        return;
    }
    licenseSubmit.disabled = true;
    setLicenseMessage("등록 중...", "info");
    try {
        const result = await Api.redeem_code(code);
        if (result.ok) {
            setLicenseMessage(result.message || "라이선스 등록 완료", "success");
            // 잠깐 메시지 보여주고 메인 화면으로
            setTimeout(async () => {
                const status = await Api.get_license_status();
                showMainScreen(status);
            }, 800);
        } else {
            setLicenseMessage(result.message || "등록 실패", "error");
            licenseSubmit.disabled = false;
        }
    } catch (e) {
        setLicenseMessage(`오류: ${e.message}`, "error");
        licenseSubmit.disabled = false;
    }
});

async function checkLicenseGate() {
    // 라이선스 게이트 — 진입 시 한 번만 호출
    try {
        const status = await Api.get_license_status();
        if (!status.has_license) {
            showLicenseGate();
            return false;
        }
        if (status.verify_ok) {
            showMainScreen(status);
            return true;
        }
        // verify fail — grace 안이면 메인 진입 (경고 배지), 아니면 게이트
        if (status.grace_ok) {
            showMainScreen(status);
            return true;
        }
        // grace 도 만료 — 게이트 진입 + 사유 표시
        showLicenseGate();
        const reason = status.verify_error === "expired" ? "라이선스가 만료되었습니다" :
                       status.verify_error === "pc_mismatch" ? "다른 PC 에서 사용 중인 코드입니다" :
                       status.verify_error === "voided" ? "무효화된 코드입니다 — 관리자에게 문의" :
                       status.verify_error === "bad_token" ? "라이선스 토큰이 손상되었습니다 — 재발급 필요" :
                       status.verify_error === "network" ? "네트워크 오프라인 + grace 만료 — 인터넷 연결 후 재시도" :
                       `라이선스 무효 (${status.verify_error || "unknown"})`;
        setLicenseMessage(reason, "error");
        return false;
    } catch (e) {
        // get_license_status 자체가 깨졌으면 일단 게이트 진입
        log(`라이선스 확인 실패: ${e.message}`);
        showLicenseGate();
        setLicenseMessage(`라이선스 확인 실패: ${e.message}`, "error");
        return false;
    }
}

window.addEventListener("pywebviewready", async () => {
    Api = window.pywebview.api;
    await loadVersionInfo();
    log("Launcher 시작");

    // v0.4.66 (G-2b): 라이선스 게이트 — main-screen 진입 전 통과 필수.
    const licenseOk = await checkLicenseGate();
    if (!licenseOk) {
        return;  // 게이트에 멈춤 — 코드 입력 대기
    }

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
