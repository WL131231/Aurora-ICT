/**
 * Aurora-ICT i18n + timezone — 4개어 (한/영/중간/일) + 4개 시간대.
 *
 * 2026-05-30 파트너 요청 — 사이드바 + killzone 배지 다국어 + 사용자 timezone.
 * localStorage 저장: aurora_ict_lang / aurora_ict_tz.
 *
 * 사용:
 *   - HTML 에 data-i18n="key" 속성 박은 element 들 자동 번역
 *   - JS 에서 t("key") 로 동적 텍스트 가져오기
 *   - 시간 변환: nyTimeToUserTz(h, m) → 사용자 timezone "HH:mm"
 */

(function () {
  const LANG_KEY = "aurora_ict_lang";
  const TZ_KEY = "aurora_ict_tz";

  // 지원 언어 — 한국어 default.
  const SUPPORTED_LANGS = ["ko", "en", "zh", "ja"];
  // 지원 시간대 (IANA tz name).
  const SUPPORTED_TZS = {
    "Asia/Seoul": "한국 (KST)",
    "America/New_York": "미국 NY (EST/EDT)",
    "Asia/Tokyo": "일본 (JST)",
    "Asia/Shanghai": "중국 (CST)",
  };

  // ============================================================
  // 번역 dictionary
  // ============================================================
  const DICT = {
    ko: {
      // 사이드바 섹션
      "sidebar.mode": "MODE",
      "sidebar.tradingPair": "TRADING PAIR",
      "sidebar.tradeTimeframe": "TRADE TIMEFRAME",
      "sidebar.apiCredentials": "API CREDENTIALS",
      "sidebar.equity": "EQUITY",
      "sidebar.dailyLossLimit": "DAILY LOSS LIMIT",
      "sidebar.control": "CONTROL",
      "sidebar.license": "LICENSE",
      "sidebar.preferences": "설정",
      // 버튼
      "btn.demo": "DEMO",
      "btn.live": "LIVE",
      "btn.start": "START",
      "btn.stop": "STOP",
      "btn.register": "등록",
      "btn.reregister": "재등록",
      "btn.set": "SET",
      "btn.viewTrades": "매매 기록 보기",
      // 라벨
      "label.todayPnl": "Today PnL",
      "label.status": "Status",
      "label.realtimeSec": "실시간 (5초)",
      "label.limitPct": "Limit %",
      "label.off": "0 = off",
      "label.language": "언어",
      "label.timezone": "시간대",
      // hint
      "hint.pairSelect": "선택한 페어만 매매 — 복수 선택 시 동시 진입 가능",
      "hint.tradeTf": "진입 봉 단위 — 선택한 TF 이상 봉으로 매매 결정 (1m 은 노이즈로 제외)",
      // killzone
      "kz.asian": "Asian",
      "kz.london": "London",
      "kz.ny_am": "NY AM",
      "kz.pm": "NY PM",
    },
    en: {
      "sidebar.mode": "MODE",
      "sidebar.tradingPair": "TRADING PAIR",
      "sidebar.tradeTimeframe": "TRADE TIMEFRAME",
      "sidebar.apiCredentials": "API CREDENTIALS",
      "sidebar.equity": "EQUITY",
      "sidebar.dailyLossLimit": "DAILY LOSS LIMIT",
      "sidebar.control": "CONTROL",
      "sidebar.license": "LICENSE",
      "sidebar.preferences": "Preferences",
      "btn.demo": "DEMO",
      "btn.live": "LIVE",
      "btn.start": "START",
      "btn.stop": "STOP",
      "btn.register": "Register",
      "btn.reregister": "Re-register",
      "btn.set": "SET",
      "btn.viewTrades": "View Trade History",
      "label.todayPnl": "Today PnL",
      "label.status": "Status",
      "label.realtimeSec": "Realtime (5s)",
      "label.limitPct": "Limit %",
      "label.off": "0 = off",
      "label.language": "Language",
      "label.timezone": "Timezone",
      "hint.pairSelect": "Trade selected pairs only — multi-select allows simultaneous entries",
      "hint.tradeTf": "Entry bar timeframe — trades decided on bars ≥ this TF (1m excluded as noise)",
      "kz.asian": "Asian",
      "kz.london": "London",
      "kz.ny_am": "NY AM",
      "kz.pm": "NY PM",
    },
    zh: {
      "sidebar.mode": "模式",
      "sidebar.tradingPair": "交易对",
      "sidebar.tradeTimeframe": "交易周期",
      "sidebar.apiCredentials": "API 密钥",
      "sidebar.equity": "权益",
      "sidebar.dailyLossLimit": "每日亏损上限",
      "sidebar.control": "控制",
      "sidebar.license": "授权",
      "sidebar.preferences": "设置",
      "btn.demo": "模拟",
      "btn.live": "实盘",
      "btn.start": "启动",
      "btn.stop": "停止",
      "btn.register": "注册",
      "btn.reregister": "重新注册",
      "btn.set": "设置",
      "btn.viewTrades": "查看交易记录",
      "label.todayPnl": "今日盈亏",
      "label.status": "状态",
      "label.realtimeSec": "实时 (5秒)",
      "label.limitPct": "上限 %",
      "label.off": "0 = 关闭",
      "label.language": "语言",
      "label.timezone": "时区",
      "hint.pairSelect": "仅交易已选交易对 — 多选可同时进入",
      "hint.tradeTf": "入场K线周期 — 选定TF以上的K线决策 (1m作为噪音排除)",
      "kz.asian": "亚洲",
      "kz.london": "伦敦",
      "kz.ny_am": "纽约早市",
      "kz.pm": "纽约午市",
    },
    ja: {
      "sidebar.mode": "モード",
      "sidebar.tradingPair": "取引ペア",
      "sidebar.tradeTimeframe": "取引時間枠",
      "sidebar.apiCredentials": "API キー",
      "sidebar.equity": "残高",
      "sidebar.dailyLossLimit": "日次損失上限",
      "sidebar.control": "操作",
      "sidebar.license": "ライセンス",
      "sidebar.preferences": "設定",
      "btn.demo": "デモ",
      "btn.live": "本番",
      "btn.start": "開始",
      "btn.stop": "停止",
      "btn.register": "登録",
      "btn.reregister": "再登録",
      "btn.set": "設定",
      "btn.viewTrades": "取引履歴を見る",
      "label.todayPnl": "本日損益",
      "label.status": "ステータス",
      "label.realtimeSec": "リアルタイム (5秒)",
      "label.limitPct": "上限 %",
      "label.off": "0 = オフ",
      "label.language": "言語",
      "label.timezone": "タイムゾーン",
      "hint.pairSelect": "選択したペアのみ取引 — 複数選択で同時エントリー可能",
      "hint.tradeTf": "エントリー足単位 — 選択TF以上の足で判定 (1mはノイズ除外)",
      "kz.asian": "アジア",
      "kz.london": "ロンドン",
      "kz.ny_am": "NY 前場",
      "kz.pm": "NY 後場",
    },
  };

  // ============================================================
  // 언어 / 시간대 storage
  // ============================================================
  function getLang() {
    try {
      const v = localStorage.getItem(LANG_KEY);
      if (SUPPORTED_LANGS.includes(v)) return v;
    } catch (e) {}
    return "ko";
  }
  function setLang(lang) {
    if (!SUPPORTED_LANGS.includes(lang)) return;
    try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
  }
  function getTz() {
    try {
      const v = localStorage.getItem(TZ_KEY);
      if (v && Object.prototype.hasOwnProperty.call(SUPPORTED_TZS, v)) return v;
    } catch (e) {}
    return "Asia/Seoul";
  }
  function setTz(tz) {
    if (!Object.prototype.hasOwnProperty.call(SUPPORTED_TZS, tz)) return;
    try { localStorage.setItem(TZ_KEY, tz); } catch (e) {}
  }

  // ============================================================
  // 번역 + DOM 갱신
  // ============================================================
  function t(key) {
    const lang = getLang();
    const dict = DICT[lang] || DICT.ko;
    return dict[key] !== undefined ? dict[key] : (DICT.ko[key] || key);
  }

  function applyI18nToDOM() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      el.textContent = t(key);
    });
    // title 속성도 i18n 처리.
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      const key = el.getAttribute("data-i18n-title");
      if (key) el.setAttribute("title", t(key));
    });
  }

  // ============================================================
  // 시간대 — NY local 시각을 사용자 tz "HH:mm" 으로 변환
  // ============================================================

  /** 특정 tz 의 현재 UTC offset (분 단위, EDT=-240, EST=-300, KST=540). */
  function _getTzOffsetMinutes(tz, date) {
    try {
      const dtf = new Intl.DateTimeFormat("en-US", {
        timeZone: tz,
        timeZoneName: "shortOffset",
      });
      const parts = dtf.formatToParts(date);
      const tzName = parts.find((p) => p.type === "timeZoneName")?.value || "";
      // "GMT-4" / "GMT+9" / "GMT-04:30"
      const m = tzName.match(/GMT([+-])(\d{1,2})(?::(\d{2}))?/);
      if (!m) return 0;
      const sign = m[1] === "+" ? 1 : -1;
      const h = parseInt(m[2], 10);
      const min = m[3] ? parseInt(m[3], 10) : 0;
      return sign * (h * 60 + min);
    } catch (e) {
      return 0;
    }
  }

  /**
   * NY local 의 (hh:mm) 을 사용자 tz "HH:mm" 으로 변환.
   * DST 자동 처리 (Intl 사용).
   *
   * 사용 예: nyTimeToUserTz(19, 0) → KST "08:00" (EDT), "09:00" (EST)
   */
  function nyTimeToUserTz(hh, mm, userTz) {
    const tz = userTz || getTz();
    const now = new Date();
    // 1) 오늘 NY local 의 연/월/일 추출.
    const nyDate = new Intl.DateTimeFormat("en-CA", {
      timeZone: "America/New_York",
      year: "numeric", month: "2-digit", day: "2-digit",
    }).format(now); // "2026-05-30"
    const [y, mo, d] = nyDate.split("-").map(Number);
    // 2) NY tz 의 현재 offset (분).
    const nyOffsetMin = _getTzOffsetMinutes("America/New_York", now);
    // 3) NY local (y,mo,d,hh,mm) 을 UTC ms 로:
    //    Date.UTC 로 만든 ms 는 그 시각을 UTC 로 간주한 값이므로, NY offset 만큼 빼야
    //    실제 UTC.
    const utcMs = Date.UTC(y, mo - 1, d, hh, mm) - nyOffsetMin * 60_000;
    // 4) 그 UTC ms 를 userTz 의 "HH:mm" 으로:
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: tz,
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(new Date(utcMs));
  }

  // ============================================================
  // 언어 / 시간대 select 만들기 (사이드바 어디에든 마운트 가능)
  // ============================================================

  function makeLangSelect() {
    const sel = document.createElement("select");
    sel.id = "i18n-lang-select";
    sel.className = "i18n-select";
    [
      ["ko", "한국어"],
      ["en", "English"],
      ["zh", "中文 (简体)"],
      ["ja", "日本語"],
    ].forEach(([code, label]) => {
      const opt = document.createElement("option");
      opt.value = code; opt.textContent = label;
      sel.appendChild(opt);
    });
    sel.value = getLang();
    sel.addEventListener("change", () => {
      setLang(sel.value);
      applyI18nToDOM();
      // killzone 라벨도 즉시 갱신.
      window.dispatchEvent(new CustomEvent("aurora-i18n-changed"));
    });
    return sel;
  }

  function makeTzSelect() {
    const sel = document.createElement("select");
    sel.id = "i18n-tz-select";
    sel.className = "i18n-select";
    Object.entries(SUPPORTED_TZS).forEach(([tz, label]) => {
      const opt = document.createElement("option");
      opt.value = tz; opt.textContent = label;
      sel.appendChild(opt);
    });
    sel.value = getTz();
    sel.addEventListener("change", () => {
      setTz(sel.value);
      // killzone 시간 즉시 갱신.
      window.dispatchEvent(new CustomEvent("aurora-tz-changed"));
    });
    return sel;
  }

  // ============================================================
  // export to window
  // ============================================================
  window.AuroraI18n = {
    t,
    applyI18nToDOM,
    getLang, setLang,
    getTz, setTz,
    nyTimeToUserTz,
    makeLangSelect, makeTzSelect,
    SUPPORTED_LANGS,
    SUPPORTED_TZS,
  };

  // DOMContentLoaded 후 자동 1회 적용.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyI18nToDOM);
  } else {
    applyI18nToDOM();
  }
})();
