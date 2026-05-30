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

  // 2026-05-30 파트너 요청: 언어 변경 시 해당 국가 시간대 자동 설정.
  // 사용자가 별도로 시간대 select 로 override 가능.
  const LANG_TO_TZ = {
    "ko": "Asia/Seoul",
    "en": "America/New_York",
    "zh": "Asia/Shanghai",
    "ja": "Asia/Tokyo",
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
      "btn.delete": "삭제",
      "confirm.deleteKey": "{mode} API 키를 삭제할까요?",
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
      // 라이선스 카드
      "license.title": "LICENSE",
      "license.type": "Type",
      "license.period": "Period",
      "license.daysLeft": "남음",
      "license.unlimited": "무기한",
      "license.referral": "레퍼럴",
      "license.sub_30d": "구독 30일",
      "license.sub_90d": "구독 90일",
      "license.sub_365d": "구독 365일",
      // 매매 기록 페이지 (trades.html)
      "trades.title": "매매 기록",
      "trades.colTime": "시각",
      "trades.colType": "유형",
      "trades.colMode": "모드",
      "trades.colDirection": "방향",
      "trades.colSymbol": "심볼",
      "trades.colPrice": "가격",
      "trades.colQty": "수량",
      "trades.colPnl": "PnL (USDT)",
      "trades.colReason": "사유",
      "trades.loading": "로딩 중…",
      "trades.empty": "데이터 없음",
      "trades.range": "기간",
      "trades.all": "전체",
      "trades.refresh": "새로고침",
      "trades.csv": "CSV 다운로드",
      "trades.sectionEvent": "이벤트",
      "trades.sectionEntryReason": "진입 사유 (당시 봇 판단)",
      "trades.sectionCloseJudgment": "청산 시점 봇 판단 스냅샷",
      "trades.sectionCloseDiagnostics": "청산 시점 봇 진단 카운터",
      "trades.sectionConfluences": "Confluences (작동한 지표)",
      // 봇 판단 패널
      "judgment.title": "봇의 현재 판단",
      "judgment.reason": "Reason",
      "judgment.entryCondition": "Entry Condition",
      // 세션 / 기타
      "session.title": "Session",
      "session.logout": "로그아웃",
      "equity.realtimeNote": "봇 시작 후 실시간 표시",
      "positions.empty": "포지션 없음 — 봇 가동 후 진입 시 표시됩니다",
      "positions.pending": "대기 중",
      "td.entryPrice": "진입가",
      "td.sl": "SL",
      "td.tp": "TP",
      "td.qty": "수량",
      "td.leverage": "레버리지",
      "td.rr": "RR",
      "td.confluenceScore": "Confluence Score",
      "td.source": "Source",
      "td.window": "Window",
      "td.closePrice": "청산가",
      "td.movePct": "가격 변화 (%)",
      "td.realizedPnl": "실현 PnL",
      "td.closeReason": "청산 사유",
      "td.killzone": "Killzone",
      "td.dolDraw": "DOL Draw",
      "td.htfBullWeight": "HTF FVG Bull Weight",
      "td.htfBearWeight": "HTF FVG Bear Weight",
      "td.htfFvgCount": "HTF FVG 개수",
      "judgment.directionUncertain": "방향 불확실 (혼조)",
      "judgment.directionLong": "롱 우세",
      "judgment.directionShort": "숏 우세",
      "judgment.recentEntries": "최근 진입 분포",
      "judgment.entryWaitGrade": "등급 3 이상 & RR 2.0 이상 setup 대기",
      "judgment.entryNoSetup": "현재 활성 setup 없음 (HTF 게이트 미달로 skip)",
      "judgment.accumulating": "분석 자료 누적 중…",
      "judgment.interpretation": "해석",
      "judgment.long": "롱",
      "judgment.short": "숏",
      "judgment.recentLast": "최근",
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
      "btn.delete": "Delete",
      "confirm.deleteKey": "Delete {mode} API key?",
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
      "license.title": "LICENSE",
      "license.type": "Type",
      "license.period": "Period",
      "license.daysLeft": "left",
      "license.unlimited": "Unlimited",
      "license.referral": "Referral",
      "license.sub_30d": "30-Day Subscription",
      "license.sub_90d": "90-Day Subscription",
      "license.sub_365d": "1-Year Subscription",
      "trades.title": "Trade History",
      "trades.colTime": "Time",
      "trades.colType": "Type",
      "trades.colMode": "Mode",
      "trades.colDirection": "Side",
      "trades.colSymbol": "Symbol",
      "trades.colPrice": "Price",
      "trades.colQty": "Qty",
      "trades.colPnl": "PnL (USDT)",
      "trades.colReason": "Reason",
      "trades.loading": "Loading…",
      "trades.empty": "No data",
      "trades.range": "Range",
      "trades.all": "All",
      "trades.refresh": "Refresh",
      "trades.csv": "Download CSV",
      "trades.sectionEvent": "Event",
      "trades.sectionEntryReason": "Entry Reason (Bot Judgment)",
      "trades.sectionCloseJudgment": "Bot Judgment Snapshot at Close",
      "trades.sectionCloseDiagnostics": "Bot Diagnostic Counters at Close",
      "trades.sectionConfluences": "Confluences (Active Signals)",
      "judgment.title": "Bot Judgment",
      "judgment.reason": "Reason",
      "judgment.entryCondition": "Entry Condition",
      "session.title": "Session",
      "session.logout": "Logout",
      "equity.realtimeNote": "Live after bot start",
      "positions.empty": "No positions — shown after bot starts an entry",
      "positions.pending": "Pending",
      "td.entryPrice": "Entry",
      "td.sl": "SL",
      "td.tp": "TP",
      "td.qty": "Qty",
      "td.leverage": "Leverage",
      "td.rr": "RR",
      "td.confluenceScore": "Confluence Score",
      "td.source": "Source",
      "td.window": "Window",
      "td.closePrice": "Close Price",
      "td.movePct": "Move (%)",
      "td.realizedPnl": "Realized PnL",
      "td.closeReason": "Close Reason",
      "td.killzone": "Killzone",
      "td.dolDraw": "DOL Draw",
      "td.htfBullWeight": "HTF FVG Bull Weight",
      "td.htfBearWeight": "HTF FVG Bear Weight",
      "td.htfFvgCount": "HTF FVG Count",
      "judgment.directionUncertain": "Direction Uncertain (mixed)",
      "judgment.directionLong": "Long Bias",
      "judgment.directionShort": "Short Bias",
      "judgment.recentEntries": "Recent Entries",
      "judgment.entryWaitGrade": "Wait for setup ≥ Grade 3 & RR ≥ 2.0",
      "judgment.entryNoSetup": "No active setup (HTF gate skip)",
      "judgment.accumulating": "Analyzing…",
      "judgment.interpretation": "Reading",
      "judgment.long": "Long",
      "judgment.short": "Short",
      "judgment.recentLast": "last",
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
      "license.title": "授权",
      "license.type": "类型",
      "license.period": "有效期",
      "license.daysLeft": "剩余",
      "license.unlimited": "无限期",
      "license.referral": "推荐",
      "license.sub_30d": "30天订阅",
      "license.sub_90d": "90天订阅",
      "license.sub_365d": "365天订阅",
      "trades.title": "交易记录",
      "trades.colTime": "时间",
      "trades.colType": "类型",
      "trades.colMode": "模式",
      "trades.colDirection": "方向",
      "trades.colSymbol": "交易对",
      "trades.colPrice": "价格",
      "trades.colQty": "数量",
      "trades.colPnl": "盈亏 (USDT)",
      "trades.colReason": "原因",
      "trades.loading": "加载中…",
      "trades.empty": "无数据",
      "trades.range": "范围",
      "trades.all": "全部",
      "trades.refresh": "刷新",
      "trades.csv": "下载CSV",
      "trades.sectionEvent": "事件",
      "trades.sectionEntryReason": "进场原因 (当时机器人判断)",
      "trades.sectionCloseJudgment": "平仓时机器人判断快照",
      "trades.sectionCloseDiagnostics": "平仓时机器人诊断计数",
      "trades.sectionConfluences": "Confluences (生效指标)",
      "judgment.title": "机器人当前判断",
      "judgment.reason": "原因",
      "judgment.entryCondition": "入场条件",
      "session.title": "会话",
      "session.logout": "退出",
      "equity.realtimeNote": "机器人启动后实时显示",
      "positions.empty": "无仓位 — 机器人启动并进场后显示",
      "positions.pending": "等待中",
      "td.entryPrice": "入场价",
      "td.sl": "止损",
      "td.tp": "止盈",
      "td.qty": "数量",
      "td.leverage": "杠杆",
      "td.rr": "RR",
      "td.confluenceScore": "Confluence Score",
      "td.source": "来源",
      "td.window": "窗口",
      "td.closePrice": "平仓价",
      "td.movePct": "价格变动 (%)",
      "td.realizedPnl": "已实现盈亏",
      "td.closeReason": "平仓原因",
      "td.killzone": "Killzone",
      "td.dolDraw": "DOL Draw",
      "td.htfBullWeight": "HTF FVG 多方权重",
      "td.htfBearWeight": "HTF FVG 空方权重",
      "td.htfFvgCount": "HTF FVG 数量",
      "judgment.directionUncertain": "方向不明 (混合)",
      "judgment.directionLong": "看多",
      "judgment.directionShort": "看空",
      "judgment.recentEntries": "最近进场分布",
      "judgment.entryWaitGrade": "等级 ≥ 3 & RR ≥ 2.0 setup 等待",
      "judgment.entryNoSetup": "无活跃setup (HTF门控跳过)",
      "judgment.accumulating": "分析中…",
      "judgment.interpretation": "解读",
      "judgment.long": "多",
      "judgment.short": "空",
      "judgment.recentLast": "最近",
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
      "license.title": "ライセンス",
      "license.type": "タイプ",
      "license.period": "有効期間",
      "license.daysLeft": "残り",
      "license.unlimited": "無期限",
      "license.referral": "リファラル",
      "license.sub_30d": "30日サブスク",
      "license.sub_90d": "90日サブスク",
      "license.sub_365d": "365日サブスク",
      "trades.title": "取引履歴",
      "trades.colTime": "時刻",
      "trades.colType": "種別",
      "trades.colMode": "モード",
      "trades.colDirection": "方向",
      "trades.colSymbol": "シンボル",
      "trades.colPrice": "価格",
      "trades.colQty": "数量",
      "trades.colPnl": "損益 (USDT)",
      "trades.colReason": "理由",
      "trades.loading": "読み込み中…",
      "trades.empty": "データなし",
      "trades.range": "期間",
      "trades.all": "すべて",
      "trades.refresh": "更新",
      "trades.csv": "CSVダウンロード",
      "trades.sectionEvent": "イベント",
      "trades.sectionEntryReason": "エントリー理由 (当時のボット判断)",
      "trades.sectionCloseJudgment": "決済時点のボット判断スナップショット",
      "trades.sectionCloseDiagnostics": "決済時点のボット診断カウンター",
      "trades.sectionConfluences": "Confluences (有効指標)",
      "judgment.title": "ボットの現在の判断",
      "judgment.reason": "理由",
      "judgment.entryCondition": "エントリー条件",
      "session.title": "セッション",
      "session.logout": "ログアウト",
      "equity.realtimeNote": "ボット起動後にリアルタイム表示",
      "positions.empty": "ポジションなし — ボット起動後エントリー時に表示",
      "positions.pending": "待機中",
      "td.entryPrice": "エントリー価格",
      "td.sl": "SL",
      "td.tp": "TP",
      "td.qty": "数量",
      "td.leverage": "レバレッジ",
      "td.rr": "RR",
      "td.confluenceScore": "Confluence Score",
      "td.source": "ソース",
      "td.window": "ウィンドウ",
      "td.closePrice": "決済価格",
      "td.movePct": "価格変動 (%)",
      "td.realizedPnl": "実現損益",
      "td.closeReason": "決済理由",
      "td.killzone": "Killzone",
      "td.dolDraw": "DOL Draw",
      "td.htfBullWeight": "HTF FVG ロング重み",
      "td.htfBearWeight": "HTF FVG ショート重み",
      "td.htfFvgCount": "HTF FVG 数",
      "judgment.directionUncertain": "方向不確実 (混在)",
      "judgment.directionLong": "ロング優勢",
      "judgment.directionShort": "ショート優勢",
      "judgment.recentEntries": "最近のエントリー分布",
      "judgment.entryWaitGrade": "等級 3 以上 & RR 2.0 以上 setup 待ち",
      "judgment.entryNoSetup": "アクティブsetupなし (HTFゲートskip)",
      "judgment.accumulating": "分析中…",
      "judgment.interpretation": "解釈",
      "judgment.long": "ロング",
      "judgment.short": "ショート",
      "judgment.recentLast": "最近",
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
      // 2026-05-30 파트너 요청: 언어 변경 시 해당 국가 시간대 자동 설정.
      const autoTz = LANG_TO_TZ[sel.value];
      if (autoTz) {
        setTz(autoTz);
        // tz select element 도 동기화.
        const tzSel = document.getElementById("i18n-tz-select");
        if (tzSel) tzSel.value = autoTz;
        window.dispatchEvent(new CustomEvent("aurora-tz-changed"));
      }
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
