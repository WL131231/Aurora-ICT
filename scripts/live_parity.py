"""라이브 정합 백테 계층 — 모든 Origo 연구는 **이 모듈을 통해** 백테를 돌린다.

## 왜 만들었나 (2026-07-30 전면 감사 결과)

라이브 강제설정 19개 중 **10개가 공용 하니스(replay.py)에 미구현**이었다.
각 기능은 개별 연구 스크립트에서 사후 필터로 검증하고 배포했으나 공용 하니스에
통합되지 않아, **다음 연구는 그 기능이 없는 상태**로 돌아갔다.
배포는 누적되는데 하니스는 따라가지 않아 정합 오류가 줄줄이 발생했다.

## 2026-08-08 전수 감사 — 규칙의 적용 범위가 좁았다

라이브 매매기록 실측(Origo 진입 56건)으로 **백테와 라이브가 다른 봇**임이 드러났다.
7/30 규칙은 "replay 옵션으로 표현되는 것"만 감사했고, 봇이 셋업 생성 **후** 점수를
고치는 경로(_apply_htf_supporting_boost 등)와 셋업 **생성 경로 자체**(Phase B
4소스, prefer_direction)는 감사 대상이 아니었다.

전수 감사 결과(data/parity/audit.json): 45경로 중 missing 28 · different 12 ·
present 5. 대표적으로

  · Phase B 4소스(turtle/mitigation/implied/rejection) — 라이브 진입의 **68%**
    가 백테가 생성조차 못 하는 셋업이었다.
  · _apply_htf_supporting_boost — 라이브 진입의 **93%** 가 +1~+3 을 무상으로
    받았다 → 실효 문턱이 5가 아니라 2~4였다.
  · _apply_dol_bias — 라이브는 역방향 **-2 페널티**, 백테는 정방향 **+1 보너스**.
  · align 점수 계산식 — 라이브는 670봉 SMA시드+미완성봉, 백테는 전체 ewm+shift.
  · 트레일 무장 시 분할익절 무효 — 라이브는 항상 비활성, 백테는 동시 적용.

이 모듈의 LIVE_BASE 는 그 감사 결과를 반영해 갱신됐다.

## 규칙 (프로세스)

1. **라이브에 무언가를 배포하면 이 모듈도 같은 PR 에서 갱신한다.** 갱신 없는 배포 금지.
   범위는 "replay 옵션"이 아니라 **봇 step() 이 진입/청산에 이르는 모든 경로**다.
2. 모든 Origo 연구는 `run_live_parity()` 로 기준선을 잡는다. `BASE` 직접 사용 금지.
3. 결과 보고 시 `parity_report()` 를 함께 출력해 라이브 실측과 대조한다.
4. 여기서 **미구현으로 표시된 항목(GAPS)은 결론에 한계로 명시**한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

# ── 라이브 강제값 (config/settings.py `_enforce_license_tier_policy` sub_ 분기 실측
#    + bot/multi_user_manager.py BotIctInstance 생성 인자 실측) ──
#
# ⚠️ 2026-08-08 감사 이후 추가/수정된 것에는 [08-08] 표시. 각 항목 옆에 라이브 출처.
LIVE_BASE: dict = dict(
    htf_ema_bias="align",       # htf_ema_bias_enabled=True + htf_ema_align_enabled=True
    htf_align_threshold=2,      # settings.htf_ema_align_threshold 기본 2
    align_live_formula=True,    # [08-08] #3 라이브 EMA 계산식(670봉 SMA시드·미완성봉)
    prefer_direction_select=True,  # [08-08] #2 방향 일치 셋업 중 최신 선택
    phase_b_sources=True,       # [08-08] #1 Phase B 4소스 (라이브 진입의 68%)
    sl_liq_cap=True,            # #LIQ-CAP
    min_confluence=5,           # sub_ 강제 (>=5)
    min_rr=2.0,                 # sub_ 강제 (>=2.0)
    sl_dist_mult=4.0,           # sub_ 강제 (>=4.0), #ORIGO-1.3
    sl_dist_mult_ct=4.0,        # #CT-SL 역추세 x4
    ct_trend_threshold=0.0,
    setup_stale_bars=3,         # #FRESH-30 (5m×3=15분) — settings 559-562
    entry_ttl_bars=6,           # #ORIGO-1 ttl 1800초 = 5m×6봉 (BTC 는 아래 override)
    leverage=7,                 # [08-08] #LEV-7 (settings 545-548). 없어서 cfg 기본
                                #   20 이 쓰였고 net%·MDD 가 ~2.9배 부풀려 출력됐다.
    apply_cisd=True,            # _apply_cisd_boost
    apply_po3=True,             # _apply_po3_boost
    apply_ote=True,             # [08-08] _apply_ote_boost — replay 에 구현은 있었는데
                                #   LIVE_BASE 에 키가 없어 OFF 였다.
    apply_dol_counter=True,     # [08-08] #8 _apply_dol_bias 는 **-2 페널티**
    dol_counter_penalty=2,      #   bot_ict_instance.py:280 _DOL_COUNTER_PENALTY
    apply_dol=False,            #   기존 +1 보너스 경로는 반드시 꺼 둘 것(부호 반대)
    htf_fvg_support=True,       # [08-08] #5 _apply_htf_supporting_boost (+1~+3)
    htf_ltf_conflict_guard_ratio=1.10,  # [08-08] #HTF-LTF-CONFLICT (settings 기본)
    htf_fvg_flip=True,          # [08-08] #7 htf_override_mode="C" flip 청산
    flip_min_r=1.5,             # #FLIP-MIN-R (Origo 2.3, sub_ 강제 >=1.5)
    disable_time_filter=False,  # sub_ 강제 False (킬존 필터 on)
    entry_killzone_gate=True,   # [08-08] #9 #KZ-ENTRY 진입 '시점' 킬존 재확인
    exclude_nypm=True,          # [08-08] #NYPM-GATE — NY local 13:30-16:00 (DST 반영)
    max_entry_distance_pct=0.005,  # [08-08] settings.max_entry_distance_pct 기본
    sweep_gate_days=2,          # [08-08] #SWEEP-GATE sub_ 강제 2일
    size_pct=0.9,
    ote_level=0.707,            # 기본 OTE
    ote_up_level=0.786,         # [08-08] #REGIME-OTE 상승 국면 (sub_ 강제 0.786)
    tp_rr_override=0.0,
    trail_trigger=2.0,          # #TRAIL-EXCHANGE (>=2.0R)
    trail_dist=1.5,             # (>=1.5R)
    be_trigger=1.0,             # #BE-LOCK (>=1.0R)
    partial_tp_rr=1.5,
    partial_be=True,
    trail_supersedes_partial=True,  # [08-08] #6 트레일 무장 시 분할익절 **항상 비활성**
    regime_filter=True,         # [08-08] #REGIME 를 재생 루프 안으로 (사후 필터 X)
    cond_align=True,            # [08-08] #COND-ALIGN 동일
    regime_rolling=True,        # [08-08] 라이브 롤링 33/70분위(표본>=20, 최근 150)
)

# #FLIP-MIN-R (Origo 2.3, 2026-07-30 배포) — flip 발동 시 이익이 이 R 미만이면
# flip 무시하고 원래 청산까지 홀드. 이제 replay 가 flip 을 직접 시뮬하므로
# LIVE_BASE["flip_min_r"] 가 정본이다. 이 상수는 구 스크립트 호환용.
LIVE_FLIP_MIN_R: float = 1.5

# 페어별 진입대기 ttl(5m 봉). #ORIGO-1 은 1800초(6봉)인데 BTC 만 manager 가
# 3600초로 override 한다(config/settings.py PAIR_TTL_OVERRIDES →
# origo1_ttl_for_symbol, multi_user_manager.py:467). [08-08] 추가.
PAIR_TTL_BARS: dict[str, int] = {"BTCUSDT": 12}

# 페어별 분위 (bot_ict_instance.py:480/486 ClassVar 실측 — 롤링 분위의 fallback).
REGIME_Q33 = {"BTCUSDT": 0.230, "ETHUSDT": 0.268, "SOLUSDT": 0.396, "XRPUSDT": 0.271,
              "DOGEUSDT": 0.275, "LINKUSDT": 0.315, "HYPEUSDT": 0.527}
STRONG_Q70 = {"BTCUSDT": 0.401, "ETHUSDT": 0.365, "SOLUSDT": 0.598, "XRPUSDT": 0.381,
              "DOGEUSDT": 0.721, "LINKUSDT": 1.036, "HYPEUSDT": 0.924}

# 아직 이식 못 한 것 — **근사하지 않고 그대로 남긴다**. 결론에 반드시 명시할 한계.
# (2026-08-08 전수 감사 기준. 이식된 항목은 목록에서 제거됨.)
GAPS: dict[str, str] = {
    "미완성 봉(60초 폴링)": "미구현 — 라이브는 5m 봉 하나를 5번 평가하고 매번 "
        "'형성 중'인 봉을 마지막 봉으로 쓴다(_fetch_ohlcv_tf 가 마지막 봉을 안 버림). "
        "셋업 검출·CISD·PO3·추세·DOL·OTE 입력이 전부 다르다. 봉 단위 엔진으로는 "
        "구조적으로 재현 불가 — 1m 재생 엔진이 따로 필요하다. "
        "HTF FVG map 의 '미완성 TF 봉이 FVG 3번째 봉/체결 판정에 쓰이는 경우'도 "
        "이 GAP 의 부분집합이라 닫힌 봉만 쓴다.",
    "MMBM 2번째 모델": "미구현 — settings 555 가 sub_ 에서 origo_mmbm_enabled=True 를 "
        "강제하고 multi_user_manager.py:472 가 실제로 연결한다(#MMBM-WIRE 2026-08-06). "
        "SB 셋업이 없는 봉에서 별도 모델이 돌며 confluence 게이트를 전면 우회한다. "
        "라이브 56건 표본 밖이라 빈도 미상 — 별도 이식 과제.",
    "risk_based_sizing / smart_size": "미구현 — 라이브는 건당 리스크가 score 연동 "
        "3~6%, 여기에 품질 배수(0.4~1.4)까지 곱한다. 백테는 size_pct 고정. "
        "거래 선별에는 영향 0(빈도·타점 불변), **복리 자산곡선에만** 영향.",
    "margin guard / min_entry_qty_ratio 0.2": "미구현 — 가용잔고 기반. 라이브 실측 "
        "진입의 5.3% 가 여기서 차단된다. 잔고 상태 시뮬이 필요.",
    "daily_loss_limit 15% / daily_profit_limit / dd_throttle 25%": "미구현 — 계좌 "
        "단위 서킷·스로틀. 페어별 독립 백테로는 표현 불가, 포트폴리오 복리 시뮬"
        "(scripts/origo_leverage_verify.py)에서만 반영된다.",
    "daily_pair_loss_limit_r 2.0": "미구현 — 기본값 2.0 이라 **항상 활성**인데 기존 "
        "GAPS 목록에도 없던 누락. 페어당 하루 손실이 -2R 로 캡되므로 백테 낙폭은 "
        "과대추정이다.",
    "detect 입력 길이 500 vs 라이브 1000": "미이식 — 라이브 settings.ohlcv_limit=1000 "
        "이라 generate_ict_signal 이 매번 1000봉을 본다. 백테 cfg.window 는 500. "
        "swing/OB/DOL/turtle 의 lookback 모집단이 달라 셋업 자체가 달라진다. "
        "고치는 건 한 줄(window=1000)이지만 detect 비용 2배 + 모든 timeline 캐시 "
        "무효 + 과거 연구와 비교 불가라 파트너 결정 대기. 전수 감사 45항목에도 "
        "빠져 있던 항목이다.",
    "bias 주입(HTF+daily)": "미구현 — 라이브는 _combine_with_daily 결과를 "
        "generate_ict_signal 의 bias 로 넘긴다. #BIAS-DIRECTION 이후 방향 강제엔 "
        "안 쓰이고 confluence 'bias=' +1 에만 쓰이지만, 백테는 bias=None(구조 자동 "
        "추정)이라 그 +1 이 붙는 봉이 다르다. 라이브 56건 중 bias 는 20%.",
    "포지션 상태 경로": "미구현 — #MANUAL-POS-RESPECT / #REENTRY-BLOCK(4h 쿨다운) / "
        "_recovery_failed / _ensure_protective_sl / 최소수량 skip. 전부 거래소·사용자 "
        "상호작용 의존이라 백테에 대응물이 없다.",
    "flip zone 재조회(진입 후 생성분)": "부분이식 — [08-09] replay 는 flip target 을 "
        "**진입 시점에 한 번 고정**한다(replay.py:877-883). 라이브 _maybe_flip 은 매 "
        "봉 HTF FVG map 을 다시 만들어 **진입 후 새로 생긴 zone 으로도** flip 한다. "
        "그래서 백테 flip 청산 건수가 과소추정된다(BASE 402건 중 22건). 건당R 은 "
        "flip_min_r 되돌림 실측에서 불변이었으므로(-0.067 vs -0.069) ΔR·순열 p 에는 "
        "영향이 없고, **승률·RR·복리·MDD·파산만** 이 GAP 의 영향권이다.",
}


def _dir_sign(t) -> float:
    v = getattr(t.direction, "value", t.direction)
    return 1.0 if str(v).lower() in ("long", "buy") else -1.0


def gate_regime(t, sym: str) -> bool:
    """#REGIME (6/23) — |entry_trend_pct| < 페어 q33 이면 진입 차단. True=통과.

    ⚠️ 2026-08-08 이후 이 게이트는 replay 재생 루프 안(cfg.regime_filter)에서
    돌아간다. 이 함수는 **구 스크립트 호환용**이며 run_live_parity 는 쓰지 않는다.
    사후 필터로 걸면 라이브가 거르는 셋업이 백테에선 포지션 슬롯을 먹어 뒤 셋업을
    가리는 인공물이 생긴다(그게 이 함수의 원래 문제였다).
    """
    floor = REGIME_Q33.get(sym, 0.0)
    if floor <= 0:
        return True
    return abs(float(t.entry_trend_pct or 0.0)) >= floor


def gate_cond_align(t, sym: str) -> bool:
    """#COND-ALIGN (7/17) — 약/중추세에서 역추세 진입만 차단. True=통과.

    ⚠️ gate_regime 과 같은 이유로 **구 스크립트 호환용**. 현행 경로는
    cfg.cond_align (replay 루프 안).
    """
    strong = STRONG_Q70.get(sym, 0.0)
    if strong <= 0:
        return True
    tr = float(t.entry_trend_pct or 0.0)
    signed = tr * _dir_sign(t)
    return not (abs(tr) < strong and signed < 0)


def gate_nypm(ts: pd.Timestamp) -> bool:
    """#NYPM-GATE (7/16) — NY_PM 진입 차단. True=통과.

    ⚠️ **구 스크립트 호환용**. 고정 UTC 17-21 근사라 서머타임 구간에서 라이브
    (NY local 13:30-16:00)와 최대 1.5시간 어긋난다. 현행 경로는 cfg.exclude_nypm
    (replay 가 zoneinfo 로 NY local 판정).
    """
    return not (17 <= ts.hour < 21)


def live_cfg(sym: str, extra: dict | None = None) -> BacktestConfig:
    """페어별 라이브 정합 cfg — LIVE_BASE + 페어별 강제값(ttl·분위 fallback).

    Args:
        sym: 심볼 ("BTCUSDT").
        extra: 덮어쓸 필드 (연구 변형용).
    Returns:
        BacktestConfig.
    """
    kw = dict(LIVE_BASE)
    kw["entry_ttl_bars"] = PAIR_TTL_BARS.get(sym, LIVE_BASE["entry_ttl_bars"])
    kw["regime_floor"] = REGIME_Q33.get(sym, 0.0)
    kw["strong_trend_floor"] = STRONG_Q70.get(sym, 0.0)
    if extra:
        kw.update(extra)
    return BacktestConfig(**kw)


def load_funding(sym: str):
    """펀딩 요율 로드 — 없으면 None (기존 동작 유지).

    #FUNDING 2026-08-09: 백테가 수수료·슬리피지는 반영하면서 펀딩은 아예 없었다.
    SB 보유는 중앙 13.3시간·평균 30.4시간이라 **62% 가 8시간 정산 주기를 넘고**
    거래당 평균 3.8회 정산을 겪는다. 실측 건당 +0.0096R(BTC+ETH)·+0.0037R(알트5).
    작지만 롱만 보면 +0.019R 이라 **롱숏 비교를 왜곡**하므로 기록해 둔다.
    """
    import os

    import pandas as _pd

    # _load_full 과 같은 규약 — 저장소 루트에서 실행하는 것을 전제로 한 상대 경로.
    f = f"data/{sym}_funding.parquet"
    if not os.path.exists(f):
        return None
    d = _pd.read_parquet(f)
    if "rate" not in d.columns or not isinstance(d.index, _pd.DatetimeIndex):
        return None
    return d[["rate"]].sort_index()


def run_live_parity(sym: str, extra: dict | None = None, *, funding: bool = True):
    """라이브 정합 백테 1페어. 반환: (df5, trade 목록, 게이트 통계).

    2026-08-08 이후 모든 게이트(regime·cond_align·nypm·killzone·sweep·충돌)가
    replay 재생 루프 **안**에서 라이브 순서대로 돌아간다. 사후 필터 없음.

    Args:
        sym: 심볼.
        extra: cfg 덮어쓰기 (연구 변형).
        funding: True 면 펀딩 정산을 net 에 반영하고 Trade.funding_pct 에 기록한다.
            raw_pnl_pct 는 불변이므로 raw 기반 R 결과는 바뀌지 않는다.
    Returns:
        (df5, trades, stats). stats 는 구 호출부 호환용 dict.
    """
    df5 = _resample(_load_full(sym))
    cfg = live_cfg(sym, extra)
    tl = cached_setup_timeline(df5, cfg, sym)
    # 펀딩은 net_pnl_pct 와 Trade.funding_pct 에만 반영된다. raw_pnl_pct 는 불변이라
    # raw 기반 R 을 쓰는 기존 연구 결과와 직접 비교가 깨지지 않는다.
    bt = run_backtest_from_timeline(
        df5, tl, cfg, funding=load_funding(sym) if funding else None,
    )
    stats = {"전체": len(bt.trades), "통과": len(bt.trades),
             "regime": 0, "cond_align": 0, "nypm": 0,
             "비고": "게이트는 replay 루프 내부에서 적용 — 사후 카운트 없음"}
    return df5, list(bt.trades), stats


def stat(pairs_trades: list[tuple[pd.Timestamp, float, str]]):
    """(ts, net_pct, sym) 리스트 → 집계."""
    if not pairs_trades:
        return None
    tr = sorted(pairs_trades)
    nets = [p for _, p, _ in tr]
    net = sum(nets)
    w = [p for p in nets if p > 0]
    l = [p for p in nets if p < 0]
    eq = pk = mdd = 0.0
    for p in nets:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for ts, p, _ in tr:
        ys[ts.year] = ys.get(ts.year, 0.0) + p
    sy: dict[str, float] = {}
    for _, p, s in tr:
        sy[s] = sy.get(s, 0.0) + p
    half = len(nets) // 2
    return dict(
        n=len(tr), net=net * 100, wr=100 * len(w) / len(tr),
        rr=(np.mean(w) / abs(np.mean(l))) if w and l else float("nan"),
        avgw=(np.mean(w) * 100 if w else 0.0), avgl=(np.mean(l) * 100 if l else 0.0),
        mdd=mdd * 100, h1=sum(nets[:half]) * 100, h2=sum(nets[half:]) * 100,
        ypos=sum(1 for v in ys.values() if v > 0), ytot=len(ys),
        spos=sum(1 for v in sy.values() if v > 0), stot=len(sy),
        ys=" ".join(f"{k}:{v * 100:+.1f}" for k, v in sorted(ys.items())),
        sy=" ".join(f"{k.replace('USDT', '')}:{v * 100:+.1f}" for k, v in sy.items()),
    )


def line(s) -> str:
    if s is None:
        return "거래 없음"
    return (f"n={s['n']:4d} net={s['net']:+8.2f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
            f"익={s['avgw']:+5.2f} 손={s['avgl']:+5.2f} MDD={s['mdd']:6.2f} "
            f"H1={s['h1']:+7.2f} H2={s['h2']:+7.2f} 연도{s['ypos']}/{s['ytot']} "
            f"페어{s['spos']}/{s['stot']}")


def parity_report(s, live: dict | None = None) -> None:
    """라이브 실측과 대조 출력. live 예: {"wr":46,"rr":0.94}"""
    print("\n  [라이브 정합 대조]", flush=True)
    if s:
        print(f"    백테 승률 {s['wr']:.0f}% / RR {s['rr']:.2f}", flush=True)
    if live:
        print(f"    라이브 승률 {live.get('wr', '?')}% / RR {live.get('rr', '?')}"
              "  (2026-05-28~07-30 Origo 2.2, 청산 100건)", flush=True)
    print("\n  [미구현 한계 — 결론에 명시할 것]", flush=True)
    for k, v in GAPS.items():
        print(f"    · {k}: {v}", flush=True)


__all__ = ["LIVE_BASE", "PAIRS", "GAPS", "REGIME_Q33", "STRONG_Q70", "PAIR_TTL_BARS",
           "gate_regime", "gate_cond_align", "gate_nypm", "live_cfg",
           "run_live_parity", "stat", "line", "parity_report"]
