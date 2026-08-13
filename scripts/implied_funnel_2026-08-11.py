"""#IMPLIED 2026-08-11 (1단계): implied_fvg 가 어디서 걸러지나 — 깔때기 분석.

## 왜 이 소스인가
소스별 분리 결과 **성적 1위**다:
    본표본 BTC+ETH   31건 +0.508R [+0.002 ~ +1.039] ★0초과 · 심볼 2/2
    홀드아웃 알트5    72건 +0.319R [-0.003 ~ +0.650] · **심볼 4/4** · 롱숏 양쪽 +0.3
일관성이 모든 소스 중 가장 좋다(홀드아웃 심볼 4/4). 문제는 **빈도**로,
전체 진입의 3% 뿐이라 표본이 얇아 구간이 0 을 걸친다.

## 그런데 늘리기 전에
오늘 빈도 확대를 네 번 시도해 **전부 기각**됐다(킬존·창당1회·FVG재사용·대기창).
전부 "열면 나쁜 진입이 들어온다"였다. 그래서 이번에는 **어디서 걸러지는지 먼저**
세고, 그 관문을 푸는 게 이 소스에 한해 말이 되는지 본다. 무작정 완화하지 않는다.

## 세는 단계 (라이브 순서 그대로)
    ① 검출          detect_implied_fvgs
    ② 시간 필터      미장 안의 킬존/매크로/SB (또는 킬존 전면)
    ③ 최근 3개 컷    다른 소스와 같은 규약 — `[-3:]` 로 잘린다
    ④ 목표 스윙 없음  _find_target_swing 실패
    ⑤ min_rr 2.0    손익비 미달
    ⑥ 진입 후보     여기까지 통과한 것
그 뒤 replay 단계(stale 3봉 · confluence 5 · 체결)는 별도로 센다.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.indicators.implied_fvg import detect_implied_fvgs  # noqa: E402
from aurora_ict.indicators.swing_points import detect_swing_points  # noqa: E402
from aurora_ict.strategy.silver_bullet import (  # noqa: E402
    _build_implied_fvg_setup,
    _phase_b_in_time_window,
)

SYM = "BTCUSDT"
BARS = 120_000        # 최근 약 14개월
WINDOW = 500


def main() -> int:
    cfg = live_cfg(SYM)
    df = _resample(_load_full(SYM)).tail(BARS)
    print(f"=== implied_fvg 깔때기 — {SYM} 최근 {len(df):,}봉", flush=True)

    # 라이브는 매 봉 window 를 잘라 보지만, 깔때기 분석은 전 구간 1회로 충분하다
    # (어느 관문이 몇 %를 걸러내는지가 목적).
    swings = detect_swing_points(df)
    ifvgs = detect_implied_fvgs(df)
    n1 = len(ifvgs)
    print(f"  ① 검출                {n1:>7,}건", flush=True)

    # ② 시간 필터 (라이브 = 미장 안, 파트너 지시로 전면 개방도 같이 본다)
    # ImpliedFVG 는 ts_ms 를 안 들고 idx 만 있다 — 인덱스로 시각을 만든다.
    def _ts_at(i: int) -> int:
        v = df.index[min(int(i), len(df) - 1)]
        return int(v.value // 10**6)

    keep_sub, keep_wide = [], []
    for f in ifvgs:
        t = _ts_at(f.idx)
        if _phase_b_in_time_window(t, False, True):
            keep_sub.append(f)
        if _phase_b_in_time_window(t, False, False):
            keep_wide.append(f)
    print(f"  ② 시간 필터 (현행)     {len(keep_sub):>7,}건"
          f"  ({100 * len(keep_sub) / max(n1, 1):.1f}%)", flush=True)
    print(f"     시간 필터 (전면개방) {len(keep_wide):>7,}건"
          f"  ({100 * len(keep_wide) / max(n1, 1):.1f}%)", flush=True)

    # ③ "최근 3개" 컷 — 봉마다 최근 3개만 후보가 되므로, 실제로는 각 봉의 window
    #    안에서 마지막 3개만 산다. 전 구간 기준으로 몇 개가 살아남는지 근사한다.
    #    (라이브 재현이 아니라 관문 크기 파악이 목적)
    print(f"  ③ 최근 3개 컷         봉마다 상위 3개만 — 검출 대비 상시 제한", flush=True)

    # ④⑤ 빌더 통과율 — 목표 스윙 + min_rr
    atr = float(np.nanmean(
        (df["high"] - df["low"]).tail(200).to_numpy(float),
    )) if len(df) > 200 else 0.0
    built = no_target = under_rr = 0
    rr_vals = []
    for f in keep_sub:
        st = _build_implied_fvg_setup(f, df, swings, min_rr=0.0, atr_val=atr)
        if st is None:
            no_target += 1
            continue
        rr_vals.append(float(st.risk_reward))
        if st.risk_reward < cfg.min_rr:
            under_rr += 1
        else:
            built += 1
    print(f"  ④ 목표 스윙 없음       {no_target:>7,}건 탈락", flush=True)
    print(f"  ⑤ min_rr {cfg.min_rr} 미달    {under_rr:>7,}건 탈락", flush=True)
    print(f"  ⑥ 진입 후보           {built:>7,}건"
          f"  ({100 * built / max(n1, 1):.2f}%)", flush=True)

    if rr_vals:
        rv = np.array(rr_vals)
        print(f"\n  손익비 분포 — 중앙 {np.median(rv):.2f} · 평균 {rv.mean():.2f}"
              f" · 2.0 이상 {100 * (rv >= 2.0).mean():.1f}%"
              f" · 1.5 이상 {100 * (rv >= 1.5).mean():.1f}%", flush=True)

    print("\n  판정 — 가장 크게 깎는 관문이 이 소스에 정당한지가 다음 질문이다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
