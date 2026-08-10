"""#INTRABAR 2026-08-10 (1단계): 미완성 봉이 셋업을 얼마나 바꾸나 — 영향 측정.

## 배경
라이브 봇은 60초마다 돌면서 **아직 만들어지는 중인 5분봉**을 마지막 봉으로 쓴다
(`_fetch_ohlcv_tf` 가 마지막 봉을 버리지 않는다). 백테는 닫힌 봉만 본다.
5분봉 하나를 라이브는 5번 평가하고, 그때마다 고가·저가·종가가 다르다.

파트너 결정: "최대한 라이브랑 환경을 똑같이 만들어야지."

## 왜 바로 전면 구현을 안 하나
평가 횟수가 5배가 되어 타임라인 빌드가 페어당 2시간 → **10시간**이 된다.
그 비용을 쓰기 전에 **효과 크기부터 잰다** — 8/7 피보나치 때 세운 원칙
(스윕 전에 구조적 상한 먼저)을 그대로 적용한다. 셋업이 거의 안 바뀌면
GAP 으로 남기고, 크게 바뀌면 전면 구현한다.

## 방법
1분봉에서 5분봉을 **부분 구성**해 라이브가 보는 창을 그대로 만든다.
    · 완성분 [0..k-1] + 형성중 봉(1~5분 누적: open=첫, high=max, low=min, close=현재)
매 1분 시점마다 `detect_silver_bullet_setups` 를 돌려, 닫힌 봉만 볼 때와
  ① 셋업이 새로 잡히는가(라이브만 보는 셋업)
  ② 같은 셋업의 진입가가 달라지는가
를 센다.

## 판정
새 셋업 비율이 5% 미만이고 진입가 차이가 손절폭 대비 미미하면 GAP 유지.
그 이상이면 전면 구현(1분 재생 엔진).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.strategy.silver_bullet import detect_silver_bullet_setups  # noqa: E402

SYM = "BTCUSDT"
PROBE_5M = 6000          # 검사할 5분봉 수 (약 3주) — 1분 시점 3만 회 평가
WINDOW = 500             # 라이브가 보는 창 (cfg.window)


def five_min_frames(df1: pd.DataFrame):
    """1분봉 → (완성 5분봉 df, 각 1분 시점의 형성중 봉) 생성기.

    라이브가 60초마다 보는 창을 그대로 만든다. 형성중 봉은
    open=구간 첫 시가 · high=지금까지 최고 · low=최저 · close=현재 종가.
    """
    o = df1["open"].to_numpy(float)
    h = df1["high"].to_numpy(float)
    lo = df1["low"].to_numpy(float)
    c = df1["close"].to_numpy(float)
    v = df1["volume"].to_numpy(float) if "volume" in df1.columns else np.zeros(len(df1))
    idx = df1.index
    # 5분 경계 = 인덱스 분(minute) % 5 == 0 인 지점에서 시작
    start = int(np.argmax((idx.minute % 5) == 0))
    for k in range(start, len(df1) - 5, 5):
        yield k, (o[k], h[k], lo[k], c[k], v[k])


def main() -> int:
    print("=== 미완성 봉 영향 측정 (1단계)", flush=True)
    cfg = live_cfg(SYM)
    df1 = _load_full(SYM)
    # 최근 구간만
    need_1m = PROBE_5M * 5 + WINDOW * 5 + 100
    df1 = df1.tail(need_1m)
    print(f"  1분봉 {len(df1):,} · 검사 5분봉 {PROBE_5M:,} (약 {PROBE_5M * 5 / 1440:.0f}일)",
          flush=True)

    # 완성 5분봉
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df5 = df1.resample("5min").agg(agg).dropna()
    print(f"  완성 5분봉 {len(df5):,}", flush=True)

    o1 = df1["open"].to_numpy(float)
    h1 = df1["high"].to_numpy(float)
    l1 = df1["low"].to_numpy(float)
    c1 = df1["close"].to_numpy(float)
    v1 = df1["volume"].to_numpy(float) if "volume" in df1.columns else np.zeros(len(df1))

    # df5 의 각 봉이 df1 의 어느 구간인지
    pos5 = {ts: i for i, ts in enumerate(df5.index)}
    kw = dict(bias=None, min_rr=cfg.min_rr, fvg_min_size_pct=cfg.fvg_min_size_pct,
              expand_to_killzone=cfg.expand_to_killzone,
              disable_time_filter=cfg.disable_time_filter,
              min_sl_distance_pct=cfg.min_sl_distance_pct, ote_level=cfg.ote_level)

    t0 = time.time()
    n_eval = closed_only = intrabar_new = same = swapped = 0
    entry_diffs: list[float] = []

    start5 = max(WINDOW, len(df5) - PROBE_5M)
    for b in range(start5, len(df5) - 1):
        # ① 닫힌 봉만 (백테 현행) — b 까지 완성
        win_closed = df5.iloc[b + 1 - WINDOW : b + 1]
        s_closed = detect_silver_bullet_setups(win_closed, **kw)
        last_closed = s_closed[-1] if s_closed else None
        if last_closed is not None:
            closed_only += 1

        # ② 라이브 — b+1 봉이 형성되는 동안 1~4분 시점
        ts_next = df5.index[b + 1]
        j0 = df1.index.get_indexer([ts_next])[0]
        if j0 < 0:
            continue
        for m in range(1, 5):          # 1,2,3,4분 경과 시점 (5분이면 완성 = ①)
            if j0 + m > len(df1):
                break
            seg = slice(j0, j0 + m)
            part = pd.DataFrame(
                {"open": [o1[j0]], "high": [h1[seg].max()], "low": [l1[seg].min()],
                 "close": [c1[j0 + m - 1]], "volume": [v1[seg].sum()]},
                index=[ts_next],
            )
            win_live = pd.concat([win_closed.iloc[1:], part])
            s_live = detect_silver_bullet_setups(win_live, **kw)
            n_eval += 1
            last_live = s_live[-1] if s_live else None
            if last_live is None:
                continue
            if last_closed is None:
                intrabar_new += 1
                continue
            # ★ 같은 셋업인지 먼저 확인한다. 앞선 판에서는 양쪽의 '최신 셋업'을
            # 그냥 비교해 **서로 다른 셋업의 가격 차이**를 재고 있었다(중앙 2.68R).
            # ts_ms 가 같아야 같은 셋업이다.
            if int(last_live.ts_ms) != int(last_closed.ts_ms):
                swapped += 1
                continue
            d = abs(float(last_live.entry) - float(last_closed.entry))
            if d > 1e-9:
                risk = abs(float(last_closed.entry) - float(last_closed.stop_loss))
                if risk > 0:
                    entry_diffs.append(d / risk)
            else:
                same += 1

    el = time.time() - t0
    print(f"\n  1분 시점 평가 {n_eval:,}회 ({el:.0f}초)", flush=True)
    print(f"  닫힌 봉 기준 셋업 보유 봉 {closed_only:,}", flush=True)
    print(f"  ★ 라이브만 보는 셋업(닫힌 봉엔 없음) {intrabar_new:,}회"
          f" — 평가 대비 {100 * intrabar_new / max(n_eval, 1):.2f}%", flush=True)
    print(f"  같은 셋업·진입가 동일 {same:,}회", flush=True)
    print(f"  ★ 최신 셋업이 **다른 것으로 바뀜** {swapped:,}회"
          f" — 평가 대비 {100 * swapped / max(n_eval, 1):.2f}%", flush=True)
    if entry_diffs:
        d = np.array(entry_diffs)
        print(f"  진입가 차이 {len(d):,}건 — 중앙 {np.median(d):.4f}R ·"
              f" 평균 {d.mean():.4f}R · 95분위 {np.percentile(d, 95):.4f}R", flush=True)
    else:
        print("  진입가 차이 0건", flush=True)

    print("\n  판정 — 라이브 전용 셋업이 5% 미만이고 진입가 차이가 0.05R 미만이면"
          " GAP 유지가 합리적이다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
