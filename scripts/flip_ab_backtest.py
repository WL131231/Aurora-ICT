"""#AUTONOMOUS 2026-07-30: HTF FVG flip 백테 이식 + A/B 처방 검증 (파트너 지시).

배경 — 라이브 Origo 2.2 실측(live_flip_rr.py):
  flip 청산이 평균 **+0.61R** 에서 발생(86%가 1R 미만), 고정 TP 는 2.0R.
  홀드 반사실: 2R TP 선착 72% / SL 선착 28% → 기대값 +0.61R vs **+1.17R**.
  손익분기 RR 이 1.17 이므로 이 개선만으로 흑자권 진입 가능.
문제 — replay.py 에 **HTF FVG flip 경로가 미구현**(liq/swing/liquidity 0회).
  라이브 익절 2/3 를 만드는 경로가 백테에 없어 처방 검증 수단이 없었다.

이 스크립트: replay.py 를 건드리지 않고(라이브 공유 코드 — 위험) 백테 trade 결과에
**flip 을 사후 적용**해 시나리오를 비교한다. 같은 trade 집합에 청산 규칙만 바꿔
끼우므로 A/B 가 정확히 대조된다.

라이브 정합 파라미터(bot_ict_instance.py):
  htf_fvg_tfs=("15m","1h","2h","4h","1d","1w") / TF_WEIGHT 5m1·15m2·1h4·2h6·4h10·1d20·1w40
  threshold = max(LTF weight×3, 6) = 6 (LTF=5m)
  _FLIP_TARGET_MIN_WEIGHT = 4 (1h+ 존만 target — Origo 1.3 #360 FLIP-REFINE)
  max_touch_count = 3 / fvg_min_size_pct = 0.0005

시나리오:
  F0 flip 없음            — 현행 백테(= 고정 TP/SL/트레일만)
  F1 flip 라이브 정합      — min_weight 4 (현행 라이브 재현)
  F2 min_weight 10 (4h+)  — FLIP-REFINE 의 연장(더 먼 존만)
  A  flip 시 50% 부분청산  — 나머지는 원래 청산까지 홀드
  B  flip 최소 R 요구      — 이익이 thr R 미만이면 flip 무시(0.5/1.0/1.5R)
인과성: FVG 는 3봉 패턴이라 **마지막 봉 완결 후** 인지 가능(created_idx = idx+1).
        활성 판정은 created 이후 ~ 첫 접촉 이전. 미래 정보 차단.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402
from aurora_ict.indicators.fvg import FVGType, detect_fvgs  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
TF_WEIGHT = {"15m": 2, "1h": 4, "2h": 6, "4h": 10, "1d": 20, "1w": 40}
HTF_TFS = ("15min", "1h", "2h", "4h", "1D", "1W")   # pandas 규칙("15m"=15개월 함정 회피)
TF_KEY = {"15min": "15m", "1h": "1h", "2h": "2h", "4h": "4h", "1D": "1d", "1W": "1w"}
THRESHOLD = 6
FVG_MIN_PCT = 0.0005
# 라이브 build_htf_fvg_map(limit=200) 정합 — 각 TF 는 **최근 200봉만** fetch 하므로
# 그보다 오래된 FVG 는 후보에서 사라진다. 5m 봉 수로 환산한 유효 수명.
# (이 제한을 빼면 5년 전 미체결 FVG 까지 살아남아 zone 이 4만개가 되고, flip 무장률이
#  85~98%[라이브 30%]로 폭증해 net/RR 이 비현실적으로 부풀었다 — 7/30 1차 실행 오류.)
TF_LIFE_5M = {"15m": 200 * 3, "1h": 200 * 12, "2h": 200 * 24,
              "4h": 200 * 48, "1d": 200 * 288, "1w": 200 * 2016}
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)


def build_fvg_zones(df5: pd.DataFrame) -> list[dict]:
    """전 HTF 의 FVG 를 (활성구간, zone, 가중치, 타입) 으로 평탄화. 5m idx 기준.

    인과: created5 = HTF 3봉 패턴 마지막 봉이 **완결된** 5m idx.
          touched5 = 가격이 zone 에 처음 닿은 5m idx (그 이후 비활성).
    """
    h5 = df5["high"].to_numpy(); l5 = df5["low"].to_numpy()
    n5 = len(df5)
    out: list[dict] = []
    for tf in HTF_TFS:
        w = TF_WEIGHT[TF_KEY[tf]]
        d = df5.resample(tf).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna()
        if len(d) < 5:
            continue
        try:
            fvgs = detect_fvgs(d, min_size_pct=FVG_MIN_PCT)
        except Exception:  # noqa: BLE001
            continue
        # HTF idx → 5m idx 매핑 (해당 HTF 봉의 종료 시점)
        tf_end = d.index.to_list()
        d_hi = d["high"].to_numpy(); d_lo = d["low"].to_numpy()
        d_cl = d["close"].to_numpy()
        for fv in fvgs:
            k = fv.idx + 1                       # 3봉 패턴 마지막 봉
            if k >= len(tf_end):
                continue
            # 그 HTF 봉이 닫히는 시점 = 다음 HTF 봉 시작. 그 시점의 5m idx.
            close_ts = tf_end[k + 1] if k + 1 < len(tf_end) else tf_end[k]
            created5 = int(df5.index.searchsorted(close_ts))
            if created5 >= n5 - 1:
                continue
            hi, lo = float(fv.high), float(fv.low)
            mid = (hi + lo) / 2.0
            bull = fv.type is FVGType.BULLISH
            # ── 라이브 정합 (indicators/fvg.mark_filled_and_invalidated + build_htf_fvg_map
            #    + find_opposite_htf_fvg 실측) — 후보 제외 조건이 **세 갈래**다:
            #   ① filled      : bullish low<=mid / bearish high>=mid (50% retest). HTF 봉 기준.
            #   ② invalidated : bullish **close**<low / bearish **close**>high (zone 이탈).
            #                   ← 종가 기준이라 5m 로 대체 불가. HTF 종가로만 판정.
            #   ③ touch_count : zone 경계 "밖→안" 전환 3회 도달 시 약화 제외.
            #   검사 시작은 fv.idx+2 (중간봉 다음다음) — 라이브와 동일.
            # 이 셋을 하나로 뭉개거나(7/30 1~4차: 첫 접촉=즉시 비활성) invalidated 를
            # 빼면(5차) zone 수명이 어긋나 flip 발동 R·승률이 라이브와 벌어진다.
            inv_k = fil_k = None
            for j in range(fv.idx + 2, len(d)):
                if fil_k is None:
                    if (bull and d_lo[j] <= mid) or ((not bull) and d_hi[j] >= mid):
                        fil_k = j
                if (bull and d_cl[j] < lo) or ((not bull) and d_cl[j] > hi):
                    inv_k = j
                    break
            def _to5(kk: int | None) -> int:
                """HTF 봉 idx → 그 봉이 닫히는 5m idx (없으면 데이터 끝)."""
                if kk is None or kk + 1 >= len(tf_end):
                    return n5
                return int(df5.index.searchsorted(tf_end[kk + 1]))
            filled5, invalid5 = _to5(fil_k), _to5(inv_k)
            # touch_count 는 틱 감시(FlipWatcher)라 5m 해상도로 근사한다.
            seg_hi = h5[created5:]; seg_lo = l5[created5:]
            inside = (seg_lo <= hi) & (seg_hi >= lo)
            enters = np.flatnonzero(inside & ~np.concatenate([[False], inside[:-1]]))
            touch3 = int(created5 + enters[2]) if enters.size >= 3 else n5
            expire5 = created5 + TF_LIFE_5M[TF_KEY[tf]]   # fetch limit=200 수명
            out.append(dict(tf=TF_KEY[tf], w=w, typ=fv.type, hi=hi, lo=lo, c5=created5,
                            t5=min(filled5, invalid5, touch3, expire5)))
    return out


def flip_target_at(zones: list[dict], idx: int, direction: int, price: float,
                   min_w: int) -> dict | None:
    """진입 시점 flip target — 봇 _evaluate_htf_override 이식.

    롱(direction=1) → 반대 = bearish, 가격 **위쪽** zone. 숏 → bullish, 아래쪽.
    합산 가중치 > THRESHOLD 이어야 하고, target 은 weight>=min_w 중 가장 가까운 것.
    """
    opp = FVGType.BEARISH if direction == 1 else FVGType.BULLISH
    cands = []
    for z in zones:
        if z["typ"] is not opp:
            continue
        if not (z["c5"] <= idx < z["t5"]):      # 활성 구간만 (인과)
            continue
        if direction == 1 and z["lo"] <= price:
            continue
        if direction == -1 and z["hi"] >= price:
            continue
        cands.append(z)
    if sum(z["w"] for z in cands) <= THRESHOLD:
        return None
    cands.sort(key=lambda z: abs((z["hi"] + z["lo"]) / 2 - price))
    for z in cands:
        if z["w"] >= min_w:
            return z
    return None


def simulate(df5: pd.DataFrame, trades, zones, mode: str, min_w: int = 4,
             min_r: float = 0.0, partial: float = 0.0):
    """원본 trade 에 flip 규칙을 사후 적용해 net% 리스트 반환.

    mode: "off" | "flip" | "partial" | "minr"
    """
    h5 = df5["high"].to_numpy(); l5 = df5["low"].to_numpy()
    out = []
    fired: list[float] = []          # flip 실제 발동 건의 R — 라이브(평균 0.61R·발동 30%) 대조
    for t in trades:
        base_net = t.net_pnl_pct
        if mode == "off":
            out.append((df5.index[t.entry_idx], base_net))
            continue
        d = _dir_of(t)
        entry = float(t.entry)
        z = flip_target_at(zones, t.entry_idx, d, entry, min_w)
        if z is None:
            out.append((df5.index[t.entry_idx], base_net))
            continue
        # flip zone 도달 시점 — 보유 구간 내에서만
        lo_i, hi_i = t.entry_idx + 1, min(t.exit_idx, len(df5) - 1)
        if hi_i <= lo_i:
            out.append((df5.index[t.entry_idx], base_net))
            continue
        seg_hi = h5[lo_i:hi_i + 1]; seg_lo = l5[lo_i:hi_i + 1]
        hit = np.flatnonzero((seg_lo <= z["hi"]) & (seg_hi >= z["lo"]))
        if not hit.size:
            out.append((df5.index[t.entry_idx], base_net))
            continue
        fi = lo_i + int(hit[0])
        # flip 청산가 = zone 근접 edge.
        # 근거: FlipWatcher 는 WS 틱(0.2s)으로 감시하며 "FVG zone 1회 touch = mitigation
        # 인정 = **즉시** flip, wick 도 touch" 다(flip_watcher.py 모듈 docstring).
        # 즉 봉 종가를 기다리지 않고 zone 경계에 닿는 순간 시장가 청산 → edge 가 정합.
        # (3차 실행에서 close 로 바꿨던 것은 라이브 동작 오해였다 — 되돌림.)
        fpx = z["lo"] if d == 1 else z["hi"]
        raw = (fpx - entry) / entry * d
        # 위험폭(R) — 원 trade 의 SL 거리를 entry_atr_pct 로 근사 못 하므로
        # base 결과의 outcome 대신 sl_dist 를 직접 못 얻는다 → R 은 raw/risk 로 계산.
        risk = abs(entry - _sl_of(t, entry, d))
        r_at_flip = (raw * entry / risk) if risk > 0 else 0.0
        # 원 trade 의 비용분(net-raw, 음수)을 그대로 승계 — 진입/청산 각 1회는 동일.
        cost = base_net - t.raw_pnl_pct
        flip_net = raw * 100 + cost
        fired.append(r_at_flip)          # flip 이 실제 발동한 건 — 라이브 대조용
        if mode == "flip":
            out.append((df5.index[t.entry_idx], flip_net))
        elif mode == "minr":
            out.append((df5.index[t.entry_idx],
                        flip_net if r_at_flip >= min_r else base_net))
        elif mode == "partial":
            out.append((df5.index[t.entry_idx],
                        partial * flip_net + (1 - partial) * base_net))
        else:
            out.append((df5.index[t.entry_idx], base_net))
    return out, fired


def _dir_of(t) -> int:
    """trade.direction 은 str 또는 enum — 둘 다 수용해 +1/-1 로 정규화."""
    v = getattr(t.direction, "value", t.direction)
    return 1 if str(v).lower() in ("long", "buy") else -1


def _sl_of(t, entry: float, d: int) -> float:
    """원 trade 의 **실제 초기 SL** (replay.Trade.entry_sl, 2026-07-30 추가).

    이전에는 ATR×4.0 근사를 썼고 그 때문에 flip 발동 R 이 라이브 0.61R vs 백테 1.44R
    로 어긋났다(2.4배). 이제 sl_dist_mult·sl_liq_cap 적용 후 실측값을 쓴다.
    """
    sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
    if sl > 0:
        return sl
    # 구 결과 호환 — 필드 없으면 SL 청산가, 그것도 없으면 ATR 근사.
    if str(getattr(t, "outcome", "")).lower().startswith("sl"):
        return float(t.exit_price)
    atrp = float(getattr(t, "entry_atr_pct", 0.0) or 0.0)
    return entry * (1 - d * (atrp / 100 * 4.0 if atrp > 0 else 0.015))


def stat(tr):
    if not tr:
        return None
    tr = sorted(tr)
    nets = [p for _, p in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(nets[:half]); h2 = sum(nets[half:])
    ys: dict[int, float] = {}
    for ts, p in tr:
        ys[ts.year] = ys.get(ts.year, 0.0) + p
    wins = [p for p in nets if p > 0]; los = [p for p in nets if p < 0]
    rr = (np.mean(wins) / abs(np.mean(los))) if wins and los else float("nan")
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, rr=rr,
                ypos=sum(1 for v in ys.values() if v > 0), ytot=len(ys),
                ys=" ".join(f"{k}:{v:+.0f}" for k, v in sorted(ys.items())))


def line(s):
    if s is None:
        return "없음"
    return (f"n={s['n']:4d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
            f"H1={s['h1']:+7.1f} H2={s['h2']:+7.1f} 연도{s['ypos']}/{s['ytot']}")


def main() -> int:
    scen = {}
    fired_all: list[float] = []
    n_closed = 0
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        zones = build_fvg_zones(df5)
        armed = sum(1 for t in bt.trades
                    if flip_target_at(zones, t.entry_idx, _dir_of(t),
                                      float(t.entry), 4) is not None)
        print(f"  {sym}: trade {len(bt.trades)}건 / FVG zone {len(zones)} / "
              f"flip 무장 {armed}건 ({100 * armed / max(len(bt.trades), 1):.0f}%)", flush=True)
        for name, kw in (
            ("F0 flip 없음", dict(mode="off")),
            ("F1 flip 정합(1h+)", dict(mode="flip", min_w=4)),
            ("F2 flip 4h+ 만", dict(mode="flip", min_w=10)),
            ("A 50% 부분청산", dict(mode="partial", min_w=4, partial=0.5)),
            ("A 30% 부분청산", dict(mode="partial", min_w=4, partial=0.3)),
            ("B 최소 0.5R", dict(mode="minr", min_w=4, min_r=0.5)),
            ("B 최소 1.0R", dict(mode="minr", min_w=4, min_r=1.0)),
            ("B 최소 1.5R", dict(mode="minr", min_w=4, min_r=1.5)),
        ):
            res, fired = simulate(df5, bt.trades, zones, **kw)
            scen.setdefault(name, []).extend(res)
            if name == "F1 flip 정합(1h+)":
                fired_all.extend(fired)
                n_closed += len(bt.trades)
    # ★ 정합 검증 — 라이브 실측(발동률 30%, 평균 +0.61R, <1R 86%)과 대조.
    # 이 두 수치가 안 맞으면 아래 net/RR 은 신뢰할 수 없다(7/30 1·2차 실행에서 학습).
    print("\n\n===== ★ 라이브 정합 검증 =====", flush=True)
    if fired_all:
        fa = np.array(fired_all)
        print(f"  flip 발동률: 백테 {100 * len(fa) / max(n_closed, 1):.0f}%  vs 라이브 30%",
              flush=True)
        print(f"  발동 시 평균 R: 백테 {fa.mean():+.2f}R  vs 라이브 +0.61R", flush=True)
        print(f"  1R 미만 비율 : 백테 {100 * (fa < 1).mean():.0f}%  vs 라이브 86%", flush=True)
    print("\n\n===== 시나리오 비교 (7페어 5년 합산) =====", flush=True)
    base = stat(scen["F0 flip 없음"])
    for name in ("F0 flip 없음", "F1 flip 정합(1h+)", "F2 flip 4h+ 만",
                 "A 50% 부분청산", "A 30% 부분청산",
                 "B 최소 0.5R", "B 최소 1.0R", "B 최소 1.5R"):
        s = stat(scen[name])
        mark = ""
        if s and base and name != "F1 flip 정합(1h+)":
            f1 = stat(scen["F1 flip 정합(1h+)"])
            if f1 and s["net"] > f1["net"]:
                mark = " ★현행대비 개선"
        print(f"  {name:<18} {line(s)}{mark}", flush=True)
    print("\n  ※ 현행 라이브 = F1. F1 대비 개선되는 처방이 배포 후보.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
