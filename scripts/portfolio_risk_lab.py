"""포트폴리오 리스크 층 연구 — 동시 진입·상관·DD 스로틀 (파트너 승인, 2026-07-10).

배경: 판 단위 로직은 1.7 로 완성 — 남은 구멍은 묶음 단위 (7/4 숏 클러스터 전멸:
페어당 6%라도 동시 5개면 그 순간 실질 30%). 진단 + 처방 그리드:

    진단: 5년 병합 스트림의 동시 포지션 분포, 동일방향 클러스터, 최악의 날.
    처방 (복리 시뮬, 리스크 6%≈m3.6, 사망=시드 5%):
      - 동시 포지션 상한 N (초과 진입 skip)
      - 동일방향 상한 N
      - 상관 가드: BTC·ETH 동일방향 중복 skip
      - DD 스로틀: 현재 낙폭 > X% 면 신규 진입 리스크 ×0.5 (anti-martingale)
      - 조합
근사: 페어 간 독립(한 페어의 skip 이 타 페어 결과 불변) — 실제와 일치.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/portfolio_risk_lab.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
M = 3.6            # 리스크 6% 근사 노셔널 배수 (net 은 18배 기준 → ×M/18)
BT_N = 18.0
DEAD = 0.05
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    out = [(df5.index[t.entry_idx], df5.index[t.exit_idx], str(t.direction),
            t.net_pnl_pct, sym) for t in bt.trades]
    print(f"  {sym} done n={len(out)}", flush=True)
    return out


def simulate(trades, cap_total=99, cap_dir=99, corr_guard=False,
             dd_throttle=0.0, throttle_factor=0.5, daily_stop=0.0):
    """시간순 이벤트 시뮬 — 반환 (최종배수, MDD, 사망ts, skip수, 최악일손실%)."""
    events = []
    for k, (e_ts, x_ts, d, net, sym) in enumerate(trades):
        events.append((e_ts, 0, k))   # 0=entry (동시각이면 entry 먼저 — 보수)
        events.append((x_ts, 1, k))
    events.sort(key=lambda x: (x[0], x[1]))
    eq = peak = 1.0
    mdd = 0.0
    dead = None
    active: dict[int, float] = {}      # k -> risk_scale
    n_skip = 0
    daily: dict[str, float] = {}
    corr = {"BTCUSDT", "ETHUSDT"}
    for ts, kind, k in events:
        e_ts, x_ts, d, net, sym = trades[k]
        if kind == 0:
            if dead:
                continue
            # 일일 서킷브레이커 — 당일 실현손실이 임계 초과면 신규 진입 중단.
            if daily_stop > 0 and daily.get(str(ts)[:10], 0.0) <= -daily_stop:
                n_skip += 1
                continue
            same_dir = sum(1 for j in active if trades[j][2] == d)
            corr_dup = corr_guard and sym in corr and any(
                trades[j][4] in corr and trades[j][2] == d and trades[j][4] != sym
                for j in active)
            if len(active) >= cap_total or same_dir >= cap_dir or corr_dup:
                n_skip += 1
                continue
            dd_now = 1.0 - eq / peak
            scale = throttle_factor if (dd_throttle > 0 and dd_now > dd_throttle) else 1.0
            active[k] = scale
        else:
            if k not in active:
                continue
            scale = active.pop(k)
            delta = net * (M / BT_N) * scale
            eq *= (1.0 + delta)
            day = str(ts)[:10]
            daily[day] = daily.get(day, 0.0) + delta
            peak = max(peak, eq)
            mdd = max(mdd, 1.0 - eq / peak)
            if eq <= DEAD and not dead:
                dead = str(ts)[:10]
    worst_day = min(daily.values()) if daily else 0.0
    return eq, mdd, dead, n_skip, worst_day, daily


def main() -> int:
    with Pool(4) as p:
        per_pair = p.map(_pair_worker, PAIRS)
    trades = sorted([t for lst in per_pair for t in lst], key=lambda x: x[0])

    # --- 진단: 동시 포지션 분포 / 동일방향 클러스터 ---
    events = []
    for k, (e, x, d, n, s) in enumerate(trades):
        events.append((e, 1, d))
        events.append((x, -1, d))
    events.sort(key=lambda v: (v[0], -v[1]))
    cur = {"long": 0, "short": 0}
    hist: dict[int, int] = {}
    max_same = 0
    for _ts, delta, d in events:
        cur[d] += delta
        tot = cur["long"] + cur["short"]
        hist[tot] = hist.get(tot, 0) + (1 if delta > 0 else 0)
        max_same = max(max_same, cur["long"], cur["short"])

    lines = ["===== 포트폴리오 리스크 층 (7페어 5년, 리스크6% 복리, 시드 1.0) =====",
             f"트레이드 {len(trades)}건 · 진입시점 동시보유 분포 {dict(sorted(hist.items()))}"
             f" · 동일방향 최대 동시 {max_same}개",
             "",
             f"{'변형':<26}{'최종배수':>8}{'MDD':>7}{'최악일':>8}{'skip':>6}{'생존':>10}"]
    variants = [
        ("기준 (무제한)", dict()),
        ("일일스탑 -12%", dict(daily_stop=0.12)),
        ("일일스탑 -15%", dict(daily_stop=0.15)),
        ("일일스탑 -18%", dict(daily_stop=0.18)),
        ("DD스로틀 25%/x0.7", dict(dd_throttle=0.25, throttle_factor=0.7)),
        ("DD스로틀 30%/x0.7", dict(dd_throttle=0.30, throttle_factor=0.7)),
        ("DD스로틀 35%/x0.7", dict(dd_throttle=0.35, throttle_factor=0.7)),
        ("일일스탑15+DD30/x0.7", dict(daily_stop=0.15, dd_throttle=0.30,
                                   throttle_factor=0.7)),
        ("일일스탑15+DD25/x0.7", dict(daily_stop=0.15, dd_throttle=0.25,
                                   throttle_factor=0.7)),
    ]
    mid_ts = trades[len(trades) // 2][0]
    for label, kw in variants:
        eq, mdd, dead, n_skip, worst, daily = simulate(trades, **kw)
        surv = f"사망 {dead}" if dead else "생존"
        # 반기 수익 (일별 합 분리 — 로그수익 아닌 단순합 근사)
        h1 = sum(v for d, v in daily.items() if d < str(mid_ts)[:10])
        h2 = sum(v for d, v in daily.items() if d >= str(mid_ts)[:10])
        lines.append(f"{label:<24}{eq:>7.2f}x{mdd * 100:>6.0f}%{worst * 100:>+7.1f}%"
                     f"{n_skip:>6d}{surv:>12}  전/후반 {h1 * 100:+.0f}/{h2 * 100:+.0f}%")
    # 최악일 감사 — 기준의 하위 5일
    _eq, _m, _d, _s2, _w, base_daily = simulate(trades)
    lines.append("")
    lines.append("-- 기준 최악 5일 (일일 실현손익) --")
    for d, v in sorted(base_daily.items(), key=lambda kv: kv[1])[:5]:
        lines.append(f"  {d}  {v * 100:+.1f}%")
    txt = "\n".join(lines)
    with open("portfolio_risk_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
