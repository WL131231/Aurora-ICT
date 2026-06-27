"""볼린저 평균회귀 — TF 단축(5m/15m) 재검증. bb_meanrev_bt 모듈 재사용.

파트너(6/27): 1h 평균회귀가 수수료0에도 적자였는데, 더 짧은 TF(5m/15m)에서
횡보 평균회귀가 더 빈번해 엣지가 있는지 확인. TF 만 바꾸고 전략·비용 모델은 동일.

실행: cwd=Aurora-ICT-research, PYTHONPATH=../Aurora-ICT/src, argv[1]=TF분(기본 15).
담당: 지영민.
"""
from __future__ import annotations

import sys

import bb_meanrev_bt as bb
import pandas as pd


def _load_tf(sym: str, tf_min: int) -> pd.DataFrame:
    """1m parquet → tf_min 분봉 OHLCV 리샘플."""
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    rule = f"{tf_min}min"
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    lo = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def main() -> int:
    tf_min = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    # 펀딩은 봉당 = 시간당 × (분/60). ttl 은 동일 48시간 유지(봉수 환산).
    bb.FUNDING_PER_HOUR = (0.0001 / 8) * (tf_min / 60.0)
    ttl = int(48 * 60 / tf_min)
    data = {}
    for sym in bb.PAIRS:
        try:
            d = _load_tf(sym, tf_min)
            if len(d) >= 500:
                data[sym] = d
        except Exception as e:  # noqa: BLE001
            print(f"(로드 실패 {sym}: {e})")

    lines = [
        f"===== 볼린저 평균회귀 {tf_min}m 재검증 (7페어, 시드1000, 20x) =====",
        f"ttl={ttl}봉(48h), 펀딩봉당={bb.FUNDING_PER_HOUR:.2e}",
        "",
        f"{'익절':>5} {'진입σ':>5} {'수수료':>6} {'USDT':>9} {'승률':>6} {'RR':>5} {'거래':>7}",
    ]

    def _agg(entry_mult: float, exit_mode: str, fee_pct: float) -> dict:
        agg = {"net": 0.0, "wr": 0.0, "rr": 0.0, "n": 0}
        yearly: dict[int, float] = {}
        nz = 0
        for d in data.values():
            tr = bb._run(d, adx_thr=20.0, sl_mult=3.0, use_sto=True,
                         entry_mult=entry_mult, exit_mode=exit_mode,
                         fee_pct=fee_pct, ttl=ttl)
            s = bb._stats([t[0] for t in tr])
            agg["net"] += s["net"]
            agg["wr"] += s["wr"]
            agg["rr"] += s["rr"]
            agg["n"] += int(s["n"])
            nz += 1
            for net, yr in tr:
                yearly[yr] = yearly.get(yr, 0.0) + net
        agg["nz"] = max(nz, 1)
        agg["yearly"] = yearly
        return agg

    # 핵심 조합만(1h 패턴 = 진입 깊을수록·반대밴드일수록 개선). taker vs maker(수수료0).
    # fast(argv[2]): 봉 많은 5m 용 — 진입 깊은 3σ만(이전 TF 에서 최선).
    fast = len(sys.argv) > 2 and sys.argv[2] == "fast"
    entry_mults = (3.0,) if fast else (2.0, 2.5, 3.0)
    best_taker = None
    for exit_mode in ("mid", "opp"):
        for entry_mult in entry_mults:
            for fee_pct in (0.0004, 0.0):
                a = _agg(entry_mult, exit_mode, fee_pct)
                nz = a["nz"]
                tag = "taker" if fee_pct > 0 else "maker"
                lines.append(
                    f"{exit_mode:>5} {entry_mult:5.1f} {tag:>6} "
                    f"{a['net'] * bb.SEED:+9.0f} {a['wr'] / nz:5.0f}% "
                    f"{a['rr'] / nz:5.2f} {a['n']:7d}")
                if fee_pct > 0 and (best_taker is None or a["net"] > best_taker[0]):
                    best_taker = (a["net"], exit_mode, entry_mult, a)

    if best_taker is not None:
        _, em, en, a = best_taker
        lines.append("")
        lines.append(f"--- 최선 taker 조합({em}, {en:.1f}σ) 연도별 net ---")
        for yr in sorted(a["yearly"]):
            lines.append(f"  {yr}: {a['yearly'][yr] * bb.SEED:+9.0f} USDT")

    txt = "\n".join(lines)
    out = f"bb_meanrev_{tf_min}m_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + f"\n→ {out}\nDONE")
    except UnicodeEncodeError:
        print(f"(결과는 {out})\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
