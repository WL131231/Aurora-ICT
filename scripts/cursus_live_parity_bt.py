"""#AUTONOMOUS 2026-07-30: Cursus **라이브 정합** 백테 — 원본 엔진 그대로 (파트너 지적).

★ 왜 새로 짜는가
  기존 dst_trend_bt*.py 는 "우리원칙 버전"(4분할TP 제거 + ST 라인 트레일 스탑)이었다.
  그러나 **라이브 Cursus 는 원본 엔진 그대로**다(파트너 6/27 복원, 로직 미변경):
      고정 SL 2% + 4분할 TP 1/2/3/4% ×25% + TP 래더 트레일 + REVERSE
  즉 그동안의 Cursus 백테(-24,808% 등)는 **라이브와 다른 전략을 측정**한 것이고,
  그 위에서 낸 결론(출범검증 무효·CSI 게이트 5.5%)도 함께 무효다.
  덧붙여 라이브는 SL 이 **고정 2%** 라 7/27 "유령 체결" 버그가 구조적으로 불가능하다
  (그 버그는 ST 라인을 스탑으로 쓰는 연구 스크립트 고유 문제였다).

라이브 스펙 (bot_trend_instance.py / strategy/dual_st.py 실측)
  TF 1h · ATR14 · ST1 ×2.0 · ST2 ×3.0 · 둘 다 정렬 시 진입 · 양방향
  초기 SL = entry × (1 ∓ 0.02)          (고정 2%, 청산가 캡은 안전망)
  TP      = entry × (1 ± 0.01/0.02/0.03/0.04), 각 25% 부분 익절
  래더    = TP2 체결 후 SL→TP1, TP3 체결 후 SL→TP2, TP4 전량 종료
  REVERSE = 반대 정렬 신호 시 잔량 청산 후 역진입
  레버리지 20x · size_pct 0.9 · 페어당 동시 1포지션

인과 방어: 신호는 봉 종가 확정 후 → **다음 봉 시가 진입**(1봉 지연).
           같은 봉에서 SL·TP 동시 도달 시 **SL 우선**(보수).
비용: taker 0.04%×2 + 변동성 슬리피지 + 펀딩(보유 시간당) — 부분 청산마다 개별 적용.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dst_trend_bt_clamped as DST  # noqa: E402  (데이터 로드·ST·신호 재사용)

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402

PAIRS = DST.PAIRS
LEVERAGE = 20.0
SIZE_PCT = 0.9
ATR_PERIOD = 14
SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
FUNDING_PER_HOUR = getattr(DST, "FUNDING_PER_HOUR", 0.0)


def run_live_parity(df: pd.DataFrame, csi: np.ndarray | None = None,
                    csi_thr: float = 1.0,
                    sig_df: pd.DataFrame | None = None,
                    ) -> tuple[list[tuple[float, int]], int]:
    """라이브 엔진 시뮬. 반환: [(net_pnl_비율, 연도)], 게이트 skip 수.

    net 은 **거래 1건 전체**(부분 청산 합산) 기준 시드 대비 비율.

    Args:
        df: 체결·손익 기준 캔들(**항상 실제 OHLC**).
        sig_df: 신호(ST·buy/sell) 계산용 캔들. None 이면 df 로 계산.
            하이켄아시 실험처럼 **신호는 변환 캔들, 체결은 실가격**으로 나눌 때 쓴다.
            (2026-07-31: 이걸 분리하지 않아 HA 결과가 현행과 소수점까지 동일하게
             나왔다 — 내부에서 _signals(df) 를 재계산해 주입한 컬럼을 덮어썼다.)
    """
    sig_src = df if sig_df is None else sig_df
    sig = DST._signals(sig_src)
    if sig_df is not None:
        # 신호는 sig_df 에서, 가격(OHLC)은 실제 df 에서 — 인덱스 정렬 후 교체.
        for col in ("open", "high", "low", "close"):
            sig[col] = df[col].reindex(sig.index).to_numpy()
    h = sig["high"].to_numpy(); lo = sig["low"].to_numpy()
    c = sig["close"].to_numpy(); o = sig["open"].to_numpy()
    years = sig.index.year.to_numpy()
    # 1봉 지연 — 종가 확정 신호를 다음 봉 시가에 집행(라이브 정합).
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    trades: list[tuple[float, int]] = []
    skipped = 0

    side: str | None = None
    entry = 0.0
    stop = 0.0
    tps: list[float] = []
    hits = 0            # 체결된 TP 개수
    remain = 1.0        # 잔량 비율
    acc = 0.0           # 이번 거래 누적 net(비율)
    entry_i = 0

    def close_all(i: int, price_raw: float, sd: str) -> float:
        """잔량 전량 청산 → net 비율 반환."""
        slp = slip_pct(h[i], lo[i], c[i])
        px = apply_slippage(price_raw, sd, "exit", slp)
        raw = (px - entry) / entry
        if sd == "short":
            raw = -raw
        net, _ = apply_costs(raw, SIZE_PCT * remain, LEVERAGE)
        net -= (i - entry_i) * FUNDING_PER_HOUR * SIZE_PCT * remain * LEVERAGE
        return net

    for i in range(1, len(c)):
        if side is not None:
            rev = bool(sell[i]) if side == "long" else bool(buy[i])
            # ---- SL 우선 판정(같은 봉 동시 도달 시 보수) ----
            sl_hit = (lo[i] <= stop) if side == "long" else (h[i] >= stop)
            if sl_hit:
                acc += close_all(i, stop, side)
                trades.append((acc, int(years[i])))
                side = None; acc = 0.0; remain = 1.0; hits = 0
            else:
                # ---- 4분할 TP 순차 체결 ----
                closed_all = False
                while hits < len(tps):
                    tp = tps[hits]
                    reached = (h[i] >= tp) if side == "long" else (lo[i] <= tp)
                    if not reached:
                        break
                    hits += 1
                    if hits >= len(tps):
                        acc += close_all(i, tp, side)      # TP4 = 전량 종료
                        trades.append((acc, int(years[i])))
                        side = None; acc = 0.0; remain = 1.0; hits = 0
                        closed_all = True
                        break
                    # 부분 익절 — TP_FRAC 만큼
                    slp = slip_pct(h[i], lo[i], c[i])
                    px = apply_slippage(tp, side, "exit", slp)
                    raw = (px - entry) / entry
                    if side == "short":
                        raw = -raw
                    part, _ = apply_costs(raw, SIZE_PCT * TP_FRAC, LEVERAGE)
                    acc += part
                    remain = max(remain - TP_FRAC, 0.0)
                    # ---- 래더 트레일: TP2 체결 후 SL→TP1, TP3 후 SL→TP2 ----
                    if hits >= 2:
                        ladder = tps[hits - 2]
                        if (side == "long" and ladder > stop) or \
                           (side == "short" and ladder < stop):
                            stop = ladder
                if closed_all:
                    pass
                elif side is not None and rev:
                    acc += close_all(i, o[i], side)        # REVERSE — 시가 청산
                    trades.append((acc, int(years[i])))
                    side = None; acc = 0.0; remain = 1.0; hits = 0
        # ---- 신규 진입 (REVERSE 직후 같은 봉 역진입 포함) ----
        if side is None and (buy[i] or sell[i]):
            if csi is not None and csi_thr < 1.0 and not np.isnan(csi[i]) and csi[i] >= csi_thr:
                skipped += 1
                continue
            sd = "long" if buy[i] else "short"
            slp = slip_pct(h[i], lo[i], c[i])
            entry = apply_slippage(o[i], sd, "entry", slp)
            side = sd
            sign = 1.0 if sd == "long" else -1.0
            stop = entry * (1 - sign * SL_PCT)
            tps = [entry * (1 + sign * p) for p in TP_PCTS]
            hits = 0; remain = 1.0; acc = 0.0; entry_i = i
    return trades, skipped


def stat(tr):
    if not tr:
        return None
    nets = [n for n, _ in tr]
    net = sum(nets)
    w = [n for n in nets if n > 0]; l = [n for n in nets if n < 0]
    rr = (np.mean(w) / abs(np.mean(l))) if w and l else float("nan")
    eq = pk = mdd = 0.0
    for n in nets:
        eq += n; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for n, y in tr:
        ys[y] = ys.get(y, 0.0) + n
    half = len(nets) // 2
    return dict(n=len(tr), net=net * 100, wr=100 * len(w) / len(tr), rr=rr,
                avgw=(np.mean(w) * 100 if w else 0.0),
                avgl=(np.mean(l) * 100 if l else 0.0), mdd=mdd * 100,
                h1=sum(nets[:half]) * 100, h2=sum(nets[half:]) * 100,
                ypos=sum(1 for v in ys.values() if v > 0), ytot=len(ys),
                ys=" ".join(f"{k}:{v * 100:+.0f}" for k, v in sorted(ys.items())))


def line(s):
    if s is None:
        return "거래 없음"
    return (f"n={s['n']:5d} net={s['net']:+9.1f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
            f"익평균={s['avgw']:+6.2f} 손평균={s['avgl']:+6.2f} MDD={s['mdd']:7.1f} "
            f"H1={s['h1']:+8.1f} H2={s['h2']:+8.1f} 연도{s['ypos']}/{s['ytot']}")


def main() -> int:
    print("=== Cursus 라이브 정합 백테 (원본 엔진: 고정SL 2% + 4분할TP + 래더 + REVERSE) ===",
          flush=True)
    print("  ※ 기존 dst_trend_bt*.py(트레일ST·분할없음)는 라이브와 다른 전략이었음\n", flush=True)
    allt: list[tuple[float, int]] = []
    per: dict[str, float] = {}
    for sym in PAIRS:
        df = DST._load_1h(sym)
        tr, _ = run_live_parity(df)
        allt.extend(tr)
        per[sym] = sum(n for n, _ in tr) * 100
        s = stat(tr)
        print(f"  {sym:<10} {line(s)}", flush=True)
    print("\n----- 합계 -----", flush=True)
    tot = stat(allt)
    print(f"  {line(tot)}", flush=True)
    if tot:
        print(f"  연도별: {tot['ys']}", flush=True)
        be = (100 - tot["wr"]) / max(tot["wr"], 1e-9)
        print(f"\n  승률 {tot['wr']:.0f}% → 손익분기 RR {be:.2f} / 실현 RR {tot['rr']:.2f} "
              f"({'흑자권' if tot['rr'] > be else '적자 구조'})", flush=True)
    print("\n  페어별 net: " + " ".join(f"{k.replace('USDT', '')}:{v:+.0f}"
                                       for k, v in per.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
