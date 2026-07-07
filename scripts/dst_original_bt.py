"""Cursus 원본(매매기법.py) 엔진 백테 — 실매매 정합 우선 (2026-07-07 파트너 지시).

라이브 봇(#366 복원)과 1:1 동작 정합:
    - 신호: 마감 1h 봉의 ST1×2 & ST2×3 정렬 신규발생 (dst._signals 동일).
      **다음 봉 시가**에 시장가 진입 (라이브 = 마감봉 감지 후 시장가 ≈ 차봉 시가).
    - SL: 고정 2% (entry×(1∓0.02)). 거래소 상주 주문 → 봉 저/고가 터치 시 체결.
    - TP: 1/2/3/4% ×25% 부분익절 — 라이브는 봇 폴링 reduce_only 시장가 → TP 가격에
      슬리피지·테이커 수수료 적용(보수).
    - 래더: TP2 체결 후 SL→TP1, TP3 후 SL→TP2. 라이브는 체결 다음 step 에 SL 이동
      → 백테도 **다음 봉부터** 반영(보수).
    - 동일 봉에서 SL·TP 둘 다 터치 가능하면 **SL 우선**(비관적 — 1h 봉 내부 순서
      불가지론에서 안전한 쪽).
    - REVERSE: 반대 신호 봉 시가에 잔량 청산 후 역진입.
    - 비용: 테이커 수수료(진입+청산×조각), 변동성 비례 슬리피지, 펀딩비(보유시간).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/dst_original_bt.py
담당: 지영민 (Cursus 원본 정합 검증).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dst_trend_bt as dst  # noqa: E402

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402

SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
TRAIL_TRIGGER = 2  # TP2 체결부터 래더


def _run_original(df):
    """원본 엔진 시뮬 — 포지션 단위 net(시드 비율) 리스트 + 부분체결 수 반환."""
    sig = dst._signals(df)
    h = sig["high"].values
    low = sig["low"].values
    c = sig["close"].values
    o = sig["open"].values
    # 마감봉 신호 → 다음 봉 행동 (dst 와 동일 shift)
    buy = np.concatenate([[False], sig["buy_sig"].values[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].values[:-1]])

    trades: list[float] = []   # 포지션 단위 net
    side = None                # "long"/"short"
    entry = stop = 0.0
    tps: list[float] = []
    filled = [False] * 4
    entry_i = 0
    pos_net = 0.0              # 현재 포지션 누적 net (조각 합)
    ladder_stop_pending = None  # 다음 봉부터 적용할 래더 SL

    def _frac_exit(exit_px_raw: float, frac: float, i: int, market: bool) -> float:
        """조각 청산 net — 수수료·슬리피지·펀딩 반영 (시드 비율)."""
        slp = slip_pct(h[i], low[i], c[i]) if market else 0.0
        px = apply_slippage(exit_px_raw, side, "exit", slp) if market else exit_px_raw
        raw = (px - entry) / entry
        if side == "short":
            raw = -raw
        net, _ = apply_costs(raw, dst.SIZE_PCT * frac, dst.LEVERAGE)
        net -= (i - entry_i) * dst.FUNDING_PER_HOUR * dst.SIZE_PCT * frac * dst.LEVERAGE
        return net

    def _open_pos(direction: str, i: int):
        nonlocal side, entry, stop, tps, filled, entry_i, pos_net, ladder_stop_pending
        slp = slip_pct(h[i], low[i], c[i])
        entry = apply_slippage(o[i], direction, "entry", slp)
        side = direction
        sign = 1.0 if direction == "long" else -1.0
        stop = entry * (1 - sign * SL_PCT)
        tps = [entry * (1 + sign * p) for p in TP_PCTS]
        filled = [False] * 4
        entry_i = i
        pos_net = 0.0
        ladder_stop_pending = None

    n_parts = 0
    for i in range(1, len(c)):
        if side is not None:
            # 래더 SL — 직전 봉 TP 체결분은 이번 봉부터 반영(라이브 step 지연 정합).
            if ladder_stop_pending is not None:
                better = ladder_stop_pending > stop if side == "long" \
                    else ladder_stop_pending < stop
                if better:
                    stop = ladder_stop_pending
                ladder_stop_pending = None
            remaining = 1.0 - TP_FRAC * sum(filled)
            # 1) SL 우선 (동일 봉 SL·TP 모호성은 비관적으로)
            hit_sl = low[i] <= stop if side == "long" else h[i] >= stop
            if hit_sl and remaining > 1e-9:
                pos_net += _frac_exit(stop, remaining, i, market=False)
                trades.append(pos_net)
                side = None
            else:
                # 2) TP 순차 체결 (봇 폴링 reduce_only 시장가 — 슬리피지 반영)
                for k in range(4):
                    if filled[k]:
                        continue
                    reached = h[i] >= tps[k] if side == "long" else low[i] <= tps[k]
                    if not reached:
                        break
                    filled[k] = True
                    n_parts += 1
                    pos_net += _frac_exit(tps[k], TP_FRAC, i, market=True)
                hits = sum(filled)
                if hits >= 4:
                    trades.append(pos_net)
                    side = None
                elif hits >= TRAIL_TRIGGER:
                    ladder_stop_pending = tps[hits - 2]
            # 3) REVERSE — 반대 신호 (이번 봉 시가 기준 행동)
            if side is not None:
                rev = bool(sell[i]) if side == "long" else bool(buy[i])
                if rev:
                    remaining = 1.0 - TP_FRAC * sum(filled)
                    if remaining > 1e-9:
                        pos_net += _frac_exit(o[i], remaining, i, market=True)
                    trades.append(pos_net)
                    side = None
                    _open_pos("short" if bool(sell[i]) else "long", i)
        if side is None and (buy[i] or sell[i]) and not (buy[i] and sell[i]):
            # 신규 진입 (REVERSE 로 이미 열렸으면 side 가 차 있어 skip)
            _open_pos("long" if buy[i] else "short", i)
    return trades, n_parts


def main() -> int:
    lines = ["===== Cursus 원본 엔진 백테 (실매매 정합: 차봉시가 진입·SL우선·래더 차봉 반영) =====",
             f"1h 7페어 5년 · SL2% · TP1~4%×25% · 래더(TP2→TP1) · REVERSE · 시드1000 {dst.LEVERAGE:.0f}x",
             "",
             f"{'페어':<10}{'net(USDT)':>10}{'승률':>6}{'RR':>6}{'포지션':>7}{'조각':>7}"]
    tot_net = 0.0
    tot_n = 0
    all_halves = [0.0, 0.0]
    for sym in dst.PAIRS:
        try:
            df = dst._load_1h(sym)
        except Exception as e:  # noqa: BLE001
            lines.append(f"{sym:<10} 로드 실패: {e}")
            continue
        if len(df) < 200:
            continue
        trades, n_parts = _run_original(df)
        s = dst._stats(trades)
        tot_net += s["net"] * dst.SEED
        tot_n += int(s["n"])
        # walk-forward 반분
        half = len(trades) // 2
        all_halves[0] += sum(trades[:half]) * dst.SEED
        all_halves[1] += sum(trades[half:]) * dst.SEED
        lines.append(f"{sym:<10}{s['net'] * dst.SEED:>+10.0f}{s['wr']:>5.0f}%"
                     f"{s['rr']:>6.2f}{int(s['n']):>7d}{n_parts:>7d}")
        print(lines[-1], flush=True)
    lines.append("")
    lines.append(f"합계 net {tot_net:+.0f} USDT / 포지션 {tot_n} / "
                 f"전반 {all_halves[0]:+.0f} · 후반 {all_halves[1]:+.0f}")
    txt = "\n".join(lines)
    with open("dst_original_bt_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
