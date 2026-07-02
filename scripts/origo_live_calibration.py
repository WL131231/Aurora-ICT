"""Origo 라이브 vs 백테스트 괴리 정밀 진단 — FST #2 자율연구 (2026-07-02).

파트너 지시: "실제랑 백테스트랑 다를 수도 있으니까" — conf5+sl4(+124) 백테 검증이
라이브에서 재현될지, 라이브에만 존재하는 메커니즘이 엣지를 깎는지 정량화.

백테 replay 의 청산 = 고정 TP / SL / ttl 뿐. 라이브 Origo 는 추가로:
    - flip_close: HTF FVG flip watcher 가 목표 도달 전 조기 청산
    - partial TP(분할익절), sync/reconcile, 수동
이 존재. 첫 스캔에서 Origo 1.2 청산 166건 중 tp_hit 4건(2.4%)·flip 62건(37%)
발견 → flip 이 승자를 어디서 얼마나 자르는지가 핵심 질문.

분석: 청산 사유별 net/건수/평균/승률, flip 세부(어느 TF flip 이 얼마나),
킬존별, 페어별, 실현 RR vs 설정 RR(2.5) 괴리.
사용: PYTHONIOENCODING=utf-8 python scripts/origo_live_calibration.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

P = "data/live/fst_snapshots/fst_2026-07-02_all_users.csv"
KST = timezone(timedelta(hours=9))

# 킬존 (KST) — killzone_research_2026-06-24 와 동일 구분.
KZ = [
    ("asia", 8, 12), ("london", 16, 20), ("ny_am", 21, 24),
    ("ny_pm", 2, 5), ("etc", -1, -1),
]


def kz_of(hour: int) -> str:
    for name, a, b in KZ[:-1]:
        if a <= hour < b:
            return name
    return "etc"


def main() -> int:
    df = pd.read_csv(P)
    df["pnl_usdt"] = pd.to_numeric(df["pnl_usdt"], errors="coerce")
    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(KST)

    lines = ["===== Origo 라이브 vs 백테스트 괴리 진단 (2026-07-02 스냅샷) ====="]

    for model in ("Origo 1.2", "Origo 1.1"):
        m = df[(df["model"] == model) & (df["mode"] == "live")]
        # 순수 청산 = pnl 있는 비-ENTRY/비-RECOVERED. SL실패 비상청산(과분류) 제외 위해
        # reason 에 'SL 적용 실패' 포함 행 별도 집계.
        ex = m[m["pnl_usdt"].notna() & ~m["event_type"].isin(["ENTRY", "RECOVERED"])].copy()
        fail = ex[ex["reason"].fillna("").str.contains("SL 적용 실패|적용 실패")]
        ex = ex.drop(fail.index)
        if ex.empty:
            continue
        wins = ex[ex["pnl_usdt"] > 0]["pnl_usdt"]
        losses = ex[ex["pnl_usdt"] < 0]["pnl_usdt"]
        rr = (wins.mean() / -losses.mean()) if len(losses) and len(wins) else 0.0
        lines.append(f"\n[{model}] 청산 {len(ex)}건 net={ex['pnl_usdt'].sum():+.1f} "
                     f"승률={(ex['pnl_usdt'] > 0).mean() * 100:.0f}% "
                     f"평균익={wins.mean():+.2f} 평균손={losses.mean():+.2f} RR={rr:.2f}"
                     f" (설정 RR 2.5)")
        if len(fail):
            lines.append(f"  (SL실패 비상청산 별도: {len(fail)}건 {fail['pnl_usdt'].sum():+.1f})")

        # 청산 사유별
        lines.append("  --- 청산 사유별 ---")
        g = ex.groupby("event_type")["pnl_usdt"].agg(
            net="sum", n="count", avg="mean", wr=lambda x: (x > 0).mean() * 100)
        for k, r in g.sort_values("net").iterrows():
            lines.append(f"  {k:<12} net={r['net']:>+8.1f} n={int(r['n']):>3} "
                         f"avg={r['avg']:>+6.2f} 승률={r['wr']:.0f}%")

        # flip 세부 — 어느 TF flip 이 얼마나 자르나
        fl = ex[ex["event_type"] == "flip_close"].copy()
        if len(fl):
            fl["tf"] = fl["reason"].fillna("").apply(
                lambda s: (re.search(r"@(\w+)", s) or [None, "?"])[1])
            lines.append("  --- flip_close TF 별 (조기청산 주범 추적) ---")
            g2 = fl.groupby("tf")["pnl_usdt"].agg(
                net="sum", n="count", avg="mean", wr=lambda x: (x > 0).mean() * 100)
            for k, r in g2.sort_values("net").iterrows():
                lines.append(f"  flip@{k:<5} net={r['net']:>+8.1f} n={int(r['n']):>3} "
                             f"avg={r['avg']:>+6.2f} 승률={r['wr']:.0f}%")
            # 승자 절단 정량 — flip 승리건 평균익 vs sl_hit 평균손 (TP2.5R 대비)
            fw = fl[fl["pnl_usdt"] > 0]["pnl_usdt"]
            sl_avg = ex[ex["event_type"] == "sl_hit"]["pnl_usdt"].mean()
            if len(fw) and sl_avg < 0:
                lines.append(f"  flip 승리 {len(fw)}건 평균 +{fw.mean():.2f} — SL평균 "
                             f"{sl_avg:+.2f} 대비 실현 {fw.mean() / -sl_avg:.2f}R "
                             f"(TP설계 2.5R 의 {fw.mean() / -sl_avg / 2.5 * 100:.0f}%)")

        # 킬존별
        lines.append("  --- 킬존별 (KST) ---")
        ex["kz"] = ex["dt"].dt.hour.apply(kz_of)
        g3 = ex.groupby("kz")["pnl_usdt"].agg(
            net="sum", n="count", wr=lambda x: (x > 0).mean() * 100)
        for k, r in g3.sort_values("net").iterrows():
            lines.append(f"  {k:<8} net={r['net']:>+8.1f} n={int(r['n']):>3} 승률={r['wr']:.0f}%")

        # 페어별 상/하위
        g4 = ex.groupby("symbol")["pnl_usdt"].agg(net="sum", n="count")
        g4 = g4.sort_values("net")
        lines.append("  --- 페어 하위3 / 상위3 ---")
        for k, r in pd.concat([g4.head(3), g4.tail(3)]).iterrows():
            lines.append(f"  {k:<18} net={r['net']:>+8.1f} n={int(r['n']):>3}")

    txt = "\n".join(lines)
    with open("origo_live_calibration_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
