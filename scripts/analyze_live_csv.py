"""실제 라이브 거래 CSV 진단 — 페어/방향/이벤트별 net·승률 + entry_trend 역산 가능성.

파트너(6/23): 배포 보류하고 실 거래 데이터로 횡보 임계 학습/검증. 먼저 실제 성과
파악 + 진입 시점(setup_ts) OHLCV 역산 가능 범위 확인.
"""
from __future__ import annotations

import pandas as pd

P = r"C:\Users\지영민\Downloads\trades_all_users (1).csv"
df = pd.read_csv(P)
df["pnl_usdt"] = pd.to_numeric(df["pnl_usdt"], errors="coerce")

closed = df[df["pnl_usdt"].notna() & (df["mode"] == "live")].copy()
lines = []
lines.append("===== 실제 라이브 거래 진단 =====")
lines.append(f"전체 live 실현손익: net={closed['pnl_usdt'].sum():+.1f} USDT  거래={len(closed)}  "
             f"승률={(closed['pnl_usdt'] > 0).mean() * 100:.0f}%")

o = closed[closed["model"] == "Origo 1.1"]
old = closed[closed["model"] != "Origo 1.1"]
lines.append(f"[Origo 1.1] net={o['pnl_usdt'].sum():+.1f}  거래={len(o)}  승률={(o['pnl_usdt'] > 0).mean() * 100:.0f}%")
lines.append(f"[이전버전]  net={old['pnl_usdt'].sum():+.1f}  거래={len(old)}  승률={(old['pnl_usdt'] > 0).mean() * 100:.0f}%")

lines.append("\n=== 이벤트별 (Origo 1.1) ===")
ev = o.groupby("event_type")["pnl_usdt"].agg(["sum", "count", lambda x: (x > 0).mean() * 100])
ev.columns = ["net", "건수", "승률%"]
lines.append(ev.round(1).to_string())

lines.append("\n=== 페어별 net (Origo 1.1, 시드 달라 절대값보다 부호/승률 위주) ===")
sym = o.groupby("symbol")["pnl_usdt"].agg(["sum", "count", lambda x: (x > 0).mean() * 100])
sym.columns = ["net", "건수", "승률%"]
lines.append(sym.sort_values("net").round(1).to_string())

lines.append("\n=== 방향별 (Origo 1.1) ===")
d = o.groupby("direction")["pnl_usdt"].agg(["sum", "count", lambda x: (x > 0).mean() * 100])
d.columns = ["net", "건수", "승률%"]
lines.append(d.round(1).to_string())

# sl_hit 만 (손실 패턴)
sl = o[o["event_type"] == "sl_hit"]
lines.append(f"\n=== 손절(sl_hit) {len(sl)}건 페어별 ===")
lines.append(sl.groupby("symbol")["pnl_usdt"].agg(["sum", "count"]).round(1).to_string())

# entry 시점 범위 (역산 가능성)
ent = df[(df["event_type"] == "entry") & (df["mode"] == "live")].copy()
ent["sts"] = pd.to_datetime(ent["setup_ts_ms"], unit="ms")
lines.append(f"\n=== entry 역산 가능성 ===")
lines.append(f"entry 건수={len(ent)}  페어수={ent['symbol'].nunique()}")
lines.append(f"setup_ts 범위: {ent['sts'].min()} ~ {ent['sts'].max()}")
lines.append(f"페어별 entry 건수:\n{ent['symbol'].value_counts().to_string()}")

txt = "\n".join(lines)
with open("analyze_live_csv_result.txt", "w", encoding="utf-8") as f:
    f.write(txt + "\n")
try:
    print(txt + "\nDONE")
except UnicodeEncodeError:
    print("(결과는 analyze_live_csv_result.txt)\nDONE")
