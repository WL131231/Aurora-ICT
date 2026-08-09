"""킬존 연구 — 라이브 실거래(24h)의 진입 킬존별 net/승률. 어느 장에 봇이 강한지.

파트너(6/24): 킬존들 중 어느 장에 봇이 강한지. 라이브 봇은 24h 진입(disable_time_
filter)이라 실거래가 모든 킬존에 흩어짐 → setup_ts_ms(진입 직전 시각, UTC) 를
ICT 킬존(KST 기준 봇 UI)으로 분류해 net/승률/빈도 집계. 고래③ 런던집중 가설 검증.
"""
from __future__ import annotations

import pandas as pd

P = r"C:\Users\지영민\Downloads\trades_all_users (1).csv"
df = pd.read_csv(P)
df["pnl"] = pd.to_numeric(df["pnl_usdt"], errors="coerce")
# flip_close(작은 확정수익, 거의 100%승)·sync_close(복구) 제외 — 실제 진입매매
# (tp_hit/sl_hit)만 봐야 킬존 강점이 왜곡 없이 보임.
c = df[df["pnl"].notna() & (df["mode"] == "live") & (df["setup_ts_ms"] > 0)
       & df["event_type"].isin(["tp_hit", "sl_hit"])].copy()
c["hour"] = pd.to_datetime(c["setup_ts_ms"], unit="ms", utc=True).dt.hour


def kz(h: int) -> str:
    # 봇 UI 킬존(KST=UTC+9). London 16-18KST=07-09UTC, NY_AM 20-22=11-13,
    # LDN마감 23-01=14-16, NY_PM 02-05=17-20, Asian 08-12:50=23-03.
    if 7 <= h < 10:
        return "London(16-18KST)"
    if 11 <= h < 13:
        return "NY_AM(20-22KST)"
    if 14 <= h < 16:
        return "LDN_Close(23-01KST)"
    if 17 <= h < 21:
        return "NY_PM(02-05KST)"
    if h >= 22 or h < 4:
        return "Asian(07-12KST)"
    return "기타(장간)"


c["kz"] = c["hour"].apply(kz)
order = ["Asian(07-12KST)", "London(16-18KST)", "NY_AM(20-22KST)",
         "LDN_Close(23-01KST)", "NY_PM(02-05KST)", "기타(장간)"]


def _tbl(sub: pd.DataFrame, title: str) -> list[str]:
    lines = [f"\n=== {title} ===", f"  {'킬존':<20} {'net':>9} {'거래':>5} {'승률':>6} {'평균':>7}"]
    g = sub.groupby("kz")["pnl"].agg(["sum", "count", lambda x: (x > 0).mean() * 100, "mean"])
    g.columns = ["net", "n", "wr", "avg"]
    for k in order:
        if k in g.index:
            r = g.loc[k]
            lines.append(f"  {k:<20} {r['net']:+9.1f} {int(r['n']):5d} {r['wr']:5.0f}% {r['avg']:+7.2f}")
    return lines


lines = ["===== 킬존별 봇 강점 (라이브 실거래, setup_ts 진입 시각 기준) ====="]
lines += _tbl(c, "전체 live")
lines += _tbl(c[c["model"] == "Origo 1.1"], "Origo 1.1 (최근)")
# UTC hour별 세밀 (Origo 1.1, 실매매) — NY_PM(17-20UTC) 어느 시각 최악인지
oh = c[c["model"] == "Origo 1.1"].groupby("hour")["pnl"].agg(["sum", "count", lambda x: (x > 0).mean() * 100])
oh.columns = ["net", "n", "wr"]
lines.append("\n=== UTC hour별 (Origo 1.1 실매매, 표본작아 방향만) ===")
for h in range(24):
    if h in oh.index and oh.loc[h, "n"] >= 2:
        r = oh.loc[h]
        kst = (h + 9) % 24
        lines.append(f"  {h:02d}UTC({kst:02d}KST) net{r['net']:+7.1f} n{int(r['n']):3d} 승{r['wr']:3.0f}%")

lines.append("\n※ net 높고 승률 높은 킬존 = 봇이 강한 장. 고래③ 런던킬존 집중 가설 대조.")
lines.append("  시드 사용자별 달라 절대 net 보다 승률·평균·부호 위주 해석.")

txt = "\n".join(lines)
with open("kz_analysis_result.txt", "w", encoding="utf-8") as f:
    f.write(txt + "\n")
try:
    print(txt + "\nDONE")
except UnicodeEncodeError:
    print("(결과는 kz_analysis_result.txt)\nDONE")
