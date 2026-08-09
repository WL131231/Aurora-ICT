"""#AUTONOMOUS 2026-07-31: Cursus 개발자 변경사항 백테 — 하이켄아시 · 페어 · 노출.

개발자 전달(파트너 경유):
  ① 1시간봉 기준            → 현행과 동일 (변경 없음)
  ② 차트는 **하이켄아시**    → "눌림 가격대에 들어갈 수 있음, 지정가"
  ③ 종목 변경               → **TRX 필수 추가**, LINK 제외
  ④ 레버리지 **10배**        (현행 20)
  ⑤ 종목당 **10% 진입**      (현행 90%)

⚠️ 원본 `매매기법.py`(163줄) 확인 결과 하이켄아시·지정가·매수지점 코드가 **없다**.
   ST 는 일반 캔들 hl2 기준, `build_order_plan(side, entry, cfg)` 는 진입가를 인자로
   받아 진입가 결정이 원본 밖이다. 즉 이번 건은 원본에 없던 **신규 요구**다.
   지정가 가격 기준은 개발자 회신 대기 — 이 스크립트는 나머지만 측정한다.

★ 노출 축소(④⑤)는 **비용 비율을 바꾸지 않는다.**
  20×0.9=18배 → 10×0.1=1배 로 노출이 1/18 이 되지만 gross 와 비용이 같은 비율로
  줄어 gross/비용 = 0.149 가 그대로다(7/30 진단: gross +0.497%/거래 vs 비용 3.33%).
  net% 가 1/18 로 스케일될 뿐 적자 구조는 불변 → **지정가(maker)가 진짜 지렛대**다
  (taker 0.04%→maker 0.01% + 슬리피지 제거 = 비용 5~9배↓ → 본전권).

측정: 하이켄아시 ST 의 신호 품질 변화 + 페어 교체 효과. 라이브 정합 엔진
      (cursus_live_parity_bt: 고정SL 2% + 4분할TP + 래더 + REVERSE) 위에서 비교한다.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import cursus_live_parity_bt as M  # noqa: E402
import dst_trend_bt_clamped as DST  # noqa: E402

CUR_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
NEW_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "TRXUSDT", "HYPEUSDT"]


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """하이켄아시 캔들 변환.

    HA_close = (O+H+L+C)/4
    HA_open  = (직전 HA_open + 직전 HA_close)/2   (첫 봉은 (O+C)/2)
    HA_high  = max(H, HA_open, HA_close)
    HA_low   = min(L, HA_open, HA_close)

    평활 효과로 추세가 매끄러워져 노이즈 반전이 줄어든다 — 추세추종에 유리하다는
    것이 개발자 논지. 단 HA_open/close 는 **계산값이라 실제 체결 가격이 아니다**
    (지정가를 HA 값에 걸 때 주의 — 실제로 그 가격에 거래되지 않을 수 있다).
    """
    o = df["open"].to_numpy(dtype=float); h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float); c = df["close"].to_numpy(dtype=float)
    n = len(df)
    ha_c = (o + h + lo + c) / 4.0
    ha_o = np.empty(n)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([lo, ha_o, ha_c])
    out = pd.DataFrame({"open": ha_o, "high": ha_h, "low": ha_l, "close": ha_c},
                       index=df.index)
    if "volume" in df:
        out["volume"] = df["volume"].to_numpy()
    return out


def run_variant(pairs: list[str], use_ha: bool):
    """라이브 정합 엔진 + (선택)하이켄아시 신호. 반환 (합계 stat, 페어별 net)."""
    allt: list[tuple[float, int]] = []
    per: dict[str, float] = {}
    for sym in pairs:
        try:
            df = DST._load_1h(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym} 로드 실패: {e}", flush=True)
            continue
        # 신호는 HA, **체결·손익은 실제 캔들** — run_live_parity(sig_df=) 로 분리.
        tr, _ = M.run_live_parity(df, sig_df=heikin_ashi(df) if use_ha else None)
        allt.extend(tr)
        per[sym] = sum(n for n, _ in tr) * 100
    return M.stat(allt), per


def main() -> int:
    print("=== Cursus 개발자 변경사항 백테 (라이브 정합 엔진) ===", flush=True)
    print("  ※ 레버리지/사이즈 축소는 net% 를 1/18 로 스케일할 뿐 구조 불변 —", flush=True)
    print("     비용 비율(gross/비용 0.149)이 그대로라 여기서는 측정 제외.\n", flush=True)

    variants = [
        ("현행 (일반캔들·LINK)", CUR_PAIRS, False),
        ("HA 신호 (LINK 유지)", CUR_PAIRS, True),
        ("페어교체만 (TRX·LINK제외)", NEW_PAIRS, False),
        ("HA + 페어교체 (개발자안)", NEW_PAIRS, True),
    ]
    results = {}
    for label, pairs, ha in variants:
        print(f"[{label}] 실행...", flush=True)
        s, per = run_variant(pairs, ha)
        results[label] = (s, per)
        print(f"  {M.line(s)}", flush=True)

    print("\n\n===== 요약 =====", flush=True)
    base = results["현행 (일반캔들·LINK)"][0]
    for label, _, _ in variants:
        s = results[label][0]
        if s is None:
            continue
        be = (100 - s["wr"]) / max(s["wr"], 1e-9)
        d = "" if label.startswith("현행") else f"  현행대비 {s['net'] - base['net']:+.0f}%"
        print(f"  {label:<26} net={s['net']:+9.0f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
              f"분기={be:4.2f} {'흑자권' if s['rr'] > be else '적자'}{d}", flush=True)

    print("\n[페어별 — 개발자안]", flush=True)
    _, per = results["HA + 페어교체 (개발자안)"]
    for k, v in per.items():
        print(f"  {k.replace('USDT', ''):<6} {v:+9.0f}%", flush=True)
    print("\n[LINK vs TRX 직접 비교 (일반캔들 기준)]", flush=True)
    _, per_cur = results["현행 (일반캔들·LINK)"]
    _, per_new = results["페어교체만 (TRX·LINK제외)"]
    print(f"  LINK {per_cur.get('LINKUSDT', 0):+9.0f}%   TRX {per_new.get('TRXUSDT', 0):+9.0f}%",
          flush=True)

    print("\n\n⚠️ 지정가 진입은 개발자 회신 대기 — 이 결과는 **시장가 기준**이다.", flush=True)
    print("   비용의 대부분이 taker+슬리피지이므로 지정가 적용 시 결과가 크게 달라진다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
