"""범용 백테 러너 — 설정만 바꿔 A/B + 검증 배터리를 한 번에 돌린다.

작성 의도 (2026-08-16): 지금까지 연구마다 일회용 스크립트를 새로 짰다(scripts/ 에
수십 개). 그 결과 ① 검증 배터리를 매번 다시 구현하고 ② 빼먹기도 하고
③ 사후필터 같은 함정을 각자 다시 밟았다. 이 러너 하나로 통일한다.

기준선은 **항상** live_parity.run_live_parity() — LIVE_BASE(라이브 강제설정)에서
출발한다. 연구자는 바꿀 필드만 --set 으로 얹는다.

## 사용법

    # 1) 단일 설정 변경 A/B (BTC+ETH 본표본)
    python scripts/bt_run.py --set turtle_soup_enabled=false

    # 2) 홀드아웃까지 (본표본 통과 시에만 의미 있음 — 홀드아웃은 1회용이다)
    python scripts/bt_run.py --set turtle_soup_enabled=false --holdout

    # 3) 여러 필드 동시 + 라벨
    python scripts/bt_run.py --set min_rr=1.5 --set fvg_body_mult=2.0 --label "정통FVG"

    # 4) 페어 직접 지정 / 병렬도 조절
    python scripts/bt_run.py --set killzone_preset=crypto --pairs BTCUSDT,SOLUSDT --nproc 4

결과는 data/axis/run_<label>.json 에 저장되고, 같은 라벨로 다시 돌리면 재사용한다
(--fresh 로 무시).

## 판정 4관문 (all_sources 규약 계승)

    ① 95% 구간이 0 초과      ② 심볼 일관성 (과반 이상 같은 부호)
    ③ 롱/숏 양쪽 개선        ④ 순열검정 p < 0.05

넷 다 통과한 것만 홀드아웃으로 넘긴다. 홀드아웃도 통과해야 배포 후보다.

## 주의 (밟았던 함정)

- **사후필터 금지**: "거래를 낸 뒤 걸러내기"는 라이브가 거르는 셋업이 백테에선
  포지션 슬롯을 먹어 뒤 셋업을 가리는 인공물을 만든다. 반드시 cfg 로 끄고 재실행.
- **캐시 키**: detect 인자를 새로 추가하면 bt_par.cached_setup_timeline 의 키와
  replay._detect_params 양쪽에 넣어야 한다. 안 넣으면 "바꿨는데 안 바뀐" 결과가
  나온다. 기본값일 때는 키를 넣지 않는 규약(기존 캐시 보존).
- **레버리지**: LIVE_BASE 는 leverage=7. net% 는 레버 반영 시드 대비다. R 은
  raw 기준이라 레버와 무관 — 둘을 섞지 말 것.
- **홀드아웃은 1회용**: 같은 페어로 두 번 재면 그건 더 이상 홀드아웃이 아니다.

담당: 연구 공용 (장수 supervision)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# 콘솔 코드페이지가 cp949 면 em dash 같은 문자에서 UnicodeEncodeError 로 죽는다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 경로 순서가 중요하다. 이 저장소는 봇 본체(Desktop/Aurora-ICT)의 **git 워크트리**라
# 같은 이름의 패키지가 두 벌 있다. src 를 먼저 넣지 않으면 봇 본체의 aurora_ict 가
# 잡혀서 **조용히 다른 코드가 돌아간다**(연구용 함수가 없으면 ImportError 로 터지지만,
# 이름만 같고 내용이 다른 경우엔 아무 경고 없이 틀린 결과가 나온다).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from live_parity import run_live_parity  # noqa: E402

# 본표본 / 홀드아웃 분리 — 홀드아웃은 탐색에 쓰지 않은 페어여야 한다.
MAIN_PAIRS = ["BTCUSDT", "ETHUSDT"]
HOLDOUT_PAIRS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]

N_BOOT = 20000
N_PERM = 20000
MIN_N = 30
RNG = np.random.default_rng(20260816)


def parse_value(s: str):
    """--set KEY=VALUE 의 VALUE 를 파이썬 값으로. bool/int/float/None/str 순 시도."""
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def r_of(t) -> float:
    """R 환산 = raw_pnl_pct / 진입~손절 거리.

    ⚠️ 단위 함정: 이름이 `_pct` 지만 raw_pnl_pct / net_pnl_pct 는 **비율**이다
    (0.76 = 76%). 그래서 손절 거리도 비율로 맞춰야 한다. 여기서 `*100` 을 하면
    R 이 100배 작게 나온다(8/16에 실제로 밟음 — 건당 +0.001R 로 찍혀서 0 처럼 보였다).

    net_pnl_pct 는 레버(7배)·포지션 크기가 반영된 시드 대비라 R 과 다른 지표다.
    포지션 크기가 confluence 에 따라 달라지므로 **R 평균과 net 합의 방향이 갈릴 수
    있다** — 둘 다 보고할 것.
    """
    if not t.entry or not t.entry_sl:
        return 0.0
    risk = abs(t.entry - t.entry_sl) / t.entry
    return (t.raw_pnl_pct / risk) if risk > 0 else 0.0


def collect(pairs: list[str], extra: dict) -> dict[str, list[dict]]:
    """페어별로 백테를 돌려 거래 행을 모은다."""
    out: dict[str, list[dict]] = {}
    for sym in pairs:
        t0 = time.time()
        print(f"    {sym} …", end="", flush=True)
        _, trades, _ = run_live_parity(sym, extra or None)
        out[sym] = [
            {
                "r": r_of(t),
                "net": float(t.net_pnl_pct or 0),
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "conf": list(getattr(t, "confluences", ()) or ()),
            }
            for t in trades
        ]
        print(f" {len(out[sym])}건 ({time.time() - t0:.0f}초)", flush=True)
    return out


def ci(r: np.ndarray) -> tuple[float, float]:
    """부트스트랩 95% 신뢰구간."""
    if len(r) < MIN_N:
        return (float("nan"), float("nan"))
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def perm_p(base: np.ndarray, var: np.ndarray) -> float:
    """순열검정 — var 가 base 보다 나은 게 우연일 확률 (한쪽 꼬리).

    두 집합을 합쳐 무작위로 다시 가른 뒤 관측 차이 이상이 나오는 비율.
    """
    if len(base) < MIN_N or len(var) < MIN_N:
        return float("nan")
    obs = var.mean() - base.mean()
    both = np.concatenate([base, var])
    nb = len(base)
    cnt = 0
    for _ in range(N_PERM):
        p = RNG.permutation(both)
        if (p[nb:].mean() - p[:nb].mean()) >= obs:
            cnt += 1
    return cnt / N_PERM


def flat(rows: dict[str, list[dict]], key="r", where=None) -> np.ndarray:
    vals = [x[key] for sym in rows for x in rows[sym] if (where is None or where(x))]
    return np.array(vals, dtype=float)


def report(base_rows: dict, var_rows: dict, label: str, pairs: list[str]) -> dict:
    """A/B 리포트 + 4관문 판정. 반환 = 판정 dict."""
    b, v = flat(base_rows), flat(var_rows)
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")
    print(f"  {'':>10s} {'거래':>7s} {'승률':>7s} {'건당R':>9s} {'95% 구간':>22s} {'net합%':>10s}")
    for tag, rows, arr in (("기준선", base_rows, b), ("변형", var_rows, v)):
        lo, hi = ci(arr)
        # net_pnl_pct 는 **비율**(0.0053 = 0.53%)이라 표시할 때 100을 곱한다.
        net = sum(x["net"] for sym in rows for x in rows[sym]) * 100
        print(f"  {tag:>10s} {len(arr):7,d} {(arr > 0).mean() * 100:6.1f}% "
              f"{arr.mean():+8.3f}R   [{lo:+.3f} ~ {hi:+.3f}] {net:+9.2f}%")

    diff = v.mean() - b.mean()
    p = perm_p(b, v)
    lo, hi = ci(v)
    print(f"\n  차이 {diff:+.3f}R   순열 p={p:.4f}")

    # ① 구간 0 초과
    g1 = bool(np.isfinite(lo) and lo > 0)
    # ② 심볼 일관성 — 과반 이상에서 개선
    sym_ok = 0
    print(f"\n  [심볼별]")
    for sym in pairs:
        a = np.array([x["r"] for x in base_rows.get(sym, [])], dtype=float)
        c = np.array([x["r"] for x in var_rows.get(sym, [])], dtype=float)
        if len(a) == 0 or len(c) == 0:
            continue
        better = c.mean() > a.mean()
        sym_ok += better
        print(f"    {sym:>10s}  기준 {len(a):4d}건 {a.mean():+.3f}R  →  변형 "
              f"{len(c):4d}건 {c.mean():+.3f}R  {'개선' if better else '악화'}")
    g2 = sym_ok > len(pairs) / 2
    # ③ 롱숏 양쪽
    print(f"\n  [롱/숏 분리]")
    dir_ok = 0
    for d in ("long", "short"):
        a = flat(base_rows, where=lambda x: x["dir"] == d)
        c = flat(var_rows, where=lambda x: x["dir"] == d)
        if len(a) == 0 or len(c) == 0:
            continue
        better = c.mean() > a.mean()
        dir_ok += better
        print(f"    {d:>6s}  기준 {len(a):4d}건 {a.mean():+.3f}R  →  변형 "
              f"{len(c):4d}건 {c.mean():+.3f}R  {'개선' if better else '악화'}")
    g3 = dir_ok == 2
    # ④ 순열 p
    g4 = bool(np.isfinite(p) and p < 0.05)

    print(f"\n  [4관문]  ①구간0초과 {'O' if g1 else 'X'}   ②심볼일관 {'O' if g2 else 'X'}"
          f"   ③롱숏양쪽 {'O' if g3 else 'X'}   ④순열p<0.05 {'O' if g4 else 'X'}")
    verdict = all((g1, g2, g3, g4))
    print(f"  ▶ {'통과 — 홀드아웃으로' if verdict else '탈락'}")
    return {"diff": diff, "p": p, "gates": [g1, g2, g3, g4], "pass": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aurora-ICT 범용 백테 러너")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="cfg 필드 덮어쓰기 (여러 번 가능)")
    ap.add_argument("--pairs", default=None, help="쉼표 구분. 기본 BTCUSDT,ETHUSDT")
    ap.add_argument("--holdout", action="store_true", help="홀드아웃 페어로도 검증")
    ap.add_argument("--label", default=None, help="결과 파일/리포트 라벨")
    ap.add_argument("--nproc", type=int, default=1, help="(예약) 병렬 프로세스 수")
    ap.add_argument("--fresh", action="store_true", help="저장된 결과 무시하고 재실행")
    args = ap.parse_args()

    if not args.set:
        ap.error("--set 이 최소 하나 필요하다 (예: --set turtle_soup_enabled=false)")

    extra = {}
    for kv in args.set:
        if "=" not in kv:
            ap.error(f"--set 형식은 KEY=VALUE (받은 값: {kv})")
        k, v = kv.split("=", 1)
        extra[k.strip()] = parse_value(v.strip())

    label = args.label or ",".join(f"{k}={v}" for k, v in extra.items())
    pairs = args.pairs.split(",") if args.pairs else MAIN_PAIRS
    out_path = f"data/axis/run_{label.replace('/', '_').replace(' ', '_')}.json"

    print(f"\n[bt_run] 변형: {extra}")
    print(f"[bt_run] 본표본: {pairs}")

    if os.path.exists(out_path) and not args.fresh:
        print(f"[bt_run] 저장된 결과 재사용: {out_path}  (--fresh 로 무시)")
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("\n  [기준선] LIVE_BASE …")
        base_rows = collect(pairs, {})
        print("\n  [변형] …")
        var_rows = collect(pairs, extra)
        data = {"extra": {k: str(v) for k, v in extra.items()},
                "pairs": pairs, "base": base_rows, "var": var_rows}
        os.makedirs("data/axis", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"\n[bt_run] 저장: {out_path}")

    res = report(data["base"], data["var"], f"본표본 — {label}", data["pairs"])

    if args.holdout:
        if not res["pass"]:
            print("\n[bt_run] 본표본 탈락 — 홀드아웃 생략 (홀드아웃은 1회용이다)")
            return
        print(f"\n\n  [홀드아웃] {HOLDOUT_PAIRS}")
        ho_path = out_path.replace(".json", "_holdout.json")
        if os.path.exists(ho_path) and not args.fresh:
            with open(ho_path, encoding="utf-8") as f:
                hd = json.load(f)
        else:
            hb = collect(HOLDOUT_PAIRS, {})
            hv = collect(HOLDOUT_PAIRS, extra)
            hd = {"base": hb, "var": hv, "pairs": HOLDOUT_PAIRS}
            with open(ho_path, "w", encoding="utf-8") as f:
                json.dump(hd, f)
        report(hd["base"], hd["var"], f"홀드아웃 — {label}", hd["pairs"])


if __name__ == "__main__":
    main()
