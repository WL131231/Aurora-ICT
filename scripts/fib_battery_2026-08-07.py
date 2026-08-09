"""
피보나치 되돌림 진입 깊이(ote_level) 레벨 간 차이 검증 배터리 — 2026-08-07

무엇을 재나:
  data/fib/lv{level}.json (Backtest 단계 산출물)에 저장된 레벨별 거래를 읽어,
  현행 라이브 기본값 0.707 과 대안 레벨(0.500 / 0.618 / 0.786 / 0.886)의
  건당 R 배수 차이가 통계적으로 구분 가능한지 판정한다.

왜 재나:
  레벨을 바꾸면 SL 폭이 같이 바뀌므로 %수익 비교는 왜곡된다(팀 표준 4번).
  또한 레벨 스윕은 같은 원본 셋업을 조금씩 다른 진입가로 재체결한 것이라
  "레벨이 좋아서"가 아니라 "표본이 흔들려서" 순위가 뒤집힐 수 있다.
  그래서 아래 4종을 전부 돌린다.

측정 항목:
  1) 순열검정 (20000회, 팀 표준 1번)
     - 비대응(unpaired): 두 레벨 거래를 합쳐 원래 크기로 무작위 분할 → 건당R 차이 분포
     - 대응(paired): (sym, ts)가 같은 거래만 짝지어 R 차이의 부호를 무작위로 뒤집음
       레벨 스윕은 같은 셋업을 공유하므로 대응 검정이 검출력이 훨씬 높다.
       두 검정 결과를 병기해 어느 쪽으로도 p<0.05 를 못 넘으면 기각.
  2) 연도별 일관성 (팀 표준 3번) — 최선 레벨이 매년 최선인지, 승리 연도 수로 집계
  3) 롱/숏 분리 (2026-07-29 검증 필수 3종) — 방향별 최적 레벨이 갈리는지
  4) 페어별 (BTC/ETH) — 두 페어에서 결론이 같은지

표본 규칙(팀 표준 5번): 셀 건수 < 30 이면 "표본부족"으로 표시하고 결론을 내지 않는다.
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

LEVELS = ["0.500", "0.618", "0.707", "0.786", "0.886"]
BASE = "0.707"          # 현행 라이브 기본값
N_PERM = 20000          # 순열 반복 횟수 (팀 표준)
MIN_N = 30              # 표본 하한
SEED = 20260807

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fib")


def load():
    """레벨별 거래 로드. 각 거래에 연도(year) 필드를 붙인다."""
    out = {}
    for lv in LEVELS:
        with open(os.path.join(DATA_DIR, "lv%s.json" % lv), "r", encoding="utf-8") as f:
            d = json.load(f)
        for t in d["trades"]:
            t["year"] = datetime.fromtimestamp(t["ts"] / 1000, tz=timezone.utc).year
        out[lv] = d["trades"]
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=3):
    return "표본0" if x != x else ("%+.*f" % (nd, x))


# ---------------------------------------------------------------- 순열검정
def perm_unpaired(a, b, rng, n=N_PERM):
    """비대응 순열검정: a,b 를 합쳐 원래 크기로 무작위 분할.
    관측 |평균차| 이상이 나오는 비율 = 양측 p값."""
    obs = mean(a) - mean(b)
    pool = list(a) + list(b)
    na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = mean(pool[:na]) - mean(pool[na:])
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def perm_paired(diffs, rng, n=N_PERM):
    """대응 순열검정(부호 뒤집기): 같은 셋업 짝의 R 차이 부호를 무작위로 반전.
    귀무가설 = 레벨 차이가 무의미(차이 분포가 0 대칭)."""
    if not diffs:
        return float("nan"), float("nan")
    obs = mean(diffs)
    cnt = 0
    for _ in range(n):
        s = mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(s) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def pair_up(ta, tb):
    """(sym, ts) 키가 같은 거래만 짝지어 R 차이 리스트 반환."""
    ma = {(t["sym"], t["ts"]): t["r"] for t in ta}
    mb = {(t["sym"], t["ts"]): t["r"] for t in tb}
    keys = sorted(set(ma) & set(mb))
    return [ma[k] - mb[k] for k in keys], len(keys)


# ---------------------------------------------------------------- 리포트
def line(*cols):
    print(" | ".join(cols))


def main():
    rng = random.Random(SEED)
    tr = load()

    print("=" * 96)
    print("[0] 레벨별 전체 요약 (R 배수 기준)")
    print("=" * 96)
    line("레벨".ljust(7), "건수".rjust(4), "건당R".rjust(8), "총R".rjust(8),
         "승률".rjust(6), "표본판정")
    for lv in LEVELS:
        rs = [t["r"] for t in tr[lv]]
        wr = sum(1 for r in rs if r > 0) / len(rs) * 100
        line(lv.ljust(7), str(len(rs)).rjust(4), fmt(mean(rs)).rjust(8),
             fmt(sum(rs), 2).rjust(8), ("%.1f%%" % wr).rjust(6),
             "OK" if len(rs) >= MIN_N else "표본부족(<%d)" % MIN_N)

    # ---------------- 1. 순열검정
    print()
    print("=" * 96)
    print("[1] 순열검정 — 현행 %s vs 각 대안 (%d회, 양측)" % (BASE, N_PERM))
    print("    비대응 = 합쳐서 무작위 분할 / 대응 = 공통 셋업 짝의 R차 부호 뒤집기")
    print("=" * 96)
    line("대안".ljust(7), "n(대안)".rjust(7), "건당R차(대안-0.707)".rjust(19),
         "p_비대응".rjust(9), "짝수".rjust(5), "짝평균R차".rjust(10), "p_대응".rjust(8), "판정")
    base_r = [t["r"] for t in tr[BASE]]
    for lv in LEVELS:
        if lv == BASE:
            continue
        alt_r = [t["r"] for t in tr[lv]]
        obs_u, p_u = perm_unpaired(alt_r, base_r, rng)
        diffs, npair = pair_up(tr[lv], tr[BASE])
        obs_p, p_p = perm_paired(diffs, rng)
        sig = (p_u < 0.05) or (p_p == p_p and p_p < 0.05)
        verdict = "구분됨" if sig else "기각(노이즈와 구분 불가)"
        line(lv.ljust(7), str(len(alt_r)).rjust(7), fmt(obs_u).rjust(19),
             ("%.4f" % p_u).rjust(9), str(npair).rjust(5),
             fmt(obs_p).rjust(10), ("%.4f" % p_p).rjust(8), verdict)

    # ---------------- 2. 연도별
    print()
    print("=" * 96)
    print("[2] 연도별 건당R — 최선 레벨이 매년 최선인가 (팀 표준 3번)")
    print("=" * 96)
    years = sorted({t["year"] for lv in LEVELS for t in tr[lv]})
    line("연도".ljust(6), *[lv.rjust(9) for lv in LEVELS],
         "  최선", "  건수(0.707)")
    win = defaultdict(int)
    for y in years:
        cells = []
        best, bestv = None, None
        for lv in LEVELS:
            rs = [t["r"] for t in tr[lv] if t["year"] == y]
            m = mean(rs)
            cells.append(("%s(%d)" % (fmt(m, 2), len(rs))).rjust(9))
            if rs and (bestv is None or m > bestv):
                best, bestv = lv, m
        if best:
            win[best] += 1
        nb = len([t for t in tr[BASE] if t["year"] == y])
        line(str(y).ljust(6), *cells, ("  " + (best or "-")).ljust(8),
             "  %d %s" % (nb, "표본부족" if nb < MIN_N else ""))
    print()
    print("  연도 승리 횟수: " + ", ".join("%s=%d/%d" % (lv, win[lv], len(years)) for lv in LEVELS))
    print("  판정: 특정 레벨이 %d개 연도 중 과반 승리 못 하면 연도 일관성 없음" % len(years))

    # ---------------- 3. 롱/숏
    print()
    print("=" * 96)
    print("[3] 롱/숏 분리 — 방향별 최적 레벨 (2026-07-29 검증 필수 3종)")
    print("=" * 96)
    for d in ("long", "short"):
        line(("[%s]" % d).ljust(8), *[lv.rjust(11) for lv in LEVELS])
        cells, best, bestv = [], None, None
        for lv in LEVELS:
            rs = [t["r"] for t in tr[lv] if t["dir"] == d]
            m = mean(rs)
            cells.append(("%s(n%d)" % (fmt(m, 2), len(rs))).rjust(11))
            if rs and (bestv is None or m > bestv):
                best, bestv = lv, m
        line("건당R".ljust(8), *cells)
        nb = len([t for t in tr[BASE] if t["dir"] == d])
        print("   최선=%s / 0.707 표본 n=%d → %s" %
              (best, nb, "OK" if nb >= MIN_N else "표본부족(<%d), 결론 보류" % MIN_N))
        # 방향 내 대응 순열검정 (현행 vs 최선)
        if best and best != BASE:
            ma = {(t["sym"], t["ts"]): t["r"] for t in tr[best] if t["dir"] == d}
            mb = {(t["sym"], t["ts"]): t["r"] for t in tr[BASE] if t["dir"] == d}
            ks = sorted(set(ma) & set(mb))
            obs, p = perm_paired([ma[k] - mb[k] for k in ks], rng)
            print("   대응 순열검정 %s vs %s: 짝%d, 평균R차 %s, p=%.4f → %s" %
                  (best, BASE, len(ks), fmt(obs), p,
                   "구분됨" if p < 0.05 else "기각"))
        print()

    # ---------------- 4. 페어별
    print("=" * 96)
    print("[4] 페어별 — BTC 와 ETH 에서 결론이 같은가")
    print("=" * 96)
    for sym in ("BTCUSDT", "ETHUSDT"):
        line(("[%s]" % sym).ljust(10), *[lv.rjust(11) for lv in LEVELS])
        cells, best, bestv = [], None, None
        for lv in LEVELS:
            rs = [t["r"] for t in tr[lv] if t["sym"] == sym]
            m = mean(rs)
            cells.append(("%s(n%d)" % (fmt(m, 2), len(rs))).rjust(11))
            if rs and (bestv is None or m > bestv):
                best, bestv = lv, m
        line("건당R".ljust(10), *cells)
        nb = len([t for t in tr[BASE] if t["sym"] == sym])
        print("   최선=%s / 0.707 표본 n=%d → %s" %
              (best, nb, "OK" if nb >= MIN_N else "표본부족(<%d), 결론 보류" % MIN_N))
        if best and best != BASE:
            ma = {(t["sym"], t["ts"]): t["r"] for t in tr[best] if t["sym"] == sym}
            mb = {(t["sym"], t["ts"]): t["r"] for t in tr[BASE] if t["sym"] == sym}
            ks = sorted(set(ma) & set(mb))
            obs, p = perm_paired([ma[k] - mb[k] for k in ks], rng)
            print("   대응 순열검정 %s vs %s: 짝%d, 평균R차 %s, p=%.4f → %s" %
                  (best, BASE, len(ks), fmt(obs), p, "구분됨" if p < 0.05 else "기각"))
        print()

    # ---------------- 보조 A: 레벨 간 셋업 중복도
    print("=" * 96)
    print("[보조 A] 레벨 간 셋업 중복도 — 레벨 스윕이 독립 표본이 아님을 보이는 지표")
    print("=" * 96)
    bs = {(t["sym"], t["ts"]) for t in tr[BASE]}
    for lv in LEVELS:
        s = {(t["sym"], t["ts"]) for t in tr[lv]}
        print("  0.707(n%d) ∩ %s(n%d) = %d 건 (%.0f%% 공유)" %
              (len(bs), lv, len(s), len(bs & s), len(bs & s) / len(bs) * 100))

    # ---------------- 보조 B: 대응 순열검정 p값 해부
    # Why: [1]에서 대응 p가 0.000x 로 나왔는데 평균 R차는 0.01~0.07 에 불과했다.
    #      공통 셋업 대부분이 "같은 outcome, 진입가만 미세하게 다름"이라 R차가 거의 0에
    #      몰려 있고, 그 미세차의 부호만 일정하면 부호뒤집기 검정은 자동으로 유의해진다.
    #      즉 이 p값은 "레벨을 바꾸면 돈이 더 벌린다"가 아니라 "산술적으로 R이 조금 달라진다"를
    #      검출한 것. 이걸 수치로 드러내기 위해 결과가 갈린 짝(outcome flip)만 따로 본다.
    print()
    print("=" * 96)
    print("[보조 B] 대응 순열검정 해부 — p값이 경제적 우위인지 산술 잡음인지 분리")
    print("=" * 96)
    out_base = {(t["sym"], t["ts"]): t["outcome"] for t in tr[BASE]}
    r_base = {(t["sym"], t["ts"]): t["r"] for t in tr[BASE]}
    line("대안".ljust(7), "짝수".rjust(4), "결과동일".rjust(6), "|R차|<0.05".rjust(9),
         "동일짝 평균R차".rjust(13), "flip짝수".rjust(7), "flip 평균R차".rjust(11), "flip p")
    for lv in LEVELS:
        if lv == BASE:
            continue
        ma = {(t["sym"], t["ts"]): t for t in tr[lv]}
        ks = sorted(set(ma) & set(r_base))
        same = [ma[k]["r"] - r_base[k] for k in ks if ma[k]["outcome"] == out_base[k]]
        flip = [ma[k]["r"] - r_base[k] for k in ks if ma[k]["outcome"] != out_base[k]]
        tiny = sum(1 for k in ks if abs(ma[k]["r"] - r_base[k]) < 0.05)
        _, p_flip = perm_paired(flip, rng) if flip else (float("nan"), float("nan"))
        line(lv.ljust(7), str(len(ks)).rjust(4), str(len(same)).rjust(6),
             str(tiny).rjust(9), fmt(mean(same)).rjust(13),
             str(len(flip)).rjust(7), fmt(mean(flip)).rjust(11),
             ("%.4f" % p_flip) if p_flip == p_flip else "n/a")
    print()
    print("  해석 규칙: '결과동일' 짝은 진입가만 미세하게 다른 사실상 같은 거래다.")
    print("  손익을 실제로 바꾸는 것은 flip 짝뿐이며, flip 짝수가 30 미만이면 표본부족.")


if __name__ == "__main__":
    main()
