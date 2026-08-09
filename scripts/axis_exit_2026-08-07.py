"""Origo 청산 로직 축(트레일·본전이동·부분익절·HTF FVG flip) 검증 — 2026-08-07.

## 무엇을 재나
현행 라이브 청산 설정(트레일 2.0R 발동/1.5R 간격 · 본전이동 1.0R ·
부분익절 1.5R 50% · flip 최소 1.5R)을 기준선으로 잡고, 각 손잡이를 단독으로
움직였을 때 **건당 R 배수**가 얼마나 달라지는지 잰다. 유망 조합은 교차 검증하고
순열검정·연도 일관성·국면×방향 통제·복리 시뮬까지 통과해야 후보로 남긴다.

## 왜 이 순서인가 (직전 피보 연구의 교훈)
2026-08-07 오전에 종결된 ote_level(피보 되돌림) 연구는 **손잡이의 물리적 상한이
건당 0.027R** 이라 스윕 자체가 무의미했다. FVG 갭이 가격의 0.123% 뿐인데 SL 은
1.5×ATR 바닥 때문에 1.9% 로 고정돼, 레벨을 끝까지 밀어도 진입가가 0.05% 밖에
안 움직였기 때문이다.

그래서 [1단계]에서 **청산 손잡이의 구조적 상한**을 먼저 산출한다.
청산 파라미터는 전부 risk0=|진입가-SL| 의 배수로 동작하므로(replay._simulate_exit)
피보와 달리 R 을 직접 움직인다. 상한 = "모든 거래를 최대유리폭(MFE)에서
청산했을 때의 건당 R". 이 상한이 표본 노이즈(건당 R 의 표준오차)보다 작으면
거기서 멈추고 그 사실만 보고한다.

## 하니스
live_parity.run_live_parity 만 쓴다(2026-07-30 규칙). 다만 같은 설정을 수십 번
돌리므로 _load_full/_resample/cached_setup_timeline 세 함수를 **프로세스 내
메모이즈**만 한다(계산 내용은 동일). [0단계]에서 메모이즈 결과가 원본과
일치하는지 assert 로 확인한다.

## 측정 단위
- R = (raw_pnl_pct - 2×taker수수료) / (|진입가-SL|/진입가)
  → SL 폭이 변형마다 달라져도 비교 가능(팀 표준 4번).
- 복리 판정은 별도로 raw_pnl_pct 에 레버리지 7x·size 0.9·동시보유 분할·
  DD 스로틀(-25%→×0.7)·파산(시드 20%)을 적용해서 낸다.

## 파라미터별 구현 경로
- trail_trigger / trail_dist / be_trigger / partial_tp_rr
    → BacktestConfig 필드. 캐시 키에 없으므로 재생만 다시 돌면 된다(페어 7개 4초).
- 부분익절 **비율**
    → replay._exit 가 0.5 로 하드코딩돼 있어 config 로는 못 바꾼다. 다만
      partial_be=True 라 잔여분 경로(청산 시점·가격)가 비율과 무관하므로,
      f=0.5 결과에서 잔여 청산가를 역산해 임의 비율 f 로 **정확히** 재가중할 수
      있다. 슬리피지는 곱셈이라 역산 가능. [2-E]에서 f=0.5 재구성이 원본과
      1e-12 이내로 일치하는지 검증한 뒤 사용한다.
- flip 최소 R
    → replay 에 flip 경로가 없어 flip_verdict.apply_flip 과 같은 방식으로 사후
      적용한다. 단 R 비교를 위해 net% 대신 raw 를 반환하도록 다시 구현했고,
      flip 청산 시점(봉 인덱스)도 함께 돌려받아 복리 시뮬의 동시보유 계산에 쓴다.

## 판정 게이트 (하나라도 못 넘기면 기각)
1. 순열검정 p<0.05 (대응 20000회 + 비대응 20000회 병기)
2. 국면×방향 기저 통제 — 셀별 ΔR 부호
3. 연도 일관성 — 특정 연도 몰빵이면 기각
4. R 배수로 측정
5. n<30 셀은 표본부족 표기 후 결론 제외
6. 다중비교 — 시도한 조합 수를 전부 집계해 출력
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import live_parity as LP  # noqa: E402
from flip_ab_backtest import build_fvg_zones, flip_target_at  # noqa: E402

from aurora.backtest.cost import (  # noqa: E402
    SLIP_NORMAL_PCT, SLIP_VOLATILE_PCT, TAKER_FEE_PCT, VOLATILE_THRESHOLD,
)

# ────────────────────────────────────────────────────────────── 상수
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "axis")
OUT_PATH = os.path.join(OUT_DIR, "exit.json")

PAIRS = LP.PAIRS                      # 7페어 — 통계 검정력용
LIVE_PAIRS = ["BTCUSDT", "ETHUSDT"]   # 2026-08-07 실제 라이브 편성

SEED = 20260807
N_PERM = 20000        # 순열검정 반복 (팀 표준 1번)
N_BOOT = 2000         # 복리 부트스트랩
MIN_N = 30            # 표본 하한 (팀 표준 5번)

LEV = 7.0             # 라이브 레버리지
SIZE = 0.9            # 라이브 size_pct
RUIN = 0.20           # 파산 판정 = 시드의 20%
DD_PCT, DD_FACTOR = 0.25, 0.7

FLIP_MIN_W = 4        # 라이브 정합 (1h 이상 HTF FVG)
FLIP_OFF = 999.0      # flip 을 아예 끄는 센티넬

# 현행 라이브 청산 설정 = 기준선
BASE_EXIT = dict(trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, partial_tp_rr=1.5)
BASE_FLIP_R = LP.LIVE_FLIP_MIN_R      # 1.5
BASE_PFRAC = 0.5                      # replay 하드코딩

RNG = np.random.default_rng(SEED)

# 시도한 설정 수 집계 (다중비교 명시용)
TRIED: dict[str, int] = {"백테재생": 0, "청산평가": 0}


# ────────────────────────────────────────────── [0] 메모이즈 (계산 동일, 재사용만)
def _install_memo() -> None:
    """live_parity 내부 3함수를 프로세스 캐시로 감싼다. 결과는 원본과 동일."""
    _df: dict = {}
    _o_load = LP._load_full

    def load(sym):
        if sym not in _df:
            _df[sym] = _o_load(sym)
        return _df[sym]

    _rs: dict = {}
    _o_res = LP._resample

    def resample(d, rule="5min"):
        k = (id(d), rule)
        if k not in _rs:
            _rs[k] = _o_res(d, rule)
        return _rs[k]

    _tl: dict = {}
    _o_tl = LP.cached_setup_timeline

    def timeline(df, cfg, sym):
        # bt_par 의 캐시 키와 동일 구성 (df 범위는 페어당 고정이라 sym 으로 갈음)
        k = (sym, round(cfg.ote_level, 4), round(cfg.min_rr, 3),
             round(cfg.fvg_min_size_pct, 6), bool(cfg.expand_to_killzone),
             bool(cfg.disable_time_filter), round(cfg.min_sl_distance_pct, 6),
             int(cfg.window), len(df))
        if k not in _tl:
            _tl[k] = _o_tl(df, cfg, sym)
        return _tl[k]

    LP._load_full = load
    LP._resample = resample
    LP.cached_setup_timeline = timeline


# ────────────────────────────────────────────────────────────── 유틸
def _sign(t) -> float:
    v = getattr(t.direction, "value", t.direction)
    return 1.0 if str(v).lower() in ("long", "buy") else -1.0


def _risk_frac(t) -> float:
    """진입가 대비 초기 위험폭 비율. entry_sl 미기록이면 0."""
    if t.entry_sl <= 0 or t.entry <= 0:
        return 0.0
    return abs(t.entry - t.entry_sl) / t.entry


def _slip_of(df5, idx: int) -> float:
    h = float(df5["high"].iat[idx]); lo = float(df5["low"].iat[idx])
    c = float(df5["close"].iat[idx])
    if c <= 0:
        return SLIP_NORMAL_PCT
    return SLIP_VOLATILE_PCT if (h - lo) / c > VOLATILE_THRESHOLD else SLIP_NORMAL_PCT


def _apply_exit_slip(price: float, d: float, slip: float) -> float:
    """청산 슬리피지 — 롱 청산은 불리하게 ↓, 숏 청산은 ↑ (cost.apply_slippage 동일)."""
    return price * (1.0 + slip) if d < 0 else price * (1.0 - slip)


def _r_of(raw: float, rf: float) -> float:
    """raw 가격변동비율 → 수수료 차감 R 배수."""
    if rf <= 0:
        return 0.0
    return (raw - 2.0 * TAKER_FEE_PCT) / rf


# ────────────────────────────────────────────── 페어 캐시
class PairData:
    """페어당 1회만 만드는 것들 — 5분봉·flip zone·numpy 뷰."""

    def __init__(self, sym: str, df5):
        self.sym = sym
        self.df5 = df5
        self.idx = df5.index
        self.hi = df5["high"].to_numpy()
        self.lo = df5["low"].to_numpy()
        self.zones = build_fvg_zones(df5)


PD: dict[str, PairData] = {}


def run_cfg(extra: dict) -> dict[str, list]:
    """라이브 정합 백테 1회(7페어). 반환: sym -> 통과 trade 목록."""
    TRIED["백테재생"] += 1
    out = {}
    for sym in PAIRS:
        df5, kept, _ = LP.run_live_parity(sym, extra)
        if sym not in PD:
            PD[sym] = PairData(sym, df5)
        out[sym] = kept
    return out


# ────────────────────────────────────────────── 부분익절 비율 재가중
def repartial(t, pd_: PairData, partial_tp_rr: float, f: float):
    """부분익절 비율을 0.5 → f 로 정확히 재가중한 raw 를 돌려준다.

    partial_be=True 이므로 잔여분의 청산 시점·가격은 비율과 무관하다.
    replay._exit 는 blend = 0.5*tp1 + 0.5*px 를 만든 뒤 슬리피지를 곱한다.
    슬리피지가 곱셈이라 역산해 px 를 얻고, f 로 다시 섞으면 된다.
    """
    if f == 0.5 or not str(t.outcome).startswith("p_"):
        return t.raw_pnl_pct
    d = _sign(t)
    slip = _slip_of(pd_.df5, min(t.exit_idx, len(pd_.df5) - 1))
    blend = t.exit_price / ((1.0 + slip) if d < 0 else (1.0 - slip))
    risk0 = abs(t.entry - t.entry_sl)
    tp1 = t.entry + d * partial_tp_rr * risk0
    px = (blend - 0.5 * tp1) / 0.5          # 잔여분 실제 청산가 역산
    new_blend = f * tp1 + (1.0 - f) * px
    new_px = _apply_exit_slip(new_blend, d, slip)
    return (new_px - t.entry) / t.entry * d


# ────────────────────────────────────────────── flip 사후 적용
def apply_flip_raw(pd_: PairData, t, min_r: float):
    """HTF FVG flip 을 raw 단위로 사후 적용. 반환 (raw, 청산봉idx, flip발동여부).

    flip_verdict.apply_flip 과 같은 규칙(zone 경계 1회 touch 즉시 청산)이되
    net% 대신 raw 를 돌려주고, 청산 봉 인덱스도 함께 준다(동시보유 계산용).
    min_r=FLIP_OFF 면 flip 자체를 끈다.
    """
    base = (t.raw_pnl_pct, min(t.exit_idx, len(pd_.idx) - 1), False)
    if min_r >= FLIP_OFF:
        return base
    d = _sign(t)
    z = flip_target_at(pd_.zones, t.entry_idx, int(d), float(t.entry), FLIP_MIN_W)
    if z is None:
        return base
    lo_i, hi_i = t.entry_idx + 1, min(t.exit_idx, len(pd_.idx) - 1)
    if hi_i <= lo_i:
        return base
    seg_hi = pd_.hi[lo_i:hi_i + 1]; seg_lo = pd_.lo[lo_i:hi_i + 1]
    hit = np.flatnonzero((seg_lo <= z["hi"]) & (seg_hi >= z["lo"]))
    if not hit.size:
        return base
    fpx = z["lo"] if d > 0 else z["hi"]
    raw = (fpx - t.entry) / t.entry * d
    rf = _risk_frac(t)
    r_at = (raw / rf) if rf > 0 else 0.0
    if r_at < min_r:                        # #FLIP-MIN-R — 얕은 flip 은 무시하고 홀드
        return base
    return raw, int(lo_i + hit[0]), True


# ────────────────────────────────────────────── 시나리오 평가
def evaluate(trades_by_sym, *, partial_tp_rr: float, pfrac: float, flip_r: float,
             pairs=None):
    """한 청산 시나리오의 거래 리스트를 만든다.

    반환: [{sym, ts(ns), ex_ts(ns), raw, r, year, dir, trend}]
    """
    TRIED["청산평가"] += 1
    use = pairs or PAIRS
    rows = []
    for sym in use:
        pd_ = PD[sym]
        for t in trades_by_sym[sym]:
            rf = _risk_frac(t)
            if rf <= 0:
                continue                       # entry_sl 미기록 — R 산출 불가
            raw_p = repartial(t, pd_, partial_tp_rr, pfrac) if partial_tp_rr > 0 \
                else t.raw_pnl_pct
            # flip 은 부분익절 재가중 이전 경로에 얹힌다(flip 이 먼저 터지면 그게 청산).
            raw_f, ex_i, fired = apply_flip_raw(pd_, t, flip_r)
            raw = raw_f if fired else raw_p
            ts = pd_.idx[t.entry_idx]
            rows.append(dict(
                sym=sym, ts=int(ts.value), ex_ts=int(pd_.idx[ex_i].value),
                raw=float(raw), r=float(_r_of(raw, rf)), year=int(ts.year),
                dir=("long" if _sign(t) > 0 else "short"),
                trend=float(t.entry_trend_pct or 0.0), fired=bool(fired),
            ))
    rows.sort(key=lambda x: x["ts"])
    return rows


def summ(rows) -> dict:
    """R 기준 요약."""
    if not rows:
        return dict(n=0)
    r = np.array([x["r"] for x in rows])
    w = r[r > 0]
    lo = r[r < 0]
    return dict(
        n=len(r), r_mean=float(r.mean()), r_sum=float(r.sum()),
        r_se=float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan"),
        wr=float(100.0 * len(w) / len(r)),
        rr=float(w.mean() / abs(lo.mean())) if len(w) and len(lo) else float("nan"),
    )


# ────────────────────────────────────────────── 복리 시뮬 (레버리지 7x)
def concurrency(rows) -> np.ndarray:
    """각 거래 보유 구간에 겹친 거래 수(자기 포함). 라이브의 자본 분할 재현."""
    n = len(rows)
    s = np.array([x["ts"] for x in rows], dtype=np.int64)
    e = np.array([x["ex_ts"] for x in rows], dtype=np.int64)
    out = np.empty(n)
    for i in range(n):
        out[i] = 1.0 + int(np.count_nonzero((s <= e[i]) & (e >= s[i]))) - 1
    return out


def sim(raws, scale, lev=LEV):
    """복리 — (최종배수, MDD%, 파산여부). DD 스로틀·파산 판정 포함."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(raws)):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = raws[i] * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True
    return float(eq), 100.0 * mdd, False


def compound(rows, n_boot=N_BOOT):
    """복리 결과 + 부트스트랩(복원추출) 중앙값·5%분위·파산확률."""
    if len(rows) < 5:
        return dict(compound=float("nan"), mdd=float("nan"), ruin=float("nan"),
                    boot_p50=float("nan"), boot_p5=float("nan"))
    raws = np.array([x["raw"] for x in rows])
    sc = 1.0 / concurrency(rows)
    eq, mdd, _ = sim(raws, sc)
    fin = np.empty(n_boot); ruin = 0
    n = len(raws)
    for k in range(n_boot):
        i = RNG.integers(0, n, size=n)
        e, _, r_ = sim(raws[i], sc[i])
        fin[k] = e; ruin += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    return dict(compound=float(eq), mdd=float(mdd), ruin=float(100.0 * ruin / n_boot),
                boot_p50=float(p50), boot_p5=float(p5))


# ────────────────────────────────────────────── 순열검정
def perm_unpaired(a, b, n=N_PERM):
    """비대응 순열검정 — 두 R 묶음을 합쳐 원래 크기로 무작위 분할.

    ※ 청산 스윕에서는 두 묶음이 같은 진입을 공유해 상관이 매우 높다. 비대응
       검정은 그 상관을 무시하므로 **과도하게 보수적**이다(검출력 낮음).
       참고용으로 병기하되 주 판정은 대응 검정으로 한다.
    """
    a = np.asarray(a); b = np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a); tot = len(pool)
    # 벡터화: n×tot 무작위 순위로 한 번에 분할
    order = RNG.random((n, tot)).argsort(axis=1)
    pm = pool[order]
    dist = pm[:, :na].mean(axis=1) - pm[:, na:].mean(axis=1)
    cnt = int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12))
    return float(obs), float((cnt + 1) / (n + 1))


def perm_paired(diffs, n=N_PERM):
    """대응 순열검정 — 같은 셋업 짝의 R 차이 부호를 무작위 반전.

    청산 스윕은 같은 진입을 공유하므로 대응 검정의 검출력이 훨씬 높다.
    """
    d = np.asarray(diffs)
    if d.size == 0:
        return float("nan"), float("nan")
    obs = d.mean()
    sgn = RNG.integers(0, 2, size=(n, d.size)) * 2 - 1
    dist = (sgn * d).mean(axis=1)
    p = (int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12)) + 1) / (n + 1)
    return float(obs), float(p)


def pair_diffs(cand, base):
    """(sym, 진입ts) 가 같은 거래만 짝지어 R 차이."""
    mb = {(x["sym"], x["ts"]): x["r"] for x in base}
    out = []
    for x in cand:
        k = (x["sym"], x["ts"])
        if k in mb:
            out.append(x["r"] - mb[k])
    return out


# ────────────────────────────────────────────── 출력 헬퍼
def hdr(s):
    print("\n" + "=" * 92, flush=True)
    print(s, flush=True)
    print("=" * 92, flush=True)


def row(label, s, base_r=None, extra=""):
    if not s or s.get("n", 0) == 0:
        print(f"  {label:<22} 거래 없음", flush=True)
        return
    d = "" if base_r is None else f" ΔR={s['r_mean'] - base_r:+.4f}"
    flag = "" if s["n"] >= MIN_N else "  [표본부족]"
    print(f"  {label:<22} n={s['n']:4d} 건당R={s['r_mean']:+.4f}±{s['r_se']:.4f} "
          f"총R={s['r_sum']:+8.1f} 승률={s['wr']:4.1f}% RR={s['rr']:5.2f}"
          f"{d}{flag}{extra}", flush=True)


# ══════════════════════════════════════════════════════════════ 본문
def main() -> int:
    t_start = time.time()

    hdr("[0] 하니스 준비 · 메모이즈 정합 확인")
    # 메모이즈 **설치 전** 원본 경로로 1페어를 돌려 대조군을 만든다.
    _, ref, _ = LP.run_live_parity("BTCUSDT", dict(BASE_EXIT))
    _install_memo()

    base_bt = run_cfg(dict(BASE_EXIT))
    n_all = sum(len(v) for v in base_bt.values())
    print(f"  기준선 백테 통과 {n_all}건 / {len(PAIRS)}페어  "
          f"({time.time() - t_start:.0f}s)", flush=True)
    got = base_bt["BTCUSDT"]
    same = (len(got) == len(ref)) and all(
        a.entry_idx == b.entry_idx and a.exit_idx == b.exit_idx
        and abs(a.raw_pnl_pct - b.raw_pnl_pct) < 1e-15
        for a, b in zip(got, ref))
    assert same, "메모이즈가 결과를 바꿨다 — 중단"
    print(f"  메모이즈 정합: OK (원본 경로 BTCUSDT {len(ref)}건과 완전 일치)", flush=True)
    for s in PAIRS:
        print(f"    {s.replace('USDT', ''):<6} {len(base_bt[s]):3d}건", flush=True)

    # 기준선 시나리오 (현행 라이브 = 부분익절 1.5R/50% + flip 최소 1.5R)
    BASE = evaluate(base_bt, partial_tp_rr=1.5, pfrac=BASE_PFRAC, flip_r=BASE_FLIP_R)
    base_s = summ(BASE)
    base_r = base_s["r_mean"]

    # ───────────────────────────────────────── [1] 구조적 상한
    hdr("[1] 구조적 상한 — 청산 손잡이가 R 을 물리적으로 얼마나 움직일 수 있나")
    print("  방법: 트레일·본전·부분익절을 모두 끈 설정(TP/SL 만)의 보유구간이", flush=True)
    print("        모든 청산 변형의 상위집합이다(청산 로직은 더 일찍 나올 뿐).", flush=True)
    print("        그 구간의 최대유리폭(MFE)에서 청산 = 도달 불가능한 천장.", flush=True)
    off_bt = run_cfg(dict(trail_trigger=0.0, trail_dist=1.5, be_trigger=0.0,
                          partial_tp_rr=0.0))
    mfe, mae, cur_r, reach = [], [], [], {1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0, 3.0: 0, 4.0: 0}
    n_off = 0
    for sym in PAIRS:
        pd_ = PD[sym]
        for t in off_bt[sym]:
            rf = _risk_frac(t)
            if rf <= 0:
                continue
            n_off += 1
            d = _sign(t)
            a, b = t.entry_idx + 1, min(t.exit_idx, len(pd_.idx) - 1)
            if b < a:
                continue
            if d > 0:
                best = float(pd_.hi[a:b + 1].max()); worst = float(pd_.lo[a:b + 1].min())
            else:
                best = float(pd_.lo[a:b + 1].min()); worst = float(pd_.hi[a:b + 1].max())
            # 수수료를 차감한 R 로 환산 — 기준선 R 과 같은 단위여야 비교가 성립.
            m = _r_of((best - t.entry) / t.entry * d, rf)
            w = _r_of((worst - t.entry) / t.entry * d, rf)
            mfe.append(m); mae.append(w)
            cur_r.append(_r_of(t.raw_pnl_pct, rf))
            for k in reach:
                if m >= k:
                    reach[k] += 1
    mfe = np.array(mfe); mae = np.array(mae); cur_r = np.array(cur_r)
    ceil_r = float(mfe.mean())   # 이미 수수료 차감된 R
    print(f"\n  청산기능 OFF 기준선: n={n_off} 건당R={cur_r.mean():+.4f}", flush=True)
    print(f"  현행 라이브 기준선 : n={base_s['n']} 건당R={base_r:+.4f} "
          f"(표준오차 ±{base_s['r_se']:.4f})", flush=True)
    print(f"\n  MFE 평균 = {mfe.mean():+.3f}R  (중앙 {np.median(mfe):+.3f}R, "
          f"최대 {mfe.max():+.1f}R)", flush=True)
    print(f"  MAE 평균 = {mae.mean():+.3f}R", flush=True)
    print(f"  → 완전예지 청산(MFE 체결) 건당R 천장 ≈ {ceil_r:+.3f}R", flush=True)
    room_up = ceil_r - base_r
    room_dn = base_r - float(mae.mean())
    print(f"  → 현행 대비 위쪽 여유 {room_up:+.3f}R / 아래쪽 여유 {-room_dn:+.3f}R",
          flush=True)
    noise = base_s["r_se"]
    print(f"  → 표본 노이즈(건당R 표준오차) = ±{noise:.4f}R", flush=True)
    ratio = room_up / noise if noise > 0 else float("inf")
    print(f"\n  판정: 상한/노이즈 = {ratio:.1f}배", flush=True)
    if ratio < 2.0:
        print("  ★ 상한이 노이즈 2배 미만 — 스윕 무의미. 여기서 중단한다.", flush=True)
        return 0
    print("  ★ 상한이 노이즈보다 충분히 크다 → 스윕 진행 (피보 축과 다름)", flush=True)

    print(f"\n  [손잡이별 사정거리] MFE 가 임계에 도달한 거래 비율 (n={len(mfe)})",
          flush=True)
    for k in sorted(reach):
        tag = {1.0: "본전이동 발동선", 1.5: "부분익절 발동선",
               2.0: "트레일 발동선"}.get(k, "")
        print(f"    MFE ≥ {k:.1f}R : {reach[k]:4d}건 ({100 * reach[k] / len(mfe):4.1f}%)"
              f"  {tag}", flush=True)

    # ───────────────────────────────────────── [2] 단독 스윕
    hdr("[2] 단독 스윕 — 한 손잡이만 움직이고 나머지는 현행값 고정")
    row("현행(기준선)", base_s, None, "  ← 비교 기준")

    results: dict[str, dict] = {}   # name -> {params, rows, summ}

    def record(name, params, rows):
        results[name] = dict(params=params, rows=rows, s=summ(rows))
        return results[name]["s"]

    record("현행", dict(**BASE_EXIT, partial_frac=BASE_PFRAC, flip_min_r=BASE_FLIP_R),
           BASE)

    # A. trail_trigger
    print("\n  [A] 트레일 발동 R (trail_trigger) — 나머지 현행", flush=True)
    for v in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        if v == 2.0:
            row("  2.0 (현행)", base_s, base_r)
            continue
        bt = run_cfg({**BASE_EXIT, "trail_trigger": v})
        rows = evaluate(bt, partial_tp_rr=1.5, pfrac=BASE_PFRAC, flip_r=BASE_FLIP_R)
        s = record(f"trail_trigger={v}", {**BASE_EXIT, "trail_trigger": v,
                                          "partial_frac": BASE_PFRAC,
                                          "flip_min_r": BASE_FLIP_R}, rows)
        row(f"  {v:.1f}{' (끔)' if v == 0 else ''}", s, base_r)

    # B. trail_dist
    print("\n  [B] 트레일 간격 R (trail_dist) — 나머지 현행", flush=True)
    for v in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        if v == 1.5:
            row("  1.5 (현행)", base_s, base_r)
            continue
        bt = run_cfg({**BASE_EXIT, "trail_dist": v})
        rows = evaluate(bt, partial_tp_rr=1.5, pfrac=BASE_PFRAC, flip_r=BASE_FLIP_R)
        s = record(f"trail_dist={v}", {**BASE_EXIT, "trail_dist": v,
                                       "partial_frac": BASE_PFRAC,
                                       "flip_min_r": BASE_FLIP_R}, rows)
        row(f"  {v:.2f}", s, base_r)

    # C. be_trigger
    print("\n  [C] 본전이동 발동 R (be_trigger) — 나머지 현행", flush=True)
    for v in (0.0, 0.5, 0.75, 1.0, 1.5, 2.0):
        if v == 1.0:
            row("  1.0 (현행)", base_s, base_r)
            continue
        bt = run_cfg({**BASE_EXIT, "be_trigger": v})
        rows = evaluate(bt, partial_tp_rr=1.5, pfrac=BASE_PFRAC, flip_r=BASE_FLIP_R)
        s = record(f"be_trigger={v}", {**BASE_EXIT, "be_trigger": v,
                                       "partial_frac": BASE_PFRAC,
                                       "flip_min_r": BASE_FLIP_R}, rows)
        row(f"  {v:.2f}{' (끔)' if v == 0 else ''}", s, base_r)

    # D. partial_tp_rr
    print("\n  [D] 부분익절 위치 R (partial_tp_rr, 비율 50% 고정) — 나머지 현행",
          flush=True)
    for v in (0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        if v == 1.5:
            row("  1.5 (현행)", base_s, base_r)
            continue
        bt = run_cfg({**BASE_EXIT, "partial_tp_rr": v})
        rows = evaluate(bt, partial_tp_rr=v, pfrac=BASE_PFRAC, flip_r=BASE_FLIP_R)
        s = record(f"partial_tp_rr={v}", {**BASE_EXIT, "partial_tp_rr": v,
                                          "partial_frac": BASE_PFRAC,
                                          "flip_min_r": BASE_FLIP_R}, rows)
        row(f"  {v:.2f}{' (끔)' if v == 0 else ''}", s, base_r)

    # E. 부분익절 비율 (역산 재가중)
    print("\n  [E] 부분익절 비율 (partial_frac) — 1.5R 위치 고정", flush=True)
    chk = evaluate(base_bt, partial_tp_rr=1.5, pfrac=0.5, flip_r=BASE_FLIP_R)
    err = max(abs(a["raw"] - b["raw"]) for a, b in zip(chk, BASE))
    print(f"      역산 검증: f=0.5 재구성 오차 최대 {err:.2e} "
          f"({'OK' if err < 1e-12 else '⚠ 불일치'})", flush=True)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        if f == 0.5:
            row("  0.50 (현행)", base_s, base_r)
            continue
        rows = evaluate(base_bt, partial_tp_rr=1.5, pfrac=f, flip_r=BASE_FLIP_R)
        s = record(f"partial_frac={f}", {**BASE_EXIT, "partial_frac": f,
                                         "flip_min_r": BASE_FLIP_R}, rows)
        row(f"  {f:.2f}", s, base_r)

    # F. flip 최소 R
    print("\n  [F] HTF FVG flip 최소 R (flip_min_r) — 나머지 현행", flush=True)
    for v in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, FLIP_OFF):
        if v == 1.5:
            row("  1.5 (현행)", base_s, base_r)
            continue
        rows = evaluate(base_bt, partial_tp_rr=1.5, pfrac=BASE_PFRAC, flip_r=v)
        nf = sum(1 for x in rows if x["fired"])
        s = record(f"flip_min_r={v}", {**BASE_EXIT, "partial_frac": BASE_PFRAC,
                                       "flip_min_r": v}, rows)
        lab = "flip 끔" if v >= FLIP_OFF else ("0.0 (무조건)" if v == 0 else f"{v:.1f}")
        row(f"  {lab}", s, base_r, f"  flip발동 {nf}건")
    nf_base = sum(1 for x in BASE if x["fired"])
    print(f"      (현행 1.5R 의 flip 발동 = {nf_base}건 / {len(BASE)}건)", flush=True)

    # ───────────────────────────────────────── [3] 교차 조합
    hdr("[3] 축별 단조성 점검 + 교차 조합")
    # 단조성 = "한 값만 튀는 게 아니라 방향이 일관되게 좋아지는가". 단일 스파이크는
    # 체결 우연일 확률이 높다(2026-08-02 기각 사례).
    print("  [단조성] 축을 따라 ΔR 이 한 방향으로 움직이는지", flush=True)
    axis_grid = {
        "trail_trigger": (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
        "trail_dist": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
        "be_trigger": (0.0, 0.5, 0.75, 1.0, 1.5, 2.0),
        "partial_tp_rr": (0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0),
        "partial_frac": (0.0, 0.25, 0.5, 0.75, 1.0),
        "flip_min_r": (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, FLIP_OFF),
    }
    for ax, grid in axis_grid.items():
        vals = []
        for g in grid:
            k = f"{ax}={g}"
            m = base_r if k not in results else results[k]["s"]["r_mean"]
            vals.append(m - base_r)
        rng_ = max(vals) - min(vals)
        print(f"    {ax:<15} " + " ".join(f"{g:g}:{v:+.4f}" for g, v in zip(grid, vals))
              + f"   폭={rng_:.4f}R ({rng_ / base_s['r_se']:.1f}×SE)", flush=True)

    print("\n  [조합]", flush=True)
    singles = {k: v for k, v in results.items() if k != "현행"}
    ranked = sorted(singles.items(), key=lambda kv: -kv[1]["s"]["r_mean"])
    print("  단독 스윕 상위 8 (건당R 순):", flush=True)
    for k, v in ranked[:8]:
        print(f"    {k:<24} 건당R={v['s']['r_mean']:+.4f} "
              f"(ΔR {v['s']['r_mean'] - base_r:+.4f})", flush=True)

    # 각 축의 최선값 추출 (현행보다 나은 것만)
    best_by_axis: dict[str, float] = {}
    for k, v in singles.items():
        ax, val = k.split("=")
        val = float(val)
        if v["s"]["r_mean"] <= base_r:
            continue
        if ax not in best_by_axis or v["s"]["r_mean"] > results[f"{ax}={best_by_axis[ax]}"]["s"]["r_mean"]:
            best_by_axis[ax] = val
    # partial_tp_rr=0 과 partial_frac=0 은 **같은 상태(부분익절 끔)** 다. 둘 다
    # 남기면 한쪽을 빼도 다른 쪽이 살아 있어 축 제거 실험이 성립하지 않는다.
    if best_by_axis.get("partial_tp_rr") == 0.0:
        best_by_axis.pop("partial_frac", None)
    print(f"\n  현행을 넘은 축·값: {best_by_axis if best_by_axis else '없음'}", flush=True)

    combos: list[tuple[str, dict]] = []
    cfg_axes = {k: v for k, v in best_by_axis.items()
                if k in ("trail_trigger", "trail_dist", "be_trigger", "partial_tp_rr")}
    post_axes = {k: v for k, v in best_by_axis.items()
                 if k in ("partial_frac", "flip_min_r")}
    allb = {**cfg_axes, **post_axes}
    if allb:
        combos.append(("전축 최선 결합", dict(allb)))
    # 축을 하나씩 빼 본다(어느 축이 실제 기여인지 = 조합의 취약점 확인)
    for drop in list(allb):
        sub = {k: v for k, v in allb.items() if k != drop}
        if sub:
            combos.append((f"최선결합 −{drop}", sub))
    # 최소 조합 — 노이즈 수준(ΔR < 표준오차 절반) 축을 뺀 "설명 가능한" 후보
    minimal = {k: v for k, v in allb.items()
               if results[f"{k}={v}"]["s"]["r_mean"] - base_r > 0.5 * base_s["r_se"]}
    if minimal and minimal != allb:
        combos.append(("최소조합(노이즈축 제거)", minimal))
    if not combos:
        combos.append(("(단독에서 개선 없음 — 조합 생략)", {}))

    for name, cb in combos:
        if not cb:
            print(f"  {name}", flush=True)
            continue
        cfg_part = {**BASE_EXIT}
        for k in ("trail_trigger", "trail_dist", "be_trigger", "partial_tp_rr"):
            if k in cb:
                cfg_part[k] = cb[k]
        bt = run_cfg(cfg_part)
        rows = evaluate(bt, partial_tp_rr=cfg_part["partial_tp_rr"],
                        pfrac=cb.get("partial_frac", BASE_PFRAC),
                        flip_r=cb.get("flip_min_r", BASE_FLIP_R))
        s = record(name, {**cfg_part, "partial_frac": cb.get("partial_frac", BASE_PFRAC),
                          "flip_min_r": cb.get("flip_min_r", BASE_FLIP_R)}, rows)
        row(name, s, base_r)

    # ───────────────────────────────────────── [4] 후보 선별 → 검정
    hdr("[4] 후보 검정 — 순열검정 / 연도 / 국면×방향 / 복리")
    # 후보 = ① 각 축의 단독 최선(해석 가능한 1파라미터 가설) + ② 조합 상위.
    # 단독을 반드시 포함하는 이유: 조합만 보면 39개 argmax 라 승자의 저주를 못 건넌다.
    picked: list[tuple[str, dict]] = []
    seen = set()
    for ax, val in best_by_axis.items():
        k = f"{ax}={val}"
        if k in results and results[k]["s"]["r_mean"] > base_r and k not in seen:
            seen.add(k); picked.append((k, results[k]))
    for k, v in sorted(((k, v) for k, v in results.items()
                        if "=" not in k and k != "현행"
                        and v["s"]["r_mean"] > base_r),
                       key=lambda kv: -kv[1]["s"]["r_mean"]):
        if k not in seen:
            seen.add(k); picked.append((k, v))
    cands = [(k, v) for k, v in picked if v["s"]["n"] >= MIN_N]
    n_cfg_tried = len(results) - 1
    print(f"  시도한 청산 설정 = {n_cfg_tried}개 (백테 재생 {TRIED['백테재생']}회 / "
          f"청산 평가 {TRIED['청산평가']}회)", flush=True)
    print(f"  현행보다 건당R 이 높은 설정 = {sum(1 for k, v in results.items() if k != '현행' and v['s']['r_mean'] > base_r)}개", flush=True)
    print(f"  → 정밀 검정 대상 {len(cands)}개 = 축별 단독 최선 + 조합. "
          "단독을 반드시 포함해 조합 argmax 편향을 대조한다.", flush=True)
    # 다중비교 보정 임계 (Bonferroni)
    alpha_adj = 0.05 / max(n_cfg_tried, 1)
    print(f"  Bonferroni 보정 임계 α = 0.05/{n_cfg_tried} = {alpha_adj:.5f}", flush=True)

    base_comp = compound(BASE)
    base_comp_live = compound([x for x in BASE if x["sym"] in LIVE_PAIRS])
    print(f"\n  [기준선 복리 · 7페어] {base_comp['compound']:.2f}배 "
          f"MDD {base_comp['mdd']:.1f}% 파산 {base_comp['ruin']:.1f}% "
          f"부트중앙 {base_comp['boot_p50']:.2f}배 5%분위 {base_comp['boot_p5']:.2f}배",
          flush=True)
    print(f"  [기준선 복리 · BTC+ETH] {base_comp_live['compound']:.2f}배 "
          f"MDD {base_comp_live['mdd']:.1f}% 파산 {base_comp_live['ruin']:.1f}% "
          f"부트중앙 {base_comp_live['boot_p50']:.2f}배 "
          f"5%분위 {base_comp_live['boot_p5']:.2f}배", flush=True)

    verdicts = []
    for name, v in cands:
        rows, s = v["rows"], v["s"]
        print("\n" + "-" * 92, flush=True)
        print(f"  후보: {name}   params={v['params']}", flush=True)
        row("    성적", s, base_r)

        # 순열검정 (대응 + 비대응)
        d = pair_diffs(rows, BASE)
        obs_p, p_paired = perm_paired(d)
        obs_u, p_unp = perm_unpaired([x["r"] for x in rows], [x["r"] for x in BASE])
        print(f"    순열검정 대응  : 짝 {len(d)}건 ΔR={obs_p:+.4f} p={p_paired:.4f}",
              flush=True)
        print(f"    순열검정 비대응: ΔR={obs_u:+.4f} p={p_unp:.4f}  (상관 무시 — 참고용)",
              flush=True)
        # 라이브 편성(BTC+ETH)만으로도 같은 방향인지 — 실제 배포 대상 확인
        lrows = [x for x in rows if x["sym"] in LIVE_PAIRS]
        lbase = [x for x in BASE if x["sym"] in LIVE_PAIRS]
        dl = pair_diffs(lrows, lbase)
        obs_l, p_live = perm_paired(dl) if dl else (float("nan"), float("nan"))
        ltag = "" if len(dl) >= MIN_N else " [표본부족]"
        print(f"    순열검정 BTC+ETH: 짝 {len(dl)}건 ΔR={obs_l:+.4f} "
              f"p={p_live:.4f}{ltag}", flush=True)
        p_best = p_paired
        g1 = p_best < 0.05
        g1b = p_best < alpha_adj

        # 연도 일관성
        yrs = sorted({x["year"] for x in BASE} | {x["year"] for x in rows})
        y_win = 0; y_tot = 0; y_share = {}
        tot_gain = s["r_sum"] - base_s["r_sum"]
        print("    연도별 총R  " + "".join(f"{y:>9d}" for y in yrs), flush=True)
        lb = [sum(x["r"] for x in BASE if x["year"] == y) for y in yrs]
        lc = [sum(x["r"] for x in rows if x["year"] == y) for y in yrs]
        print("      현행      " + "".join(f"{x:>+9.1f}" for x in lb), flush=True)
        print("      후보      " + "".join(f"{x:>+9.1f}" for x in lc), flush=True)
        print("      차이      " + "".join(f"{c - b:>+9.1f}" for b, c in zip(lb, lc)),
              flush=True)
        for y, b, c in zip(yrs, lb, lc):
            y_tot += 1
            if c - b > 0:
                y_win += 1
            y_share[y] = (c - b) / tot_gain if abs(tot_gain) > 1e-9 else 0.0
        top_year = max(y_share, key=lambda k: y_share[k]) if y_share else None
        top_sh = y_share.get(top_year, 0.0)
        print(f"      → 개선 연도 {y_win}/{y_tot}, 최대 기여 연도 {top_year} "
              f"({100 * top_sh:.0f}%)", flush=True)
        # 연도 게이트: 부호 다수결. 크기 집중은 아래 연도제외 강건성(g5)으로 본다
        # — 기여율 임계 하나로 자르면 6/6 전부 개선인 경우까지 잘못 죽인다.
        g2 = y_win >= (y_tot + 1) // 2

        # 국면×방향 기저 통제
        print("    국면×방향 (셀별 ΔR)", flush=True)
        cells_ok = 0; cells_tot = 0
        mb = {(x["sym"], x["ts"]): x for x in BASE}
        for dr in ("long", "short"):
            for rg, lab in ((1, "상승국면"), (-1, "하락국면")):
                dd = [x["r"] - mb[(x["sym"], x["ts"])]["r"] for x in rows
                      if x["dir"] == dr and (1 if x["trend"] >= 0 else -1) == rg
                      and (x["sym"], x["ts"]) in mb]
                if not dd:
                    continue
                cells_tot += 1
                m = float(np.mean(dd))
                if m > 0:
                    cells_ok += 1
                tag = "" if len(dd) >= MIN_N else " [표본부족]"
                print(f"      {dr:<6}{lab}  n={len(dd):3d} ΔR={m:+.4f}{tag}", flush=True)
        print(f"      → 개선 셀 {cells_ok}/{cells_tot}", flush=True)
        g3 = cells_ok >= (cells_tot + 1) // 2

        # 페어별
        p_win = 0
        pl = []
        for sym in PAIRS:
            b = sum(x["r"] for x in BASE if x["sym"] == sym)
            c = sum(x["r"] for x in rows if x["sym"] == sym)
            pl.append(f"{sym.replace('USDT', '')}:{c - b:+.1f}")
            if c - b > 0:
                p_win += 1
        print(f"    페어별 ΔR   {' '.join(pl)}  → 개선 {p_win}/{len(PAIRS)}", flush=True)
        g4 = p_win >= 4

        # 연도/거래 제외 강건성 — "특정 연도·특정 거래 몰빵"의 직접 검사.
        # 팀 표준 3번(연도 일관성)은 총합 기여율만 보는데, 그것만으로는 부호가
        # 6/6 로 일관된 경우를 과하게 벌준다. 한 해를 통째로 빼고도 유의한지 본다.
        dp = []
        for x in rows:
            k2 = (x["sym"], x["ts"])
            if k2 in mb:
                dp.append((x["year"], x["r"] - mb[k2]["r"]))
        loo_min_d, loo_max_p, loo_worst_y = None, 0.0, None
        for y in sorted({yy for yy, _ in dp}):
            sub = [v for yy, v in dp if yy != y]
            if len(sub) < MIN_N:
                continue
            m2, p2 = perm_paired(sub, n=5000)
            if loo_min_d is None or m2 < loo_min_d:
                loo_min_d, loo_worst_y = m2, y
            loo_max_p = max(loo_max_p, p2)
        dv = np.array([v for _, v in dp])
        top1 = float((dv.sum() - dv[np.argmax(np.abs(dv))]) / (len(dv) - 1)) if len(dv) > 1 else float("nan")
        print(f"    강건성 연도제외: 최악 {loo_worst_y} 제외 시 ΔR={loo_min_d:+.4f} "
              f"(6회 중 최대 p={loo_max_p:.4f})", flush=True)
        print(f"    강건성 거래제외: 최대기여 1건 제외 시 ΔR={top1:+.4f}", flush=True)
        g5 = (loo_min_d is not None and loo_min_d > 0 and loo_max_p < 0.05
              and top1 > 0)

        # 복리
        cp = compound(rows)
        cpl = compound([x for x in rows if x["sym"] in LIVE_PAIRS])
        print(f"    복리 7페어  {cp['compound']:.2f}배 (현행 {base_comp['compound']:.2f}배) "
              f"MDD {cp['mdd']:.1f}% 파산 {cp['ruin']:.1f}% "
              f"부트중앙 {cp['boot_p50']:.2f}배 5%분위 {cp['boot_p5']:.2f}배", flush=True)
        print(f"    복리 BTC+ETH {cpl['compound']:.2f}배 "
              f"(현행 {base_comp_live['compound']:.2f}배) MDD {cpl['mdd']:.1f}% "
              f"파산 {cpl['ruin']:.1f}%", flush=True)

        ok = g1 and g2 and g3 and g4 and g5
        verdict = "유망" if (ok and g1b) else ("조건부(다중비교 미통과)" if ok else "기각")
        why = []
        if not g1:
            why.append(f"순열 p={p_best:.3f}≥0.05")
        elif not g1b:
            why.append(f"p={p_best:.4f} < 0.05 이나 Bonferroni {alpha_adj:.5f} 미달")
        if not g2:
            why.append(f"연도 {y_win}/{y_tot}")
        if not g3:
            why.append(f"국면×방향 셀 {cells_ok}/{cells_tot}")
        if not g4:
            why.append(f"페어 {p_win}/{len(PAIRS)}")
        if not g5:
            why.append(f"연도제외 강건성 실패(최악 ΔR={loo_min_d:+.4f}, 최대 p={loo_max_p:.3f})")
        print(f"    ▶ 판정: {verdict}  {('— ' + ', '.join(why)) if why else ''}",
              flush=True)

        verdicts.append(dict(
            name=name, params=v["params"], n=s["n"], r_mean=s["r_mean"],
            r_se=s["r_se"], r_sum=s["r_sum"], wr=s["wr"], rr=s["rr"],
            d_r=s["r_mean"] - base_r,
            compound=cp["compound"], mdd=cp["mdd"], ruin=cp["ruin"],
            boot_p50=cp["boot_p50"], boot_p5=cp["boot_p5"],
            compound_live=cpl["compound"], mdd_live=cpl["mdd"], ruin_live=cpl["ruin"],
            p_perm=p_best, p_paired=p_paired, p_unpaired=p_unp,
            p_paired_live=p_live, d_r_live=obs_l, n_live=len(dl),
            year_win=f"{y_win}/{y_tot}", year_top_share=top_sh,
            loo_worst_year=loo_worst_y, loo_min_d_r=loo_min_d, loo_max_p=loo_max_p,
            drop_top_trade_d_r=top1,
            cells=f"{cells_ok}/{cells_tot}", pairs=f"{p_win}/{len(PAIRS)}",
            verdict=verdict, reasons=why,
        ))

    # ───────────────────────────────────────── [5] 저장
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = dict(
        axis="exit (trail / breakeven / partial TP / HTF FVG flip min R)",
        date="2026-08-07",
        harness="scripts/live_parity.py run_live_parity (7페어, 2021-07~2026-06)",
        structural_bound=dict(
            mfe_mean_r=float(mfe.mean()), mfe_median_r=float(np.median(mfe)),
            mae_mean_r=float(mae.mean()), ceiling_r=ceil_r,
            baseline_r=base_r, headroom_r=room_up, noise_se_r=noise,
            ratio_bound_over_noise=ratio,
            reach_pct={str(k): 100.0 * reach[k] / len(mfe) for k in sorted(reach)},
        ),
        baseline=dict(
            name="현행 라이브",
            params=dict(**BASE_EXIT, partial_frac=BASE_PFRAC, flip_min_r=BASE_FLIP_R),
            n=base_s["n"], r_mean=base_r, r_se=base_s["r_se"], r_sum=base_s["r_sum"],
            wr=base_s["wr"], rr=base_s["rr"],
            compound=base_comp["compound"], mdd=base_comp["mdd"],
            ruin=base_comp["ruin"], boot_p50=base_comp["boot_p50"],
            boot_p5=base_comp["boot_p5"],
            compound_live=base_comp_live["compound"], mdd_live=base_comp_live["mdd"],
            ruin_live=base_comp_live["ruin"],
        ),
        candidates=verdicts,
        all_configs={k: dict(params=v["params"], n=v["s"]["n"],
                             r_mean=v["s"]["r_mean"], r_sum=v["s"]["r_sum"],
                             wr=v["s"]["wr"], rr=v["s"]["rr"])
                     for k, v in results.items()},
        multiple_comparisons=dict(
            n_configs=n_cfg_tried, n_backtest_runs=TRIED["백테재생"],
            n_exit_evals=TRIED["청산평가"],
            n_better_than_baseline=sum(1 for k, v in results.items()
                                       if k != "현행" and v["s"]["r_mean"] > base_r),
            n_axes=6,
            alpha_bonferroni_configs=alpha_adj,       # 0.05/39 — 설정 단위(가장 엄격)
            alpha_bonferroni_axes=0.05 / 6.0,         # 0.05/6 — 축 단위(설정끼리 강상관)
            note=("39개 설정은 6개 축의 격자라 서로 강하게 상관돼 있다. 설정 단위 "
                  "Bonferroni 는 과보정이므로 축 단위(0.0083)를 함께 본다."),
        ),
        notes=(
            "청산 파라미터는 전부 risk0=|진입-SL| 배수로 동작해 R 을 직접 움직인다"
            "(피보 축과 구조가 다름). 한계: ① 부분익절 비율은 replay 하드코딩 0.5 를 "
            "슬리피지 역산으로 재가중한 값(f=0.5 재구성 오차 <1e-12 검증). "
            "② flip 은 replay 에 경로가 없어 사후 적용이며 flip 청산가에 슬리피지 미반영"
            "(flip_verdict.apply_flip 과 동일 관행). ③ live_parity.GAPS 의 미구현 5종"
            "(ote_up_level 0.786 / sweep_gate / smart_size / dd_throttle 진입단 / "
            "daily_loss_limit)은 여전히 백테 미반영. ④ 라이브는 BTC+ETH 2페어라 "
            "7페어 결과는 검정력 확보용이며 복리는 양쪽 모두 보고."
        ),
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n\n저장: {OUT_PATH}", flush=True)
    print(f"총 소요 {time.time() - t_start:.0f}s", flush=True)

    LP.parity_report(None, {"wr": 46, "rr": 0.94})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
