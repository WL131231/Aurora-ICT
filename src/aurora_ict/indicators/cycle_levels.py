"""Cycle(순환매) 지지/저항 레벨 — 나뇨띠 자료 기반. 차트 표시용.

2026-08-18 파트너 요청: 순환매가 보는 지지/저항을 차트에서 눈으로 확인.

레벨 소스
  ① 구름대 스팬 — 일목 선행스팬 A/B (9/26/52, offset = displacement-1).
                   **여러 TF** 를 본다. 나뇨띠 매물대 자료의 "상위 프레임일수록
                   지지/저항이 훨씬 강력하다"가 근거 — 어느 TF 구름인지가
                   그 자체로 정보라서 터치 마커에 TF 라벨을 함께 실어 보낸다.
  ② 매물대      — 가격대별 거래량 상위 구간 (시간대별이 아니라 **가격대별**)
  ③ 2468        — 1K 단위 안의 200/400(상방) · 600/800(하방). 방향 필터 적용
                   (2,4,6,8 타점매매 PDF: "추세가 상방이면 2·4, 하방이면 6·8")

터치 판정은 봉이 레벨을 스치고 그 방향으로 버텼는지로 본다.
지지 터치(아래에서 받침) / 저항 터치(위에서 눌림)를 구분해 반환한다.

상위 TF 는 차트 봉을 리샘플해 만든다. 두 가지를 지킨다.
  · 차트 TF 의 **정수배** 만 (3분봉으로 5분봉은 못 만든다)
  · 리샘플 결과가 구름 계산에 필요한 표본(52+26봉)을 넘을 때만
그래서 3분 차트(5,000봉 = 약 10일)에서는 1D·1W 가 자동으로 빠진다.

담당: 연구 공용
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONV, BASE, SPANB, DISP = 9, 26, 52, 26

# 같은 가격대 재터치를 다시 표시하기까지 기다리는 봉 수 (차트 표시 전용).
COOLDOWN = 20

# 얇은 구름은 지지/저항으로 안 친다 (파트너 2026-08-18: "저런 데는 뚫리기 쉽다").
# 정통 일목도 얇은 구름을 약한 지지/저항으로 본다. 스팬 A/B 가 교차하는 근처가
# 그런 구간이다.
#
# 임계는 **그 TF 자기 두께 대비 비율**로 잡는다. 고정 % 로 하면 못 쓴다 — 실측
# 두께 중앙값이 3m 0.13% / 1h 0.75% / 4h 1.63% / 1D 6.10% 로 TF 마다 자릿수가
# 달라서, 0.1% 같은 값을 쓰면 3m 은 절반이 잘리고 1D 는 하나도 안 잘린다.
# 중앙값은 rolling 으로 구한다(전체 표본 백분위는 미래참조).
THIN_RATIO = 0.35
THIN_WINDOW = 200

# 나뇨띠가 보는 TF — 라벨: 분. 차트 TF 의 정수배만 실제로 계산된다.
TF_MINUTES: dict[str, int] = {
    "3m": 3, "5m": 5, "15m": 15, "1h": 60,
    "2h": 120, "4h": 240, "1D": 1440, "1W": 10080,
}
# 리샘플 규칙. volume 이 없는 df 로도 불리므로(마커 테스트·일부 호출부) 있는
# 컬럼만 골라 쓴다 — 없는 컬럼을 넣으면 pandas 가 KeyError 로 차트를 통째로 깬다.
_AGG = {"high": "max", "low": "min", "close": "last", "volume": "sum"}


def _agg_for(df: pd.DataFrame) -> dict[str, str]:
    return {k: v for k, v in _AGG.items() if k in df.columns}


def cloud_spans(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """차트에 그려져 있는 선행스팬 A/B (이미 시프트 적용된 값)."""
    h, lo = df["high"], df["low"]
    conv = (h.rolling(CONV).max() + lo.rolling(CONV).min()) / 2
    base = (h.rolling(BASE).max() + lo.rolling(BASE).min()) / 2
    a = ((conv + base) / 2).shift(DISP - 1)
    b = ((h.rolling(SPANB).max() + lo.rolling(SPANB).min()) / 2).shift(DISP - 1)
    return a.to_numpy(), b.to_numpy()


def base_tf_minutes(df: pd.DataFrame) -> int:
    """차트 df 의 봉 간격(분) 추정 — 중앙값이라 결측 한두 개엔 안 흔들린다."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 3:
        return 0
    diffs = pd.Series(df.index).diff().dropna()
    if diffs.empty:
        return 0
    return max(int(round(diffs.median().total_seconds() / 60)), 1)


def multi_tf_clouds(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """차트 df 에서 만들 수 있는 TF 별 구름 스팬 — df 인덱스에 정렬해 반환.

    한 봉 미루지 **않는다**. 선행스팬은 이미 25봉 전 데이터로 만들어져 진행 중인
    봉의 고저를 안 쓰므로 미래참조가 아니다. 안전하겠거니 하고 ``shift(1)`` 을
    걸었더니 상위 TF 한 봉만큼(4h 면 4시간) 레벨이 통째로 밀려, 화면 구름과 안
    맞는 자리에 터치가 찍혔다(트뷰 대조로 확인).
    """
    base_m = base_tf_minutes(df)
    if base_m <= 0:
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    need = SPANB + DISP + 5
    for label, minutes in TF_MINUTES.items():
        if minutes < base_m or minutes % base_m != 0:
            continue
        res = df if minutes == base_m else df.resample(
            f"{minutes}min",
        ).agg(_agg_for(df)).dropna()
        if len(res) < need:
            continue
        a, b = cloud_spans(res)
        sa = pd.Series(a, index=res.index)
        sb = pd.Series(b, index=res.index)
        # 얇은 구간은 아예 NaN 으로 지운다 — find_touches 가 유한값만 보므로
        # 그 구간 구름은 후보 레벨에서 자동으로 빠진다.
        thick = (sa - sb).abs() / res["close"]
        # 중앙값 창은 표본에 맞춘다. 창이 표본보다 길면(3분 차트의 1D 처럼) 중앙값이
        # 사실상 전체 평균이 돼 47% 가 잘려나갔다.
        win = min(THIN_WINDOW, max(30, len(res) // 3))
        med = thick.rolling(win, min_periods=min(30, win)).median()
        thin = thick < med * THIN_RATIO
        sa = sa.mask(thin)
        sb = sb.mask(thin)
        out[label] = (
            sa.reindex(df.index, method="ffill").to_numpy(),
            sb.reindex(df.index, method="ffill").to_numpy(),
        )
    return out


def volume_profile_levels(df: pd.DataFrame, lookback: int = 300,
                          bins: int = 40, top: int = 5) -> list[float]:
    """매물대 — 최근 lookback 봉을 가격대로 쪼개 거래량 상위 구간 중심가."""
    if "volume" not in df.columns:
        return []          # 거래량 없는 df — 매물대는 계산할 수 없다(구름·2468 만).
    w = df.iloc[-lookback:]
    if len(w) < 20:
        return []
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    mid = ((w["high"] + w["low"]) / 2).to_numpy()
    vol = w["volume"].to_numpy()
    idx = np.clip(np.digitize(mid, edges) - 1, 0, bins - 1)
    agg = np.zeros(bins)
    np.add.at(agg, idx, vol)
    return [float(centers[b]) for b in np.argsort(agg)[-top:]]


def levels_2468(price: float, direction: str, span: int = 2) -> list[float]:
    """2468 — PDF 원문 기준. 1K 단위 안의 200/400(상방) · 600/800(하방)."""
    k = int(price // 1000)
    offs = (200, 400) if direction == "up" else (600, 800)
    out: list[float] = []
    for d in range(-span, span + 1):
        b = (k + d) * 1000
        out.extend(float(b + o) for o in offs)
    return sorted(out)


def touched(prev_close: float, high: float, low: float, close: float,
            level: float, tol: float, direction: str) -> bool:
    """레벨 터치 판정.

    지지(long): 위에서 내려와 레벨에 닿고 그 위에서 버팀
    저항(short): 아래서 올라가 레벨에 닿고 그 아래로 밀림
    """
    band = level * tol
    if direction == "long":
        return prev_close > level and low <= level + band and close >= level - band
    return prev_close < level and high >= level - band and close <= level + band


def find_touches(df: pd.DataFrame, tol: float = 0.002,
                 max_bars: int = 300) -> list[dict]:
    """최근 max_bars 구간에서 지지/저항 터치 지점을 찾아 반환.

    Returns:
        [{"idx": int, "price": float, "kind": "support"|"resistance",
          "source": str, "tf": str, "count": int}]
        ``tf`` 는 마커에 붙일 라벨 — 구름이면 TF 이름("1h"), 매물대는 "매물",
        2468 은 "2468". ``count`` 는 그 자리에 겹친 **출처 종류** 수(1~3).
    """
    n = len(df)
    if n < BASE + DISP + 5:
        return []
    clouds = multi_tf_clouds(df)
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    start = max(BASE + DISP, n - max_bars)
    out: list[dict] = []
    prof: list[float] = []
    recent: list[tuple[int, float]] = []
    for i in range(start, n):
        px = c[i]
        if not np.isfinite(px):
            continue
        # 매물대는 20봉마다만 다시 잡는다. 봉마다 히스토그램을 돌리면 차트 한 번에
        # 수백 ms 가 날아가는데, 창이 300봉이라 20봉 이동으로는 값이 거의 안 변한다.
        if not prof or (i - start) % 20 == 0:
            prof = volume_profile_levels(df.iloc[: i + 1])
        for direction, kind in (("long", "support"), ("short", "resistance")):
            lvls: list[tuple[float, str, str]] = []   # (가격, 출처, TF라벨)
            for tf_label, (a, b) in clouds.items():
                for arr in (a, b):
                    if i < len(arr) and np.isfinite(arr[i]):
                        lvls.append((float(arr[i]), "cloud", tf_label))
            for p in prof:
                lvls.append((p, "profile", "매물"))
            for p in levels_2468(px, "up" if direction == "long" else "down"):
                lvls.append((p, "2468", "2468"))
            hits = [x for x in lvls
                    if touched(c[i - 1], h[i], lo[i], px, x[0], tol, direction)]
            if not hits:
                continue
            # 같은 자리를 여럿이 잡으면 **상위 TF 구름**을 라벨로 쓴다 — 나뇨띠
            # 자료가 상위 프레임 지지/저항을 더 강하게 보기 때문. 동률이면 현재가에
            # 가까운 쪽.
            best = max(
                hits,
                key=lambda x: (TF_MINUTES.get(x[2], 0), -abs(x[0] - px)),
            )
            # 겹침은 **출처 종류 수**로 센다(±0.4%). 레벨 개수로 세면 매물대가 한
            # 자리에 여럿 몰려 늘 2 이상이 나와 변별력이 없었다.
            cnt = len({
                src for p, src, _ in lvls
                if abs(p - best[0]) / best[0] <= 0.004
            })
            # 쿨다운 — 같은 자리를 연속으로 스치면 봉마다 마커가 찍혀 차트가 뭉갠다.
            if any(i - j <= COOLDOWN and abs(p - best[0]) / best[0] <= 0.004
                   for j, p in recent):
                break
            recent.append((i, best[0]))
            out.append({"idx": i, "price": best[0], "kind": kind,
                        "source": best[1], "tf": best[2], "count": cnt})
            break
    return out


def _pack_series(ts: np.ndarray, arr: np.ndarray) -> list[dict]:
    """(timestamp, 값) → 프론트가 바로 setData 할 수 있는 형태. NaN 은 뺀다."""
    return [
        {"time": int(t), "value": float(v)}
        for t, v in zip(ts, arr, strict=False)
        if np.isfinite(v)
    ]


def cloud_series(df: pd.DataFrame, tf_label: str | None = None) -> dict:
    """차트에 그릴 구름 — 선행스팬 A/B 시계열.

    tf_label 미지정이면 차트 TF 자신의 구름. 반환 형태는
    ``{"tf": "3m", "a": [{"time": ms, "value": px}, ...], "b": [...]}``.
    """
    empty = {"tf": tf_label or "", "a": [], "b": []}
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < SPANB + DISP:
        return empty
    base_m = base_tf_minutes(df)
    if tf_label and TF_MINUTES.get(tf_label, 0) != base_m:
        clouds = multi_tf_clouds(df)
        if tf_label not in clouds:
            return empty
        a, b = clouds[tf_label]
    else:
        tf_label = tf_label or next(
            (k for k, v in TF_MINUTES.items() if v == base_m), f"{base_m}m",
        )
        a, b = cloud_spans(df)
    ts = df.index.astype("int64").to_numpy() // 10**6
    return {"tf": tf_label, "a": _pack_series(ts, a), "b": _pack_series(ts, b)}
