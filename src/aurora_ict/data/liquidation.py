"""Hyperliquid 청산 지도 — 고래 포지션의 실제 청산가를 가격대별로 집계.

2026-08-20 파트너 요청: "코인글라스처럼 청산맵을 사이트에 붙이자".

코인글라스와의 차이를 먼저 적어둔다. 코인글라스 히트맵은 전체 거래소 미결제약정에
레버리지 분포를 가정해 **추정**한 값이다. 여기서 만드는 건 Hyperliquid 공개 API가
알려주는 **실제 청산가**다 — 추정이 아니라 실측이지만, 대신 Hyperliquid 한 거래소
안의, 리더보드에 잡히는 지갑만 본다. 시장 전체가 아니다.

수집 경로 (전부 무료·무인증)
  1. stats-data.hyperliquid.xyz/Mainnet/leaderboard  → 지갑 주소 목록(4만 개대)
  2. api.hyperliquid.xyz/info · clearinghouseState    → 지갑별 포지션 + liquidationPx
  3. 코인별로 모아 가격대(bin)별 **청산 시 사라질 명목금액**을 합산

스캔이 지갑 수에 비례해 오래 걸리므로(전량이면 수십 분) 백그라운드로 갱신하고
API 는 캐시에서만 읽는다. 캐시가 비어 있으면 빈 결과를 준다 — 요청을 붙잡지 않는다.

담당: 연구 공용
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_INFO_URL = "https://api.hyperliquid.xyz/info"
_LB_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

# 스캔 지갑 수. 전량(4.2만)은 수십 분이 걸려 주기 갱신에 안 맞는다. 계좌가치 상위
# 순으로 자르되, 근거리 청산은 소액 고레버 지갑이 만들므로 너무 적게 잡으면
# 현재가 근처가 비어버린다 — 8/20 실측: 상위 3,000개에서 BTC 포지션 87개.
SCAN_WALLETS = 8000
# 병렬 조회 스레드. 공개 API 라 예의상 과하지 않게.
WORKERS = 16
# 가격대 폭 — 현재가 대비 비율. 0.005 = 0.5% 간격.
BIN_PCT = 0.005
# 현재가에서 이 배율을 벗어난 청산가는 버린다(차트에 그릴 수 없는 먼 값).
RANGE_LO, RANGE_HI = 0.5, 1.8
# 캐시 수명 — 이보다 오래된 스냅샷은 stale 로 표시해 내려보낸다.
STALE_SEC = 30 * 60


@dataclass
class LiqSnapshot:
    """한 코인의 청산 지도 스냅샷."""

    coin: str
    mid: float = 0.0
    ts: int = 0
    scanned: int = 0                     # 조회한 지갑 수
    positions: int = 0                   # 그 코인 포지션 수
    bins: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        age = int(time.time()) - self.ts if self.ts else None
        return {
            "coin": self.coin,
            "mid": self.mid,
            "ts": self.ts,
            "age_sec": age,
            "stale": age is None or age > STALE_SEC,
            "scanned": self.scanned,
            "positions": self.positions,
            "bins": self.bins,
        }


_cache: dict[str, LiqSnapshot] = {}
_lock = threading.Lock()
_running: set[str] = set()


def _post(payload: dict[str, Any], timeout: int = 20) -> Any:
    req = urllib.request.Request(
        _INFO_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _wallets(limit: int) -> list[str]:
    """계좌가치 상위 지갑 주소."""
    rows = _get(_LB_URL).get("leaderboardRows", [])
    rows = [r for r in rows if float(r.get("accountValue", 0) or 0) > 0]
    rows.sort(key=lambda r: float(r["accountValue"]), reverse=True)
    return [r["ethAddress"] for r in rows[:limit]]


def _positions(addr: str, coin: str, mid: float) -> list[tuple[float, float, bool]]:
    """(청산가, 명목$, 롱여부) 목록. 실패하면 빈 목록 — 한 지갑 때문에 안 멈춘다."""
    try:
        st = _post({"type": "clearinghouseState", "user": addr})
    except Exception:
        return []
    out = []
    for p in st.get("assetPositions", []):
        pos = p.get("position", {})
        if pos.get("coin") != coin:
            continue
        liq = pos.get("liquidationPx")
        if not liq:
            continue                      # 청산가 없음 = 사실상 청산 위험 없는 포지션
        try:
            sz = float(pos["szi"])
            out.append((float(liq), abs(sz) * mid, sz > 0))
        except (TypeError, ValueError):
            continue
    return out


def scan(coin: str = "BTC", wallets: int = SCAN_WALLETS) -> LiqSnapshot:
    """실제 스캔 — 오래 걸린다. 호출자는 백그라운드에서 부를 것."""
    mid = float(_post({"type": "allMids"})[coin])
    addrs = _wallets(wallets)
    hits: list[tuple[float, float, bool]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(lambda a: _positions(a, coin, mid), addrs):
            hits.extend(res)

    step = mid * BIN_PCT
    agg: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for liq, notional, is_long in hits:
        if not (RANGE_LO * mid <= liq <= RANGE_HI * mid):
            continue
        agg[round(liq / step)][0 if is_long else 1] += notional

    bins = [
        {
            "price": round(k * step, 2),
            "pct": round((k * step / mid - 1) * 100, 2),
            "long": round(v[0], 2),
            "short": round(v[1], 2),
            "total": round(v[0] + v[1], 2),
        }
        for k, v in sorted(agg.items())
    ]
    return LiqSnapshot(coin=coin, mid=mid, ts=int(time.time()),
                       scanned=len(addrs), positions=len(hits), bins=bins)


def get_cached(coin: str = "BTC") -> dict[str, Any]:
    """API 가 읽는 곳 — 캐시만 본다. 비어 있으면 빈 결과."""
    with _lock:
        snap = _cache.get(coin)
    return snap.to_dict() if snap else LiqSnapshot(coin=coin).to_dict()


def refresh(coin: str = "BTC", wallets: int = SCAN_WALLETS) -> None:
    """백그라운드 갱신 1회. 같은 코인 스캔이 이미 돌면 건너뛴다."""
    with _lock:
        if coin in _running:
            return
        _running.add(coin)
    try:
        snap = scan(coin, wallets)
        with _lock:
            _cache[coin] = snap
        logger.info(
            "청산지도 갱신 %s — 지갑 %d개 · 포지션 %d개 · 구간 %d개",
            coin, snap.scanned, snap.positions, len(snap.bins),
        )
    except Exception as e:  # noqa: BLE001 — 갱신 실패로 봇을 멈추지 않는다
        logger.warning("청산지도 갱신 실패 %s: %s", coin, e)
    finally:
        with _lock:
            _running.discard(coin)


def refresh_async(coin: str = "BTC") -> None:
    """요청 스레드를 막지 않고 갱신을 띄운다."""
    threading.Thread(target=refresh, args=(coin,), daemon=True).start()
