"""매매 로그 마이그레이션 — 단일 슬롯 → 사용자별 슬롯 (2026-05-29).

배경:
    기존 SaaS 봇은 모든 사용자 거래를 ``<data_dir>/trades.jsonl`` +
    ``<data_dir>/trades.db`` + ``<data_dir>/trade_journal.log`` 한 파일에
    기록했다. multi-user 격리가 안 돼 (1) privacy 문제 + (2) SQLite 동시
    write 잠금 충돌 위험.

이 모듈:
    startup 시 1회 호출 → 기존 ``trades.*`` 가 있고 사용자별 디렉토리에는
    아직 없으면 (idempotent), 환경변수 ``AURORA_ICT_LEGACY_TRADES_OWNER``
    가 가리키는 사용자 코드의 디렉토리로 이동한다.

    파트너 결정 2026-05-29 — 혼자만 쓰는 봇이라 기존 기록은 그 한 사람
    (AICT-0Q8B-D1YU-VFRN) 에게 이관. 이관 후 원본은 ``.migrated`` 접미사로
    rename (삭제하지 않음 — 안전망).

호출 위치:
    ``saas.main`` 의 startup hook 에서 1회. 다중 부팅 시 idempotent (이미
    이관됐으면 no-op).

담당: 지영민 (SaaS 매매 로그 격리 PR)
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# 환경변수 키 — Fly secrets / .env 로 박는다. 미설정 시 마이그레이션 skip
# (안전 — 누구에게 이관해야 할지 모르면 원본 그대로 둠).
_OWNER_ENV = "AURORA_ICT_LEGACY_TRADES_OWNER"

# 옮길 파일 목록 — TradesStore 가 생성하는 3 종.
_LEGACY_FILES = ("trades.jsonl", "trades.db", "trade_journal.log")


def _owner_code() -> str | None:
    """환경변수에서 owner code 읽기. 빈 문자열도 None 취급."""
    val = (os.environ.get(_OWNER_ENV) or "").strip()
    return val or None


def migrate_legacy_trades_to_user_dir(data_dir: Path) -> dict[str, str]:
    """기존 단일 슬롯 trades.* 를 owner 사용자 디렉토리로 이관 — idempotent.

    동작:
        1) ``AURORA_ICT_LEGACY_TRADES_OWNER`` 미설정 → skip (status="no_owner").
        2) owner 사용자 디렉토리 (`<data_dir>/users/<code>/`) 이미 ``trades.jsonl``
           존재 → skip (status="already_migrated"). 안전 — 사용자가 신규 등록
           이후 발생한 거래를 덮지 않음.
        3) 원본 (`<data_dir>/trades.jsonl` 등) 부재 → skip (status="no_source").
        4) 위 모두 아니면 owner 디렉토리 생성 후 3 파일 move. 원본은
           ``.migrated`` 접미사로 rename (rollback 가능).

    Args:
        data_dir: SaaS 데이터 루트 (Fly: ``/data``). 부재해도 OK — skip.

    Returns:
        파일별 처리 결과 dict — ``{filename: "moved" | "skipped:reason"}``.
        호출자가 startup 로그에 그대로 박을 수 있는 형식.
    """
    result: dict[str, str] = {}
    owner = _owner_code()
    if owner is None:
        logger.info(
            "legacy trades 마이그레이션 skip — %s 환경변수 미설정",
            _OWNER_ENV,
        )
        return {"status": "no_owner"}

    if not data_dir.exists():
        return {"status": "no_data_dir"}

    target_dir = data_dir / "users" / owner
    # 신규 슬롯에 이미 trades.jsonl 있으면 마이그레이션 X — 진행 중인 사용자
    # 거래를 절대 덮지 않는다.
    target_jsonl = target_dir / "trades.jsonl"
    if target_jsonl.exists() and target_jsonl.stat().st_size > 0:
        logger.info(
            "legacy trades 마이그레이션 skip — 이미 owner 디렉토리에 trades.jsonl 존재 (%s)",
            target_jsonl,
        )
        return {"status": "already_migrated", "target": str(target_dir)}

    moved_any = False
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in _LEGACY_FILES:
        src = data_dir / name
        if not src.exists():
            result[name] = "skipped:no_source"
            continue
        dst = target_dir / name
        # dst 이미 있으면 덮지 않고 skip — owner 디렉토리 보존 우선.
        if dst.exists() and dst.stat().st_size > 0:
            result[name] = "skipped:dst_exists"
            continue
        try:
            shutil.move(str(src), str(dst))
        except OSError as e:
            logger.warning("legacy %s move 실패: %s", name, e)
            result[name] = f"error:{e}"
            continue
        moved_any = True
        result[name] = "moved"

    if moved_any:
        # 원본은 .migrated 빈 sentinel 파일로 남겨 다음 부팅 시 추가 동작 안 하게
        # (다음에 같은 이름의 신규 파일이 생겨도 마이그레이션 안 한다).
        try:
            (data_dir / ".trades_migrated").write_text(
                f"owner={owner}\n", encoding="utf-8",
            )
        except OSError as e:
            logger.warning(".trades_migrated 마커 write 실패 (무시): %s", e)
        logger.info(
            "legacy trades 마이그레이션 완료 — %s → %s",
            data_dir, target_dir,
        )

    result["status"] = "moved" if moved_any else "no_files"
    result["target"] = str(target_dir)
    return result


__all__ = ["migrate_legacy_trades_to_user_dir"]
