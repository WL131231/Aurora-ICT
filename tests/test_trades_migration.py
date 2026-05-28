"""trades_migration — 레거시 단일 파일 → 사용자 디렉토리 이관.

검증:
    1. AURORA_ICT_LEGACY_TRADES_OWNER 미설정 → skip.
    2. 원본 (`/data/trades.jsonl`) 부재 → skip.
    3. 정상 이관 — 3 파일 모두 owner dir 로 move.
    4. 멱등 — 두 번째 호출 시 이미 이관됐으면 안 덮어쓴다.

담당: 지영민 (매매 로그 격리 PR)
"""

from __future__ import annotations

from aurora_ict.interfaces.trades_migration import migrate_legacy_trades_to_user_dir


def test_skip_when_owner_env_missing(tmp_path, monkeypatch):
    """OWNER env 미설정 → skip (안전, 데이터 보존)."""
    monkeypatch.delenv("AURORA_ICT_LEGACY_TRADES_OWNER", raising=False)
    (tmp_path / "trades.jsonl").write_text("dummy\n")
    result = migrate_legacy_trades_to_user_dir(tmp_path)
    assert result == {"status": "no_owner"}
    # 원본 그대로
    assert (tmp_path / "trades.jsonl").exists()


def test_skip_when_no_source_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_ICT_LEGACY_TRADES_OWNER", "AICT-OWNR-OWNR-OWNR")
    result = migrate_legacy_trades_to_user_dir(tmp_path)
    assert result["status"] == "no_files"


def test_full_migration_moves_three_files(tmp_path, monkeypatch):
    """원본 3 파일이 target dir 로 이동, 원본은 삭제됨."""
    code = "AICT-OWNR-OWNR-OWNR"
    monkeypatch.setenv("AURORA_ICT_LEGACY_TRADES_OWNER", code)
    (tmp_path / "trades.jsonl").write_text('{"event":"entry"}\n')
    (tmp_path / "trades.db").write_bytes(b"SQLite format 3\x00fake")
    (tmp_path / "trade_journal.log").write_text("2026-05-29 ...")

    result = migrate_legacy_trades_to_user_dir(tmp_path)

    assert result["status"] == "moved"
    target = tmp_path / "users" / code
    assert (target / "trades.jsonl").exists()
    assert (target / "trades.db").exists()
    assert (target / "trade_journal.log").exists()
    # 원본 제거
    assert not (tmp_path / "trades.jsonl").exists()
    assert not (tmp_path / "trades.db").exists()
    # 마커 파일 생성
    assert (tmp_path / ".trades_migrated").exists()
    assert "owner=" + code in (tmp_path / ".trades_migrated").read_text()


def test_idempotent_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    """이미 target dir 에 trades.jsonl 있으면 원본을 덮지 않는다."""
    code = "AICT-IDEM-IDEM-IDEM"
    monkeypatch.setenv("AURORA_ICT_LEGACY_TRADES_OWNER", code)
    # target 에 이미 데이터 있음.
    target = tmp_path / "users" / code
    target.mkdir(parents=True, exist_ok=True)
    (target / "trades.jsonl").write_text('{"existing":"row"}\n')
    # 원본도 있음 (잘못된 상태 시뮬레이션).
    (tmp_path / "trades.jsonl").write_text('{"legacy":"row"}\n')

    result = migrate_legacy_trades_to_user_dir(tmp_path)

    assert result["status"] == "already_migrated"
    # target 에 있던 데이터 그대로 유지.
    assert (target / "trades.jsonl").read_text() == '{"existing":"row"}\n'
    # 원본도 안 건드림 (안전).
    assert (tmp_path / "trades.jsonl").read_text() == '{"legacy":"row"}\n'


def test_returns_no_data_dir_when_path_missing(tmp_path, monkeypatch):
    """data_dir 자체가 없으면 skip — Fly 첫 부팅 대비."""
    monkeypatch.setenv("AURORA_ICT_LEGACY_TRADES_OWNER", "AICT-X-X-X")
    nonexistent = tmp_path / "does_not_exist"
    result = migrate_legacy_trades_to_user_dir(nonexistent)
    assert result == {"status": "no_data_dir"}
