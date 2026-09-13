"""로컬 CSV 분석의 계약·보수적 집계·비밀 보호를 합성 입력으로 검증한다.

담당: Codex. 외부 네트워크와 거래소 호출, 외부 mock을 사용하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import trade_analysis as analysis

START = 1_800_000_000_000


def row(kind="entry", offset=0, **changes):
    """합성 CSV 한 행을 만든다.

    Args: kind: 이벤트. offset: 시간 차. changes: 열 덮어쓰기.
    Returns: 행 사전.
    Raises: 없음.
    """
    result = dict(zip(analysis.HEADERS, (
        "PRIVATE_ACCOUNT_A", str(START + offset), kind, "live", "Cursus 1.0",
        "BTC/USDT:USDT", "long", "100", "1", "" if kind == "entry" else "1", "", "",
    ), strict=True))
    result.update(changes)
    return result


def payload(*rows):
    """표준 CSV 작성기로 인용과 개행을 포함한 합성 바이트를 만든다.

    Args: rows: 합성 행.
    Returns: UTF-8 CSV.
    Raises: 없음.
    """
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=analysis.HEADERS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def report(*rows):
    """CSV 파싱부터 분석까지 실제 순수 함수를 호출한다.

    Args: rows: 합성 행.
    Returns: 분석 결과.
    Raises: AnalysisError: 잘못된 합성 데이터.
    """
    return analysis.analyze(analysis.parse_csv(payload(*rows)))


def test_four_partial_exits_are_one_losing_position():
    result = report(
        row(), row("tp_hit", 1, qty="0.25", pnl_usdt="1"),
        row("tp_hit", 2, qty="0.25", pnl_usdt="1"),
        row("tp_hit", 3, qty="0.25", pnl_usdt="1"),
        row("sl_hit", 4, qty="0.25", pnl_usdt="-4"),
    )
    assert result["summary"]["positions"] == 1
    assert result["summary"]["recorded_win_rate_pct"] == 0
    assert result["summary"]["recorded_pnl_usdt"] == "-1"
    assert result["positions"][0]["close_events"] == 4
    assert result["net_pnl_usdt"] is None
    assert result["r_multiple"] is None
    assert "PRIVATE_ACCOUNT_A" not in json.dumps(result)


def test_multiple_cursus_positions_without_setup_are_not_merged():
    result = report(
        row("entry", 0), row("flip_close", 1, pnl_usdt="-2"),
        row("entry", 2, direction="short"),
        row("tp_hit", 3, direction="short", pnl_usdt="3"),
    )
    assert result["summary"]["positions"] == 2
    assert result["summary"]["recorded_win_rate_pct"] == 50
    assert result["summary"]["recorded_profit_factor"] == 1.5


def test_late_extra_close_invalidates_previously_matched_quantity():
    result = report(row(), row("tp_hit", 1), row("sl_hit", 2, qty="0.5", pnl_usdt="-20"))
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["closed_quantity_exceeds_entry"] == 1


def test_late_old_setup_after_new_entry_invalidates_both_candidates():
    first, second = str(START - 100), str(START - 50)
    result = report(
        row(setup_ts_ms=first), row("tp_hit", 1, setup_ts_ms=first, pnl_usdt="10"),
        row("entry", 2, setup_ts_ms=second),
        row("sl_hit", 3, setup_ts_ms=first, qty="0.5", pnl_usdt="-20"),
        row("tp_hit", 4, setup_ts_ms=second),
    )
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["setup_crosses_entry_boundaries"] == 2


def test_partial_record_gap_is_unknown_not_a_live_open_position():
    result = report(row(), row("sync_close", 1, qty="0.5"))
    assert result["summary"]["positions"] == 0
    assert result["quality"]["incomplete_or_open_unknown_candidates"] == 1


def test_small_unclosed_quantity_cannot_round_to_complete():
    result = report(row(qty="0.0000000000001"))
    assert result["summary"]["positions"] == 0
    assert result["quality"]["incomplete_or_open_unknown_candidates"] == 1


@pytest.mark.parametrize("kind", ["recovered", "sync_close"])
def test_recovery_or_alarm_does_not_prove_quantity_close(kind):
    result = report(row(), row(kind, 1, pnl_usdt="", reason="ALARM: close failed"))
    assert result["summary"]["positions"] == 0
    assert result["quality"]["incomplete_or_open_unknown_candidates"] == 1


@pytest.mark.parametrize("reason", ["reconcile: repair", "backfill: repair", "ALARM: failure"])
def test_repair_records_are_quarantined(reason):
    result = report(row(), row("sync_close", 1, reason=reason))
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["repair_or_alarm_record"] == 1


@pytest.mark.parametrize("changes", [
    {"direction": "short"}, {"model": "Origo 2.2"}, {"setup_ts_ms": str(START - 1)},
])
def test_conflicting_position_metadata_is_quarantined(changes):
    result = report(row(), row("sl_hit", 1, **changes))
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["position_metadata_mismatch"] == 1


def test_setup_zero_is_valid_unknown_identifier():
    events = analysis.parse_csv(payload(row(setup_ts_ms="0")))
    assert events[0].setup is None


def test_origo_flip_open_is_an_entry():
    result = report(
        row("flip_open", 0, model="Origo 2.2", setup_ts_ms=str(START), pnl_usdt=""),
        row("sync_close", 1, model="Origo 2.2", setup_ts_ms=str(START)),
    )
    assert result["summary"]["positions"] == 1


def test_duplicates_are_flagged_not_silently_removed():
    exit_row = row("tp_hit", 1)
    result = report(row(), exit_row, exit_row)
    assert result["input"]["rows"] == 3
    assert result["quality"]["exact_duplicate_extra_rows"] == 1
    assert result["summary"]["positions"] == 0


def test_same_timestamp_reverse_order_is_ambiguous():
    result = report(row("sl_hit", 0), row())
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["ambiguous_same_timestamp"] == 1


def test_overlapping_entries_are_quarantined():
    result = report(row(), row("entry", 1), row("sl_hit", 2))
    assert result["summary"]["positions"] == 0
    assert result["quality"]["excluded_candidates"] == 2


def test_late_recovery_invalidates_quantity_matched_candidate():
    result = report(row(), row("tp_hit", 1), row("recovered", 2, pnl_usdt=""))
    assert result["summary"]["positions"] == 0
    assert result["excluded_reasons"]["recovery_during_position"] == 1


@pytest.mark.parametrize("changes", [{"mode": "demo"}, {"mode": ""}, {"model": ""}])
def test_nonlive_or_unknown_metadata_is_excluded(changes):
    result = report(row(**changes), row("sl_hit", 1, **changes))
    assert result["summary"]["positions"] == 0


def test_user_and_symbol_streams_are_isolated_and_time_sorted():
    result = report(
        row("sl_hit", 1, pnl_usdt="-2"), row(),
        row("entry", 0, user_code="PRIVATE_ACCOUNT_B"),
        row("tp_hit", 1, user_code="PRIVATE_ACCOUNT_B", pnl_usdt="3"),
        row("sync_close", 2, symbol="ETH/USDT:USDT"),
    )
    assert result["summary"]["positions"] == 2
    assert len(result["by"]["account"]) == 2
    assert result["quality"]["unmatched_close_events"] == 1


def test_breakeven_is_not_a_loss_and_pf_without_losses_is_null():
    result = report(row(), row("tp_hit", 1, pnl_usdt="0"))
    assert result["summary"]["breakeven"] == 1
    assert result["summary"]["losses"] == 0
    assert result["summary"]["recorded_profit_factor"] is None


def test_month_is_final_close_in_kst():
    close_ts = int(datetime(2026, 8, 31, 15, 0, tzinfo=UTC).timestamp() * 1000)
    result = report(row(ts_ms=str(close_ts - 1)), row("tp_hit", 1, ts_ms=str(close_ts)))
    assert result["positions"][0]["month_kst"] == "2026-09"


def test_empty_export_is_valid_but_not_a_zero_percent_win_rate():
    result = report()
    assert result["input"]["rows"] == 0
    assert result["summary"]["recorded_win_rate_pct"] is None


def test_bom_quoted_comma_and_multiline_reason():
    events = analysis.parse_csv(b"\xef\xbb\xbf" + payload(row(reason='한국어, "인용"\n다음 줄')))
    assert events[0].reason == '한국어, "인용"\n다음 줄'


def test_actual_server_serializer_matches_parser_contract():
    from aurora_ict.api.trades_router import _trades_to_csv

    exported = _trades_to_csv([
        row(ts_ms=START, qty=1.0, price=100.0, pnl_usdt=None, setup_ts_ms=0),
    ], with_user_code=True)
    events = analysis.parse_csv(exported.encode("utf-8"))
    assert events[0].setup is None
    assert events[0].pnl is None


@pytest.mark.parametrize("field,value", [
    ("qty", "NaN"), ("price", "Infinity"), ("pnl_usdt", "-Infinity"),
    ("qty", "0"), ("price", "-1"), ("pnl_usdt", "1e100"),
    ("pnl_usdt", "1e999999999"),
    ("pnl_usdt", "1e-99999"), ("ts_ms", "1.5"), ("ts_ms", "0"),
    ("ts_ms", "9" * 5000), ("setup_ts_ms", "-1"), ("direction", "buy"),
    ("event_type", "new_unknown_type"), ("user_code", ""), ("symbol", ""),
])
def test_invalid_values_fail_closed_without_echoing_values(field, value):
    with pytest.raises(analysis.AnalysisError):
        analysis.parse_csv(payload(row(**{field: value})))


@pytest.mark.parametrize("body", [
    b"<html>login</html>", b"user_code,ts_ms\nuser,123\n", b"\xff",
    payload(row()).replace(b",100,1,", b",100,1,extra,"),
    payload(row()) + b'"unterminated',
])
def test_malformed_csv_rejected(body):
    with pytest.raises(analysis.AnalysisError):
        analysis.parse_csv(body)


def test_server_cap_warns_without_claiming_complete_history():
    event = analysis.parse_csv(payload(row("recovered", pnl_usdt="")))[0]
    result = analysis.analyze([event] * analysis.EXPORT_ROW_CAP)
    assert result["quality"]["accounts_at_export_cap"] == 1


def test_token_is_only_a_header_on_fixed_get_url():
    request = analysis.make_request("synthetic-test-token")
    assert request.method == "GET"
    assert str(request.url) == analysis.EXPORT_URL
    assert request.headers["X-Admin-Token"] == "synthetic-test-token"
    assert request.content == b""
    assert "synthetic-test-token" not in str(request.url)


@pytest.mark.parametrize("token", ["", "contains space", "newline\r\nattack", "한글"])
def test_invalid_tokens_never_form_a_request(token):
    with pytest.raises(analysis.AnalysisError):
        analysis.make_request(token)


@pytest.mark.parametrize("status,mime,length", [
    (401, "text/csv", ""), (403, "text/csv", ""), (302, "text/csv", ""),
    (500, "text/csv", ""), (200, "text/html", ""),
    (200, "text/csv", str(analysis.MAX_BYTES + 1)),
    (200, "text/csv", "invalid"), (200, "text/csv", "9" * 5000),
    (200, "text/csv", "²"),
])
def test_response_rejects_auth_redirect_html_and_oversize(status, mime, length):
    with pytest.raises(analysis.AnalysisError):
        analysis.validate_response(status, mime, length)


def test_valid_response_contract():
    analysis.validate_response(200, "text/csv; charset=utf-8", "123")


def test_snapshots_do_not_overwrite_or_merge_and_have_provenance(tmp_path):
    raw = payload(row(), row("tp_hit", 1))
    result = analysis.analyze(analysis.parse_csv(raw))
    first = analysis.save_snapshot(raw, result, "local_csv", root=tmp_path)
    second = analysis.save_snapshot(raw, result, "local_csv", root=tmp_path)
    assert first != second
    assert (first / "raw.csv").read_bytes() == raw
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"]
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "PRIVATE_ACCOUNT_A" not in (first / "report.json").read_text(encoding="utf-8")
    assert "PRIVATE_ACCOUNT_A" not in (first / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        analysis._write_new(first / "raw.csv", b"overwrite")


def test_cli_offline_end_to_end_and_missing_auth_without_network(tmp_path):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "trade_analysis.py"
    shutil.copyfile(Path(analysis.__file__), script)
    source = tmp_path / "synthetic.csv"
    source.write_bytes(payload(row(), row("tp_hit", 1)))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop(analysis.TOKEN_ENV, None)
    run = subprocess.run(
        [sys.executable, str(script), "analyze", str(source)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=20, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert len(list((tmp_path / "data" / "trade_analysis").glob("*/manifest.json"))) == 1
    no_auth = subprocess.run(
        [sys.executable, str(script), "sync"], capture_output=True,
        text=True, encoding="utf-8", env=env, timeout=20, check=False,
    )
    assert no_auth.returncode == 1
    assert "토큰" in no_auth.stderr
    assert "Traceback" not in no_auth.stderr
