"""관리 CSV 수집과 보수적인 포지션 단위 분석을 위한 로컬 도구.

담당: Codex. 봇 런타임과 분리하며 주문, 설정 변경, 배포, LLM 호출은 하지 않는다.
CSV의 손익은 비용 포함 여부가 혼재하므로 순손익이나 시드 수익률로 해석하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import httpx

EXPORT_URL = "https://aurora-ict-one.fly.dev/admin/trades/export-all"
TOKEN_ENV = "AURORA_ICT_ADMIN_TOKEN"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "trade_analysis"
MAX_BYTES = 50 * 1024 * 1024  # 실수로 거대 응답을 저장하지 않도록 50 MiB로 제한.
EXPORT_ROW_CAP = 200_000  # 현재 서버 내보내기의 사용자별 조회 상한.
HEADERS = (
    "user_code", "ts_ms", "event_type", "mode", "model", "symbol", "direction",
    "price", "qty", "pnl_usdt", "setup_ts_ms", "reason",
)
ENTRIES = {"entry", "flip_open"}
CLOSES = {"sl_hit", "tp_hit", "flip_close", "sync_close", "manual_close"}
EVENTS = ENTRIES | CLOSES | {"recovered"}
KST = timezone(timedelta(hours=9))
LIMITATIONS = [
    "CSV 수량 일치 재구성일 뿐 거래소 체결 원장과 대조한 확정 성과가 아니다.",
    "pnl_usdt는 거래소 값과 가격차 추정값이 혼재한다. 수수료·펀딩 포함 여부를 확정할 수 없다.",
    "이벤트 승률은 계산하지 않는다. 불확실 포지션을 제외한 표본에는 선택 편향이 있다.",
    "수량 미완결은 실제 미청산 또는 기록 누락이다. 현재 보유 포지션으로 단정하지 않는다.",
    "setup 식별자가 없는 Cursus의 지연·오배정 청산은 CSV만으로 완전히 식별할 수 없다.",
    "CSV에는 진입 시드·초기 손절·상세 체결·context_json이 없어 시드%, R, 순손익, 계좌 MDD는 미산출이다.",
    "사용자별 200000행 상한, SQLite 조회 원본, 같은 시각 순서 부재로 전체 기록 완전성은 미확인이다.",
    "스냅샷은 매번 독립 분석한다. 이전 CSV와 이어 붙이거나 중복 행을 임의 삭제하지 않는다.",
    "월별 통계는 KST 최종 청산 기록 시각 기준이며 같은 포지션의 부분 청산을 한 번만 센다.",
    "이 보고서만으로 전략을 채택하거나 배포하지 않는다. AGENTS.md 제5절 검증이 별도로 필요하다.",
]


class AnalysisError(ValueError):
    """원본 값이나 인증 정보를 포함하지 않는 사용자 표시용 오류."""


@dataclass(frozen=True)
class Event:
    """검증된 CSV 한 행. 사용자 식별자는 로컬 메모리와 원본에만 둔다."""

    account: str
    ts: int
    kind: str
    mode: str
    model: str
    symbol: str
    direction: str
    price: Decimal
    qty: Decimal
    pnl: Decimal | None
    setup: int | None
    reason: str


def _number(value: str, row: int, column: str, *, positive: bool = False) -> Decimal:
    """유한한 십진수를 검증하며 잘못된 원본 값은 오류에 노출하지 않는다.

    Args: value: 원문. row: 행 번호. column: 열 이름. positive: 양수 필수 여부.
    Returns: 검증된 십진수.
    Raises: AnalysisError: 숫자 형식, 범위 또는 부호 오류.
    """
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise AnalysisError(f"CSV {row}행 {column}: 숫자 형식 오류") from None
    if not result.is_finite() or result.copy_abs() > Decimal("1e30"):
        raise AnalysisError(f"CSV {row}행 {column}: 숫자 범위 오류")
    if result and result.adjusted() < -18:
        raise AnalysisError(f"CSV {row}행 {column}: 지원하는 최소 단위 미만")
    if positive and result <= 0:
        raise AnalysisError(f"CSV {row}행 {column}: 양수 필요")
    return result if result else Decimal(0)


def _timestamp(value: str, row: int, column: str) -> int:
    """UTC 밀리초 정수를 검증한다.

    Args: value: 원문. row: 행 번호. column: 열 이름.
    Returns: 검증된 밀리초.
    Raises: AnalysisError: 잘못된 시각.
    """
    if len(value) > 15 or not value.isascii() or not value.isdigit():
        raise AnalysisError(f"CSV {row}행 {column}: 정수 밀리초 필요")
    result = int(value)
    if not 0 < result < 253402214400000:  # KST 변환까지 datetime 범위 안에 둔다.
        raise AnalysisError(f"CSV {row}행 {column}: 시각 범위 오류")
    return result


def parse_csv(payload: bytes) -> list[Event]:
    """현재 전체 사용자 CSV 계약을 엄격하게 읽는다.

    Args: payload: BOM 유무와 무관한 UTF-8 CSV.
    Returns: 검증된 이벤트. 중복은 삭제하지 않는다.
    Raises: AnalysisError: 크기, 인코딩, 스키마 또는 행 오류.
    """
    if len(payload) > MAX_BYTES:
        raise AnalysisError("CSV가 50 MiB 상한을 초과했다.")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), strict=True)
        if reader.fieldnames != list(HEADERS):
            raise AnalysisError("전체 사용자 CSV 스키마가 다르다. 현재 내보내기 계약 확인이 필요하다.")
        events = []
        for row_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise AnalysisError(f"CSV {row_number}행: 열 개수 불일치")
            if any(not row[key].strip() for key in ("user_code", "symbol")):
                raise AnalysisError(f"CSV {row_number}행: 사용자 또는 심볼 누락")
            if row["event_type"] not in EVENTS or row["direction"] not in {"long", "short"}:
                raise AnalysisError(f"CSV {row_number}행: 이벤트 또는 방향 계약 불일치")
            events.append(Event(
                account=row["user_code"],
                ts=_timestamp(row["ts_ms"], row_number, "ts_ms"),
                kind=row["event_type"], mode=row["mode"], model=row["model"],
                symbol=row["symbol"], direction=row["direction"],
                price=_number(row["price"], row_number, "price", positive=True),
                qty=_number(row["qty"], row_number, "qty", positive=True),
                pnl=_number(row["pnl_usdt"], row_number, "pnl_usdt")
                if row["pnl_usdt"] else None,
                setup=_timestamp(row["setup_ts_ms"], row_number, "setup_ts_ms")
                if row["setup_ts_ms"] not in {"", "0"} else None,
                reason=row["reason"],
            ))
        return events
    except (UnicodeError, csv.Error):
        raise AnalysisError("CSV 인코딩 또는 인용부호 형식 오류") from None


def make_request(token: str) -> httpx.Request:
    """비밀을 고정 HTTPS 내보내기 주소의 헤더에만 싣는다.

    Args: token: 환경변수 또는 비표시 입력으로 받은 관리자 토큰.
    Returns: 아직 전송하지 않은 GET 요청.
    Raises: AnalysisError: 토큰 부재 또는 잘못된 헤더 문자.
    """
    if not token or any(not 33 <= ord(char) <= 126 for char in token):
        raise AnalysisError("관리자 토큰이 없거나 HTTP 헤더에 쓸 수 없는 문자가 있다.")
    return httpx.Request("GET", EXPORT_URL, headers={
        "X-Admin-Token": token, "Accept": "text/csv", "Accept-Encoding": "identity",
    })


def validate_response(status: int, content_type: str, content_length: str = "") -> None:
    """오류 본문과 리다이렉트를 저장하지 않고 응답 계약을 검증한다.

    Args: status: HTTP 상태. content_type: MIME 형식. content_length: 길이 헤더.
    Returns: 없음.
    Raises: AnalysisError: 인증, 상태, MIME 또는 크기 오류.
    """
    if status in {401, 403}:
        raise AnalysisError("관리자 인증이 거부됐다. 토큰을 확인해야 한다.")
    if status != 200:
        raise AnalysisError(f"CSV 요청 실패: HTTP {status}. 리다이렉트와 자동 재시도는 하지 않는다.")
    if content_type.split(";", 1)[0].strip().lower() != "text/csv":
        raise AnalysisError("CSV가 아닌 응답이다. 저장하지 않았다.")
    if content_length:
        if (len(content_length) > 20 or not content_length.isascii()
                or not content_length.isdigit() or int(content_length) > MAX_BYTES):
            raise AnalysisError("응답 크기 헤더가 잘못됐거나 50 MiB 상한을 초과했다.")


def download_csv(token: str) -> bytes:
    """관리 API를 한 번 읽는다. 프록시 환경변수와 리다이렉트는 사용하지 않는다.

    Args: token: 메모리에만 유지하는 관리자 토큰.
    Returns: 크기를 제한한 CSV 바이트.
    Raises: AnalysisError: 네트워크 또는 응답 검증 오류.
    """
    request = make_request(token)
    try:
        with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
            response = client.send(request, stream=True)
            try:
                validate_response(
                    response.status_code, response.headers.get("content-type", ""),
                    response.headers.get("content-length", ""),
                )
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > MAX_BYTES:
                        raise AnalysisError("CSV가 50 MiB 상한을 초과했다.")
                    body.extend(chunk)
                return bytes(body)
            finally:
                response.close()
    except httpx.HTTPError:
        raise AnalysisError("CSV 네트워크 요청 실패. 인증 정보와 응답 본문은 출력하지 않았다.") from None


def _iso(ts: int) -> str:
    """UTC 밀리초를 KST ISO 시각으로 바꾼다.

    Args: ts: 검증된 시각.
    Returns: KST 문자열.
    Raises: 없음.
    """
    return datetime.fromtimestamp(ts / 1000, UTC).astimezone(KST).isoformat()


def _metrics(positions: list[dict]) -> dict:
    """수량 일치 재구성 포지션만으로 기록 손익 통계를 계산한다.

    Args: positions: 불확실 표본을 제외한 포지션.
    Returns: 손익 문자열과 포지션 단위 승률/PF. 분모 부재는 None.
    Raises: 없음.
    """
    values = [Decimal(position["recorded_pnl_usdt"]) for position in positions]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gain, loss = sum(wins, Decimal(0)), -sum(losses, Decimal(0))
    return {
        "positions": len(values), "wins": len(wins), "losses": len(losses),
        "breakeven": values.count(Decimal(0)),
        "recorded_pnl_usdt": str(sum(values, Decimal(0))),
        "recorded_win_rate_pct": round(100 * len(wins) / len(values), 4) if values else None,
        "recorded_profit_factor": float(gain / loss) if loss else None,
        "mean_recorded_pnl_usdt": str(sum(values, Decimal(0)) / len(values))
        if values else None,
    }


def analyze(events: list[Event]) -> dict:
    """진입에서 청산까지 수량을 연결하고 불확실 기록은 별도 격리한다.

    Args: events: 원본 한 스냅샷의 검증된 이벤트 전체.
    Returns: 익명 계정별 포지션, 품질 경고, 모델/페어/월별 기록 통계.
    Raises: 없음.
    """
    aliases = {account: f"account_{i:03d}"
               for i, account in enumerate(sorted({event.account for event in events}), 1)}
    streams = defaultdict(list)
    for event in events:
        streams[(event.account, event.mode, event.symbol)].append(event)
    diagnostics = Counter()
    excluded = Counter()
    positions = []
    frequencies = Counter(events)
    diagnostics["exact_duplicate_extra_rows"] = sum(count - 1 for count in frequencies.values())
    account_counts = Counter(event.account for event in events)
    diagnostics["accounts_at_export_cap"] = sum(
        count >= EXPORT_ROW_CAP for count in account_counts.values()
    )

    def finish(candidate: dict) -> None:
        """다음 진입 또는 파일 끝에서만 후보를 확정해 뒤늦은 초과 청산도 검사한다.

        Args: candidate: 한 진입 경계 안의 전체 기록.
        Returns: 없음. 통계와 제외 사유를 갱신한다.
        Raises: 없음.
        """
        entry = candidate["entry"]
        tolerance = entry.qty * Decimal("0.000001")  # CSV 수량 반올림만 허용하는 상대 오차.
        if candidate["closed_qty"] < entry.qty - tolerance:
            candidate["flags"].add("incomplete_or_open_unknown")
            diagnostics["incomplete_or_open_unknown_candidates"] += 1
        if candidate["closed_qty"] > entry.qty + tolerance:
            candidate["flags"].add("closed_quantity_exceeds_entry")
        if candidate["flags"]:
            diagnostics["excluded_candidates"] += 1
            excluded.update(candidate["flags"])
            return
        positions.append({
            "account": aliases[entry.account], "model": entry.model,
            "symbol": entry.symbol, "direction": entry.direction,
            "entry_at_kst": _iso(entry.ts), "close_at_kst": _iso(candidate["last_close_ts"]),
            "month_kst": _iso(candidate["last_close_ts"])[:7], "entry_qty": str(entry.qty),
            "closed_qty": str(candidate["closed_qty"]),
            "close_events": candidate["close_events"],
            "recorded_pnl_usdt": str(candidate["pnl"]),
        })

    for stream in streams.values():
        stream.sort(key=lambda event: event.ts)
        same_time = Counter(event.ts for event in stream)
        setup_boundaries = defaultdict(set)
        boundary = 0
        for event in stream:
            if event.kind in ENTRIES:
                boundary += 1
            if event.setup is not None:
                setup_boundaries[event.setup].add(boundary)
        # 이전 setup의 지연 청산이 다음 진입 뒤에 기록돼도 이전 후보까지 제외한다.
        crossed_setups = {setup for setup, bounds in setup_boundaries.items() if len(bounds) > 1}
        active = None
        for event in stream:
            flags = set()
            if same_time[event.ts] > 1:
                flags.add("ambiguous_same_timestamp")
            if frequencies[event] > 1:
                flags.add("duplicate_rows")
            if event.setup in crossed_setups:
                flags.add("setup_crosses_entry_boundaries")
            if event.mode != "live" or not event.model:
                flags.add("nonlive_or_unknown_metadata")
            if event.reason.lower().startswith(("reconcile:", "backfill:", "alarm:")):
                flags.add("repair_or_alarm_record")
            if event.kind in ENTRIES:
                if active is not None:
                    tolerance = active["entry"].qty * Decimal("0.000001")
                    if active["closed_qty"] < active["entry"].qty - tolerance:
                        active["flags"].add("entry_before_quantity_complete")
                        flags.add("entry_before_quantity_complete")
                    finish(active)
                active = {"entry": event, "closed_qty": Decimal(0), "pnl": Decimal(0),
                          "flags": flags, "close_events": 0}
                if event.pnl is not None:
                    active["flags"].add("entry_has_pnl")
                continue
            if event.kind == "recovered":
                diagnostics["recovered_events"] += 1
                if active is not None:
                    active["flags"].update(flags | {"recovery_during_position"})
                continue
            if active is None:
                diagnostics["unmatched_close_events"] += 1
                continue
            entry = active["entry"]
            active["flags"].update(flags)
            if (event.direction, event.model, event.setup) != (
                entry.direction, entry.model, entry.setup,
            ):
                active["flags"].add("position_metadata_mismatch")
            if event.pnl is None:
                # 알람도 sync_close로 기록될 수 있으므로 수량 소진의 증거로 쓰지 않는다.
                active["flags"].add("missing_close_pnl")
                continue
            active["closed_qty"] += event.qty
            active["pnl"] += event.pnl
            active["close_events"] += 1
            active["last_close_ts"] = event.ts
        if active is not None:
            finish(active)
    grouped = {}
    for dimension in ("account", "model", "symbol", "direction", "month_kst"):
        groups = defaultdict(list)
        for position in positions:
            groups[position[dimension]].append(position)
        grouped[dimension] = {key: _metrics(group) for key, group in sorted(groups.items())}
    return {
        "schema_version": 1,
        "scope": "csv_quantity_matched_positions_not_exchange_verified",
        "input": {
            "rows": len(events), "accounts": len(aliases),
            "first_event_kst": _iso(min(event.ts for event in events)) if events else None,
            "last_event_kst": _iso(max(event.ts for event in events)) if events else None,
            "events_by_type": dict(sorted(Counter(event.kind for event in events).items())),
        },
        "quality": dict(sorted(diagnostics.items())),
        "excluded_reasons": dict(sorted(excluded.items())),
        "summary": _metrics(positions), "by": grouped, "positions": positions,
        "net_pnl_usdt": None, "seed_return_pct": None, "r_multiple": None,
        "account_mdd_pct": None,
        "limitations": LIMITATIONS,
    }


def render_markdown(report: dict) -> str:
    """원본 자유 텍스트와 계정 식별자를 싣지 않는 짧은 보고서를 만든다.

    Args: report: 분석 결과.
    Returns: Markdown 보고서.
    Raises: 없음.
    """
    summary, quality = report["summary"], report["quality"]
    lines = [
        "# CSV 분석 보고서", "",
        "실거래 검증 성과가 아닌, CSV 수량 일치 포지션의 기록 통계다.", "",
        f"- 원본: {report['input']['rows']}행 / {report['input']['accounts']}계정",
        f"- 수량 일치 재구성: {summary['positions']}포지션",
        f"- 제외한 진입 후보: {quality.get('excluded_candidates', 0)}개",
        f"- 진입과 연결하지 못한 청산: {quality.get('unmatched_close_events', 0)}행",
        f"- 기록 손익 합계: {summary['recorded_pnl_usdt']} USDT (비용 포함 여부 미확인)",
        f"- 포지션 기록 승률: {summary['recorded_win_rate_pct']}% (None은 표본 없음)",
        f"- 기록 손익 기준 PF: {summary['recorded_profit_factor']} (None은 손실 분모 없음)",
        "", "## 품질 확인", "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["quality"].items())
    lines.extend(f"- 제외 사유 {key}: {value}" for key, value in report["excluded_reasons"].items())
    lines.extend(["", "## 해석 제한", ""])
    lines.extend(f"- {line}" for line in report["limitations"])
    lines.extend(["", "모델·페어·방향·익명 계정·KST 월별 수치는 report.json에 있다.", ""])
    return "\n".join(lines)


def _write_new(path: Path, payload: bytes) -> None:
    """기존 파일을 덮어쓰지 않고 로컬 결과물을 생성한다.

    Args: path: 새 파일 경로. payload: 파일 내용.
    Returns: 없음.
    Raises: OSError: 이미 존재하거나 저장 실패.
    """
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        out.write(payload)


def save_snapshot(payload: bytes, report: dict, source: str, *, root: Path = OUTPUT_ROOT) -> Path:
    """완성 마커를 마지막에 저장하고 스냅샷을 서로 섞지 않는다.

    Args: payload: 검증된 CSV. report: 보고서. source: 취득 방식. root: 저장 루트.
    Returns: 새 스냅샷 폴더.
    Raises: OSError: 저장 실패. manifest.json이 없으면 불완전 저장이다.
    """
    now = datetime.now(UTC)
    digest = hashlib.sha256(payload).hexdigest()
    folder = root / f"{now:%Y%m%dT%H%M%S%fZ}_{digest[:12]}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, mode=0o700)
    _write_new(folder / "raw.csv", payload)
    _write_new(folder / "report.json", json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False,
    ).encode("utf-8"))
    _write_new(folder / "summary.md", render_markdown(report).encode("utf-8"))
    manifest = {
        "schema_version": 1, "created_at_utc": now.isoformat(), "source": source,
        "sha256": digest, "bytes": len(payload), "rows": report["input"]["rows"],
        "complete": True, "analysis_schema_version": report["schema_version"],
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_new(folder / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    return folder


def main(argv: list[str] | None = None) -> int:
    """한 번 수집·분석하거나 기존 CSV를 오프라인 분석한다.

    Args: argv: CLI 인자. None이면 프로세스 인자.
    Returns: 성공 0, 인증·입력·저장 오류 1.
    Raises: SystemExit: argparse 도움말 또는 사용법 오류.
    """
    parser = argparse.ArgumentParser(description="로컬 CSV 수집·품질 검사·포지션 기록 분석")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="기존 관리자 CSV를 한 번 받아 분석")
    sync.add_argument("--prompt-token", action="store_true", help="터미널 비표시 토큰 입력")
    offline = commands.add_parser("analyze", help="네트워크 없이 기존 전체 사용자 CSV 분석")
    offline.add_argument("csv_path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "sync":
            if args.prompt_token:
                if not sys.stdin.isatty():
                    raise AnalysisError("비표시 입력은 직접 연 대화형 터미널에서 실행해야 한다.")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    try:
                        token = getpass.getpass("관리자 토큰 (화면에 표시되지 않음): ")
                    except getpass.GetPassWarning:
                        raise AnalysisError("터미널 비표시 입력을 사용할 수 없어 중단했다.") from None
            else:
                token = os.environ.get(TOKEN_ENV, "")
            payload = download_csv(token)
            del token
            source = EXPORT_URL
        else:
            with args.csv_path.open("rb") as source_file:
                payload = source_file.read(MAX_BYTES + 1)
            source = "local_csv"
        events = parse_csv(payload)
        report = analyze(events)
        folder = save_snapshot(payload, report, source)
        print(f"수집·분석 결과: {folder}")
        print(f"원본 {len(events)}행 / 수량 일치 재구성 {report['summary']['positions']}포지션")
        print("비용 포함 여부와 실제 거래소 체결은 미검증. summary.md의 품질 경고를 확인해야 한다.")
        return 0
    except (AnalysisError, OSError, EOFError) as error:
        message = str(error) if isinstance(error, AnalysisError) else "로컬 입력 또는 파일 저장 실패."
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
