"""#PARITY-GUARD 2026-08-09 — 봇이 셋업 생성 '후' 점수를 고치는 경로를 자동 감시한다.

## 왜 필요한가 (2026-08-08 사건)

하니스 정합 규칙(2026-07-30)은 "라이브 배포 시 같은 PR 에서 백테를 갱신한다"였는데,
감사 범위가 **replay 옵션으로 표현되는 것**에 머물렀다. 그래서
``_apply_htf_supporting_boost`` 처럼 셋업이 만들어진 뒤 ``setup.confluence_score``
를 in-place 로 고치는 경로가 통째로 빠졌고, 라이브 진입의 93% 가 백테에 없는
+3 을 받는 상태로 두 달을 굴렀다.

사람이 기억으로 막을 수 있는 종류가 아니다. **기계가 세도록** 만든다.

## 무엇을 하나

### A. 점수 변이 경로
1. 프로덕션 소스를 AST 로 훑어 아래 두 종류의 '점수 변이'를 전부 찾는다.
   - ``<x>.confluence_score`` 에 대한 대입/증감 (Assign / AugAssign)
   - ``<x>.confluences.append(...)`` 호출
2. (함수, 연산, 횟수) 로 서명을 만들어 lock 파일과 대조한다.
3. lock 에 없는 새 경로 = **백테 미이식 의심**. 종료코드 1 로 떨군다.

lock 파일에는 각 경로가 백테 어디에 대응하는지(``backtest``)와 미이식 사유
(``status``)를 같이 적는다. 즉 이 파일 자체가 정합 대장이다.

### B. 구독 강제 설정 (2026-08-13 추가)

A 만으로는 **진입 소스를 켜고 끄는 설정**을 못 잡는다. 2026-08-11 에 turtle_soup
제거(#421)를 배포하면서 연구 사본 갱신을 빠뜨렸고, 이틀 뒤 라이브 진단 중에야
드러났다 — 점수를 건드리지 않는 변경이라 A 의 감시망 밖이었다.

그래서 `_enforce_license_tier_policy` 의 **sub_ 분기에서 강제되는 설정 전부**를
같이 센다. 새 강제값이 생기거나 값이 바뀌면 종료코드 1 이다. 그때
`scripts/live_parity.py` 의 `LIVE_BASE` 도 갱신했는지 확인하고 `--update` 한다.
(연구 사본은 별도 저장소라 CI 가 직접 볼 수 없다. 그래서 '바뀌었다'만 알리고
대조는 사람이 한다.)

## 사용법

    python scripts/parity_guard.py             # 검사 — CI 가 매 PR 에서 돌린다
    python scripts/parity_guard.py --update    # 경로를 의도적으로 바꿨을 때 lock 갱신

담당: Claude (파트너 지시, 2026-08-09)
"""

from __future__ import annotations

import ast
import json
import os
import sys

# 저장소 루트 — 이 스크립트가 scripts/ 안에 있다는 전제. CI(리눅스)와
# 로컬(윈도우) 양쪽에서 같게 동작하도록 경로 결합은 os.path 에 맡긴다.
PROD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 점수 변이가 일어날 수 있는 파일 (프로덕션 상대경로).
TARGETS = [
    os.path.join("src", "aurora_ict", "bot", "bot_ict_instance.py"),
    os.path.join("src", "aurora_ict", "strategy", "silver_bullet.py"),
    os.path.join("src", "aurora_ict", "signal", "ict_signal.py"),
    os.path.join("src", "aurora_ict", "strategy", "mmbm.py"),
]

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "parity_lock.json")


def _enclosing(tree: ast.AST) -> dict[int, str]:
    """각 줄번호 → 그 줄을 감싸는 최근접 함수 이름 매핑.

    Args:
        tree: 모듈 AST.

    Returns:
        {lineno: func_name}. 모듈 최상단 코드는 "<module>".
    """
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                # 중첩 함수가 더 안쪽이므로 나중에 덮어써도 되게 범위 좁은 쪽 우선
                prev = owner.get(ln)
                if prev is None or len(node.name) >= 0:
                    owner[ln] = node.name
    return owner


def scan(path: str) -> list[dict]:
    """한 파일에서 점수 변이 경로를 모두 찾는다.

    Args:
        path: 검사할 .py 절대경로.

    Returns:
        [{file, func, op, line}] 목록. op 는 "score=" / "score+=" / "append".
    """
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    owner = _enclosing(tree)
    out: list[dict] = []
    rel = os.path.relpath(path, PROD_ROOT)

    for node in ast.walk(tree):
        op = None
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute) \
                and node.target.attr == "confluence_score":
            op = "score+="
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "confluence_score":
                    op = "score="
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "append" \
                and isinstance(node.func.value, ast.Attribute) \
                and node.func.value.attr == "confluences":
            op = "append"
        if op:
            out.append({"file": rel.replace("\\", "/"),
                        "func": owner.get(node.lineno, "<module>"),
                        "op": op, "line": node.lineno})
    return out


def scan_tier_forced(path: str) -> dict[str, str]:
    """`_enforce_license_tier_policy` 의 sub_ 분기 강제 설정 → {필드: 값}.

    구독 정책은 사용자 설정을 덮어쓴다. 즉 **라이브가 실제로 쓰는 값**이고,
    백테(`LIVE_BASE`)가 반드시 따라와야 하는 정본이다.

    Args:
        path: settings.py 절대경로.

    Returns:
        {"origo_turtle_soup_enabled": "False", ...}. 값은 소스 표기 그대로 문자열.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out: dict[str, str] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != "_enforce_license_tier_policy":
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    try:
                        out[t.attr] = ast.unparse(node.value)
                    except Exception:  # noqa: BLE001
                        out[t.attr] = "<unparsed>"
    return out


def signature(rows: list[dict]) -> dict[str, int]:
    """(파일::함수::연산) → 횟수. 줄번호는 리팩터링에 흔들리니 서명에서 뺀다."""
    sig: dict[str, int] = {}
    for r in rows:
        k = f"{r['file']}::{r['func']}::{r['op']}"
        sig[k] = sig.get(k, 0) + 1
    return sig


SETTINGS_REL = os.path.join("src", "aurora_ict", "config", "settings.py")


def main() -> int:
    rows: list[dict] = []
    for rel in TARGETS:
        p = os.path.join(PROD_ROOT, rel)
        if os.path.exists(p):
            rows += scan(p)
    cur = signature(rows)

    # B) 구독 강제 설정 — 진입 소스 on/off 처럼 점수를 안 건드리는 변경을 잡는다.
    sp = os.path.join(PROD_ROOT, SETTINGS_REL)
    forced = scan_tier_forced(sp) if os.path.exists(sp) else {}

    if "--update" in sys.argv:
        old = {}
        old_forced = {}
        if os.path.exists(LOCK):
            with open(LOCK, encoding="utf-8") as fh:
                _d = json.load(fh)
            old = _d.get("paths", {})
            old_forced = _d.get("tier_forced", {})
        paths = {k: {"count": v,
                     "backtest": old.get(k, {}).get("backtest", "TODO"),
                     "status": old.get(k, {}).get("status", "TODO")}
                 for k, v in sorted(cur.items())}
        tier = {k: {"value": v,
                    "live_base": old_forced.get(k, {}).get("live_base", "TODO")}
                for k, v in sorted(forced.items())}
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        with open(LOCK, "w", encoding="utf-8") as fh:
            json.dump({"note": "각 경로의 backtest/status 를 직접 채울 것. "
                               "status: ported / gap / n-a. "
                               "tier_forced 의 live_base 에는 연구 사본 "
                               "scripts/live_parity.py LIVE_BASE 의 대응 키를 적는다"
                               "(없으면 GAPS 사유).",
                       "paths": paths, "tier_forced": tier}, fh,
                      ensure_ascii=False, indent=1)
        print(f"lock 갱신: 점수경로 {len(paths)} · 구독강제 {len(tier)}"
              f" → {os.path.abspath(LOCK)}")
        return 0

    if not os.path.exists(LOCK):
        print("lock 없음 — 먼저 --update 로 만들 것")
        return 2
    with open(LOCK, encoding="utf-8") as fh:
        _lock = json.load(fh)
    lock = _lock["paths"]
    lock_forced = _lock.get("tier_forced", {})

    added = {k: v for k, v in cur.items() if k not in lock}
    changed = {k: (lock[k]["count"], v) for k, v in cur.items()
               if k in lock and lock[k]["count"] != v}
    removed = [k for k in lock if k not in cur]
    todo = [k for k, v in lock.items() if v.get("status") == "TODO"]

    for k, v in added.items():
        print(f"[신규] {k} x{v}  ← 백테 이식 여부 확인 필요")
    for k, (a, b) in changed.items():
        print(f"[변경] {k} {a} → {b}")
    for k in removed:
        print(f"[삭제] {k}")
    for k in todo:
        print(f"[미분류] {k} — lock 의 backtest/status 를 채울 것")

    # B) 구독 강제 설정 대조
    f_added = {k: v for k, v in forced.items() if k not in lock_forced}
    f_changed = {k: (lock_forced[k]["value"], v) for k, v in forced.items()
                 if k in lock_forced and lock_forced[k]["value"] != v}
    f_removed = [k for k in lock_forced if k not in forced]
    f_todo = [k for k, v in lock_forced.items() if v.get("live_base") == "TODO"]

    for k, v in f_added.items():
        print(f"[강제 신규] settings sub_ 에 {k} = {v}"
              f"  ← LIVE_BASE 갱신했는지 확인")
    for k, (a, b) in f_changed.items():
        print(f"[강제 변경] {k}: {a} → {b}  ← LIVE_BASE 갱신했는지 확인")
    for k in f_removed:
        print(f"[강제 삭제] {k}")
    for k in f_todo:
        print(f"[강제 미분류] {k} — lock 의 live_base 를 채울 것")

    bad = bool(added or changed or removed or todo
               or f_added or f_changed or f_removed or f_todo)
    if bad:
        return 1
    print(f"OK — 점수 변이 경로 {len(cur)}개 · 구독 강제 설정 {len(forced)}개"
          f" 전부 대장과 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
