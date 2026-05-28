"""Aurora-ICT — ICT (Inner Circle Trader) 매매 전략을 담은 별도 모듈.

Aurora 측 ``src/aurora/`` = UI / launcher / 거래소 layer (그대로 재사용).
``src/aurora_ict/`` = ICT 전략 / indicator / signal 신규 layer.

ICT 핵심 = TIME first, PRICE second. Smart Money Concepts (IPDA 알고리즘 기반).
"""

# 2026-05-28: SaaS 전환 + UI 마일스톤 (PR #125~#135). 정적값을 현재 main 에 맞춤.
__version__ = "0.5.0"

__all__ = ["__version__"]
