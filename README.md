# Aurora-ICT

ICT (Inner Circle Trader) 매매 전략 기반 자동매매 봇 — Aurora의 UI / 런처 / 거래소 layer 위에 ICT entry models 박은 별개 모델.

> **Aurora** (지표 confluence 기반)와 **Aurora-ICT** (ICT 시간+유동성+구조 기반)는 별개로 운영되는 두 봇.

## 핵심 철학

- **TIME first, PRICE second** — ICT의 본질
- **AI API 호출 금지** — 100% 룰 기반 의사결정 (Aurora와 동일)
- **Smart Money Concepts** — IPDA 알고리즘이 박는 패턴 추종

## ICT 매매 모델 (Phase 1)

### Model A: Silver Bullet
- 매일 박힘 3개 1시간 윈도우 (London 3-4am / AM 10-11am / PM 2-3pm NY)
- 그 시간 안 박힌 첫 FVG → entry
- SL = FVG 봉 wick. TP = next BSL/SSL
- 백테스트 win rate 60-75%

### Model B: AM 8:00 Model (2024 멘토십)
- 8am NY 박은 후 Relative Equal High/Low 박힘
- Liquidity Sweep → 5m MSS → IFVG/Breaker retest → entry

## ICT Indicators (Tier 1 — 즉시 박은 거)

1. **FVG (Fair Value Gap)** — 3봉 패턴 imbalance
2. **Liquidity Sweep** — 옛 swing high/low wick 박은 거
3. **MSS / CHoCH** — pivot 박은 거 깨짐
4. **CISD** — Change in State of Delivery
5. **Killzone 시간 필터** — London / NY AM / PM / London Close
6. **Silver Bullet 윈도우** — 매일 3개 1시간
7. **Premium/Discount** — dealing range 50% equilibrium
8. **BSL/SSL** — Buy/Sell Side Liquidity marker

## 페어
- BTC/USDT, ETH/USDT (perp)
- SMT Divergence — BTC-ETH 상관 박힌 거 활용

## 거래소 (Aurora와 동일)
- Bybit V5 (Demo / Live)
- 추후 OKX, Binance, Hyperliquid 확장 (Aurora layer 재사용)

## 빠른 시작

```bash
git clone https://github.com/WL131231/Aurora-ICT.git
cd Aurora-ICT

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements-dev.txt
pip install -e .
```

## 버전 정책
- v0.1.x — ICT indicator + Silver Bullet 박는 거 (Phase 1)
- v0.2.x — AM Model + SMT Divergence (Phase 2)
- v0.3.x — Weekly Profile + Macros (Phase 3)

## 관계
- 코드 기반: Aurora `aurora-v0.3.4-pre-ict` tag 박은 거 fork
- UI / launcher / exchange layer 재사용 (점진적 ICT 박힌 거로 교체)
- strategy / indicators / signal layer만 ICT로 박음
