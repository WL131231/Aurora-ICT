# Aurora-ICT 지표 검증 가이드

봇이 차트에 표시하는 OB / FVG / Premium-Discount 등 ICT 구간이 **실제 ICT 룰 그대로** 검출되는지 사용자가 직접 검증할 수 있게 정리한 문서.

## 1. 빠른 검증 — LuxAlgo SMC 와 시각 비교

가장 신뢰할 수 있는 방법은 같은 종목 / 같은 timeframe 에서 LuxAlgo Smart Money Concepts 지표와 우리 봇 차트를 나란히 띄워 검출 위치를 비교하는 것.

### 절차
1. 봇 (`Aurora-ICT.exe`) 실행 → 차트 toolbar 의 TF 토글에서 비교할 timeframe 선택 (예: 1h)
2. TradingView 에서 같은 종목 (BTC/USDT) + 같은 timeframe 열기
3. LuxAlgo SMC 지표 추가:
   - Bullish/Bearish Internal Structure ON
   - Bullish/Bearish Swing Structure ON
   - Internal Order Blocks ON
   - Fair Value Gaps ON
   - Equal High/Low ON
   - Premium/Discount Zones ON
4. 봇 차트의 viz 토글 (`BOS` / `EQH·EQL` / `PD Zones`) 모두 켜기
5. 같은 시점 비교 — 가격 차이 1% 이내, 시간 차이 1봉 이내면 동일 검출로 인정

### 차이 발생 시 점검 포인트
| 차이 종류 | 원인 후보 |
|---|---|
| OB 위치 다름 | `atr_multiplier` 세팅 차이 (LuxAlgo 기본 2.0) |
| FVG 누락 | `fvg_min_size_pct` 너무 큼 |
| EQH/EQL 누락 | `tolerance_pct` (우리 0.1%) vs LuxAlgo `threshold` |
| BOS / CHoCH 시점 다름 | swing 크기 차이 (`left/right` 봉 수) |

## 2. 자동 회귀 테스트

`tests/test_aurora_ict_visual_regression.py` 안의 7개 케이스는 명확한 ICT 패턴 fixture 에 대해 각 indicator 가 정확히 검출되는지 검증.

```bash
pytest tests/test_aurora_ict_visual_regression.py -v
```

각 케이스 실패 메시지에 어떤 패턴이 누락됐는지 + 어느 파일을 봐야 하는지 힌트 포함.

## 3. 파라미터 sweep

`tests/test_aurora_ict_parameter_sweep.py` 는 같은 fixture 에 다양한 세팅을 적용해 결과 변화를 검증.

```bash
pytest tests/test_aurora_ict_parameter_sweep.py -v
```

### 권장 디폴트
실제 사용자 환경에서 검증한 결과 기준 권장 세팅:

| 파라미터 | 권장값 | 설명 |
|---|---|---|
| `atr_multiplier` (OB) | **2.0** | LuxAlgo 기본 — false-positive 가장 적음 |
| `atr_period` (OB) | **200** | 표준 ATR 윈도우 |
| `displacement_bars` (OB) | **3** | 너무 짧으면 noise, 너무 길면 stale |
| `fvg_min_size_pct` | **0.0005** (0.05%) | 작은 FVG 노이즈 제거 |
| `tolerance_pct` (EQH/EQL) | **0.001** (0.1%) | 표준 |
| swing `left/right` (기본) | **1** | sweep / structure / OB / trailing 이 사용 |
| swing `left/right` (internal) | **5** | LuxAlgo internal |
| swing `left/right` (large) | **50** | LuxAlgo swingsLengthInput 기본 |

## 4. 시각 점검 체크리스트

봇 차트 띄운 후 다음 항목을 눈으로 빠르게 확인:

- [ ] FVG 박스(area) 가 실제 3봉 갭 위치와 일치
- [ ] OB 가로선이 displacement 직전 반대 봉의 high/low 와 일치
- [ ] BOS/CHoCH 가로선의 가격이 돌파된 swing 가격과 일치
- [ ] Strong High/Low 가 차트 우측 끝까지 그려져 있고 라벨 표시
- [ ] Premium/Discount Zone 의 6개 가로선 비율이 95-100% / 47.5-52.5% / 0-5%
- [ ] EQH/EQL 가로선이 같은 가격대 2개 이상 swing 의 평균

## 5. 알려진 한계

- **batch 계산**: 봉 단위 streaming 이 아니라 매 호출마다 df 전체에서 재계산. 대용량 df 에서 약간의 지연 가능.
- **trailing extremes**: LuxAlgo 는 swing point 가 박힐 때마다 reset 하지만 우리 batch 버전은 last swing 이후 max/min 으로 단순화.
- **SMT divergence**: ETH OHLCV fetch 통합 미박힘 (지표 함수만 박혀있음).
