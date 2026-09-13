# AGENTS.md — Aurora-ICT 온보딩 (AI 협업자용)

이 문서는 이 리포에서 작업하는 **모든 AI 에이전트**(Codex/GPT, Claude 등)가 먼저 읽는
컨텍스트다. `CLAUDE.md` 가 코딩 규칙이라면, 여기는 **"지금 어디까지 왔고 무엇을 조심해야
하는가"** 다. 작성 2026-09-13. 큰 변화가 있으면 이 파일을 같이 갱신한다.

---

## 1. 이게 무엇인가

암호화폐 선물(Bybit USDT perp) **자동매매 봇 SaaS**. 구독 사용자 13명이 실계좌로 돌리고
있다(LIVE 전용, 데모 모드는 쓰지 않는다). 봇은 fly.io 에서 24시간 돌고 UI 는 웹
(`aurora-ict-one.fly.dev`), 알림은 텔레그램이다.

매매 모델은 **두 개가 별도로** 있고 사용자가 고른다:

| 모델 | 성격 | 핵심 파일 | 페어 |
|---|---|---|---|
| **Origo 2.2** | ICT 단타 — 킬존·FVG·유동성 스윕·MSS·confluence 등급 | `src/aurora_ict/bot/bot_ict_instance.py` (5,200줄) | BTC·ETH·SOL·XRP·DOGE·LINK·HYPE |
| **Cursus 1.0** | DualST(이중 SuperTrend) 추세추종, 고정 −2% SL + 4분할 TP + REVERSE | `src/aurora_ict/bot/bot_trend_instance.py` | BTC·ETH·SOL·XRP·DOGE·TRX |
| Cycle 1.0 | 차트 전용(매매 없음) — EMA·구름 지지/저항 별표시 | `src/aurora_ict/indicators/cycle_levels.py` | — |

**절대 원칙 (CLAUDE.md 와 같음)**: 봇 런타임에서 LLM API 호출 금지(100% 룰 기반) ·
한국어 주석/docstring · 고배율이라 **손절 등록이 최우선** · `main` 직접 커밋 금지(브랜치 +
PR, Squash 머지) · 테스트는 mock 0(결정론적 합성 입력만, self-spy 패턴).

## 2. 리포·폴더 지도

```
C:\Users\지영민\Desktop\Aurora-ICT\            ← 이 리포 (main). 배포 코드.
C:\Users\지영민\Desktop\Aurora-ICT-research\   ← 같은 리포의 git worktree (chore/backtest-edge-tools)
                                               연구 스크립트 + 데이터(data/*_1m_full.parquet, 36페어 5년 1분봉)
C:\Users\지영민\Desktop\star-alert-bot\        ← 별개 프로젝트. EMA/다이버전스 텔레그램 알림 봇 (fly: star-alert-bot)
C:\Users\지영민\Desktop\aurora-ict-license\    ← 라이선스 admin 텔레그램 봇 (fly: aurora-admin-bot, Supabase)
```

이 리포 안에서 자주 만지는 곳:

- `src/aurora_ict/bot/aurora_adapter.py` — 거래소(ccxt/Bybit) 호출을 감싸는 어댑터. **인증 실패
  카운터(`auth_fail_streak`)·쓰기 거부 카운터(`write_fail_streak`)가 여기 있다.**
- `src/aurora_ict/bot/multi_user_manager.py` — 사용자×페어 슬롯, auto_resume, 고정 페어 정합
- `src/aurora_ict/bot/pair_registry.py` — 모델별 고정 페어 목록(Origo/Cursus 분리, 근거 주석 필독)
- `src/aurora_ict/api/app.py` (3,400줄) — FastAPI. `/ict/*` 사용자 API, `/admin/*` 관리
- `src/aurora_ict/api/trades_router.py` — 매매기록(JSONL = source of truth, sqlite 는 조회용)
- `ui_ict/` — 실제 서빙되는 웹 UI (`ui/` 는 옛것, 건드리지 말 것)
- `docs/RUNBOOK_INCIDENT.md` — 장애 복구 절차(실측 기록). `docs/FST.md` — 장기 로드맵
- `docs/*.pine` — 트뷰 파인 지표(차트용). 봇 로직과 규칙을 맞춰야 하는 것들이 있다(아래 §7)

## 3. 지금 상태 (2026-09-13)

**라이브 성적 (5/29~9/3, 13계좌, 포지션 단위)**: 승률 16% → 25% → 44%(6·7·8월), 월손익
−1,790 → −1,370 → −240 USDT. **아직 누적 적자**이고 "흑자 봇"이라고 쓰면 안 된다.
개선 중이라는 게 정확한 표현. 8월 Cursus 는 PF 0.99·승률 55% 로 본전.

**이번 주 사고 두 건 (둘 다 코드 수정 배포됨, PR #438~#440)**:
1. **Bybit API 키 90일 자동 만료(33004)** — IP 제한 없이 만든 키는 90일 뒤 만료된다.
   6계좌가 9/7~8 에 걸렸다. 봇은 이제 인증 실패 연속이면 자동 정지 + 텔레그램 안내.
   **아직 남은 결정**: fly 고정 발신 IP(`fly machine egress-ip allocate`, 유료)를 사면 사용자가
   키를 IP 에 묶어 만료가 없어진다. 파트너 판단 대기. 그 전엔 사용자에게 "IP 제한 걸지 말라".
2. **읽기 전용 키 + 10005 오판 → 가짜 청산 4,202건** — 한 사용자가 거래 권한 없는 키로
   재등록. 어댑터가 청산 실패(10005)를 "포지션 있으니 성공"으로 기록해 1분마다 재입양·가짜
   청산 반복. 수정: 주문 성공은 **거래소 상태 변화**(수량 감소/증가, 대기주문 증가)로만
   인정. 쓰기 거부 3회면 자동 정지. 가짜 기록은 `POST /admin/trades/purge` 로 정리함.
   **교훈: "오류를 성공으로 우회"하는 코드는 우회 근거가 사라지면 반대로 작동한다.**

**미해결 큐**:
- 잔고 검사(2026-09-13): Bybit 통합계정은 CCXT 코인별 free 대신 계정 모드별
  실제 주문 가용액을 사용한다. 조회 실패/음수 가용액은 신규 진입만 보류하며 SL·청산은
  유지한다. `docs/BYBIT_AVAILABLE_MARGIN_FIX.md` 참고. 계좌별 동시 주문 예약은 미구현.
- 고정 egress IP 결정 (위)
- 키 만료로 정지된 슬롯이 재배포 때 `auto_resume` 으로 다시 뜨면서 안내 1통 재발송 —
  `users_db` 에 플래그 두고 키 재등록 전까지 auto_resume 제외 (미구현)
- `purge` 백업 파일명이 초 단위라 같은 초 두 번 실행 시 덮어씀 — ms 붙이기
- 손익비 0.98·PF 0.77 (8월) — 승률은 올랐는데 이익이 작다. 다음 연구 축

## 4. 운영

- **배포**: `main` 머지 → GitHub Actions `fly-deploy.yml` 자동 배포. **배포 후 2분간 fly 로그를
  직접 보고 이상 없음을 확인한 뒤 보고한다**(파트너 지시, 6/13 사고 후).
  로그는 ANSI 코드가 섞여 있어 `sed -r 's/\x1b\[[0-9;]*m//g'` 로 벗긴 뒤 `grep -a` 로 본다
  (그냥 grep 은 "Binary file matches" 로 아무것도 안 잡힌다).
- **관리 API**: `/admin/trades/export-all`(전 사용자 매매기록 CSV) · `/admin/positions`(거래소
  실제 포지션·SL 유무) · `/admin/trades/purge`. 헤더 `X-Admin-Token` 필요 — **토큰은 파트너에게
  받는다. 이 파일·코드·커밋 어디에도 적지 않는다.**
- **매매기록 수집은 파트너가 CSV 를 주기 전에 위 API 로 직접 당긴다** (허용됨).
- fly 앱 3개: `aurora-ict-one`(봇+UI, sin) · `aurora-admin-bot`(라이선스, nrt) · `star-alert-bot`(알림, sin).
  admin-bot 은 `fly deploy` 후 머신이 stopped 로 남는 버릇이 있어 `fly machine start` 필요.
- Supabase(라이선스 DB, 무료 티어)는 무활동 시 자동 정지된 적이 있다(7/7, 9/8). 증상 =
  admin 봇이 `db_error`. 복구는 파트너가 대시보드에서 Restore.

## 5. 연구할 때 지켜야 할 것 (수십 번 겪고 정리한 규칙)

새 신호·필터·규칙은 **아래를 통과하기 전엔 배포하지 않는다**:

1. **하니스 정합** — 연구는 `Aurora-ICT-research/scripts/live_parity.py` 의 `run_live_parity()` 로
   기준선을 잡는다. 라이브에 기능을 배포하면 **같은 PR 에서 live_parity 를 갱신**한다.
   (7/30 감사: 라이브 기능 19개 중 10개가 백테에 없어서 정합 오류가 줄줄이 났다.)
2. **홀드아웃은 탐색에 안 쓴 페어로, 1회용** — BTC 로 찾았으면 나머지 30페어에 얹는다.
   시간분할보다 강하다. 같은 홀드아웃을 두 번 쓰면 더 이상 홀드아웃이 아니다.
3. **플라시보(무작위 대조)는 봉도 무작위·방향 비율 유지·손절폭 유지·같은 출발선**.
   대조군이 구조적으로 불리하면(예: 오실레이터 0선 청산인데 아무 봉이나 뽑아 즉시 청산)
   가짜 통과가 난다.
4. **다중검정** — 42칸 훑고 고른 칸의 p 값은 그대로 못 쓴다. 플라시보를 통과해도 홀드아웃에서
   죽은 사례가 있다(EMA 터치, 8/24).
5. **청산은 1분봉 순차로** — TF 봉 단위로 재면 한 봉 안에서 SL·TP 순서를 못 봐 양수가
   전부 뒤집힌 적이 있다. 리샘플 라벨은 **구간 시작 시각**이라 그대로 진입 시각으로 쓰면
   신호 확정 전에 진입하는 버그가 된다.
6. **비용을 먼저 계산** — 손절폭이 좁으면 수수료가 R 을 먹는다(손절 0.4% 에 왕복 0.11% =
   0.26R). 진입은 시장가 종가보다 **레벨 지정가**가 건당 3~4%p 낫다(두 연구에서 재현).
7. **단위** — 백테 net 은 시드 대비 비율(레버 반영). 보고는 시드%(단리) + R 환산 병기.
8. **배포 전 검증 배터리 7종**: 연도별 일관 · 페어별 분산 · 파라미터 이웃 민감도 · 교차 지표
   재현 · 셔플/순열검정 p<0.05(원형이동, 독립순열은 p 과소평가) · MDD 장부 · 방향·세션
   교란 분해.

**지금까지 결론(반복 검토 금지 — 파트너가 이미 결정한 것들)**:
- 기각: 방향 예측 지표 15종·신호등·엘리어트·피보 OTE·볼린저 평균회귀·물타기/순환매·
  히든 다이버전스·EMA 터치 단독·주식 이식·횡보 필터 3종·LuxAlgo 중 Sigmoid 트레일링·
  Liquidity Sweep·IMFVG.
- 살아있는 것: **1h 구름 터치**(홀드아웃 통과했으나 비용 문턱), **Squeeze Momentum 일봉
  다이버전스**(30페어 8칸 전부 유의 + 공정 플라시보 p=0.015 — 단 수익 45%가 상위 5건, 손절폭
  12%라 레버 1.5배 상한, 연도별 분할 미실시), ICT 새 진입모델 MMBM/07·08AM(배포 판단 대기).
- **승률 착시 주의**: 4분할 익절이라 이벤트 기준 승률(64~81%)은 부풀려진다. 성능은 **포지션
  단위**로만 말한다.

## 6. 파트너(지영민)와 일하는 법

- **반말**, 호칭은 "파트너". "사장님" 같은 비즈니스 호칭 금지.
- **정확도 > 속도.** 추측·축약으로 틀린 숫자 내지 말 것. 모르면 코드·로그·데이터를 먼저 본다.
- 쉬운 한국어. 영어 약어·전문용어는 풀어서.
- 코드블록은 **복붙하면 바로 되는 명령**만. 에러·로그 인용은 코드블록 안 씀.
- 결정된 항목을 다시 제안하지 않는다(위 §5 기각 목록, 데모 모드 등).
- 사용자 대상 공지·발송은 파트너가 직접 한다. AI 는 문구를 준비한다.
- PR 은 CI green 후 스스로 머지·배포까지 해도 된다(위임됨). 단 배포 후 2분 감시는 필수.

## 7. 다른 AI 와 같은 리포에서 일할 때

이 리포는 Claude(Claude Code)와 Codex(GPT)가 **번갈아 또는 동시에** 작업한다.

- **브랜치를 따로** 판다. 한 브랜치를 두 에이전트가 만지지 않는다.
- PR 제목 앞에 누가 만들었는지 적는다: `[codex] fix: ...` / `[claude] fix: ...`
- 같은 파일을 동시에 고칠 가능성이 높은 것: `aurora_adapter.py`, `bot_*_instance.py`, `app.py`.
  작업 시작 전 `git pull` 하고, 열린 PR 목록(`gh pr list`)을 먼저 본다.
- 백그라운드로 PR/CI 를 돌리는 동안 브랜치를 바꾸지 않는다(`gh` 가 HEAD 기준이라 꼬인다).
- 파인 지표(`docs/*.pine`)와 봇 규칙은 짝이다: `ema_cloud_star.pine` ↔ `star-alert-bot/star_signals.py`
  (EMA 60/120/200/350/480/620, 터치 = 닿기만, 구름은 판정 제외). 한쪽을 바꾸면 다른 쪽도.
- 결정·사고·교훈은 코드 주석과 `docs/` 에 남긴다. 다른 에이전트의 개인 메모리는 못 본다.

## 8. 빠른 시작

```bash
cd C:\Users\지영민\Desktop\Aurora-ICT
git pull
python -m pytest tests/ -q          # 2,017 passed 가 기준선 (약 80초)
gh pr list                          # 열린 PR 확인
```

문제가 생기면 `docs/RUNBOOK_INCIDENT.md` 부터. 이 문서에 없는 판단이 필요하면 파트너에게
묻는다 — 특히 **실계좌 포지션·사용자 데이터·배포**가 걸린 일은 확인 후 진행.
