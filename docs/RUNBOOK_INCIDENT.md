# 장애 대응 런북 (단일 머신 운영)

Aurora-ICT 는 fly.io **단일 머신**으로 돈다. 그 머신이 멈추면 전면 중단이다.
이 문서는 실제로 겪은 사고들의 복구 절차를 모은 것이다 — 추정이 아니라 실측 기록.

작성 2026-08-03 (상용화 전 점검). 새 사고를 겪으면 여기 추가한다.

---

## 0. 먼저 알아둘 것 — 서버가 죽어도 손절은 살아 있다

**봇이 죽어도 거래소에 등록된 SL/TP 는 그대로 작동한다.** 진입 시 SL 을 거래소에
등록하고(등록 실패 시 비상청산), 이후 서버 상태와 무관하게 Bybit 가 집행한다.
장애 시 첫 판단은 "계좌가 무방비인가"가 아니라 **"멈춘 동안 무엇이 안 도는가"** 다.

| 서버가 멈추면 | 상태 |
|---|---|
| 손절(SL) | **작동** — 거래소 conditional |
| 익절(TP) | **작동** — 거래소에 등록된 분 |
| 트레일링 · 본전 이동 · 부분익절 후속 | 멈춤 |
| HTF flip 청산 · REVERSE | 멈춤 |
| 신규 진입 | 멈춤 |
| 지정가 진입 대기 주문 | **거래소에 남아 체결될 수 있음** (TTL 취소가 안 돎) |

즉 **큰 손실로 번지는 경로는 막혀 있고**, 놓치는 것은 기회와 최적화다.
서두르되 당황하지 않아도 된다.

---

## 1. 첫 3분 — 어디가 문제인지 가른다

```
fly status
```

`STATE` 와 `CHECKS` 를 본다.

| 보이는 것 | 의미 | 가서 볼 절 |
|---|---|---|
| `started` + `1 passing` | 서버는 정상. 봇 로직 문제 | §2 |
| `started` + `critical` | 앱은 떴는데 응답 불가 | §3 |
| `stopped` | 머신이 꺼짐 | §4 |
| 머신 목록이 비었음 | 머신 소실 (볼륨 사고 가능성) | §5 |
| 명령 자체가 안 됨 | fly agent 문제 | §6 |

배포 직후라면 1~2분은 정상적으로 `critical` 일 수 있다(구 머신 종료 ↔ 신 머신 기동).
2분 넘게 지속되면 실제 장애다.

---

## 2. 서버는 정상인데 봇이 안 도는 경우

로그에서 봇이 살아 있는지 본다.

```
fly logs | grep -a heartbeat
```

heartbeat 가 안 찍히면 봇 태스크가 죽은 것이다. 흔한 원인 두 가지:

**(a) 거래소 키 무효** — `auth_fail_streak` 누적으로 봇이 자동 정지한다.
```
fly logs | grep -aiE "키 무효|10003|10004|자동 정지"
```
사용자에게 키 재등록을 안내한다. 서버 조치는 없다.

**(b) 재가동 실패 잔존** — 10분 주기 auto_resume 이 재시도하므로 대개 저절로 복구된다.
그래도 안 되면 머신을 재시작한다. 부팅 시 `auto_resume_running_bots` 가 DB 의
가동 플래그를 읽어 전부 복원한다.
```
fly machine restart <MACHINE_ID>
```
재시작은 **약 1분간 전면 중단**을 뜻한다. §0 을 보고 그 값어치가 있는지 판단한다
(포지션은 거래소 SL 로 보호되지만 트레일링은 그동안 멈춘다).

---

## 3. 헬스체크 critical (앱은 떴는데 응답 불가)

메모리 고갈이 가장 흔하다. OHLCV 차트 캐시가 RAM 병목이다(2026-07-22 실측).

```
fly logs | grep -aiE "MemoryError|Killed|OOM|Out of memory"
fly status
```

메모리 부족이면 즉시 조치는 재시작, 근본 조치는 봇 상한 축소 또는 VM 증설이다.

```
fly machine restart <MACHINE_ID>
```

---

## 4. 머신이 stopped

`auto_stop_machines = false` 인데도 멈추는 경우가 있다(2026-07-31 실측).
배포 성공 후에도 확인이 필요한 이유다.

```
fly machine start <MACHINE_ID>
fly status
```

**배포할 때마다 `fly status` 로 STATE 를 확인한다.** 배포 명령이 성공해도
머신이 stopped 로 남을 수 있다.

---

## 5. 머신 소실 · 볼륨 붙일 호스트 없음 (2026-07-22 실사고)

배포 중 이 에러가 나면서 **머신이 파괴되고 봇이 전면 다운**됐다:

```
insufficient resources to create a new machine with existing volume
```

볼륨이 있는 물리 호스트에 여유가 없어서 새 머신을 못 만드는 상황이다.
볼륨을 **다른 존으로 fork** 해서 옮긴다.

```
fly volumes list
fly volumes fork <OLD_VOLUME_ID> --name aurora_data3
```

fork 된 볼륨 이름으로 `fly.toml` 의 `[[mounts]] source` 를 바꾸고 재배포한다.

```
fly deploy --now
fly status
```

fork 는 데이터를 그대로 복사하므로 손실이 없다. 성공 후 구 볼륨은 며칠 두었다가
지운다(즉시 삭제 금지 — 되돌릴 여지를 남긴다).

> 현재 `aurora_data`(구, 미부착)와 `aurora_data2`(현행)가 공존한다. 7/22 fork 의
> 잔재이며 구 볼륨은 1개월 전 데이터다. 정리 여부는 판단 필요.

---

## 6. fly 명령이 먹통 (agent i/o timeout)

```
fly agent restart
```

로그 감시 루프가 이걸로 자가복구하도록 되어 있다(2026-06-14).

---

## 7. 데이터 손상 — users.db 복구

라이선스·API 키·봇 가동 상태가 들어 있다. 손상되면 전 사용자가 로그인 불가다
(2026-06-12 실사고 — 스냅샷 복구로 해결).

**볼륨 스냅샷은 하루 1회 자동, 5일 보존이다.** 5일이 지나면 되돌릴 수 없다.

```
fly volumes snapshots list <VOLUME_ID>
fly volumes create aurora_restore --snapshot-id <SNAPSHOT_ID> --region sin --size 1
```

새 볼륨 이름으로 `fly.toml` 의 `source` 를 바꾸고 재배포한다.

**복구 시 유실 구간이 생긴다** — 스냅샷 이후의 매매 기록이 사라진다. 거래소
closed-pnl 로 백필할 수 있다:

```
curl -X POST -H "X-Admin-Token: $AURORA_ICT_ADMIN_TOKEN" "https://aurora-ict-one.fly.dev/admin/trades/backfill?days=7"
```

멱등이라 정상 구간에 다시 돌려도 안전하다(같은 symbol·방향·±10분·수량 ±2% 중복 skip).

---

## 8. 라이선스 DB(Supabase) 정지 (2026-07-07 실사고)

무료 티어가 비활성 기간이 길면 자동 정지된다. 증상은 라이선스 검증 실패다.

Supabase 대시보드에서 **Restore** 를 누르고 전파를 기다린다(수 분).
재발 방지로 6시간 주기 핑이 걸려 있다.

admin 봇은 배포 후 머신을 직접 켜야 한다:

```
fly deploy --now -a aurora-ict-license
fly machine start -a aurora-ict-license
```

---

## 9. 사용자 공지 판단 기준

| 상황 | 공지 |
|---|---|
| 5분 내 복구 | 불필요 |
| 15분 이상 중단 | 공지 — "신규 진입만 멈췄고 손절·익절은 거래소에서 작동 중" 을 명시 |
| 데이터 복구 수행 | 필수 — 유실 구간과 백필 여부를 알린다 |
| 포지션이 의도와 다르게 청산 | 필수 + 개별 확인 |

§0 의 표를 그대로 안내하면 불안이 크게 준다. **손절이 살아 있다는 사실이
가장 중요한 정보다.**

---

## 10. 정기 점검 (주 1회 권장)

```
fly status
fly volumes snapshots list vol_re1z79j3xzgom954
fly logs | grep -aicE "ERROR|Traceback"
```

- 스냅샷이 **어제 것까지 있는가** (없으면 백업이 끊긴 것 — 최우선 조치)
- 머신 STATE 가 started 인가
- ERROR 빈도가 평소와 다른가

---

## 알려진 구조적 한계

- **단일 머신** — 이중화되어 있지 않다. 이 문서는 "빨리 복구한다"까지가 목표이고
  "안 멈춘다"는 아니다.
- **스냅샷 5일** — 그 이전으로는 되돌릴 수 없다.
- **볼륨 1GB** — 매매 기록이 쌓이면 확장이 필요하다. `fly volumes extend`.
