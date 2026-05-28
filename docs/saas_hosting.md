# Aurora-ICT SaaS 호스팅 가이드

다중 사용자 (multi-user) 모드의 Aurora-ICT 를 Docker 컨테이너로 패키징해
Fly.io / Railway / 자체 VPS 등에 배포하는 절차를 정리한다.

> 단일 사용자 .exe 배포는 본 문서 대상 아님. `main.py` + PyInstaller spec
> (`Aurora-ICT.spec`) 흐름 그대로 사용.

---

## 1. 한눈에 보는 구조

```
+-------------------------+
| browser (HTTPS)         |   /ui/  /auth/*  /ict/*
+-----------+-------------+
            |
            v
+-------------------------+
| reverse proxy (LetsEnc) |   Fly platform / Caddy / Nginx
+-----------+-------------+
            |
            v
+-------------------------+
| python -m aurora_ict.saas|  FastAPI + uvicorn (8765)
|  - /auth/* (PIN 로그인)  |
|  - MultiUserBotManager   |  사용자별 BotIctInstance 격리
+-----------+-------------+
            |
            v
+-------------------------+
| /data 영속 볼륨          |
|  users.db   (PIN 해시 + |
|             API 키)      |
|  master.key (Fernet 키)  |
+-------------------------+
```

핵심 사실:
- 사용자별 자료 격리는 `users.db` row 단위 (`user_code`).
- 거래소 **API secret 은 Fernet (AES-128 + HMAC) 으로 암호화** 되어 저장.
- 복호화에 필요한 **마스터 키 (`master.key`)** 가 사라지면 **모든 사용자의 secret 복호화 불가**.
  반드시 백업 (또는 환경변수 `AURORA_ICT_MASTER_KEY` 로 외부 secrets manager 주입).

---

## 2. 로컬 빌드 + 실행

### 2.1 Docker

```bash
docker build -t aurora-ict-saas:latest .

# (선택) 마스터 키 미리 생성 → 컨테이너 재생성에도 동일 키.
export AURORA_ICT_MASTER_KEY=$(python -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())')

# HTTP 만 띄울 거면 secure cookie 끄기 (HTTPS reverse proxy 뒤에 둘 거면 1 유지).
docker run -d --name aurora-ict \
  -p 8765:8765 \
  -v aurora_data:/data \
  -e AURORA_ICT_MASTER_KEY=$AURORA_ICT_MASTER_KEY \
  -e AURORA_ICT_SECURE_COOKIE=0 \
  aurora-ict-saas:latest

# 브라우저로 http://localhost:8765/ui/ 접속 → 첫 PIN 설정 화면 진입.
```

### 2.2 docker-compose

```bash
# .env 만들기 (선택)
cat <<EOF > .env
AURORA_ICT_MASTER_KEY=$(python -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())')
AURORA_ICT_SECURE_COOKIE=0
EOF

docker compose up -d
docker compose logs -f aurora-ict
```

---

## 3. Fly.io 배포

가장 짧은 경로 — 1 인 ~ 소규모 (5 인 이하) 까지 무료~$5/월 안에 가능.

```bash
# 1) Fly CLI 설치 (https://fly.io/docs/hands-on/install-flyctl/)
fly auth login

# 2) 앱 이름/리전 확정 (--copy-config 로 저장된 fly.toml 사용)
fly launch --copy-config --no-deploy

# 3) 영속 볼륨 — fly.toml 의 source = "aurora_data" 와 동일 이름.
fly volumes create aurora_data --region nrt --size 1   # 1 GB

# 4) 마스터 키 secret 등록 (필수 — 분실 시 모든 secret 복호화 불가).
fly secrets set AURORA_ICT_MASTER_KEY="$(python -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())')"

# 5) 배포
fly deploy

# 6) 도메인 + HTTPS — Fly 가 LetsEncrypt 로 자동 발급. fly.toml 의 force_https=true.
fly status
open https://aurora-ict-saas.fly.dev/ui/
```

비용 추정 (2026-05 기준):
- shared-cpu-1x (256 MB RAM) + 1 GB volume + outbound 100 GB → **무료 한도 안**
- shared-cpu-1x (512 MB RAM) → 약 $2/월
- 2 인 베타 테스트: 무료 한도로 충분.

---

## 4. Railway 배포

```bash
# 1) Railway CLI 설치 (npm i -g @railway/cli)
railway login

# 2) 프로젝트 생성 + Dockerfile 빌드
railway init
railway up

# 3) Volume 추가 — Railway 대시보드에서 "Volumes" → mount path "/data".

# 4) 환경변수 등록 (대시보드 또는 CLI):
railway variables set \
  AURORA_ICT_MASTER_KEY="$(python -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())')" \
  AURORA_ICT_SECURE_COOKIE=1

# 5) Public Domain 발급 — 대시보드 "Settings" → "Generate Domain".
```

비용: starter 무료 크레딧 $5/월 (테스트 충분) → hobby $5/월.

---

## 5. 자체 VPS (Ubuntu 22.04 / 24.04)

### 5.1 Docker 설치 + 실행

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # 재로그인 필요

# 코드 + 이미지 빌드
git clone <repo> && cd Aurora-ICT
docker build -t aurora-ict-saas:latest .

# 데이터 볼륨 + 컨테이너
docker volume create aurora_data
docker run -d --name aurora-ict --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -v aurora_data:/data \
  -e AURORA_ICT_SECURE_COOKIE=1 \
  -e AURORA_ICT_MASTER_KEY=$(python3 -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())') \
  aurora-ict-saas:latest
```

### 5.2 systemd unit (Docker 없이 직접 가동)

```ini
# /etc/systemd/system/aurora-ict.service
[Unit]
Description=Aurora-ICT SaaS
After=network.target

[Service]
Type=simple
User=aurora
WorkingDirectory=/opt/aurora-ict
Environment="AURORA_ICT_DATA_DIR=/var/lib/aurora-ict"
Environment="AURORA_ICT_HOST=127.0.0.1"
Environment="AURORA_ICT_PORT=8765"
Environment="AURORA_ICT_SECURE_COOKIE=1"
EnvironmentFile=/etc/aurora-ict/env   # AURORA_ICT_MASTER_KEY 박아둠 (0600 권한)
ExecStart=/opt/aurora-ict/.venv/bin/python -m aurora_ict.saas
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aurora-ict
sudo journalctl -u aurora-ict -f
```

### 5.3 Caddy (HTTPS 자동)

```Caddyfile
ict.example.com {
    reverse_proxy 127.0.0.1:8765
}
```

---

## 6. 환경변수 명세

| 이름                          | 기본값          | 설명                                                                 |
|-------------------------------|-----------------|----------------------------------------------------------------------|
| `AURORA_ICT_DATA_DIR`         | `/data` (도커)  | `users.db` / `master.key` 보관 디렉토리. 호스트 볼륨 매핑.            |
| `AURORA_ICT_MASTER_KEY`       | (없음)          | Fernet 키 (44 글자 base64). 미설정 시 `master.key` 파일 자동 생성.    |
| `AURORA_ICT_HOST`             | `0.0.0.0`       | uvicorn 바인드 호스트. 컨테이너에선 `0.0.0.0`, 리버스프록시 뒤면 그대로. |
| `AURORA_ICT_PORT`             | `8765`          | 바인드 포트.                                                          |
| `AURORA_ICT_SECURE_COOKIE`    | `1`             | 세션 cookie `Secure` 속성. HTTPS 운영 시 `1`, 로컬 HTTP 테스트는 `0`.    |
| `AURORA_ICT_SAAS`             | `1` (도커)      | SaaS 빌드 식별 flag — 런타임 분기 X. (관측/디버그용)                 |
| `AURORA_ICT_LOG_LEVEL`        | `INFO`          | python logging level (`DEBUG` / `INFO` / `WARNING`).                  |

---

## 7. 데이터 이전 (서버 마이그레이션)

**중요**: `users.db` 와 `master.key` 를 **반드시 함께** 이전한다.
한 쪽만 옮기면 모든 사용자 거래소 secret 복호화 불가.

### 7.1 Docker volume 백업

```bash
# 원본 호스트
docker run --rm -v aurora_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/aurora_data.tar.gz -C /data .

scp aurora_data.tar.gz new-host:~/

# 새 호스트
docker volume create aurora_data
docker run --rm -v aurora_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/aurora_data.tar.gz -C /data
```

### 7.2 환경변수 방식 (권장 — 운영)

`master.key` 파일을 옮기지 않고 환경변수 `AURORA_ICT_MASTER_KEY` 로 외부 secrets
manager (Fly secrets, AWS Secrets Manager, Vault 등) 에서 주입하면 `users.db` 만
옮겨도 OK.

---

## 8. 보안 권장 사항

1. **거래소 API 권한 최소화** — 사용자에게 안내:
   - 출금 권한 **OFF** (필수)
   - 가능하면 거래소 IP whitelist 에 서버 IP 등록
   - 거래 + 조회 권한만 부여
2. **HTTPS 강제** — `AURORA_ICT_SECURE_COOKIE=1` + reverse proxy LetsEncrypt.
3. **마스터 키 백업** — `master.key` 또는 `AURORA_ICT_MASTER_KEY` 환경변수를
   secrets manager 에 별도 보관. 컨테이너 이미지 안에 박지 말 것.
4. **PIN 정책** — 8자 이상 + 영문/숫자/특수문자 혼합 강제 (백엔드 검증).
5. **PIN brute-force 완화** — 로그인 실패 시 0.5초 지연 + timing attack 완화 박혀 있음.
   IP rate-limit 은 reverse proxy 단에서 (Caddy/Nginx) 추가 권장.
6. **로그** — 평문 secret 로깅 없음. `cryptography` 의 `InvalidToken` 만 노출.
7. **컨테이너 권한** — Dockerfile 가 non-root (`uid 1000`) 로 실행.

---

## 9. 비용 추정 (2026-05 기준)

| 호스트            | 2 인 베타            | 10 인 운영             |
|-------------------|----------------------|-------------------------|
| Fly.io shared-1x  | 무료 한도            | ~$5/월 (RAM 512MB)     |
| Railway hobby     | $5/월 (starter free) | ~$5~10/월              |
| Hetzner CX11 VPS  | $4/월                | $4~8/월 (단일 인스턴스) |
| AWS Lightsail 2GB | $10/월               | $10/월                 |

> 1 사용자가 보유 자산 모니터링 + 1 페어 매매할 때 약 60~120 MB RAM, < 1% CPU.
> 10 사용자까지는 단일 shared-cpu / 1 vCPU 로 여유 있음 (사용자별 WebSocket flip
> watcher 는 multi-user 에서 기본 OFF — 부하 절감).

---

## 10. 트러블슈팅

- `InvalidToken` 에러 → master.key 가 잘못된 키. 환경변수/파일 점검.
- `/ui/` 가 404 → Docker 안에 `ui_ict/` 안 복사되었거나 mount 가 안 됨.
  로그에 `[multi-user] UI mounted from ...` 라인 확인.
- 로그인은 되는데 `/ict/start` 가 400 (`거래소 API 키가 등록되지 않았습니다`) →
  API 키 등록 화면을 먼저 통과해야 함.
- 컨테이너 재시작 시 모든 사용자 봇이 멈춰 있음 → 정상. 사용자가 다시 로그인 후
  `START` 눌러야 가동 (자동 복구 정책은 추후).

---

담당: 지영민 (SaaS 전환 PR — 2026-05-28)
