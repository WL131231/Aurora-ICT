# Aurora-ICT SaaS — 다중 사용자 FastAPI 서버 컨테이너 이미지
#
# 빌드:
#   docker build -t aurora-ict-saas:latest .
# 실행 (간단):
#   docker run -d --name aurora-ict -p 8765:8765 \
#       -v aurora_data:/data \
#       -e AURORA_ICT_SECURE_COOKIE=0 \
#       aurora-ict-saas:latest
# 실행 (HTTPS reverse proxy 뒤):
#   docker run -d --name aurora-ict -p 8765:8765 \
#       -v aurora_data:/data \
#       -e AURORA_ICT_MASTER_KEY=$(openssl rand -base64 32 | tr '+/' '-_') \
#       aurora-ict-saas:latest
#
# 데이터 영속성:
#   /data 볼륨이 users.db + master.key 보관. 절대 분리하지 말 것 — master.key
#   분실 시 모든 사용자 거래소 secret 영구 복호화 불가.

FROM python:3.11-slim AS base

WORKDIR /app

# 시스템 의존성 — cryptography (Fernet), ccxt 빌드용. python:slim 은 gcc 미포함.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# pyproject + 소스 (editable install 위해 src/ 통째 필요).
# 의존성 layer 캐시 최적화 — pyproject 만 먼저 복사하면 deps 만 받고 source 만 갈아끼울 때
# 캐시 재사용 가능. 단 editable install 은 source 도 필요해서 한 번에 처리.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# UI 정적 자원
COPY ui_ict/ ./ui_ict/

# 데이터 디렉토리 — users.db / master.key 보관. 절대 컨테이너 안에만 두지 말 것.
RUN mkdir -p /data
VOLUME /data

# 환경변수 기본값 — docker-compose / fly / Railway 에서 override 가능.
ENV AURORA_ICT_DATA_DIR=/data \
    AURORA_ICT_HOST=0.0.0.0 \
    AURORA_ICT_PORT=8765 \
    AURORA_ICT_SECURE_COOKIE=1 \
    AURORA_ICT_SAAS=1 \
    PYTHONUNBUFFERED=1

# non-root 사용자 — /data 도 같은 uid 소유로 (volume permission).
RUN useradd -m -u 1000 aurora && chown -R aurora:aurora /app /data
USER aurora

EXPOSE 8765

# Health check — /ict/health 는 인증 없이 200 응답.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/ict/health || exit 1

# python -m aurora_ict.saas — SaaS 진입점 (main.py 의 pywebview 흐름과 분리).
CMD ["python", "-m", "aurora_ict.saas"]
