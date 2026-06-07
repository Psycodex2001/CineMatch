# ──────────────────────────────────────────────────────────────────
# CineMatch — Dockerfile
# Build:  docker build -t cinematch .
# Run:    docker run -p 8000:8000 cinematch
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadados
LABEL maintainer="cinematch"
LABEL description="CineMatch — Sistema de Recomendação de Filmes"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Diretório de trabalho
WORKDIR /app

# ── Dependências do sistema ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Dependências Python (camada separada para cache eficiente) ───
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ── Código-fonte ─────────────────────────────────────────────────
COPY app/ ./app/

# ── Diretório de dados persistente ───────────────────────────────
RUN mkdir -p /app/data

# ── Usuário não-root (boas práticas de segurança) ────────────────
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# ── Porta exposta ────────────────────────────────────────────────
EXPOSE 8000

# ── Comando de inicialização ─────────────────────────────────────
# --workers 1 porque o modelo é carregado em memória (singleton)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
