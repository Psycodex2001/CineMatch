"""
Configurações centralizadas da aplicação.
Lê variáveis do arquivo .env quando disponível.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Informações da aplicação ───────────────────────────────────────────
    APP_NAME: str = "CineMatch – Sistema de Recomendação de Filmes"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "API de recomendação híbrida (filtragem colaborativa + baseada em conteúdo) "
        "utilizando o dataset MovieLens Small."
    )

    # ── Banco de dados ─────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/cinematch.db"

    # ── Dataset ────────────────────────────────────────────────────────────
    DATASET_REPO: str = "ashraq/ml-latest-small"

    # ── Hiperparâmetros do modelo ──────────────────────────────────────────
    N_SVD_COMPONENTS: int = 50       # Componentes latentes para SVD
    CF_WEIGHT: float = 0.60          # Peso da filtragem colaborativa no híbrido
    CB_WEIGHT: float = 0.40          # Peso da filtragem por conteúdo no híbrido
    MIN_RATINGS_POPULARITY: int = 20  # Mínimo de avaliações para recomendação popular

    # ── API ────────────────────────────────────────────────────────────────
    DEFAULT_TOP_N: int = 10
    MAX_TOP_N: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
