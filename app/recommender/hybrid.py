"""
Modelo Híbrido de Recomendação.

Combina Filtragem Colaborativa (SVD) e Filtragem por Conteúdo (TF-IDF)
com pesos configuráveis.

Estratégia de pontuação:
    score_híbrido = α × CF_normalizado + (1-α) × CB_normalizado

Tratamento de cold-start:
    - Usuário sem avaliações → recomendações por popularidade (Bayesian Average)
    - Usuário novo (não estava no treino, mas tem avaliações) → somente CB +
      projeção SVD
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import get_settings
from app.recommender.collaborative import CollaborativeFilter
from app.recommender.content_based import ContentBasedFilter

settings = get_settings()
logger = logging.getLogger(__name__)


class HybridRecommender:
    """Recomendador híbrido CF + CB."""

    def __init__(
        self,
        cf_weight: float = settings.CF_WEIGHT,
        cb_weight: float = settings.CB_WEIGHT,
        n_components: int = settings.N_SVD_COMPONENTS,
    ):
        if abs(cf_weight + cb_weight - 1.0) > 1e-6:
            raise ValueError("cf_weight + cb_weight deve ser igual a 1.0")

        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

        self.cf = CollaborativeFilter(n_components=n_components)
        self.cb = ContentBasedFilter()

        self.popular_movie_ids: List[int] = []
        self.all_movie_ids: set = set()

        # Estatísticas de treino
        self.n_users: int = 0
        self.n_movies: int = 0
        self.n_ratings: int = 0

        self.is_trained: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Treinamento
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        tags_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Treina ambos os sub-modelos.

        Args:
            ratings_df: [user_id, movie_id, rating]
            movies_df:  [movie_id, title, genres]
            tags_df:    [movie_id, tag]  (opcional)
        """
        # ── Sub-modelos ───────────────────────────────────────────────────
        self.cf.fit(ratings_df)
        self.cb.fit(movies_df, tags_df)

        # ── Lista de todos os filmes ───────────────────────────────────────
        self.all_movie_ids = set(movies_df["movie_id"].tolist())

        # ── Popularidade (Bayesian Average) ───────────────────────────────
        agg = (
            ratings_df.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg", "count": "cnt"})
        )
        C = agg["cnt"].mean()      # contagem média global
        m = agg["avg"].mean()      # média global de notas
        agg["bayesian"] = (agg["cnt"] * agg["avg"] + C * m) / (agg["cnt"] + C)
        self.popular_movie_ids = (
            agg.nlargest(100, "bayesian")["movie_id"].tolist()
        )

        # ── Estatísticas ──────────────────────────────────────────────────
        self.n_users = ratings_df["user_id"].nunique()
        self.n_movies = movies_df["movie_id"].nunique()
        self.n_ratings = len(ratings_df)
        self.is_trained = True

        logger.info(
            f"Modelo Híbrido treinado — {self.n_users} usuários, "
            f"{self.n_movies} filmes, {self.n_ratings} avaliações. "
            f"α(CF)={self.cf_weight} β(CB)={self.cb_weight}"
        )

    def train_from_db(self, db: Session) -> None:
        """
        Carrega os dados do banco SQLite e treina os modelos.
        Chamado na inicialização da aplicação.
        """
        from app.models.db_models import Rating, Movie, Tag

        logger.info("Carregando dados do banco para treino…")

        # Avaliações
        rows_r = db.query(
            Rating.user_id, Rating.movie_id, Rating.rating
        ).all()
        ratings_df = pd.DataFrame(rows_r, columns=["user_id", "movie_id", "rating"])

        # Filmes
        rows_m = db.query(Movie.movie_id, Movie.title, Movie.genres).all()
        movies_df = pd.DataFrame(rows_m, columns=["movie_id", "title", "genres"])

        # Tags
        rows_t = db.query(Tag.movie_id, Tag.tag).all()
        tags_df = pd.DataFrame(rows_t, columns=["movie_id", "tag"])

        if ratings_df.empty or movies_df.empty:
            logger.warning("Sem dados suficientes para treinar os modelos.")
            return

        self.fit(ratings_df, movies_df, tags_df if not tags_df.empty else None)

    # ──────────────────────────────────────────────────────────────────────
    # Recomendação
    # ──────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        rated_movies: Dict[int, float],
        n: int = settings.DEFAULT_TOP_N,
    ) -> List[Dict]:
        """
        Gera recomendações para um usuário.

        Args:
            user_id:      ID do usuário.
            rated_movies: {movie_id: rating} — filmes já avaliados (excluídos).
            n:            Número de recomendações.

        Returns:
            Lista de dicionários com chaves: movie_id, score, source.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Chame fit() ou train_from_db() primeiro.")

        # ── Cold-start completo (sem avaliações) ──────────────────────────
        if not rated_movies:
            return self._popular_recommendations(n, exclude=set())

        # ── Candidatos: todos os filmes não avaliados pelo usuário ────────
        candidates = self.all_movie_ids - set(rated_movies.keys())

        # ── Scores CF ────────────────────────────────────────────────────
        if user_id in self.cf.user_id_to_idx:
            cf_raw = self.cf.predict_for_known_user(user_id)
        else:
            cf_raw = self.cf.predict_for_new_user(rated_movies)

        # ── Scores CB ────────────────────────────────────────────────────
        cb_raw = self.cb.predict_for_user(rated_movies, exclude_rated=True)

        # ── Normaliza CF para [0, 1] ──────────────────────────────────────
        cf_vals = [cf_raw.get(m, 0.0) for m in candidates]
        cf_min, cf_max = (min(cf_vals), max(cf_vals)) if cf_vals else (0, 1)
        cf_range = (cf_max - cf_min) or 1.0
        cf_norm = {m: (cf_raw.get(m, 0.0) - cf_min) / cf_range for m in candidates}

        # ── Normaliza CB para [0, 1] ──────────────────────────────────────
        cb_vals = [cb_raw.get(m, 0.0) for m in candidates]
        cb_max = max(cb_vals) if cb_vals else 1.0
        cb_max = cb_max or 1.0
        cb_norm = {m: cb_raw.get(m, 0.0) / cb_max for m in candidates}

        # ── Combinação híbrida ────────────────────────────────────────────
        results: List[Dict] = []
        for movie_id in candidates:
            cf_s = max(0.0, cf_norm.get(movie_id, 0.0))
            cb_s = max(0.0, cb_norm.get(movie_id, 0.0))

            if cf_s == 0.0 and cb_s == 0.0:
                continue

            if cf_s > 0 and cb_s > 0:
                score = self.cf_weight * cf_s + self.cb_weight * cb_s
                source = "hybrid"
            elif cf_s > 0:
                score = cf_s
                source = "collaborative"
            else:
                score = cb_s
                source = "content_based"

            results.append({"movie_id": movie_id, "score": float(score), "source": source})

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:n]

        # Complementa com populares se necessário
        if len(top) < n:
            needed = n - len(top)
            existing_ids = {r["movie_id"] for r in top}
            for pop_id in self._popular_recommendations(needed * 2, exclude=existing_ids | set(rated_movies)):
                if needed == 0:
                    break
                top.append(pop_id)
                needed -= 1

        return top[:n]

    def _popular_recommendations(
        self,
        n: int,
        exclude: set,
    ) -> List[Dict]:
        """Retorna filmes populares como fallback de cold-start."""
        result: List[Dict] = []
        for movie_id in self.popular_movie_ids:
            if movie_id not in exclude:
                result.append({"movie_id": movie_id, "score": 0.5, "source": "popular"})
            if len(result) >= n:
                break
        return result
