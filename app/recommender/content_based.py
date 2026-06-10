"""
Filtragem Baseada em Conteúdo via TF-IDF + Similaridade por Cosseno.

Algoritmo:
1. Combina gêneros e tags de cada filme em uma string de features.
2. Aplica TF-IDF (bigramas) para vetorizar os filmes.
3. Para um usuário, constrói um perfil como média ponderada dos vetores
   TF-IDF dos filmes avaliados, com peso proporcional à nota (centrada em 3):
       peso = (nota - 3) / 2   →  nota 5 → +1, nota 3 → 0, nota 1 → -1
4. Calcula a similaridade por cosseno entre o perfil do usuário e todos os filmes.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """Modelo de filtragem baseada em conteúdo com TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.movie_features: csr_matrix | None = None   # (n_movies, n_features)
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self.trained = False

    # ──────────────────────────────────────────────────────────────────────
    # Treinamento
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, movies_df: pd.DataFrame, tags_df: pd.DataFrame | None = None) -> None:
        """
        Treina o modelo de conteúdo.

        Args:
            movies_df: DataFrame com colunas [movie_id, title, genres].
            tags_df:   DataFrame opcional com colunas [movie_id, tag].
        """
        logger.info("Treinando Filtragem por Conteúdo (TF-IDF)…")

        movies = movies_df.copy().reset_index(drop=True)

        # ── Constrói string de features ───────────────────────────────────
        # Gêneros: "Action|Sci-Fi" → "action sci-fi action_sci-fi"
        movies["feature_str"] = movies["genres"].fillna("").apply(
            lambda g: " ".join(g.lower().replace("-", "_").split("|"))
        )

        # Tags: agrega por filme
        if tags_df is not None and len(tags_df) > 0:
            agg_tags = (
                tags_df.groupby("movie_id")["tag"]
                .apply(lambda x: " ".join(x.astype(str).str.lower()))
                .reset_index()
                .rename(columns={"tag": "agg_tags"})
            )
            movies = movies.merge(agg_tags, on="movie_id", how="left")
            movies["feature_str"] = (
                movies["feature_str"] + " " + movies["agg_tags"].fillna("")
            )

        # Inclui o título (palavras relevantes)
        movies["feature_str"] = (
            movies["feature_str"]
            + " "
            + movies["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)
        )

        # Mapeamentos
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(movies["movie_id"])}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

        # TF-IDF
        self.movie_features = self.vectorizer.fit_transform(movies["feature_str"])

        self.trained = True
        logger.info(
            f"CB treinado: {len(self.movie_id_to_idx)} filmes, "
            f"{self.movie_features.shape[1]} features TF-IDF."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Predição
    # ──────────────────────────────────────────────────────────────────────

    def predict_for_user(
        self,
        rated_movies: Dict[int, float],
        exclude_rated: bool = True,
    ) -> Dict[int, float]:
        """
        Calcula scores de conteúdo para todos os filmes dado o histórico do usuário.

        Args:
            rated_movies:   {movie_id: rating}
            exclude_rated:  se True, exclui filmes já avaliados

        Returns:
            {movie_id: cosine_similarity_score}
        """
        if not self.trained or not rated_movies:
            return {}

        # ── Perfil do usuário ─────────────────────────────────────────────
        n_features = self.movie_features.shape[1]
        user_profile = np.zeros(n_features, dtype=np.float64)
        total_weight = 0.0

        for movie_id, rating in rated_movies.items():
            if movie_id not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[movie_id]
            # Peso: nota centrada em 3, normalizada para [-1, 1]
            weight = (rating - 3.0) / 2.0
            user_profile += weight * self.movie_features[idx].toarray()[0]
            total_weight += abs(weight)

        if total_weight < 1e-10:
            # Usuário avaliou tudo com nota 3 → perfil neutro → usa média
            for movie_id, _ in rated_movies.items():
                if movie_id in self.movie_id_to_idx:
                    idx = self.movie_id_to_idx[movie_id]
                    user_profile += self.movie_features[idx].toarray()[0]
            total_weight = len(rated_movies)

        user_profile /= (total_weight + 1e-10)

        # ── Similaridade por cosseno ──────────────────────────────────────
        user_profile_sparse = csr_matrix(user_profile.reshape(1, -1))
        similarities = cosine_similarity(user_profile_sparse, self.movie_features)[0]

        result: Dict[int, float] = {}
        for movie_id, idx in self.movie_id_to_idx.items():
            if exclude_rated and movie_id in rated_movies:
                continue
            score = float(similarities[idx])
            if score > 0:
                result[movie_id] = score

        return result

    def get_similar_movies(
        self,
        movie_id: int,
        top_n: int = 10,
        exclude_self: bool = True,
    ) -> Dict[int, float]:
        """
        Retorna os top_n filmes mais similares a um filme dado.

        Args:
            movie_id:     ID do filme base.
            top_n:        Número de filmes a retornar.
            exclude_self: Se True, exclui o próprio filme.

        Returns:
            {movie_id: similarity_score}
        """
        if not self.trained or movie_id not in self.movie_id_to_idx:
            return {}

        idx = self.movie_id_to_idx[movie_id]
        movie_vec = self.movie_features[idx]
        similarities = cosine_similarity(movie_vec, self.movie_features)[0]

        if exclude_self:
            similarities[idx] = 0.0

        top_indices = np.argsort(similarities)[::-1][:top_n]
        return {
            self.idx_to_movie_id[i]: float(similarities[i])
            for i in top_indices
            if float(similarities[i]) > 0
        }
