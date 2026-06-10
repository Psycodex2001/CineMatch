"""
Filtragem Colaborativa via Decomposição em Valores Singulares (SVD).

Algoritmo:
1. Monta a matriz usuário-item R (usuários × filmes) com as avaliações.
2. Centraliza R subtraindo a média de cada usuário (mean-centering).
3. Aplica TruncatedSVD: R_centrada ≈ U × Σ × Vᵀ  (k componentes latentes).
4. Reconstrói a matriz de previsões: R̂ = U × Σ × Vᵀ + médias.
5. Para usuários fora do treino (cold-start parcial), projeta o vetor de
   avaliações no espaço latente usando a pseudo-inversa de Σ × Vᵀ.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """Modelo de filtragem colaborativa baseado em SVD."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components

        # Mapeamentos índice ↔ ID
        self.user_id_to_idx: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.idx_to_item_id: Dict[int, int] = {}

        # Fatores latentes
        self.svd: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None   # (n_users, k)
        self.item_factors: np.ndarray | None = None   # (k, n_items)
        self.user_means: np.ndarray | None = None     # (n_users,)

        # Matriz de previsões pré-computada
        self.predicted_ratings: np.ndarray | None = None

        self.trained = False

    # ──────────────────────────────────────────────────────────────────────
    # Treinamento
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, ratings_df: pd.DataFrame) -> None:
        """
        Treina o modelo SVD.

        Args:
            ratings_df: DataFrame com colunas [user_id, movie_id, rating].
        """
        logger.info("Treinando Filtragem Colaborativa (SVD)…")

        unique_users = sorted(ratings_df["user_id"].unique())
        unique_items = sorted(ratings_df["movie_id"].unique())

        n_users = len(unique_users)
        n_items = len(unique_items)

        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx_to_item_id = {i: iid for iid, i in self.item_id_to_idx.items()}

        # Constrói matriz esparsa de avaliações
        rows = [self.user_id_to_idx[u] for u in ratings_df["user_id"]]
        cols = [self.item_id_to_idx[m] for m in ratings_df["movie_id"]]
        data = ratings_df["rating"].values.astype(np.float32)

        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        R_dense = R.toarray()

        # Média por usuário (ignora zeros = não avaliados)
        self.user_means = np.zeros(n_users, dtype=np.float32)
        for i in range(n_users):
            rated = R_dense[i][R_dense[i] != 0]
            self.user_means[i] = rated.mean() if len(rated) > 0 else 0.0

        # Centralização: subtrai a média do usuário apenas nas entradas avaliadas
        R_centered = R_dense.copy()
        for i in range(n_users):
            mask = R_dense[i] != 0
            R_centered[i, mask] -= self.user_means[i]

        # SVD truncado
        k = min(self.n_components, min(n_users, n_items) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=42, n_iter=10)
        self.user_factors = self.svd.fit_transform(csr_matrix(R_centered))  # (n_users, k)
        self.item_factors = self.svd.components_                             # (k, n_items)

        # Previsão: R̂ = U × Vᵀ + médias dos usuários
        self.predicted_ratings = (
            self.user_factors @ self.item_factors
            + self.user_means[:, np.newaxis]
        )

        self.trained = True
        logger.info(
            f"CF treinado: {n_users} usuários, {n_items} itens, k={k} componentes."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Predição
    # ──────────────────────────────────────────────────────────────────────

    def predict_for_known_user(self, user_id: int) -> Dict[int, float]:
        """
        Retorna previsões de avaliação para todos os itens de um usuário
        já presente no conjunto de treino.
        """
        if not self.trained or user_id not in self.user_id_to_idx:
            return {}

        idx = self.user_id_to_idx[user_id]
        preds = self.predicted_ratings[idx]
        return {self.idx_to_item_id[i]: float(preds[i]) for i in range(len(preds))}

    def predict_for_new_user(self, rated_movies: Dict[int, float]) -> Dict[int, float]:
        """
        Projeta um novo usuário (não visto no treino) no espaço latente.

        Usa o método de folding-in: o vetor de avaliações centralizado é
        multiplicado por Vᵀ para obter os fatores latentes do usuário.

        Args:
            rated_movies: {movie_id: rating}

        Returns:
            {movie_id: predicted_rating}
        """
        if not self.trained or not rated_movies:
            return {}

        n_items = len(self.item_id_to_idx)
        user_vec = np.zeros(n_items, dtype=np.float32)
        valid_ratings = []

        for movie_id, rating in rated_movies.items():
            if movie_id in self.item_id_to_idx:
                idx = self.item_id_to_idx[movie_id]
                user_vec[idx] = rating
                valid_ratings.append(rating)

        if not valid_ratings:
            return {}

        user_mean = float(np.mean(valid_ratings))

        # Centraliza o vetor do novo usuário
        centered = user_vec.copy()
        for movie_id, rating in rated_movies.items():
            if movie_id in self.item_id_to_idx:
                idx = self.item_id_to_idx[movie_id]
                centered[idx] -= user_mean

        # Projeção no espaço latente: u_lat = centered @ Vᵀᵀ (= Vᵀ transposta)
        u_latent = centered @ self.item_factors.T  # (k,)

        # Reconstrução: R̂ = u_lat @ Vᵀ + mean
        preds = u_latent @ self.item_factors + user_mean  # (n_items,)

        return {self.idx_to_item_id[i]: float(preds[i]) for i in range(n_items)}
