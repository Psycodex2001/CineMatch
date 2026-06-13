"""
Testes de integração — Recomendações, Avaliações e Modelo.

Cobre:
    - Recomendações para usuário existente
    - Cold-start (usuário sem avaliações)
    - Campos e tipos da resposta
    - Criação/atualização/remoção de avaliações
    - Status do modelo
    - Usuário inexistente → 404
    - Modelo não treinado → 503
"""

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Recomendações
# ──────────────────────────────────────────────────────────────────────────────

class TestRecommendations:
    def test_recommendations_for_existing_user(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/recommendations/{u1_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == u1_id
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert data["n_recommendations"] == len(data["recommendations"])

    def test_recommendations_response_fields(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/recommendations/{u1_id}?n=3")
        assert resp.status_code == 200
        items = resp.json()["recommendations"]
        if items:
            item = items[0]
            required = {"movie_id", "title", "genres", "score", "source"}
            assert required.issubset(item.keys())

    def test_recommendations_n_parameter(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/recommendations/{u1_id}?n=2")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) <= 2

    def test_recommendations_scores_between_0_and_1(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/recommendations/{u1_id}?n=5")
        assert resp.status_code == 200
        for item in resp.json()["recommendations"]:
            assert 0.0 <= item["score"] <= 1.0, (
                f"Score fora do intervalo [0,1]: {item['score']}"
            )

    def test_recommendations_source_valid_values(self, client):
        u1_id = client.user_ids["u1_id"]
        valid_sources = {"hybrid", "collaborative", "content_based", "popular"}
        resp = client.get(f"/recommendations/{u1_id}?n=10")
        assert resp.status_code == 200
        for item in resp.json()["recommendations"]:
            assert item["source"] in valid_sources, (
                f"Source inválido: {item['source']}"
            )

    def test_recommendations_excludes_already_rated_movies(self, client):
        """Filmes já avaliados pelo usuário não devem aparecer nas recomendações."""
        u1_id = client.user_ids["u1_id"]
        # Pega filmes já avaliados
        rated_resp = client.get(f"/users/{u1_id}/ratings")
        rated_ids = {r["movie_id"] for r in rated_resp.json()}

        rec_resp = client.get(f"/recommendations/{u1_id}?n=10")
        rec_ids = {r["movie_id"] for r in rec_resp.json()["recommendations"]}

        overlap = rated_ids & rec_ids
        assert len(overlap) == 0, (
            f"Filmes já avaliados apareceram nas recomendações: {overlap}"
        )

    def test_recommendations_cold_start_new_user(self, client):
        """Usuário sem avaliações deve receber recomendações populares."""
        created = client.post("/users/", json={"username": "usuario_cold"}).json()
        resp = client.get(f"/recommendations/{created['id']}")
        assert resp.status_code == 200
        data = resp.json()
        # Deve retornar algo (fallback popular)
        assert len(data["recommendations"]) > 0
        for item in data["recommendations"]:
            assert item["source"] == "popular"

    def test_recommendations_nonexistent_user_returns_404(self, client):
        resp = client.get("/recommendations/999999")
        assert resp.status_code == 404

    def test_recommendations_n_exceeds_max_returns_422(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/recommendations/{u1_id}?n=1000")
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Avaliações
# ──────────────────────────────────────────────────────────────────────────────

class TestRatings:
    def test_add_rating_success(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"user_id": u1_id, "movie_id": 10, "rating": 4.5}
        resp = client.post("/ratings/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["rating"] == 4.5
        assert data["movie_id"] == 10

    def test_add_rating_invalid_half_step_returns_422(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"user_id": u1_id, "movie_id": 1, "rating": 3.3}
        resp = client.post("/ratings/", json=payload)
        assert resp.status_code == 422

    def test_add_rating_above_max_returns_422(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"user_id": u1_id, "movie_id": 1, "rating": 5.5}
        resp = client.post("/ratings/", json=payload)
        assert resp.status_code == 422

    def test_add_rating_updates_existing(self, client):
        u1_id = client.user_ids["u1_id"]
        # Primeira avaliação
        client.post("/ratings/", json={"user_id": u1_id, "movie_id": 1, "rating": 3.0})
        # Atualiza
        resp = client.post("/ratings/", json={"user_id": u1_id, "movie_id": 1, "rating": 5.0})
        assert resp.status_code == 201
        assert resp.json()["rating"] == 5.0

    def test_add_rating_nonexistent_user_returns_404(self, client):
        payload = {"user_id": 999999, "movie_id": 1, "rating": 4.0}
        resp = client.post("/ratings/", json=payload)
        assert resp.status_code == 404

    def test_add_rating_nonexistent_movie_returns_404(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"user_id": u1_id, "movie_id": 999999, "rating": 4.0}
        resp = client.post("/ratings/", json=payload)
        assert resp.status_code == 404

    def test_list_ratings_filter_by_user(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/ratings/?user_id={u1_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        for r in data:
            assert r["user_id"] == u1_id

    def test_delete_rating(self, client):
        u2_id = client.user_ids["u2_id"]
        # Cria avaliação
        created = client.post(
            "/ratings/", json={"user_id": u2_id, "movie_id": 5, "rating": 3.0}
        ).json()
        # Remove
        resp = client.delete(f"/ratings/{created['id']}")
        assert resp.status_code == 204

    def test_delete_nonexistent_rating_returns_404(self, client):
        resp = client.delete("/ratings/999999")
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# Status e re-treino do modelo
# ──────────────────────────────────────────────────────────────────────────────

class TestModelStatus:
    def test_get_model_status(self, client):
        resp = client.get("/recommendations/model/status")
        assert resp.status_code == 200
        data = resp.json()
        required = {"is_trained", "n_users", "n_movies", "n_ratings", "cf_weight", "cb_weight"}
        assert required.issubset(data.keys())

    def test_model_is_trained(self, client):
        resp = client.get("/recommendations/model/status")
        assert resp.json()["is_trained"] is True

    def test_model_weights_sum_to_one(self, client):
        data = client.get("/recommendations/model/status").json()
        total = data["cf_weight"] + data["cb_weight"]
        assert abs(total - 1.0) < 1e-6

    def test_retrain_model(self, client):
        resp = client.post("/recommendations/model/retrain")
        assert resp.status_code == 200
        assert resp.json()["is_trained"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Health checks
# ──────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_trained" in data
