"""
Testes unitários e de integração — Rotas de Usuários (/users).

Cobre:
    - Criação de usuário (POST /users/)
    - Busca de usuário (GET /users/{id})
    - Listagem de avaliações (GET /users/{id}/ratings)
    - Atualização de preferências (PUT /users/{id}/preferences)
    - Remoção de usuário (DELETE /users/{id})
    - Tratamento de erros (404, 409, 400)
"""

import pytest


class TestCreateUser:
    def test_create_user_success(self, client):
        resp = client.post("/users/", json={"username": "novo_usuario"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "novo_usuario"
        assert "id" in data
        assert "created_at" in data

    def test_create_user_duplicate_returns_409(self, client):
        client.post("/users/", json={"username": "usuario_dup"})
        resp = client.post("/users/", json={"username": "usuario_dup"})
        assert resp.status_code == 409

    def test_create_user_short_username_returns_422(self, client):
        resp = client.post("/users/", json={"username": "ab"})
        assert resp.status_code == 422

    def test_create_user_missing_username_returns_422(self, client):
        resp = client.post("/users/", json={})
        assert resp.status_code == 422


class TestGetUser:
    def test_get_existing_user(self, client):
        created = client.post("/users/", json={"username": "usuario_get"}).json()
        resp = client.get(f"/users/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["username"] == "usuario_get"

    def test_get_nonexistent_user_returns_404(self, client):
        resp = client.get("/users/999999")
        assert resp.status_code == 404


class TestGetUserRatings:
    def test_get_ratings_of_seeded_user(self, client):
        u1_id = client.user_ids["u1_id"]
        resp = client.get(f"/users/{u1_id}/ratings")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for r in data:
            assert "movie_id" in r
            assert "rating" in r

    def test_get_ratings_new_user_returns_empty(self, client):
        created = client.post("/users/", json={"username": "sem_notas"}).json()
        resp = client.get(f"/users/{created['id']}/ratings")
        assert resp.status_code == 200
        assert resp.json() == []


class TestUpdatePreferences:
    def test_update_preferences_success(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {
            "ratings": [
                {"user_id": u1_id, "movie_id": 8, "rating": 3.5},
                {"user_id": u1_id, "movie_id": 9, "rating": 5.0},
            ]
        }
        resp = client.put(f"/users/{u1_id}/preferences", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ratings_map = {r["movie_id"]: r["rating"] for r in data}
        assert ratings_map[8] == 3.5
        assert ratings_map[9] == 5.0

    def test_update_preferences_invalid_movie_returns_404(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"ratings": [{"user_id": u1_id, "movie_id": 99999, "rating": 4.0}]}
        resp = client.put(f"/users/{u1_id}/preferences", json=payload)
        assert resp.status_code == 404

    def test_update_preferences_user_mismatch_returns_400(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"ratings": [{"user_id": 9999, "movie_id": 1, "rating": 4.0}]}
        resp = client.put(f"/users/{u1_id}/preferences", json=payload)
        assert resp.status_code == 400

    def test_update_preferences_invalid_rating_returns_422(self, client):
        u1_id = client.user_ids["u1_id"]
        payload = {"ratings": [{"user_id": u1_id, "movie_id": 1, "rating": 6.0}]}
        resp = client.put(f"/users/{u1_id}/preferences", json=payload)
        assert resp.status_code == 422


class TestDeleteUser:
    def test_delete_user_success(self, client):
        created = client.post("/users/", json={"username": "para_deletar"}).json()
        resp = client.delete(f"/users/{created['id']}")
        assert resp.status_code == 204
        # Confirma que não existe mais
        assert client.get(f"/users/{created['id']}").status_code == 404

    def test_delete_nonexistent_user_returns_404(self, client):
        resp = client.delete("/users/999999")
        assert resp.status_code == 404
