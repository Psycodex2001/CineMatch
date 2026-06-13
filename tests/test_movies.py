"""
Testes unitários e de integração — Rotas de Filmes (/movies).

Cobre:
    - Listagem paginada (GET /movies/)
    - Filtro por gênero
    - Busca por título (GET /movies/search)
    - Detalhes de um filme (GET /movies/{id})
    - Adição de novo filme (POST /movies/)
    - Filmes similares (GET /movies/{id}/similar)
    - Tratamento de erros
"""

import pytest


class TestListMovies:
    def test_list_movies_returns_list(self, client):
        resp = client.get("/movies/")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "movies" in data
        assert data["total"] > 0
        assert isinstance(data["movies"], list)

    def test_list_movies_pagination(self, client):
        resp = client.get("/movies/?limit=3&skip=0")
        assert resp.status_code == 200
        assert len(resp.json()["movies"]) <= 3

    def test_list_movies_filter_by_genre(self, client):
        resp = client.get("/movies/?genre=Thriller")
        assert resp.status_code == 200
        movies = resp.json()["movies"]
        for m in movies:
            assert "Thriller" in m["genres"]

    def test_list_movies_filter_no_match_returns_empty(self, client):
        resp = client.get("/movies/?genre=GeneroInexistente999")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestSearchMovies:
    def test_search_by_title_returns_results(self, client):
        resp = client.get("/movies/search?q=Toy")
        assert resp.status_code == 200
        movies = resp.json()["movies"]
        assert len(movies) > 0
        assert any("Toy" in m["title"] for m in movies)

    def test_search_case_insensitive(self, client):
        resp_upper = client.get("/movies/search?q=TOY")
        resp_lower = client.get("/movies/search?q=toy")
        assert resp_upper.status_code == 200
        assert resp_lower.status_code == 200
        assert resp_upper.json()["total"] == resp_lower.json()["total"]

    def test_search_empty_query_returns_422(self, client):
        resp = client.get("/movies/search?q=")
        assert resp.status_code == 422

    def test_search_nonexistent_title_returns_empty(self, client):
        resp = client.get("/movies/search?q=FilmeInexistenteXYZ999")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestGetMovie:
    def test_get_movie_by_movie_id(self, client):
        resp = client.get("/movies/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["movie_id"] == 1
        assert "title" in data
        assert "genres" in data

    def test_get_movie_response_fields(self, client):
        resp = client.get("/movies/1")
        data = resp.json()
        required_fields = {"id", "movie_id", "title", "genres"}
        assert required_fields.issubset(set(data.keys()))

    def test_get_nonexistent_movie_returns_404(self, client):
        resp = client.get("/movies/999999")
        assert resp.status_code == 404


class TestAddMovie:
    def test_add_movie_success(self, client):
        payload = {"title": "Inception (2010)", "genres": "Action|Sci-Fi|Thriller"}
        resp = client.post("/movies/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Inception"
        assert data["year"] == 2010
        assert "movie_id" in data

    def test_add_movie_without_year(self, client):
        payload = {"title": "Filme Sem Ano", "genres": "Drama"}
        resp = client.post("/movies/", json=payload)
        assert resp.status_code == 201
        assert resp.json()["year"] is None

    def test_add_movie_default_genre(self, client):
        payload = {"title": "Filme Sem Genero"}
        resp = client.post("/movies/", json=payload)
        assert resp.status_code == 201
        assert resp.json()["genres"] == "(no genres listed)"

    def test_add_movie_generates_unique_movie_id(self, client):
        ids = set()
        for i in range(3):
            resp = client.post("/movies/", json={"title": f"Filme Auto {i}"})
            ids.add(resp.json()["movie_id"])
        assert len(ids) == 3


class TestSimilarMovies:
    def test_similar_movies_returns_list(self, client):
        resp = client.get("/movies/1/similar?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3

    def test_similar_movies_excludes_self(self, client):
        resp = client.get("/movies/1/similar?n=5")
        ids = [m["movie_id"] for m in resp.json()]
        assert 1 not in ids

    def test_similar_movies_nonexistent_returns_404(self, client):
        resp = client.get("/movies/999999/similar")
        assert resp.status_code == 404
