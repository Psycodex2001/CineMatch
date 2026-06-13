"""
Configuração global dos testes (pytest fixtures).

Usa banco SQLite de arquivo temporário + monkeypatching do lifespan
para evitar download do dataset durante os testes.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models.db_models import Movie, Rating, User
from app.recommender.hybrid import HybridRecommender
from app.state import app_state

# ── Banco de teste (arquivo; evita conflitos de thread com in-memory) ─────────
TEST_DB_URL = "sqlite:///./test_cinematch.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    import os
    try:
        os.remove("./test_cinematch.db")
    except FileNotFoundError:
        pass


@pytest.fixture()
def db():
    """Sessão isolada por teste (rollback ao final)."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestSession(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


# ── Dados mínimos ─────────────────────────────────────────────────────────────

def _seed_db(db) -> dict:
    movies = [
        Movie(movie_id=1,  title="Toy Story",         genres="Animation|Children|Comedy",   year=1995),
        Movie(movie_id=2,  title="Jumanji",            genres="Adventure|Children|Fantasy",  year=1995),
        Movie(movie_id=3,  title="Heat",               genres="Action|Crime|Thriller",       year=1995),
        Movie(movie_id=4,  title="Seven",              genres="Mystery|Thriller",            year=1995),
        Movie(movie_id=5,  title="Usual Suspects",     genres="Crime|Mystery|Thriller",      year=1995),
        Movie(movie_id=6,  title="Braveheart",         genres="Action|Drama|War",            year=1995),
        Movie(movie_id=7,  title="Apollo 13",          genres="Adventure|Drama|Sci-Fi|IMAX", year=1995),
        Movie(movie_id=8,  title="Ace Ventura",        genres="Comedy",                      year=1994),
        Movie(movie_id=9,  title="GoodFellas",         genres="Crime|Drama",                 year=1990),
        Movie(movie_id=10, title="Shawshank",          genres="Crime|Drama",                 year=1994),
    ]
    db.add_all(movies)
    db.flush()

    u1 = User(username="test_user_1")
    u2 = User(username="test_user_2")
    u3 = User(username="test_user_3")
    db.add_all([u1, u2, u3])
    db.flush()

    for uid, mid, r in [
        (u1.id,1,5.0),(u1.id,2,3.0),(u1.id,3,4.0),
        (u1.id,4,4.5),(u1.id,5,4.0),(u1.id,6,3.5),
        (u2.id,1,4.0),(u2.id,3,5.0),(u2.id,7,4.0),
        (u2.id,8,2.0),(u2.id,9,5.0),(u2.id,10,5.0),
        (u3.id,2,3.0),(u3.id,4,4.0),(u3.id,6,5.0),
        (u3.id,7,4.5),(u3.id,9,5.0),(u3.id,10,4.5),
    ]:
        db.add(Rating(user_id=uid, movie_id=mid, rating=r))
    db.flush()

    return {"u1_id": u1.id, "u2_id": u2.id, "u3_id": u3.id}


def _build_mock_recommender(db) -> HybridRecommender:
    rows_r = db.query(Rating.user_id, Rating.movie_id, Rating.rating).all()
    rows_m = db.query(Movie.movie_id, Movie.title, Movie.genres).all()
    ratings_df = pd.DataFrame(rows_r, columns=["user_id", "movie_id", "rating"])
    movies_df  = pd.DataFrame(rows_m, columns=["movie_id", "title", "genres"])
    rec = HybridRecommender(cf_weight=0.6, cb_weight=0.4, n_components=3)
    rec.fit(ratings_df, movies_df)
    return rec


# ── TestClient com lifespan neutralizado ─────────────────────────────────────

@pytest.fixture()
def client(db, monkeypatch):
    """
    TestClient com:
    - banco de teste injetado via dependency_override
    - dataset download neutralizado (monkeypatch)
    - HybridRecommender treinado com dados mínimos
    """
    # 1. Semeia o banco ANTES de qualquer coisa
    ids = _seed_db(db)

    # 2. Override da sessão de banco — precisa estar ativo ANTES do TestClient
    def override_get_db():
        yield db
    app.dependency_overrides[get_db] = override_get_db

    # 3. Neutraliza o download do dataset no lifespan
    monkeypatch.setattr("app.main.load_and_populate_db", lambda *a, **kw: None)

    # 4. Prepara o recomendador mockado (será atribuído ao app_state.recommender
    #    pelo lifespan customizado abaixo)
    mock_rec = _build_mock_recommender(db)

    # 5. Substitui HybridRecommender por um que já está treinado
    class PreTrainedRecommender(HybridRecommender):
        def train_from_db(self, _db):
            # Já treinado; só sincroniza all_movie_ids a partir do banco de teste
            movie_ids = {m.movie_id for m in _db.query(Movie.movie_id).all()}
            self.all_movie_ids = movie_ids or self.all_movie_ids

    monkeypatch.setattr("app.main.HybridRecommender", lambda **kw: mock_rec)
    monkeypatch.setattr(mock_rec, "train_from_db", lambda _db: None)

    # 6. Garante estado pronto antes do lifespan terminar
    app_state.recommender = mock_rec
    app_state.is_ready = True

    with TestClient(app, raise_server_exceptions=True) as c:
        c.user_ids = ids
        yield c

    app.dependency_overrides.clear()
