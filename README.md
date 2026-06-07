# 🎬 CineMatch — Sistema de Recomendação de Filmes

> API RESTful híbrida de recomendação de filmes, combinando **Filtragem Colaborativa (SVD)** e **Filtragem Baseada em Conteúdo (TF-IDF)**, construída com FastAPI e containerizada com Docker.
> Dataset utilizado: [MovieLens Small](https://huggingface.co/datasets/ashraq/ml-latest-small) (~100 mil avaliações, ~9.700 filmes, ~600 usuários).

---

## Índice

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Modelo de Recomendação](#2-modelo-de-recomendação)
3. [Estrutura do Projeto](#3-estrutura-do-projeto)
4. [Pré-requisitos](#4-pré-requisitos)
5. [Instalação e Execução](#5-instalação-e-execução)
   - [Com Docker (recomendado)](#51-com-docker-recomendado)
   - [Localmente (sem Docker)](#52-localmente-sem-docker)
6. [Documentação da API](#6-documentação-da-api)
   - [Usuários](#61-usuários)
   - [Filmes](#62-filmes)
   - [Avaliações](#63-avaliações)
   - [Recomendações](#64-recomendações)
7. [Exemplos de Uso (cURL)](#7-exemplos-de-uso-curl)
8. [Executando os Testes](#8-executando-os-testes)
9. [Configuração](#9-configuração)
10. [Decisões de Design](#10-decisões-de-design)

---

## 1. Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                         Cliente                             │
│              (Swagger UI / cURL / Postman)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                       FastAPI (app/)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ /users   │ │ /movies  │ │ /ratings │ │/recommendations│  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │
│       │            │            │               │           │
│  ┌────▼────────────▼────────────▼───────┐       │           │
│  │         SQLAlchemy (SQLite)           │       │           │
│  └───────────────────────────────────────┘       │           │
│                                          ┌───────▼────────┐ │
│                                          │ HybridRecommender│ │
│                                          │  ┌───────────┐  │ │
│                                          │  │  SVD (CF) │  │ │
│                                          │  ├───────────┤  │ │
│                                          │  │TF-IDF (CB)│  │ │
│                                          │  └───────────┘  │ │
│                                          └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Fluxo de inicialização:**

1. As tabelas do banco são criadas (se não existirem).
2. O dataset MovieLens Small é baixado do HuggingFace e inserido no banco (apenas na primeira execução).
3. O modelo híbrido é treinado com todos os dados disponíveis.
4. A API fica disponível na porta `8000`.

---

## 2. Modelo de Recomendação

### 2.1 Filtragem Colaborativa — SVD (Decomposição em Valores Singulares)

A filtragem colaborativa explora padrões latentes nas avaliações dos usuários. Não precisa conhecer o conteúdo dos filmes — aprende a partir do comportamento coletivo.

**Pipeline:**

```
Matriz Usuário-Item (R)
       │
       ▼ Mean-Centering (subtrai média do usuário)
       │
       ▼ TruncatedSVD (k=50 componentes latentes)
       │      R ≈ U × Σ × Vᵀ
       │
       ▼ Reconstrução: R̂ = U × Vᵀ + médias
       │
       ▼ Previsão de nota para qualquer par (usuário, filme)
```

Para **novos usuários** (não presentes no treino), o vetor de avaliações é projetado no espaço latente via *folding-in*:

```
u_latente = vetor_centralizado @ Vᵀᵀ
nota_prevista = u_latente @ Vᵀ + média_usuário
```

### 2.2 Filtragem Baseada em Conteúdo — TF-IDF + Cosseno

Recomenda filmes similares àqueles que o usuário gostou, com base nos atributos do conteúdo.

**Representação dos filmes:**
```
feature_string = gêneros (normalizados) + tags (agregadas) + título
     → TF-IDF com bigramas → vetor esparso
```

**Perfil do usuário:**
```
perfil = Σ [ peso_i × vetor_TF-IDF(filme_i) ]
  onde: peso_i = (nota_i − 3) / 2
  # nota 5 → peso +1 (adorou),  nota 3 → 0 (neutro),  nota 1 → -1 (detestou)
```

**Similaridade:**
```
score_CB(filme) = cosseno(perfil_usuário, vetor_filme)
```

### 2.3 Modelo Híbrido

Combina os dois modelos com pesos configuráveis:

```
score_híbrido = α × CF_normalizado + (1−α) × CB_normalizado
```

| Parâmetro | Valor padrão | Descrição |
|-----------|-------------|-----------|
| `α` (CF_WEIGHT) | 0.60 | Peso da Filtragem Colaborativa |
| `1−α` (CB_WEIGHT) | 0.40 | Peso da Filtragem por Conteúdo |
| `k` (N_SVD_COMPONENTS) | 50 | Dimensões do espaço latente |

**Tratamento de cold-start:**

| Situação | Estratégia |
|----------|------------|
| Usuário sem nenhuma avaliação | Retorna filmes populares (Bayesian Average) |
| Usuário novo com ≥1 avaliação | Projeção SVD (folding-in) + CB |
| Filme sem avaliações | Incluído apenas via CB |

O campo `source` em cada recomendação indica qual estratégia foi usada: `hybrid`, `collaborative`, `content_based` ou `popular`.

---

## 3. Estrutura do Projeto

```
cinematch/
├── app/
│   ├── main.py                  # Ponto de entrada FastAPI + lifespan
│   ├── config.py                # Configurações centralizadas (pydantic-settings)
│   ├── database.py              # Engine e sessões SQLAlchemy
│   ├── state.py                 # Estado global (recomendador singleton)
│   ├── models/
│   │   ├── db_models.py         # ORM: User, Movie, Rating, Tag
│   │   └── schemas.py           # Pydantic schemas (request/response)
│   ├── recommender/
│   │   ├── collaborative.py     # Filtragem Colaborativa (SVD)
│   │   ├── content_based.py     # Filtragem por Conteúdo (TF-IDF)
│   │   └── hybrid.py            # Modelo Híbrido + cold-start
│   ├── routers/
│   │   ├── users.py             # CRUD de usuários
│   │   ├── movies.py            # Listagem, busca, filmes similares
│   │   ├── ratings.py           # Avaliações
│   │   └── recommendations.py  # Recomendações + gestão do modelo
│   └── services/
│       └── data_loader.py       # Download e importação do MovieLens
├── tests/
│   ├── conftest.py              # Fixtures: banco em memória, mock recommender
│   ├── test_users.py            # Testes de usuários
│   ├── test_movies.py           # Testes de filmes
│   └── test_recommendations.py # Testes de recomendações e avaliações
├── data/                        # Banco SQLite (criado em runtime)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 4. Pré-requisitos

| Ferramenta | Versão mínima | Docker | Local |
|------------|--------------|--------|-------|
| Docker | 24.x | ✅ | — |
| Docker Compose | 2.x | ✅ | — |
| Python | 3.11+ | — | ✅ |
| pip | 23+ | — | ✅ |

---

## 5. Instalação e Execução

### 5.1 Com Docker (recomendado)

```bash
# 1. Clone o repositório
git clone <url-do-repositório>
cd cinematch

# 2. (Opcional) Ajuste variáveis de ambiente
cp .env.example .env
# edite .env se necessário

# 3. Build e inicialização
docker-compose up --build
```

> ⏳ **Na primeira execução**, o sistema irá:
> - Baixar o dataset MovieLens Small do HuggingFace (~25 MB)
> - Popular o banco de dados (~100 mil avaliações)
> - Treinar o modelo híbrido
>
> Isso pode levar de **1 a 3 minutos**. Aguarde a mensagem:
> ```
> ✅  CineMatch pronto! Acesse /docs para a documentação.
> ```

**Acesse:**
- Swagger UI (interativo): [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc:                   [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health check:            [http://localhost:8000/health](http://localhost:8000/health)

**Parar a aplicação:**
```bash
docker-compose down
# Para remover também o volume de dados:
docker-compose down -v
```

### 5.2 Localmente (sem Docker)

```bash
# 1. Clone e entre no diretório
git clone <url-do-repositório>
cd cinematch

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Configure variáveis de ambiente
cp .env.example .env

# 5. Inicie a API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 6. Documentação da API

A documentação interativa completa está disponível em `/docs` (Swagger UI) após a inicialização.

### 6.1 Usuários

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/users/` | Cria novo usuário |
| `GET` | `/users/{user_id}` | Retorna dados do usuário |
| `GET` | `/users/{user_id}/ratings` | Lista avaliações do usuário |
| `PUT` | `/users/{user_id}/preferences` | Cria/atualiza múltiplas avaliações |
| `DELETE` | `/users/{user_id}` | Remove usuário e suas avaliações |

### 6.2 Filmes

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/movies/` | Lista filmes (paginado, filtro por gênero) |
| `GET` | `/movies/search?q=<termo>` | Busca por título |
| `GET` | `/movies/{movie_id}` | Detalhes de um filme |
| `POST` | `/movies/` | Adiciona novo filme ao catálogo |
| `GET` | `/movies/{movie_id}/similar` | Filmes similares (CB) |

### 6.3 Avaliações

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/ratings/` | Adiciona ou atualiza avaliação |
| `GET` | `/ratings/` | Lista avaliações (filtros: user_id, movie_id) |
| `DELETE` | `/ratings/{rating_id}` | Remove avaliação |

### 6.4 Recomendações

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/recommendations/{user_id}?n=10` | Recomendações personalizadas |
| `GET` | `/recommendations/model/status` | Status e métricas do modelo |
| `POST` | `/recommendations/model/retrain` | Re-treina o modelo |

**Schema de resposta de recomendação:**
```json
{
  "user_id": 1,
  "n_recommendations": 10,
  "model_trained": true,
  "recommendations": [
    {
      "movie_id": 318,
      "title": "The Shawshank Redemption",
      "genres": "Crime|Drama",
      "year": 1994,
      "score": 0.847392,
      "source": "hybrid"
    }
  ]
}
```

---

## 7. Exemplos de Uso (cURL)

```bash
# Criar usuário
curl -X POST http://localhost:8000/users/ \
  -H "Content-Type: application/json" \
  -d '{"username": "maria_silva"}'

# Obter recomendações (top 5)
curl "http://localhost:8000/recommendations/1?n=5"

# Adicionar avaliação
curl -X POST http://localhost:8000/ratings/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 1, "rating": 4.5}'

# Atualizar preferências em lote
curl -X PUT http://localhost:8000/users/1/preferences \
  -H "Content-Type: application/json" \
  -d '{
    "ratings": [
      {"user_id": 1, "movie_id": 296, "rating": 5.0},
      {"user_id": 1, "movie_id": 318, "rating": 5.0},
      {"user_id": 1, "movie_id": 527, "rating": 4.5}
    ]
  }'

# Buscar filmes
curl "http://localhost:8000/movies/search?q=matrix"

# Filmes similares a Toy Story (movie_id=1)
curl "http://localhost:8000/movies/1/similar?n=5"

# Status do modelo
curl http://localhost:8000/recommendations/model/status
```

---

## 8. Executando os Testes

Os testes usam um banco SQLite em memória e um recomendador treinado com dados mínimos — **não requerem conexão com a internet**.

```bash
# Instale as dependências (se ainda não instalou)
pip install -r requirements.txt

# Execute todos os testes
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ -v --tb=short

# Arquivo específico
pytest tests/test_recommendations.py -v
```

**Saída esperada:**
```
tests/test_users.py::TestCreateUser::test_create_user_success            PASSED
tests/test_users.py::TestCreateUser::test_create_user_duplicate_returns_409 PASSED
...
tests/test_recommendations.py::TestRecommendations::test_recommendations_cold_start_new_user PASSED
tests/test_recommendations.py::TestHealth::test_health_endpoint          PASSED

========== XX passed in X.XXs ==========
```

---

## 9. Configuração

Todas as configurações são lidas do arquivo `.env` (ou variáveis de ambiente no Docker):

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `DATABASE_URL` | `sqlite:///./data/cinematch.db` | URL de conexão com o banco |
| `DATASET_REPO` | `ashraq/ml-latest-small` | Repositório HuggingFace do dataset |
| `N_SVD_COMPONENTS` | `50` | Dimensões do espaço latente (SVD) |
| `CF_WEIGHT` | `0.6` | Peso da Filtragem Colaborativa |
| `CB_WEIGHT` | `0.4` | Peso da Filtragem por Conteúdo |
| `DEFAULT_TOP_N` | `10` | Nº padrão de recomendações |
| `MAX_TOP_N` | `50` | Nº máximo de recomendações |

---

## 10. Decisões de Design

### Por que SQLite?
SQLite elimina a necessidade de um serviço de banco externo, simplificando a execução local e em Docker. O `docker-compose.yml` inclui (comentado) uma alternativa com PostgreSQL para ambientes de produção.

### Por que TruncatedSVD do scikit-learn?
É nativo do ecossistema Python/scikit-learn, sem dependências extras. A implementação expõe os fatores `U` e `Vᵀ` necessários para o *folding-in* de novos usuários.

### Por que bigramas no TF-IDF?
Gêneros como `"Sci-Fi"` e pares de palavras em tags (ex.: `"based on book"`) carregam mais semântica como unidade do que unigramas isolados.

### Por que Bayesian Average para popularidade?
A média bayesiana penaliza filmes com poucas avaliações, evitando que um filme com uma única nota 5,0 apareça como "mais popular" do que filmes bem avaliados por centenas de usuários.

### Por que `workers=1` no Docker?
O modelo de recomendação é carregado em memória como um singleton. Múltiplos workers criariam cópias independentes do modelo, causando inconsistências e alto consumo de RAM. Para escalar, use uma fila de mensagens (ex.: Celery + Redis) ou sirva o modelo via um serviço dedicado.

### Atualização do modelo
O modelo não é re-treinado a cada nova avaliação (custo proibitivo). O endpoint `POST /recommendations/model/retrain` permite re-treinamento manual. Em produção, recomenda-se agendar re-treinos periódicos (ex.: via cron ou Airflow).
