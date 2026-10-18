# Backend FastAPI

API de recherche qui interroge Elasticsearch et Ollama.

## Endpoints

- `GET /search?q=...&k=5` : recherche avec la question en paramètre.
- `POST /search` : même recherche, avec un corps JSON `{"query": "...", "k": 5}`.
- `GET /search/{cluster_id}?q=...&k=5` : recherche dans l'index `data_<cluster_id>`.

Documentation interactive : http://localhost:8000/docs

## Fichiers principaux

- `src/fastapi_backend/main.py` : routes, recherche kNN et hybride, seuil de pertinence.
- `src/fastapi_backend/rerank.py` : reranking par `qwen3.5:0.8b`, avec sortie JSON contrainte et score calculé à partir des logprobs.

## Lancement

Avec Docker, depuis `iasearch` :

```powershell
docker compose up -d --build backend
```

En local, depuis ce dossier :

```powershell
uv sync
uv run uvicorn fastapi_backend.main:app --reload --port 8000
```

En local, Elasticsearch et Ollama doivent tourner sur `localhost`. Les variables `ES_URL` et `OLLAMA_URL` permettent de changer les adresses.