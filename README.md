# IA Search

Recherche sémantique sur des données de marchés publics.

## Stack
- Elasticsearch 9.3.2 (http://localhost:9200)
- Ollama : embedding (qwen3-embedding:0.6b) et rerank (qwen3.5:0.8b) (http://localhost:11434)
- Backend FastAPI (http://localhost:8000/docs)
- Front React (http://localhost:3000)

## Lancement

1. Démarrer les services :
   docker compose up -d --build

2. Télécharger les modèles dans Ollama :
   docker exec -it iasearch-ollama-1 ollama pull qwen3-embedding:0.6b
   docker exec -it iasearch-ollama-1 ollama pull qwen3.5:0.8b

3. Indexer les données (depuis index_data) :
   cd index_data
   uv sync
   uv run python src/index_data/main.py

4. Ouvrir http://localhost:3000 et lancer une recherche.