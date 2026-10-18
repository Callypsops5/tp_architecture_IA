# IA Search

Moteur de recherche sémantique sur des intitulés de marchés publics.
Le projet combine des embeddings, une recherche hybride dans Elasticsearch,
et un reranking par un LLM qui décide si chaque résultat répond à la requête.

## Architecture

- **Ollama** : embeddings (`qwen3-embedding:0.6b`, 1024 dimensions) et reranking (`qwen3.5:0.8b`).
- **Elasticsearch 9.3.2** : un index `data_<n>` par cluster, et un index `data_clusters` avec le centroïde de chaque cluster.
- **Backend FastAPI** : recherche en deux temps, puis reranking.
- **Front React (Vite + TypeScript)** : champ de recherche et affichage des résultats.
- **index_data** : script d'indexation (téléchargement, embeddings, clustering K-means, indexation).

## Fonctionnement de la recherche

1. La question est vectorisée par Ollama.
2. Le cluster le plus proche est trouvé par kNN dans `data_clusters`.
3. Les candidats du cluster sont cherchés par recherche hybride : BM25 (`match`) et kNN.
4. Chaque candidat est évalué par le modèle de rerank.
5. Les résultats sont filtrés : seuls ceux dont la décision est `yes` et dont le score atteint 0,4 sont retenus.

## Reranking

- Une requête par candidat, exécutée en parallèle.
- Le modèle répond selon un schéma JSON contraint : `{"answer": "yes"}` ou `{"answer": "no"}`.
- La décision et le score viennent de la même génération.
- Score : `p_yes / (p_yes + p_no)`, calculé à partir des logprobs de la position où `yes` ou `no` apparaît.
- Seuil de pertinence : 0,4.

### Limite connue

Le modèle `qwen3.5:0.8b` est petit. Il juge pertinentes certaines lignes proches sans rapport avec la requête, et ses scores hors sujet restent proches du seuil (environ 0,4). Le seuil sépare correctement les cas testés, mais avec une marge faible. Un modèle plus grand améliorerait la discrimination.

## Lancement

### 1. Démarrer les services

Depuis le dossier `iasearch` :

```powershell
docker compose up -d --build
```

### 2. Télécharger les modèles Ollama

```powershell
docker exec -it iasearch-ollama-1 ollama pull qwen3-embedding:0.6b
docker exec -it iasearch-ollama-1 ollama pull qwen3.5:0.8b
```

### 3. Indexer les données

Depuis le dossier `index_data` :

```powershell
uv sync
uv run python src/index_data/main.py
```

### 4. Tester

- API : http://localhost:8000/docs (`GET /search`, `POST /search`, `GET /search/{cluster_id}`)
- Front : http://localhost:3000
- Elasticsearch : http://localhost:9200

Exemple :

```powershell
curl.exe -G "http://localhost:8000/search" --data-urlencode "q=Travaux de voirie" --data-urlencode "k=5"
```

## Structure

```
iasearch/
├── docker-compose.yml
├── index_data/          # indexation : embeddings, clustering, Elasticsearch
├── fastapi-backend/     # API de recherche et reranking
└── react-front/         # interface utilisateur
```

## Auteur

NGELE BISSENGUE Jeannette, ONYANGO Charlotte, TROUILLET Steve — Master Data et IA, EPSI Montpellier.