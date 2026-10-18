# Indexation des données

Script qui prépare les données pour la recherche.

## Étapes

1. Téléchargement des intitulés depuis pCloud.
2. Calcul des embeddings avec Ollama (`qwen3-embedding:0.6b`, 1024 dimensions).
3. Clustering K-means en 10 groupes.
4. Indexation dans Elasticsearch : un index `data_1` à `data_10` par cluster, et `data_clusters` avec le centroïde de chaque cluster.

## Lancement

Ollama et Elasticsearch doivent tourner (`docker compose up -d` depuis `iasearch`).

```powershell
uv sync
uv run python src/index_data/main.py
```

## Fichiers principaux

- `src/index_data/main.py` : fonctions de téléchargement, embedding, clustering et indexation.