#src\fastapi_backend\main.py

import os

import requests
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi_backend.rerank import rerank

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

es = Elasticsearch(ES_URL)

app = FastAPI(title="IA Search Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)


def embed_query(text: str) -> list[float]:
    """Calcule l'embedding de la question avec Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [text]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def knn_search(index_name: str, vector: list[float], k: int) -> list[dict]:
    """Recherche les k documents les plus proches du vecteur dans un index."""
    response = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(k * 10, 100),
        },
        source_excludes=["embedding"],
    )   
    return response["hits"]["hits"]

def hybrid_search(index_name: str, vector: list[float], text: str, k: int) -> list[dict]:
    """Recherche hybride : combine la recherche texte (match) et la recherche vectorielle (knn)."""
    response = es.search(
        index=index_name,
        query={"match": {"line": text}},
        knn={
            "field": "embedding",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(k * 10, 100),
        },
        size=k,
        source_excludes=["embedding"],
    )
    return response["hits"]["hits"]


def format_hits(hits: list[dict]) -> list[dict]:
    return [{"line": hit["_source"]["line"], "score": hit["_score"]} for hit in hits]

RERANK_THRESHOLD = 0.4


RERANK_THRESHOLD = 0.4


def keep_relevant(ranked: list[dict], k: int) -> list[dict]:
    """Garde les candidats acceptés par le JSON et dépassant le seuil."""
    return [
        item
        for item in ranked
        if item.get("answer") == "yes"
        and item.get("score_error") is None
        and item.get("score", 0.0) >= RERANK_THRESHOLD
    ][:k]


def run_search(query: str, k: int) -> dict:
    """Recherche les candidats et renvoie les résultats avec leur score."""

    vector = embed_query(query)

    cluster_hits = knn_search("data_clusters", vector, 1)

    if not cluster_hits:
        raise HTTPException(
            status_code=404,
            detail="Aucun cluster trouvé",
        )

    data_index = cluster_hits[0]["_source"]["data_index"]

    candidates = format_hits(
        hybrid_search(data_index, vector, query, max(k * 4, 20))
    )

    # Évaluer tous les candidats, y compris ceux qui seront rejetés.
    all_ranked = rerank(
        query,
        [candidate["line"] for candidate in candidates],
        include_rejected=True,
    )

    # Filtrer les résultats réellement pertinents.
    ranked = keep_relevant(all_ranked, k)

    # Meilleur score parmi TOUS les candidats évalués.
    best = max(
        all_ranked,
        key=lambda item: item["score"],
        default=None,
    )

    return {
        "query": query,
        "cluster_index": data_index,
        "results": ranked,
        "best_score": round(best["score"], 4) if best else None,
        "best_score_percentage": (
            round(best["score"] * 100, 2) if best else None
        ),
        "best_candidate": best["line"] if best else None,
        "best_answer": best["answer"] if best else None,
        "message": (
            None
            if ranked
            else "Ce que vous avez demandé n'est pas inscrit dans la base."
        ),
    }

@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
    return run_search(q, k)


@app.post("/search")
def search_post(body: SearchRequest):
    return run_search(body.query, body.k)


@app.get("/search/{cluster_id}")
def search_in_cluster(
    cluster_id: int,
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
):
    index_name = f"data_{cluster_id}"
    if not es.indices.exists(index=index_name):
        raise HTTPException(status_code=404, detail=f"Index {index_name} introuvable")

    vector = embed_query(q)
    candidates = format_hits(hybrid_search(index_name, vector, q, max(k * 4, 20)))
    ranked = keep_relevant(rerank(q, [c["line"] for c in candidates]), k)
    message = None if ranked else f"Aucun résultat ne dépasse le seuil de pertinence ({RERANK_THRESHOLD})."
    return {"query": q, "cluster_index": index_name, "results": ranked, "message": message}