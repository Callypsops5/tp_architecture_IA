import requests
import json
import numpy as np

API_URL = "https://apilux3.pcloud.com/getpublinkdownload?fileid=93128819642&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k&linkpassword=&forcedownload=1"

VECTOR_DIMS = 1024


def create_index(index_name: str) -> None:
    """Supprime l'index s'il existe, puis le recrée avec un champ embedding de type dense_vector."""
    requests.delete(f"{ES_URL}/{index_name}")

    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine",
                }
            }
        }
    }
    response = requests.put(f"{ES_URL}/{index_name}", json=mapping)
    response.raise_for_status()


def get_data_from_cloud() -> list[str]:
    """Télécharge le fichier de lignes depuis pCloud et renvoie la liste des lignes non vides."""
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()

    download_url = f"https://{data['hosts'][0]}{data['path']}"

    r = requests.get(download_url)
    r.raise_for_status()

    lines = [line.strip() for line in r.content.decode("utf-8").splitlines() if line.strip()]
    print(f"Nombre de lignes lues : {len(lines)}")
    return lines



OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def get_data_embeddings(lines: list[str], batch_size: int = 64) -> list[list[float]]:
    """Calcule l'embedding de chaque ligne via le conteneur Ollama."""
    embeddings: list[list[float]] = []

    for start in range(0, len(lines), batch_size):
        batch = lines[start:start + batch_size]
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        )
        response.raise_for_status()
        embeddings.extend(response.json()["embeddings"])
        print(f"{len(embeddings)}/{len(lines)} lignes vectorisées")

    return embeddings

from sklearn.cluster import KMeans


def cluster_data(embeddings, lines: list[str], n_clusters: int = 10) -> dict[int, list[dict]]:
    """Regroupe les lignes par cluster (K-means) et renvoie cluster -> liste de {line, embedding}."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_to_lines: dict[int, list[dict]] = {i: [] for i in range(n_clusters)}
    for label, line, embedding in zip(cluster_labels, lines, embeddings):
        cluster_to_lines[int(label)].append({
            "line": line,
            "embedding": [float(x) for x in embedding],
        })

    for cluster_id, items in cluster_to_lines.items():
        print(f"Cluster {cluster_id} : {len(items)} lignes")

    return cluster_to_lines

ES_URL = "http://localhost:9200"


def index_cluster_lines(cluster_id: int, items: list[dict]) -> None:
    """Indexe les lignes d'un cluster dans l'index data_<n> (n = cluster_id + 1)."""
    index_name = f"data_{cluster_id + 1}"

    # Repartir d'un index vide pour éviter les doublons si on relance le script
    create_index(index_name)

    # Format bulk d'Elasticsearch : une ligne d'action, puis une ligne de document
    body_lines = []
    for item in items:
        body_lines.append(json.dumps({"index": {}}))
        body_lines.append(json.dumps({"embedding": item["embedding"], "line": item["line"]}))
    body = "\n".join(body_lines) + "\n"

    response = requests.post(
        f"{ES_URL}/{index_name}/_bulk",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    response.raise_for_status()
    if response.json().get("errors"):
        print(f"Des erreurs sont survenues lors de l'indexation de {index_name}")
    else:
        print(f"{len(items)} lignes indexées dans {index_name}")


def index_clusters(cluster_to_lines: dict[int, list[dict]]) -> None:
    """Indexe le centroïde de chaque cluster dans l'index data_clusters."""
    index_name = "data_clusters"

    # Repartir d'un index vide pour éviter les doublons si on relance le script
    create_index(index_name)

    body_lines = []
    for cluster_id, items in cluster_to_lines.items():
        if not items:
            continue
        vectors = np.array([item["embedding"] for item in items])
        centroid = vectors.mean(axis=0).tolist()

        body_lines.append(json.dumps({"index": {}}))
        body_lines.append(json.dumps({
            "embedding": centroid,
            "data_index": f"data_{cluster_id + 1}",
        }))
    body = "\n".join(body_lines) + "\n"

    response = requests.post(
        f"{ES_URL}/{index_name}/_bulk",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    response.raise_for_status()
    if response.json().get("errors"):
        print(f"Des erreurs sont survenues lors de l'indexation de {index_name}")
    else:
        print(f"{len(body_lines) // 2} centroïdes indexés dans {index_name}")


if __name__ == "__main__":
    lines = get_data_from_cloud()
    line_embeddings = get_data_embeddings(lines)
    cluster_to_lines = cluster_data(line_embeddings, lines)

    for cluster_id, items in cluster_to_lines.items():
        index_cluster_lines(cluster_id, items)

    index_clusters(cluster_to_lines)
