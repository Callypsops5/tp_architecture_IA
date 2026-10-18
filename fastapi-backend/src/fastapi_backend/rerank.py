
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

RERANK_MODEL = "qwen3.5:0.8b"
REQUEST_TIMEOUT = 120

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["yes", "no"],
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _normalize(token: str) -> str:
    """Normalise une réponse ou un token yes/no."""
    return token.strip().strip('"').strip("'").lower()


def _yes_no_probabilities(data: dict) -> tuple[float, float]:
    """
    Récupère les probabilités yes/no à la position où
    le modèle génère directement sa décision.
    """
    for entry in data.get("logprobs") or []:
        generated = _normalize(str(entry.get("token", "")))

        if generated not in ("yes", "no"):
            continue

        alternatives = list(entry.get("top_logprobs") or [])

        if not any(
            _normalize(str(alt.get("token", ""))) == generated
            for alt in alternatives
        ):
            alternatives.append({
                "token": entry.get("token", ""),
                "logprob": entry.get("logprob"),
            })

        p_yes = 0.0
        p_no = 0.0

        for alt in alternatives:
            token = _normalize(str(alt.get("token", "")))
            logprob = alt.get("logprob")

            if token not in ("yes", "no") or logprob is None:
                continue

            try:
                probability = math.exp(float(logprob))
            except (TypeError, ValueError, OverflowError):
                continue

            if token == "yes":
                p_yes += probability
            else:
                p_no += probability

        return p_yes, p_no

    return 0.0, 0.0


def _build_prompt(query: str, line: str) -> str:
    """Construit le prompt de classification JSON."""
    return (
        "Tu es un système de classement de marchés publics.\n"
        "Compare le sujet précis de la requête avec l'intitulé.\n"
        "La requête peut être courte ou abrégée ; l'intitulé peut "
        "être plus détaillé ou formulé différemment.\n"
        "Réponds yes si les deux concernent réellement le même "
        "domaine et le même objet général.\n"
        "Réponds no si les objets sont différents.\n"
        "Un mot générique commun comme 'travaux' ou 'fourniture' "
        "ne suffit pas à établir une correspondance.\n"
        'Retourne uniquement {"answer": "yes"} '
        'ou {"answer": "no"}.\n\n'
        f"Requête : {query}\n"
        f"Intitulé : {line}\n"
    )


def _build_score_prompt(query: str, line: str) -> str:
    """Demande une réponse directe yes/no pour obtenir les logprobs."""
    return (
        "Tu es un système de classement de marchés publics.\n"
        "Compare le sujet précis de la requête avec l'intitulé.\n"
        "La requête peut être courte ou abrégée ; l'intitulé peut "
        "être plus détaillé ou formulé différemment.\n"
        "Réponds yes si les deux concernent réellement le même "
        "domaine et le même objet général.\n"
        "Réponds no si les objets sont différents.\n"
        "Un mot générique commun ne suffit pas à établir "
        "une correspondance.\n"
        "Réponds uniquement yes ou no, sans explication.\n\n"
        f"Requête : {query}\n"
        f"Intitulé : {line}\n"
        "Réponse :"
    )


def _generate(
    prompt: str,
    *,
    json_output: bool = False,
    with_logprobs: bool = False,
) -> dict:
    """Envoie une requête à Ollama."""
    payload = {
        "model": RERANK_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 20 if json_output else 1,
        },
    }

    if json_output:
        payload["format"] = ANSWER_SCHEMA

    if with_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    return response.json()


def _score_line(query: str, line: str) -> dict:
    """
    Effectue une classification JSON puis un scoring direct.
    La décision opérationnelle vient de la génération yes/no.
    """
    # 1. Classification JSON contrainte.
    try:
        json_data = _generate(
            _build_prompt(query, line),
            json_output=True,
        )

        parsed = json.loads(json_data.get("response", ""))

        if not isinstance(parsed, dict):
            raise ValueError("Un objet JSON était attendu.")

        json_answer = _normalize(str(parsed.get("answer", "")))

        if json_answer not in ("yes", "no"):
            raise ValueError(
                f"Réponse JSON inattendue : {json_answer!r}"
            )

    except Exception as exc:
        logger.warning(
            "Échec de classification pour %r : %s",
            line,
            exc,
        )
        return {
        "line": line,
        "score": 0.0,
        "percentage": 0.0,
        "answer": json_answer,
        "score_error": "score_unavailable",
    }

    # 2. Génération directe yes/no avec probabilités.
    direct_answer = None

    try:
        score_data = _generate(
            _build_score_prompt(query, line),
            with_logprobs=True,
        )

        direct_answer = _normalize(
            str(score_data.get("response", ""))
        )

        if direct_answer not in ("yes", "no"):
            raise ValueError(
                f"Réponse directe inattendue : {direct_answer!r}"
            )

        p_yes, p_no = _yes_no_probabilities(score_data)

        if p_yes <= 0 or p_no <= 0:
            raise ValueError(
                "Probabilités yes/no absentes ou incomplètes."
            )

        score = p_yes / (p_yes + p_no)

        if direct_answer != json_answer:
            logger.warning(
                "Désaccord pour %r : JSON=%s, décision scoring=%s",
                line,
                json_answer,
                direct_answer,
            )

        return {
            "line": line,
            "score": score,
            "percentage": round(score * 100, 2),
            "answer": json_answer,
            }

    except Exception as exc:
        logger.warning(
            "Score indisponible pour %r : %s",
            line,
            exc,
        )

        return {
    "line": line,
    "score": score,
    "percentage": round(score * 100, 2),
    "answer": json_answer,
}


def rerank(
    query: str,
    lines: list[str],
    max_workers: int = 4,
    **_: object,
) -> list[dict]:
    """
    Évalue tous les candidats et les trie par score décroissant.
    Le filtrage de pertinence reste dans main.py.
    """
    if not lines:
        return []

    workers = max(1, min(max_workers, len(lines)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = list(
            executor.map(
                lambda line: _score_line(query, line),
                lines,
            )
        )

    return sorted(
        scored,
        key=lambda item: item["score"],
        reverse=True,
    )
