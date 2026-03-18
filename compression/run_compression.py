from core.model import BodyState, generate_sequence
import os

import numpy as np
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

# ── NARRACJA ─────────────────────────────────────────

SYSTEM_PROMPT = """Obserwujesz wzorzec aktywności fizycznej pewnej osoby.
Na podstawie TYLKO tych decyzji stwórz narrację tożsamościową.

NIE odwołuj się do bólu ani ograniczeń fizycznych.

JSON:
{
  "narracja": "2-3 zdania",
  "wartosc": "jedno słowo",
  "tozsamosc": "jestem osobą która..."
}
"""

def generate_narrative(sequence):
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # fallback jeśli brak API
    if not api_key:
        dominant = max(set(sequence), key=sequence.count)
        return {
            "narracja": f"Lubię {dominant}.",
            "wartosc": "ruch",
            "tozsamosc": f"jestem osobą która wybiera {dominant}"
        }

    seq_str = " → ".join(sequence)

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": seq_str}],
        },
    )

    text = resp.json()["content"][0]["text"]
    if "{" in text:
        text = text[text.find("{"):text.rfind("}")+1]
    return json.loads(text)


# ── METRYKI ─────────────────────────────────────────

def build_matrix(narratives):
    texts = [n["narracja"] + " " + n["tozsamosc"] for n in narratives]
    X = TfidfVectorizer(max_features=100).fit_transform(texts).toarray()
    if X.shape[0] > 2:
        X = PCA(n_components=min(5, X.shape[1])).fit_transform(X)
    return X


def compression_gap(bodies, narratives):
    body = np.array([b.vector() for b in bodies])
    narr = build_matrix(narratives)

    return np.mean(np.var(body, axis=0)) - np.mean(np.var(narr, axis=0))


def identifiability(bodies, narratives):
    X = build_matrix(narratives)
    Y = np.array([b.vector() for b in bodies])

    scores = []
    for i in range(Y.shape[1]):
        model = Ridge().fit(X, Y[:, i])
        pred = model.predict(X)
        ss_res = np.sum((Y[:, i] - pred)**2)
        ss_tot = np.sum((Y[:, i] - np.mean(Y[:, i]))**2)
        scores.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)

    return np.mean(scores)


# ── SYMULACJA ─────────────────────────────────────────

def run_simulation(n_agents=10, length=8):
    bodies, narratives = [], []

    print("\nMODEL: ciało → zachowanie → narracja\n")

    for i in range(n_agents):
        body = BodyState.random()
        seq = generate_sequence(body, length)

        narr = generate_narrative(seq)

        print(f"Agent {i+1}")
        print("  seq:", seq)
        print("  narr:", narr["narracja"])
        print()

        bodies.append(body)
        narratives.append(narr)

    print("Compression gap:", round(compression_gap(bodies, narratives), 4))
    print("Identifiability:", round(identifiability(bodies, narratives), 4))


# ── RUN ─────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()