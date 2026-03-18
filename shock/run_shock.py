from core.model import BodyState, generate_sequence

import os
import numpy as np
import json
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── NARRACJA ─────────────────────────────────────────

SYSTEM_PROMPT = """Na podstawie sekwencji aktywności opisz kim jestem.

NIE wspominaj o bólu ani ograniczeniach.

JSON:
{
  "narracja": "2-3 zdania",
  "wartosc": "jedno słowo",
  "tozsamosc": "jestem osobą która..."
}
"""

def generate_narrative(sequence):
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # fallback
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


# ── METRYKA: DRYF ───────────────────────────────────

def narrative_drift(n1, n2):
    t1 = n1["narracja"] + " " + n1["tozsamosc"]
    t2 = n2["narracja"] + " " + n2["tozsamosc"]

    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform([t1, t2])
    sim = cosine_similarity(X[0], X[1])[0][0]

    return round(1 - sim, 3)


# ── TEST ───────────────────────────────────────────

def run_shock(n_agents=10, length=6):
    print("\nTEST SZOKU: blokada aktywności\n")

    drifts = []

    for i in range(n_agents):
        body = BodyState.random()

        # faza 1
        seq1 = generate_sequence(body, length)
        dominant = max(set(seq1), key=seq1.count)

        # faza 2 (blokada)
        seq2 = generate_sequence(body, length, blocked=dominant)

        narr1 = generate_narrative(seq1)
        time.sleep(0.3)
        narr2 = generate_narrative(seq2)

        drift = narrative_drift(narr1, narr2)
        drifts.append(drift)

        print(f"\nAgent {i+1}")
        print("  before:", seq1)
        print("  after :", seq2)
        print("  drift :", drift)

    print("\nŚredni dryf:", round(np.mean(drifts), 3))


# ── RUN ─────────────────────────────────────────

if __name__ == "__main__":
    run_shock()