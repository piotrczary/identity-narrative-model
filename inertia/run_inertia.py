from core.model import BodyState

import numpy as np
import os

np.random.seed(42)

# ── PROTOTYPY RUCHU ─────────────────────────────────────────

SPORT_PROTOTYPES = {
    "bieganie":    np.array([0.9, 0.8, 0.2, 0.6, 0.2]),
    "pływanie":    np.array([0.5, 0.1, 0.4, 0.9, 0.9]),
    "siłownia":    np.array([0.6, 0.7, 0.9, 0.3, 0.2]),
    "joga":        np.array([0.2, 0.1, 0.3, 0.8, 0.8]),
    "piłka nożna": np.array([0.9, 0.9, 0.4, 0.5, 0.4]),
}


# ── KOSZT ─────────────────────────────────────────

def cost_body(body, movement):
    return np.dot(body.cost_weights(), movement**2)


def cost_total(body, movement, identity_proto, alpha):
    return cost_body(body, movement) + alpha * np.linalg.norm(movement - identity_proto)


# ── WYBÓR RUCHU ───────────────────────────────────

def choose_movement(body, identity_proto=None, alpha=0.0, n=300):
    candidates = np.random.uniform(0, 1, (n, 5))

    if identity_proto is not None and alpha > 0:
        costs = [cost_total(body, c, identity_proto, alpha) for c in candidates]
    else:
        costs = [cost_body(body, c) for c in candidates]

    return candidates[np.argmin(costs)]


def closest_sport(m):
    return min(SPORT_PROTOTYPES,
               key=lambda s: np.linalg.norm(m - SPORT_PROTOTYPES[s]))


# ── FAZA ─────────────────────────────────────────

def run_phase(body, length, identity_proto=None, alpha=0.0):
    sports = []
    costs  = []

    for _ in range(length):
        m = choose_movement(body, identity_proto, alpha)
        sports.append(closest_sport(m))
        costs.append(cost_body(body, m))

    return sports, np.mean(costs)


# ── EKSPERYMENT ───────────────────────────────────

def run_experiment(n_agents=10, length=6):
    alphas = [0.0, 0.3, 0.6, 1.0]

    print("\nINERCJA TOŻSAMOŚCI\n")

    # stałe ciała
    bodies_h = [BodyState.random() for _ in range(n_agents)]
    bodies_i = [b.injured("hip") for b in bodies_h]

    identities = []

    # faza 1
    for b in bodies_h:
        sports, _ = run_phase(b, length)
        identities.append(max(set(sports), key=sports.count))

    # wyniki
    results = {a: [] for a in alphas}

    for b, identity in zip(bodies_i, identities):
        proto = SPORT_PROTOTYPES[identity]

        for alpha in alphas:
            sports, cost = run_phase(
                b,
                length,
                identity_proto=proto if alpha > 0 else None,
                alpha=alpha
            )
            results[alpha].append(cost)

    # print
    base = np.mean(results[0.0])

    print("alpha | cost | delta")
    print("----------------------")

    for a in alphas:
        mean = np.mean(results[a])
        delta = mean - base

        marker = ""
        if a > 0:
            if delta > 0.05:
                marker = " ← SZKODZI"
            elif delta > 0.01:
                marker = " ← lekki efekt"

        print(f"{a:<5} {mean:.3f} {delta:+.3f}{marker}")

    # próg
    critical = next((a for a in alphas[1:] if np.mean(results[a]) - base > 0.02), None)

    print("\nWniosek:")
    if critical:
        print(f"Próg ≈ {critical} → tożsamość zaczyna szkodzić")
    else:
        print("Brak wyraźnego progu")


# ── RUN ─────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()