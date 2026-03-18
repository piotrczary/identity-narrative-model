import numpy as np
from dataclasses import dataclass

# ── CIAŁO ─────────────────────────────────────────

@dataclass
class BodyState:
    hip_damage: float
    stress: float
    shoulder_tension: float
    breath_capacity: float

    @classmethod
    def random(cls):
        return cls(
            hip_damage=np.random.beta(2, 3),
            stress=np.random.beta(2, 2),
            shoulder_tension=np.random.beta(2, 3),
            breath_capacity=np.random.beta(3, 2),
        )

    def vector(self):
        return np.array([
            self.hip_damage,
            self.stress,
            self.shoulder_tension,
            1 - self.breath_capacity,
        ])

    def true_driver(self):
        return ["biodro", "stres", "barki", "oddech"][
            np.argmax(self.vector())
        ]


# ── KOSZT ─────────────────────────────────────────

def compute_costs(body: BodyState):
    bc = 1 - body.breath_capacity
    return {
        "piłka nożna": 2.2 * body.hip_damage**2 + 0.5 * np.sin(body.stress * np.pi) + 0.3 * body.shoulder_tension,
        "pływanie":    0.2 * body.hip_damage + 0.4 * body.stress + 0.3 * body.shoulder_tension + 0.8 * bc**2,
        "siłownia":    0.6 * body.hip_damage + 0.6 * body.stress + 0.8 * body.shoulder_tension**2,
        "joga":        0.5 * body.hip_damage + 0.4 * body.stress + 0.3 * body.shoulder_tension + 0.4 * bc,
        "bieganie":    1.1 * body.hip_damage + 0.4 * body.stress + 0.1 * body.shoulder_tension,
    }


def choose_activity(body: BodyState, noise=0.2, blocked=None):
    costs = compute_costs(body)

    if blocked and blocked in costs:
        costs[blocked] = 99.0

    noisy = {k: v + np.random.uniform(0, noise) for k, v in costs.items()}
    return min(noisy, key=noisy.get)


def generate_sequence(body: BodyState, length=8, noise=0.2, blocked=None):
    return [choose_activity(body, noise, blocked) for _ in range(length)]