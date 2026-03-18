# Identity-Narrative Model

## Idea

This project explores a simple hypothesis:

> Identity is not a cause of behavior — it is a compression of behavior.

We simulate agents where:

- The body defines constraints
- Behavior emerges from cost minimization
- Narrative is constructed after the fact

---

## Model

Pipeline:

Body → Behavior → Narrative

- Body = hidden state (damage, stress, tension, breath)
- Behavior = optimization under constraints
- Narrative = explanation built from sequence

---

## Experiments

### 1. Compression

Narrative compresses high-dimensional body states.

Result:
→ Information loss (compression gap)

---

### 2. Shock

We remove the dominant activity.

Question:
→ Does identity change?

Result:
→ Often identity stays, explanation changes

---

### 3. Identity Inertia

We add psychological cost of changing identity.

Result:

- Low alpha → flexible
- Medium → tension
- High → maladaptive

→ Identity behaves like inertia

---

## Run

```bash
python -m compression.run_compression
python -m shock.run_shock
python -m inertia.run_inertia
