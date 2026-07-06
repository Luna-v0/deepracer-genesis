"""PPO-Lagrangian pieces (plan section 4).

The lambda controller is Stooke et al.'s PID-Lagrangian (ICML 2020): a PID
loop on the constraint violation (J_cost - budget) instead of naive dual
ascent, which oscillates. The combined surrogate uses
A = (A_reward - lambda * A_cost) / (1 + lambda) — the 1/(1+lambda) scaling
keeps the effective advantage magnitude stable as lambda grows.

The PPO delta itself lives in the Trainer's lagrangian branch: a second
(cost) critic + second GAE writing cost_advantage, this combined advantage
written into the "advantage" key ClipPPOLoss reads, and a separate smooth-L1
value loss for the cost critic.
"""

from __future__ import annotations


class PIDLagrangian:
    """One scalar lambda >= 0, PID-updated once per PPO iteration."""

    def __init__(self, budget: float, kp: float = 0.05, ki: float = 0.0005,
                 kd: float = 0.1, lambda_init: float = 0.0):
        self.budget = float(budget)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = max(0.0, lambda_init / ki) if ki > 0 else 0.0
        self.prev_error = 0.0
        self.value = max(0.0, lambda_init)

    def update(self, j_cost: float) -> float:
        """Feed the current mean episode cost estimate; returns lambda."""
        error = float(j_cost) - self.budget
        self.integral = max(0.0, self.integral + error)
        derivative = max(0.0, error - self.prev_error)
        self.prev_error = error
        self.value = max(0.0, self.kp * error + self.ki * self.integral
                         + self.kd * derivative)
        return self.value

    def state_dict(self) -> dict:
        return {"integral": self.integral, "prev_error": self.prev_error,
                "value": self.value}

    def load_state_dict(self, state: dict):
        self.integral = state["integral"]
        self.prev_error = state["prev_error"]
        self.value = state["value"]
