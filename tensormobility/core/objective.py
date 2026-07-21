from __future__ import annotations

import numpy as np

from tensormobility.core.instance import STBInstance


class STBObjective:
    """Convex Beckmann-plus-behavior objective for a finite STB column pool."""

    def __init__(self, instance: STBInstance):
        self.instance = instance

    def resource_flow(self, f: np.ndarray) -> np.ndarray:
        return self.instance.A @ np.asarray(f, dtype=float)

    def resource_time(self, f: np.ndarray) -> np.ndarray:
        inst = self.instance
        v = self.resource_flow(f)
        ratio = np.maximum(v, 0.0) / inst.capacity
        return inst.free_flow_time * (1.0 + inst.bpr_alpha * ratio ** inst.bpr_beta)

    def value(self, f: np.ndarray) -> float:
        inst = self.instance
        f = np.asarray(f, dtype=float)
        v = self.resource_flow(f)
        congestion_integral = inst.free_flow_time * (
            v
            + inst.bpr_alpha
            * np.power(np.maximum(v, 0.0), inst.bpr_beta + 1.0)
            / ((inst.bpr_beta + 1.0) * np.power(inst.capacity, inst.bpr_beta))
        )
        return float(np.sum(congestion_integral) + inst.behavior_cost @ f)

    def gradient(self, f: np.ndarray) -> np.ndarray:
        return self.instance.A.T @ self.resource_time(f) + self.instance.behavior_cost

    def value_entropy(self, f: np.ndarray, temperature: float) -> float:
        value = self.value(f)
        eps = 1e-15
        for demand, idx in zip(self.instance.demands, self.instance.group_columns, strict=True):
            p = np.maximum(f[idx] / max(demand, eps), eps)
            value += temperature * demand * float(np.sum(p * np.log(p)))
        return value

    def gradient_entropy(self, f: np.ndarray, temperature: float) -> np.ndarray:
        grad = self.gradient(f)
        eps = 1e-15
        for demand, idx in zip(self.instance.demands, self.instance.group_columns, strict=True):
            p = np.maximum(f[idx] / max(demand, eps), eps)
            grad[idx] += temperature * (np.log(p) + 1.0)
        return grad
