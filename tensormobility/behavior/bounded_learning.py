from __future__ import annotations

"""Bounded, flow-independent behavioral residual recovery.

The learned term changes alternative-specific utility but is fixed during the
inner static assignment.  Therefore it does not alter the monotonicity of the
physical BPR cost map.  This is the deliberately narrow AI component used by
the single-paper prototype.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from tensormobility.core.network_core import StaticNetwork


@dataclass(frozen=True)
class BoundedLearningResult:
    metrics: pd.DataFrame
    prediction_table: pd.DataFrame
    history: pd.DataFrame
    bound: float
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


class BoundedResidualMLP(nn.Module):
    def __init__(self, n_features: int, bound: float) -> None:
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(self.net(x).squeeze(-1))


def _turn_count(network: StaticNetwork, node_sequence_text: str) -> int:
    nodes = [int(v) for v in str(node_sequence_text).split(";") if v]
    if len(nodes) < 3 or not {"x_coord", "y_coord"}.issubset(network.nodes.columns):
        return 0
    xy = network.nodes.set_index("node_id")[["x_coord", "y_coord"]]
    turns = 0
    previous = None
    for u, v in zip(nodes[:-1], nodes[1:], strict=True):
        delta = (
            np.sign(float(xy.loc[v, "x_coord"] - xy.loc[u, "x_coord"])),
            np.sign(float(xy.loc[v, "y_coord"] - xy.loc[u, "y_coord"])),
        )
        if previous is not None and delta != previous:
            turns += 1
        previous = delta
    return turns


def build_path_features(network: StaticNetwork, path_table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    table = path_table.copy().reset_index(drop=True)
    turns = np.asarray([_turn_count(network, v) for v in table["node_sequence"]], dtype=float)
    features = np.column_stack([
        table["free_flow_path_time"].to_numpy(float),
        table["n_links"].to_numpy(float),
        turns,
        table["path_cost"].to_numpy(float),
    ])
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (features - mean) / std
    feature_table = pd.DataFrame(normalized, columns=["z_free_flow", "z_links", "z_turns", "z_cost"])
    return feature_table, normalized


def _group_probabilities(cost: np.ndarray, residual: np.ndarray, indices: np.ndarray, temperature: float) -> np.ndarray:
    utility = -(cost[indices] + residual[indices]) / temperature
    utility -= utility.max()
    expv = np.exp(np.clip(utility, -700.0, 0.0))
    return expv / expv.sum()


def train_bounded_behavioral_residual(
    network: StaticNetwork,
    path_table: pd.DataFrame,
    *,
    bound: float = 0.8,
    temperature: float = 1.5,
    epochs: int = 350,
    learning_rate: float = 0.02,
    seed: int = 11,
) -> BoundedLearningResult:
    if bound <= 0 or temperature <= 0:
        raise ValueError("bound and temperature must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    table = path_table.copy().reset_index(drop=True)
    feature_table, features = build_path_features(network, table)
    cost = table["path_cost"].to_numpy(float)

    # Nonlinear but bounded synthetic truth.  It is independent of current
    # flow and queue, so it acts as an alternative-specific behavioral term.
    raw = 0.65 * features[:, 1] - 0.55 * features[:, 2] + 0.35 * features[:, 0] * features[:, 2]
    true_residual = bound * np.tanh(raw)
    groups = sorted(table["od_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    split = max(1, int(round(0.70 * len(shuffled))))
    train_groups = tuple(sorted(shuffled[:split]))
    test_groups = tuple(sorted(shuffled[split:]))
    if not test_groups:
        test_groups = (train_groups[-1],)
        train_groups = train_groups[:-1]

    group_indices = {
        group: np.flatnonzero(table["od_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    target_probability = np.zeros(len(table), dtype=float)
    baseline_probability = np.zeros(len(table), dtype=float)
    for group, idx in group_indices.items():
        target_probability[idx] = _group_probabilities(cost, true_residual, idx, temperature)
        baseline_probability[idx] = _group_probabilities(cost, np.zeros_like(true_residual), idx, temperature)

    x = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target_probability, dtype=torch.float32)
    cost_tensor = torch.tensor(cost, dtype=torch.float32)
    model = BoundedResidualMLP(features.shape[1], bound)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        residual = model(x)
        losses = []
        for group in train_groups:
            idx_np = group_indices[group]
            idx = torch.tensor(idx_np, dtype=torch.long)
            logits = -(cost_tensor[idx] + residual[idx]) / temperature
            log_probability = torch.log_softmax(logits, dim=0)
            losses.append(-(target[idx] * log_probability).sum())
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history_rows.append({"epoch": epoch, "cross_entropy": float(loss.detach().cpu())})

    with torch.no_grad():
        learned_residual = model(x).cpu().numpy()
    learned_probability = np.zeros(len(table), dtype=float)
    for group, idx in group_indices.items():
        learned_probability[idx] = _group_probabilities(cost, learned_residual, idx, temperature)

    prediction = pd.concat([table, feature_table], axis=1)
    prediction["true_residual"] = true_residual
    prediction["learned_residual"] = learned_residual
    prediction["target_probability"] = target_probability
    prediction["baseline_probability"] = baseline_probability
    prediction["learned_probability"] = learned_probability
    prediction["split"] = np.where(prediction["od_id"].astype(str).isin(train_groups), "train", "test")

    metrics_rows: list[dict[str, float | str]] = []
    eps = 1e-12
    for split_name, split_groups in (("train", train_groups), ("test", test_groups)):
        mask = prediction["od_id"].astype(str).isin(split_groups).to_numpy()
        target_p = target_probability[mask]
        baseline_p = baseline_probability[mask]
        learned_p = learned_probability[mask]
        metrics_rows.append({
            "split": split_name,
            "baseline_probability_mae": float(np.mean(np.abs(baseline_p - target_p))),
            "learned_probability_mae": float(np.mean(np.abs(learned_p - target_p))),
            "baseline_kl": float(np.mean(target_p * np.log((target_p + eps) / (baseline_p + eps)))),
            "learned_kl": float(np.mean(target_p * np.log((target_p + eps) / (learned_p + eps)))),
            "residual_mae": float(np.mean(np.abs(learned_residual[mask] - true_residual[mask]))),
            "max_abs_learned_residual": float(np.max(np.abs(learned_residual[mask]))),
        })
    return BoundedLearningResult(
        metrics=pd.DataFrame(metrics_rows),
        prediction_table=prediction,
        history=pd.DataFrame(history_rows),
        bound=float(bound),
        train_groups=train_groups,
        test_groups=test_groups,
    )
