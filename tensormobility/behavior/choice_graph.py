"""The Choice Graph: the behavior-axis core of TensorMobility.

A layered DAG whose vertices are (stage, state) choice nodes and whose
complete source->sink chains ARE the behavioral columns of the STB
tensor (= MoE expert paths, = activity-travel patterns). Utilities live
on arcs; chain choice is multinomial logit over complete chains.

Founding identity (tested): the backward soft-Bellman (logsum)
recursion on the DAG -- Dial's STOCH / recursive-logit computation --
yields EXACTLY the flat logit over enumerated chains when utilities are
arc-additive. So the choice graph is simultaneously a sequential
(Markovian, forward/backward) model and a column (flat) model; which
face you use is a computational choice, not a behavioral one.

Plugs directly into the master fixed point x = B_theta(L(x)): the
choice graph is B_theta; any loading operator (BPR, DNL, kernel import)
is L. Cross-mode competition vs cooperation is then an EQUILIBRIUM
cross-response sign, not a fixed-supply logit derivative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChoiceGraph:
    source: str
    sink: str
    nodes: dict = field(default_factory=dict)     # name -> stage label
    arcs: dict = field(default_factory=dict)      # (u,v) -> base utility
    _succ: dict = field(default_factory=dict)

    def add_node(self, name: str, stage: str = ''):
        self.nodes[name] = stage
        self._succ.setdefault(name, [])
        return self

    def add_arc(self, u: str, v: str, utility: float = 0.0):
        for n in (u, v):
            if n not in self.nodes:
                self.add_node(n)
        self.arcs[(u, v)] = float(utility)
        self._succ[u].append(v)
        return self

    # ---- columns: complete chains --------------------------------
    def chains(self) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []

        def walk(node, path):
            if node == self.sink:
                out.append(tuple(path))
                return
            for v in self._succ.get(node, []):
                walk(v, path + [v])
        walk(self.source, [self.source])
        return out

    def chain_utility(self, chain, arc_utils: dict | None = None
                      ) -> float:
        u = arc_utils or self.arcs
        return float(sum(u[(chain[i], chain[i + 1])]
                         for i in range(len(chain) - 1)))

    # ---- flat (column) face --------------------------------------
    def logit_chain_shares(self, theta: float,
                           arc_utils: dict | None = None) -> np.ndarray:
        us = np.array([self.chain_utility(c, arc_utils)
                       for c in self.chains()])
        z = np.exp(theta * (us - us.max()))
        return z / z.sum()

    # ---- sequential (Markov / recursive-logit) face --------------
    def logsum_values(self, theta: float,
                      arc_utils: dict | None = None) -> dict:
        """Backward soft-Bellman: V(sink)=0,
        V(u) = (1/theta) ln sum_v exp(theta (u_uv + V(v)))."""
        u = arc_utils or self.arcs
        order = self._topo()
        V = {self.sink: 0.0}
        for node in reversed(order):
            if node == self.sink:
                continue
            vals = [theta * (u[(node, v)] + V[v])
                    for v in self._succ.get(node, [])]
            if vals:
                m = max(vals)
                V[node] = (m + np.log(np.sum(np.exp(np.array(vals) - m)))
                           ) / theta
        return V

    def sequential_chain_shares(self, theta: float,
                                arc_utils: dict | None = None
                                ) -> np.ndarray:
        """Chain probability as the PRODUCT of per-node transition
        probabilities p(v|u) = exp(theta(u_uv + V(v) - V(u)))."""
        u = arc_utils or self.arcs
        V = self.logsum_values(theta, u)
        out = []
        for chain in self.chains():
            p = 1.0
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                p *= float(np.exp(theta * (u[(a, b)] + V[b] - V[a])))
            out.append(p)
        return np.asarray(out)

    def _topo(self) -> list[str]:
        seen, order = set(), []

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for v in self._succ.get(n, []):
                visit(v)
            order.append(n)
        visit(self.source)
        return order[::-1]

    # ---- behavior operator for the master fixed point ------------
    def behavior_operator(self, theta: float,
                          cost_to_utils):
        """Returns B(costs) -> chain shares, where cost_to_utils maps
        the loading operator's experienced costs to arc utilities
        (flow-independent theta: clean-VI side)."""
        def B(costs):
            return self.logit_chain_shares(theta, cost_to_utils(costs))
        return B


def park_and_ride_example() -> ChoiceGraph:
    """The review's illustrative multimodal network: A -> {bus station,
    park&ride, taxi stand} -> B, four complete chains."""
    g = ChoiceGraph('A', 'B')
    g.add_node('A', 'origin').add_node('B', 'destination')
    g.add_node('BusStation', 'access').add_node('ParkRide', 'access')
    g.add_node('TaxiStand', 'access')
    g.add_arc('A', 'BusStation', -10)     # walk
    g.add_arc('A', 'ParkRide', -5)        # drive
    g.add_arc('A', 'TaxiStand', -15)      # walk
    g.add_arc('ParkRide', 'BusStation', -15)  # transfer walk
    g.add_arc('BusStation', 'B', -20)     # bus
    g.add_arc('TaxiStand', 'B', -25)      # taxi
    g.add_arc('A', 'B', -38)              # drive-only chain
    return g
