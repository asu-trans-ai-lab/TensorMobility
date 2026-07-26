from __future__ import annotations

"""The typed grid activity-tour paper case: complete columns, measured
opportunity supply U[i,j,m], and mobility-tensor projections.

One common vector of complete activity-tour-path column flows f
projects to every marginal the paper uses:

    y = E f        activity-tour flow          (person_flow)
    q = R f        tour-leg OD demand          (person_flow)
    x = Delta f    arc flow                    (vehicle_flow)
    X = Gamma f    mobility tensor X[i,j,m]    (state)

against the convex integrated objective

    F(f; U) = a(U)^T f + Phi_N(Delta f) + Phi_M(E f) + Phi_U(Gamma f; U)

whose gradient and Hessian are assembled by
`tensormobility.optimization.quadratic_pricing` -- this module owns the
scenario data (grid, tours, U field, curvatures); the pricing module
owns the joint calculus.

Naming rule (paper section "Measured Land-Use Integration"): U is the
measured land-use / activity-opportunity supply; the symbol L is
reserved for network loading operators elsewhere in the framework.

Every operator is a TypedOperator with explicit target <- source
orientation and measure contract; the mobility tensor is returned as a
named-axis STBTensor over registered extension axes, never as an
anonymous array.  This is the paper's 5x5 analytical case folded into
the certified package -- not a second codebase.
"""

from dataclasses import dataclass
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, minimize
from scipy.stats import spearmanr

from tensormobility.core.axes import (
    CANONICAL_AXES,
    AxisSpec,
    Semiring,
    Status,
    extend_axes,
)
from tensormobility.core.stb_tensor import STBTensor
from tensormobility.core.tensor_contracts import (
    Axis,
    TypedOperator,
    TypedVector,
    one_hot_operator,
    sparse_operator_from_triplets,
)
from tensormobility.optimization.quadratic_pricing import (
    exact_finite_change,
    integrated_gradient,
    integrated_hessian,
    quadratic_price,
    replacement_direction,
)

Node = tuple[int, int]

# Extension axes for the paper slice, registered on top of the canon
# (the registry extends, never mutates).  cell_row / cell_col / activity
# name the mobility tensor X[i,j,m]; tour_stage names the optional
# illustrative X[i,j,m,t].
PAPER_AXES = extend_axes(
    dict(CANONICAL_AXES),
    {
        "cell_row": AxisSpec(
            "cell_row", "grid cell row index i", Status.SPECTATOR,
            Semiring.SUM_PRODUCT, "S_repr",
        ),
        "cell_col": AxisSpec(
            "cell_col", "grid cell column index j", Status.SPECTATOR,
            Semiring.SUM_PRODUCT, "S_repr",
        ),
        "activity": AxisSpec(
            "activity", "activity type H / W / S", Status.SPECTATOR,
            Semiring.SUM_PRODUCT, "S_stat",
        ),
        "tour_stage": AxisSpec(
            "tour_stage",
            "coarse tour stage (illustrative visualization, not DNL)",
            Status.SPECTATOR, Semiring.SUM_PRODUCT, "S_mech",
        ),
    },
)


@dataclass(frozen=True)
class GridCaseConfig:
    rows: int = 5
    columns: int = 5
    k_paths_direct: int = 3
    k_paths_shop: int = 2
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    entropy_weight: float = 1.2
    opportunity_weight: float = 1.2
    epsilon: float = 1.0e-8
    random_seed: int = 42


@dataclass(frozen=True)
class Scenario:
    """Measured opportunity supply and network capacity for one run.

    Supply fields populate U[i,j,m]; capacity fields populate the
    opportunity-utilization curvature D_U; the central multiplier
    scales the capacity of the central-cross arcs only.
    """

    name: str
    work_w1_supply: float = 1000.0
    work_w2_supply: float = 800.0
    shopping_supply: float = 700.0
    work_w1_capacity: float = 650.0
    work_w2_capacity: float = 550.0
    shopping_capacity: float = 550.0
    central_capacity_multiplier: float = 1.0


@dataclass(frozen=True)
class CompleteColumn:
    """One complete realization pi = (class, tour, destinations, legs):
    traveler class, activity-tour alternative, selected destinations,
    and all network path legs -- the paper's behavioral column."""

    column_id: int
    traveler_class: str
    tour_name: str
    tour_type: str
    origin: Node
    work_name: str
    work_node: Node
    shopping_node: Node | None
    legs: tuple[tuple[Node, ...], ...]

    @property
    def visits_shopping(self) -> bool:
        return self.shopping_node is not None

    @property
    def path_length(self) -> int:
        return sum(len(leg) - 1 for leg in self.legs)


@dataclass(frozen=True)
class CaseOperators:
    """The typed projection stack, orientation target <- source with the
    complete-column axis as the common source."""

    column_axis: Axis
    class_axis: Axis
    tour_axis: Axis
    leg_od_axis: Axis
    arc_axis: Axis
    state_axis: Axis
    B: TypedOperator          # class <- column, mass preserving
    E: TypedOperator          # tour alternative <- column, mass preserving
    R: TypedOperator          # tour-leg OD <- column (one hit per leg)
    Delta: TypedOperator      # arc <- column (link usage counts)
    Gamma: TypedOperator      # activity state <- column (one hit per activity)
    t0: np.ndarray
    capacity: np.ndarray


class MobilityTensorCase:
    """Build, solve, price, and export the grid paper case."""

    activity_labels = ("H", "W", "S")

    def __init__(self, config: GridCaseConfig | None = None) -> None:
        self.config = config or GridCaseConfig()
        self.origins: dict[str, Node] = {"A": (2, 1), "B": (4, 1)}
        self.demands: dict[str, float] = {"A": 500.0, "B": 500.0}
        self.work_destinations: dict[str, Node] = {"W1": (2, 5), "W2": (4, 5)}
        self.shopping_node: Node = (3, 3)
        self.graph = self._build_grid()
        self.columns = self._build_columns()
        self.operators = self._build_operators()

    # ------------------------------------------------------------------
    # Network and column construction
    # ------------------------------------------------------------------
    def _build_grid(self) -> nx.DiGraph:
        c = self.config
        graph = nx.DiGraph()
        nodes = [(i, j) for i in range(1, c.rows + 1) for j in range(1, c.columns + 1)]
        graph.add_nodes_from(nodes)
        for i, j in nodes:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (i + di, j + dj)
                if v not in graph:
                    continue
                central = (j == 3 and v[1] == 3) or (i == 3 and v[0] == 3)
                graph.add_edge(
                    (i, j), v,
                    t0=1.0,
                    capacity=350.0 if central else 700.0,
                    weight=1.0,
                    central=central,
                )
        return graph

    @staticmethod
    def _k_shortest_paths(
        graph: nx.DiGraph, source: Node, target: Node, k: int
    ) -> list[tuple[Node, ...]]:
        generator = nx.shortest_simple_paths(graph, source, target, weight="weight")
        paths: list[tuple[Node, ...]] = []
        for _ in range(k):
            try:
                paths.append(tuple(next(generator)))
            except StopIteration:
                break
        return paths

    def _build_columns(self) -> list[CompleteColumn]:
        columns: list[CompleteColumn] = []
        column_id = 0
        for traveler_class, origin in self.origins.items():
            for work_name, work_node in self.work_destinations.items():
                # H -> W -> H
                outward = self._k_shortest_paths(
                    self.graph, origin, work_node, self.config.k_paths_direct
                )
                homeward = self._k_shortest_paths(
                    self.graph, work_node, origin, self.config.k_paths_direct
                )
                for legs in itertools.product(outward, homeward):
                    columns.append(CompleteColumn(
                        column_id=column_id,
                        traveler_class=traveler_class,
                        tour_name=f"HWH-{work_name}",
                        tour_type="HWH",
                        origin=origin,
                        work_name=work_name,
                        work_node=work_node,
                        shopping_node=None,
                        legs=tuple(legs),
                    ))
                    column_id += 1

                # H -> W -> S -> H
                outward = self._k_shortest_paths(
                    self.graph, origin, work_node, self.config.k_paths_shop
                )
                work_shop = self._k_shortest_paths(
                    self.graph, work_node, self.shopping_node, self.config.k_paths_shop
                )
                shop_home = self._k_shortest_paths(
                    self.graph, self.shopping_node, origin, self.config.k_paths_shop
                )
                for legs in itertools.product(outward, work_shop, shop_home):
                    columns.append(CompleteColumn(
                        column_id=column_id,
                        traveler_class=traveler_class,
                        tour_name=f"HWSH-{work_name}",
                        tour_type="HWSH",
                        origin=origin,
                        work_name=work_name,
                        work_node=work_node,
                        shopping_node=self.shopping_node,
                        legs=tuple(legs),
                    ))
                    column_id += 1
        return columns

    def _build_operators(self) -> CaseOperators:
        c = self.config
        edge_list = list(self.graph.edges())
        edge_index = {edge: idx for idx, edge in enumerate(edge_list)}
        classes = sorted(self.origins)
        class_index = {h: idx for idx, h in enumerate(classes)}
        tour_alternatives = sorted(
            {(col.traveler_class, col.tour_name) for col in self.columns}
        )
        tour_index = {key: idx for idx, key in enumerate(tour_alternatives)}
        leg_ods = sorted(
            {(leg[0], leg[-1]) for col in self.columns for leg in col.legs}
        )
        leg_od_index = {od: idx for idx, od in enumerate(leg_ods)}
        tensor_states = [
            (i, j, m)
            for i in range(1, c.rows + 1)
            for j in range(1, c.columns + 1)
            for m in self.activity_labels
        ]
        state_index = {state: idx for idx, state in enumerate(tensor_states)}

        column_axis = Axis(
            "complete_column",
            tuple(
                f"{col.traveler_class}:{col.tour_name}:c{col.column_id}"
                for col in self.columns
            ),
            "complete activity-tour-path columns pi",
        )
        class_axis = Axis("traveler_class", tuple(classes), "traveler classes g")
        tour_axis = Axis(
            "tour_alternative",
            tuple(f"{h}:{tour}" for h, tour in tour_alternatives),
            "class x activity-tour alternatives",
        )
        leg_od_axis = Axis(
            "tour_leg_od",
            tuple(f"{o}->{d}" for o, d in leg_ods),
            "tour-leg origin-destination pairs",
        )
        arc_axis = Axis(
            "grid_arc",
            tuple(f"{u}->{v}" for u, v in edge_list),
            "directed grid arcs",
        )
        state_axis = Axis(
            "activity_state",
            tuple(f"({i},{j},{m})" for i, j, m in tensor_states),
            "grid-activity states (i, j, m)",
        )

        B = one_hot_operator(
            name="B[class<-column]",
            source=column_axis,
            target=class_axis,
            target_index_for_source=[
                class_index[col.traveler_class] for col in self.columns
            ],
            input_measure="person_flow",
            output_measure="person_flow",
            conservation="mass_preserving",
            description="traveler-class demand incidence",
        )
        E = one_hot_operator(
            name="E[tour<-column]",
            source=column_axis,
            target=tour_axis,
            target_index_for_source=[
                tour_index[(col.traveler_class, col.tour_name)]
                for col in self.columns
            ],
            input_measure="person_flow",
            output_measure="person_flow",
            conservation="mass_preserving",
            description="activity-tour incidence y = E f",
        )

        r_rows, r_cols, r_vals = [], [], []
        d_rows, d_cols, d_vals = [], [], []
        g_rows, g_cols, g_vals = [], [], []
        for col in self.columns:
            for leg in col.legs:
                r_rows.append(leg_od_index[(leg[0], leg[-1])])
                r_cols.append(col.column_id)
                r_vals.append(1.0)
                for u, v in zip(leg[:-1], leg[1:]):
                    d_rows.append(edge_index[(u, v)])
                    d_cols.append(col.column_id)
                    d_vals.append(1.0)
            g_rows.append(state_index[(*col.origin, "H")])
            g_cols.append(col.column_id)
            g_vals.append(1.0)
            g_rows.append(state_index[(*col.work_node, "W")])
            g_cols.append(col.column_id)
            g_vals.append(1.0)
            if col.shopping_node is not None:
                g_rows.append(state_index[(*col.shopping_node, "S")])
                g_cols.append(col.column_id)
                g_vals.append(1.0)

        R = sparse_operator_from_triplets(
            name="R[leg_od<-column]",
            source=column_axis,
            target=leg_od_axis,
            rows=r_rows, cols=r_cols, values=r_vals,
            input_measure="person_flow",
            output_measure="person_flow",
            description="tour-leg OD demand q = R f (one hit per leg)",
        )
        Delta = sparse_operator_from_triplets(
            name="Delta[arc<-column]",
            source=column_axis,
            target=arc_axis,
            rows=d_rows, cols=d_cols, values=d_vals,
            input_measure="person_flow",
            output_measure="vehicle_flow",
            description="arc-column incidence x = Delta f (usage counts)",
        )
        Gamma = sparse_operator_from_triplets(
            name="Gamma[state<-column]",
            source=column_axis,
            target=state_axis,
            rows=g_rows, cols=g_cols, values=g_vals,
            input_measure="person_flow",
            output_measure="state",
            description="grid-activity-state incidence X = Gamma f",
        )

        t0 = np.array([self.graph.edges[e]["t0"] for e in edge_list], dtype=float)
        capacity = np.array(
            [self.graph.edges[e]["capacity"] for e in edge_list], dtype=float
        )
        return CaseOperators(
            column_axis=column_axis,
            class_axis=class_axis,
            tour_axis=tour_axis,
            leg_od_axis=leg_od_axis,
            arc_axis=arc_axis,
            state_axis=state_axis,
            B=B, E=E, R=R, Delta=Delta, Gamma=Gamma,
            t0=t0, capacity=capacity,
        )

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def tensor_states(self) -> list[tuple[int, int, str]]:
        c = self.config
        return [
            (i, j, m)
            for i in range(1, c.rows + 1)
            for j in range(1, c.columns + 1)
            for m in self.activity_labels
        ]

    def class_columns(self, traveler_class: str) -> np.ndarray:
        return np.array(
            [c.column_id for c in self.columns if c.traveler_class == traveler_class],
            dtype=int,
        )

    # ------------------------------------------------------------------
    # Measured opportunity supply U[i,j,m] and objective components
    # ------------------------------------------------------------------
    def opportunity_supply_tensor(self, scenario: Scenario) -> STBTensor:
        """Measured land-use / activity-opportunity supply U[i,j,m]."""
        c = self.config
        U = np.zeros((c.rows, c.columns, len(self.activity_labels)))
        m_index = {m: idx for idx, m in enumerate(self.activity_labels)}
        U[self.origins["A"][0] - 1, self.origins["A"][1] - 1, m_index["H"]] = self.demands["A"]
        U[self.origins["B"][0] - 1, self.origins["B"][1] - 1, m_index["H"]] = self.demands["B"]
        U[self.work_destinations["W1"][0] - 1, self.work_destinations["W1"][1] - 1, m_index["W"]] = scenario.work_w1_supply
        U[self.work_destinations["W2"][0] - 1, self.work_destinations["W2"][1] - 1, m_index["W"]] = scenario.work_w2_supply
        U[self.shopping_node[0] - 1, self.shopping_node[1] - 1, m_index["S"]] = scenario.shopping_supply
        return STBTensor(
            U, axes=("cell_row", "cell_col", "activity"),
            measure="opportunities", registry=PAPER_AXES,
        )

    def opportunity_capacity_tensor(self, scenario: Scenario) -> np.ndarray:
        c = self.config
        K = np.full(
            (c.rows, c.columns, len(self.activity_labels)), np.inf, dtype=float
        )
        m_index = {m: idx for idx, m in enumerate(self.activity_labels)}
        K[self.work_destinations["W1"][0] - 1, self.work_destinations["W1"][1] - 1, m_index["W"]] = scenario.work_w1_capacity
        K[self.work_destinations["W2"][0] - 1, self.work_destinations["W2"][1] - 1, m_index["W"]] = scenario.work_w2_capacity
        K[self.shopping_node[0] - 1, self.shopping_node[1] - 1, m_index["S"]] = scenario.shopping_capacity
        return K

    def linear_column_cost(self, scenario: Scenario) -> np.ndarray:
        """a(U): linear column cost from measured opportunity supply."""
        costs = np.zeros(self.n_columns)
        work_supply = {"W1": scenario.work_w1_supply, "W2": scenario.work_w2_supply}
        for col in self.columns:
            cost = 0.01 * col.path_length
            cost -= 1.5 * np.log(work_supply[col.work_name] / 800.0 + self.config.epsilon)
            if col.visits_shopping:
                cost -= 2.5
                cost -= 0.5 * np.log(
                    scenario.shopping_supply / 700.0 + self.config.epsilon
                )
            costs[col.column_id] = cost
        return costs

    def scenario_arc_capacity(self, scenario: Scenario) -> np.ndarray:
        capacity = self.operators.capacity.copy()
        for idx, edge in enumerate(self.graph.edges()):
            if self.graph.edges[edge]["central"]:
                capacity[idx] *= scenario.central_capacity_multiplier
        return capacity

    def opportunity_curvature(self, scenario: Scenario) -> np.ndarray:
        """Diagonal D_U over flattened tensor states: zero at home
        states, opportunity_weight / capacity at capacitated
        opportunity cells."""
        capacity = self.opportunity_capacity_tensor(scenario).reshape(-1)
        diag = np.zeros_like(capacity)
        finite = np.isfinite(capacity)
        diag[finite] = self.config.opportunity_weight / (
            capacity[finite] + self.config.epsilon
        )
        return diag

    def feasible_interior_flow(self) -> np.ndarray:
        flow = np.zeros(self.n_columns)
        for h in sorted(self.origins):
            indices = self.class_columns(h)
            flow[indices] = self.demands[h] / len(indices)
        return flow

    # ------------------------------------------------------------------
    # Typed projections of one common flow vector
    # ------------------------------------------------------------------
    def flow_vector(self, flow: np.ndarray, name: str = "f") -> TypedVector:
        return TypedVector(
            axis=self.operators.column_axis,
            values=np.asarray(flow, dtype=float),
            measure="person_flow",
            unit="persons",
            name=name,
        )

    def projections(self, flow: np.ndarray) -> dict[str, TypedVector | STBTensor]:
        """y = E f, q = R f, x = Delta f, X = Gamma f from one flow."""
        ops = self.operators
        f = self.flow_vector(flow)
        c = self.config
        X_flat = ops.Gamma.apply(f, output_name="X")
        X = STBTensor(
            X_flat.values.reshape(c.rows, c.columns, len(self.activity_labels)),
            axes=("cell_row", "cell_col", "activity"),
            measure="persons",
            registry=PAPER_AXES,
        )
        return {
            "tour_flow": ops.E.apply(f, output_name="y"),
            "leg_od_demand": ops.R.apply(f, output_name="q"),
            "arc_flow": ops.Delta.apply(f, output_name="x", unit="vehicles"),
            "mobility_tensor": X,
        }

    def activity_tensor(self, flow: np.ndarray) -> STBTensor:
        return self.projections(flow)["mobility_tensor"]

    def tour_stage_tensor(self, flow: np.ndarray) -> STBTensor:
        """Illustrative X[i,j,m,t] over coarse tour stages (home start,
        work, shopping, home return) -- a visualization extension, not a
        dynamic network loading result."""
        c = self.config
        T = 4
        data = np.zeros((c.rows, c.columns, len(self.activity_labels), T))
        m_index = {m: idx for idx, m in enumerate(self.activity_labels)}
        for col, weight in zip(self.columns, np.asarray(flow, dtype=float)):
            if weight <= 0:
                continue
            oi, oj = col.origin
            wi, wj = col.work_node
            data[oi - 1, oj - 1, m_index["H"], 0] += weight
            data[wi - 1, wj - 1, m_index["W"], 1] += weight
            if col.shopping_node is not None:
                si, sj = col.shopping_node
                data[si - 1, sj - 1, m_index["S"], 2] += weight
            data[oi - 1, oj - 1, m_index["H"], 3] += weight
        return STBTensor(
            data, axes=("cell_row", "cell_col", "activity", "tour_stage"),
            measure="persons", registry=PAPER_AXES,
        )

    # ------------------------------------------------------------------
    # Integrated objective F(f; U) and its joint calculus
    # ------------------------------------------------------------------
    def _component_gradients(self, scenario: Scenario):
        c = self.config
        capacity = self.scenario_arc_capacity(scenario)
        D_U = self.opportunity_curvature(scenario)
        t0 = self.operators.t0

        def network_gradient(x: np.ndarray) -> np.ndarray:
            return t0 * (1.0 + c.bpr_alpha * (x / capacity) ** c.bpr_beta)

        def tour_gradient(y: np.ndarray) -> np.ndarray:
            return c.entropy_weight * np.log(y + c.epsilon)

        def opportunity_gradient(X: np.ndarray) -> np.ndarray:
            return D_U * X

        return network_gradient, tour_gradient, opportunity_gradient

    def objective(self, flow: np.ndarray, scenario: Scenario) -> float:
        c = self.config
        ops = self.operators
        f = np.asarray(flow, dtype=float)
        capacity = self.scenario_arc_capacity(scenario)
        x = np.asarray(ops.Delta.matrix @ f).ravel()
        y = np.asarray(ops.E.matrix @ f).ravel()
        X = np.asarray(ops.Gamma.matrix @ f).ravel()
        network = np.sum(
            ops.t0
            * (
                x
                + c.bpr_alpha / (c.bpr_beta + 1.0)
                * x ** (c.bpr_beta + 1.0)
                / capacity ** c.bpr_beta
            )
        )
        entropy = c.entropy_weight * np.sum(y * np.log(y + c.epsilon) - y)
        opportunity = 0.5 * np.sum(self.opportunity_curvature(scenario) * X ** 2)
        return float(self.linear_column_cost(scenario) @ f + network + entropy + opportunity)

    def gradient(self, flow: np.ndarray, scenario: Scenario) -> np.ndarray:
        ops = self.operators
        net_g, tour_g, opp_g = self._component_gradients(scenario)
        return integrated_gradient(
            flow,
            self.linear_column_cost(scenario),
            ops.Delta.matrix, ops.E.matrix, ops.Gamma.matrix,
            net_g, tour_g, opp_g,
        )

    def hessian(self, flow: np.ndarray, scenario: Scenario) -> np.ndarray:
        c = self.config
        ops = self.operators
        f = np.asarray(flow, dtype=float)
        capacity = self.scenario_arc_capacity(scenario)
        x = np.asarray(ops.Delta.matrix @ f).ravel()
        y = np.asarray(ops.E.matrix @ f).ravel()
        D_N = (
            ops.t0 * c.bpr_alpha * c.bpr_beta
            * np.maximum(x, 0.0) ** (c.bpr_beta - 1.0)
            / capacity ** c.bpr_beta
        )
        D_M = c.entropy_weight / (y + c.epsilon)
        H = integrated_hessian(
            ops.Delta.matrix, ops.E.matrix, ops.Gamma.matrix,
            D_N, D_M, self.opportunity_curvature(scenario),
        )
        return np.asarray(H.todense()) if hasattr(H, "todense") else H

    def hessian_eigenvalues(self, flow: np.ndarray, scenario: Scenario) -> np.ndarray:
        H = self.hessian(flow, scenario)
        return np.linalg.eigvalsh(0.5 * (H + H.T))

    # ------------------------------------------------------------------
    # Equilibrium solve and certificates
    # ------------------------------------------------------------------
    def solve(
        self, scenario: Scenario, initial_flow: np.ndarray | None = None
    ) -> OptimizeResult:
        if initial_flow is None:
            initial_flow = self.feasible_interior_flow()
        classes = sorted(self.origins)
        rhs = np.array([self.demands[h] for h in classes], dtype=float)
        constraint = LinearConstraint(
            self.operators.B.matrix.toarray(), rhs, rhs
        )
        result = minimize(
            fun=lambda f: self.objective(f, scenario),
            x0=initial_flow,
            jac=lambda f: self.gradient(f, scenario),
            method="SLSQP",
            constraints=[constraint],
            bounds=Bounds(np.zeros(self.n_columns), np.full(self.n_columns, np.inf)),
            options={"maxiter": 2000, "ftol": 1.0e-10, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed for {scenario.name}: {result.message}")
        return result

    def certificates(self, flow: np.ndarray, scenario: Scenario) -> list[dict]:
        """Conservation, feasibility, and stationarity certificates for a
        solved flow.  Stationarity is reported as a KKT spread (equal
        gradient over positive columns within each class), not as a
        Wardrop / FW gap -- the paper preserves that distinction."""
        f = np.asarray(flow, dtype=float)
        classes = sorted(self.origins)
        demand = np.array([self.demands[h] for h in classes])
        class_totals = np.asarray(self.operators.B.matrix @ f).ravel()
        demand_residual = float(np.max(np.abs(class_totals - demand)))
        g = self.gradient(f, scenario)
        spreads = []
        for h in classes:
            idx = self.class_columns(h)
            active = idx[f[idx] > 1e-6 * self.demands[h] / len(idx)]
            if active.size:
                spreads.append(float(np.ptp(g[active])))
        kkt_spread = max(spreads) if spreads else 0.0
        gradient_scale = float(np.mean(np.abs(g)))
        eigenvalues = self.hessian_eigenvalues(f, scenario)
        min_eig = float(np.min(eigenvalues))
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        return [
            {"name": "class_demand_conservation",
             "residual": demand_residual,
             "passed": bool(demand_residual < 1e-6)},
            {"name": "nonnegative_flow",
             "residual": float(max(-np.min(f), 0.0)),
             "passed": bool(np.min(f) > -1e-9)},
            {"name": "kkt_equal_gradient_spread",
             "residual": kkt_spread,
             "scale": gradient_scale,
             "passed": bool(kkt_spread < 5e-2 * max(gradient_scale, 1.0))},
            {"name": "hessian_positive_semidefinite",
             "min_eigenvalue": min_eig,
             "passed": bool(min_eig > -1e-8 * scale)},
        ]

    def scenario_metrics(
        self, scenario: Scenario, result: OptimizeResult
    ) -> dict[str, float | str | int]:
        c = self.config
        X = self.activity_tensor(result.x).data
        x = np.asarray(self.operators.Delta.matrix @ result.x).ravel()
        capacity = self.scenario_arc_capacity(scenario)
        arc_cost = self.operators.t0 * (
            1.0 + c.bpr_alpha * (x / capacity) ** c.bpr_beta
        )
        total = sum(self.demands.values())
        w1 = X[self.work_destinations["W1"][0] - 1, self.work_destinations["W1"][1] - 1, 1]
        w2 = X[self.work_destinations["W2"][0] - 1, self.work_destinations["W2"][1] - 1, 1]
        shop = X[self.shopping_node[0] - 1, self.shopping_node[1] - 1, 2]
        return {
            "scenario": scenario.name,
            "objective": float(result.fun),
            "objective_per_person": float(result.fun / total),
            "network_time_per_person": float(x @ arc_cost / total),
            "W1_share": float(w1 / total),
            "W2_share": float(w2 / total),
            "shopping_share": float(shop / total),
            "max_vc": float(np.max(x / capacity)),
            "iterations": int(result.nit),
        }

    # ------------------------------------------------------------------
    # Joint second-order pricing experiment
    # ------------------------------------------------------------------
    def pricing_experiment(
        self,
        scenario: Scenario,
        samples: int = 500,
        step: float = 10.0,
        reference_flow: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sample feasible donor -> candidate replacements inside each
        class and compare first-order, second-order, and exact finite
        objective changes at a common reference iterate."""
        reference = (
            self.feasible_interior_flow()
            if reference_flow is None
            else np.asarray(reference_flow, dtype=float)
        )
        g = self.gradient(reference, scenario)
        H = self.hessian(reference, scenario)
        rng = np.random.default_rng(self.config.random_seed)
        classes = sorted(self.origins)
        records: list[dict[str, float | int | str]] = []

        for sample_id in range(samples):
            h = str(rng.choice(classes))
            indices = self.class_columns(h)
            donor = int(rng.choice(indices))
            candidate = int(rng.choice(indices[indices != donor]))
            d = replacement_direction(candidate, donor, self.n_columns)
            price = quadratic_price(g, H, d, step=step)
            exact = exact_finite_change(
                lambda f: self.objective(f, scenario), reference, d, step=step
            )
            records.append({
                "sample": sample_id,
                "traveler_class": h,
                "from_column": donor,
                "to_column": candidate,
                "exact_change": exact,
                "first_order": price.first_order,
                "second_order": price.second_order,
            })

        df = pd.DataFrame(records)
        summary = pd.DataFrame({
            "metric": ["MAE", "RMSE", "Spearman rank correlation"],
            "first_order": [
                float(np.mean(np.abs(df["exact_change"] - df["first_order"]))),
                float(np.sqrt(np.mean((df["exact_change"] - df["first_order"]) ** 2))),
                float(spearmanr(df["exact_change"], df["first_order"]).statistic),
            ],
            "second_order": [
                float(np.mean(np.abs(df["exact_change"] - df["second_order"]))),
                float(np.sqrt(np.mean((df["exact_change"] - df["second_order"]) ** 2))),
                float(spearmanr(df["exact_change"], df["second_order"]).statistic),
            ],
        })
        return df, summary

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------
    def column_catalog(self) -> pd.DataFrame:
        rows = []
        for col in self.columns:
            rows.append({
                "column_id": col.column_id,
                "traveler_class": col.traveler_class,
                "tour": col.tour_name,
                "tour_type": col.tour_type,
                "origin": str(col.origin),
                "work": col.work_name,
                "work_node": str(col.work_node),
                "shopping_node": str(col.shopping_node) if col.shopping_node else "",
                "path_length": col.path_length,
                "legs": " | ".join("->".join(map(str, leg)) for leg in col.legs),
            })
        return pd.DataFrame(rows)

    def tensor_to_dataframe(self, tensor: STBTensor | np.ndarray, value_name: str) -> pd.DataFrame:
        data = tensor.data if isinstance(tensor, STBTensor) else np.asarray(tensor)
        rows = []
        for i in range(self.config.rows):
            for j in range(self.config.columns):
                for m_idx, m in enumerate(self.activity_labels):
                    rows.append({
                        "row_i": i + 1,
                        "column_j": j + 1,
                        "activity_m": m,
                        value_name: float(data[i, j, m_idx]),
                    })
        return pd.DataFrame(rows)

    def stage_tensor_to_dataframe(self, tensor: STBTensor | np.ndarray) -> pd.DataFrame:
        data = tensor.data if isinstance(tensor, STBTensor) else np.asarray(tensor)
        rows = []
        for i in range(self.config.rows):
            for j in range(self.config.columns):
                for m_idx, m in enumerate(self.activity_labels):
                    for t in range(data.shape[3]):
                        rows.append({
                            "row_i": i + 1,
                            "column_j": j + 1,
                            "activity_m": m,
                            "tour_stage_t": t,
                            "value": float(data[i, j, m_idx, t]),
                        })
        return pd.DataFrame(rows)
