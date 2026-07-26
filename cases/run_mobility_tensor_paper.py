"""The authoritative mobility-tensor paper case: every number in the
TRB paper draft (research_papers/trb_mobility_tensor/) comes from this
script.

One run executes the complete paper experiment:

  1. build the 5x5 grid with home / work / shopping opportunity
     fields U[i,j,m] and complete activity-tour-path columns;
  2. solve the integrated first-order equilibrium for three measured
     scenarios (base, employment shifted to W2, central capacity -40%);
  3. project one common column flow to y = E f, q = R f, x = Delta f,
     and the mobility tensor X = Gamma f (plus the illustrative
     tour-stage tensor X[i,j,m,t]);
  4. price 500 feasible donor -> candidate column replacements with the
     joint first- and second-order models against the exact finite
     objective change;
  5. re-run the certified grid column-generation ladder (10x10, 20x20,
     and with --full 50x50 and Chicago Sketch) for the scalability
     table -- same settings as cases/run_demo_suite.py;
  6. export all tables, five consolidated figures, and
     certificates.json.

Paper Figure 1 (the Represent -> Conserve -> Connect -> Stabilize ->
Scale -> Transfer architecture) is the repository figure
figures/framework_2.png; Figures 2-5 are generated here.

External-data note: the TRMG2 regional and IEEE corridor rows of the
README require locally held datasets and are cited in the paper as
optional practical evidence only; they are not produced by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensormobility.core.unified_networks import load_case
from tensormobility.dta.sparse_assignment import network_from_case, solve_fw
from tensormobility.optimization.mobility_tensor_case import (
    MobilityTensorCase,
    Scenario,
)

OUT = Path(__file__).parent / "outputs" / "mobility_tensor_paper"
FIG = OUT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-samples", type=int, default=500)
    parser.add_argument("--pricing-step", type=float, default=10.0)
    parser.add_argument(
        "--full", action="store_true",
        help="include the 50x50 grid and Chicago Sketch scalability rows",
    )
    parser.add_argument(
        "--skip-scaling", action="store_true",
        help="skip the column-generation scalability section entirely",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Figures 2-5
# ----------------------------------------------------------------------
def figure_typed_tensor(case: MobilityTensorCase, X, X_stage, path: Path) -> None:
    """Figure 2: the typed mobility tensor X[i,j,m] and its illustrative
    tour-stage extension X[i,j,m,t]."""
    c = case.config
    fig = plt.figure(figsize=(13.0, 6.4))
    grid = fig.add_gridspec(2, 12, hspace=0.42)
    vmax = max(float(np.max(X.data)), 1.0)
    image = None
    for idx, label in enumerate(case.activity_labels):
        ax = fig.add_subplot(grid[0, idx * 4:idx * 4 + 4])
        image = ax.imshow(X.data[:, :, idx], vmin=0, vmax=vmax)
        ax.set_title(f"X[i,j,{label}]")
        ax.set_xticks(range(c.columns), range(1, c.columns + 1))
        ax.set_yticks(range(c.rows), range(1, c.rows + 1))
        for i in range(c.rows):
            for j in range(c.columns):
                value = X.data[i, j, idx]
                if value > 1e-6:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                            fontsize=8)
    stages = ["t=0 home start", "t=1 work", "t=2 shopping", "t=3 home return"]
    spatial = X_stage.data.sum(axis=2)
    svmax = max(float(np.max(spatial)), 1.0)
    for t in range(4):
        ax = fig.add_subplot(grid[1, t * 3:t * 3 + 3])
        image = ax.imshow(spatial[:, :, t], vmin=0, vmax=svmax)
        ax.set_title(stages[t], fontsize=10)
        ax.set_xticks(range(c.columns), range(1, c.columns + 1))
        if t == 0:
            ax.set_yticks(range(c.rows), range(1, c.rows + 1))
        else:
            ax.set_yticks([])
    fig.suptitle(
        "Typed mobility tensor: named axes (cell_row, cell_col, activity)"
        " and the coarse tour-stage extension", y=0.99,
    )
    fig.colorbar(image, ax=fig.axes, shrink=0.7, label="persons")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_grid_tours(
    case: MobilityTensorCase, scenario: Scenario, flow: np.ndarray, path: Path
) -> None:
    """Figure 3: the grid, the measured opportunity field U, and the
    highest-flow complete HWH and HWSH columns."""
    pos = {node: (node[1], -node[0]) for node in case.graph.nodes()}
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    nx.draw_networkx_edges(case.graph, pos, ax=ax, arrows=False, width=0.8,
                           alpha=0.4)
    nx.draw_networkx_nodes(case.graph, pos, ax=ax, node_size=120,
                           node_color="#c9d4de", linewidths=0.6)

    U = case.opportunity_supply_tensor(scenario).data
    markers = {"H": "o", "W": "s", "S": "^"}
    for m_idx, m in enumerate(case.activity_labels):
        nodes, sizes = [], []
        for i in range(case.config.rows):
            for j in range(case.config.columns):
                if U[i, j, m_idx] > 0:
                    nodes.append((i + 1, j + 1))
                    sizes.append(170 + 0.35 * U[i, j, m_idx])
        if nodes:
            xy = np.array([pos[node] for node in nodes])
            ax.scatter(xy[:, 0], xy[:, 1], s=sizes, marker=markers[m],
                       label=f"U supply: {m}", zorder=3)

    colors = {"HWH": "#c0392b", "HWSH": "#2471a3"}
    for tour_type in ("HWH", "HWSH"):
        candidates = [c for c in case.columns if c.tour_type == tour_type]
        best = max(candidates, key=lambda c: flow[c.column_id])
        for leg in best.legs:
            xy = np.array([pos[node] for node in leg])
            ax.plot(xy[:, 0], xy[:, 1], color=colors[tour_type], linewidth=2.4,
                    alpha=0.85, zorder=2,
                    label=f"top {tour_type} column ({flow[best.column_id]:.0f} p)"
                    if leg is best.legs[0] else None)

    ax.set_title(f"Complete activity-tour-path columns on the measured "
                 f"U field: {scenario.name}")
    ax.set_xlabel("grid column j")
    ax.set_ylabel("grid row i")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks(range(1, case.config.columns + 1))
    ax.set_yticks([-i for i in range(1, case.config.rows + 1)],
                  range(1, case.config.rows + 1))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_operator_hessian(case: MobilityTensorCase, H, path: Path) -> None:
    """Figure 4: the sparse projection stack (E, R, Delta, Gamma) and
    the assembled joint Hessian H = Delta^T D_N Delta + E^T D_M E +
    Gamma^T D_U Gamma."""
    ops = case.operators
    fig, axes = plt.subplots(1, 5, figsize=(14.4, 3.3))
    for ax, (label, op) in zip(
        axes[:4],
        (("E [tour <- column]", ops.E), ("R [leg OD <- column]", ops.R),
         ("Delta [arc <- column]", ops.Delta),
         ("Gamma [state <- column]", ops.Gamma)),
    ):
        ax.spy(op.matrix, markersize=1.2, aspect="auto")
        ax.set_title(f"{label}\n{op.matrix.shape[0]}x{op.matrix.shape[1]}, "
                     f"density {op.density:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    image = axes[4].imshow(H, cmap="viridis")
    axes[4].set_title("H = Delta^T D_N Delta + E^T D_M E\n+ Gamma^T D_U Gamma",
                      fontsize=9)
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    fig.colorbar(image, ax=axes[4], shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_numerical_evidence(
    summary: pd.DataFrame,
    pricing_samples: pd.DataFrame,
    scaling: pd.DataFrame | None,
    path: Path,
) -> None:
    """Figure 5: scenario responses, pricing accuracy, and the certified
    scalability ladder in one panel."""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))

    ax = axes[0, 0]
    labels = list(summary["scenario"])
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, 100 * summary["W1_share"], width, label="W1 share")
    ax.bar(x, 100 * summary["W2_share"], width, label="W2 share")
    ax.bar(x + width, 100 * summary["shopping_share"], width,
           label="shopping-tour share")
    ax.set_ylabel("share of travelers (%)")
    ax.set_xticks(x, [l.replace(" ", "\n", 1) for l in labels], fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("(a) destination and tour response", fontsize=10)

    ax = axes[0, 1]
    ax.bar(x, summary["max_vc"], 0.45, color="#c0392b")
    ax.axhline(1.0, linestyle="--", linewidth=1, color="#5b6b7c")
    ax.set_ylabel("max volume / capacity")
    ax.set_xticks(x, [l.replace(" ", "\n", 1) for l in labels], fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("(b) peak congestion response", fontsize=10)

    ax = axes[1, 0]
    ax.scatter(pricing_samples["exact_change"], pricing_samples["first_order"],
               s=10, alpha=0.45, label="first order")
    ax.scatter(pricing_samples["exact_change"], pricing_samples["second_order"],
               s=10, alpha=0.45, label="second order")
    values = np.concatenate([
        pricing_samples["exact_change"], pricing_samples["first_order"],
        pricing_samples["second_order"],
    ])
    lower, upper = float(np.min(values)), float(np.max(values))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1,
            label="exact agreement")
    ax.set_xlabel("exact objective change")
    ax.set_ylabel("predicted objective change")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("(c) joint second-order replacement pricing", fontsize=10)

    ax = axes[1, 1]
    if scaling is not None and len(scaling):
        ax.loglog(scaling["od_pairs"], scaling["wall_s"], "o")
        for row in scaling.itertuples(index=False):
            ax.annotate(f"{row.network}\ngap {row.full_space_gap:.1e}",
                        (row.od_pairs, row.wall_s), fontsize=7,
                        textcoords="offset points", xytext=(6, -2))
        ax.set_xlabel("OD pairs")
        ax.set_ylabel("wall time (s)")
        ax.grid(alpha=0.25, which="both")
        ax.set_title("(d) certified column-generation scalability", fontsize=10)
    else:
        ax.axis("off")
        ax.set_title("(d) scalability section skipped", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Scalability ladder (same settings as run_demo_suite.py D1/D3)
# ----------------------------------------------------------------------
def run_scaling(full: bool) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    certificates: list[dict] = []
    grid_sizes = ((10, 20), (20, 60)) + (((50, 200),) if full else ())
    for size, n_od in grid_sizes:
        case = load_case("grid", rows=size, columns=size, n_od=n_od,
                         demand_per_od=400.0, base_capacity=900.0)
        start = time.perf_counter()
        r = solve_fw(network_from_case(case), max_rounds=80, tolerance=1e-4)
        wall = time.perf_counter() - start
        rows.append(dict(
            network=f"grid {size}x{size}", nodes=case.network.n_nodes,
            links=case.network.n_links, od_pairs=len(case.demand),
            columns=r.n_columns, full_space_gap=r.relative_gap,
            feasibility=r.feasibility_residual, wall_s=wall,
        ))
        certificates.append({
            "name": f"grid_{size}x{size}_full_space_gap",
            "relative_gap": float(r.relative_gap),
            "feasibility_residual": float(r.feasibility_residual),
            "passed": bool(r.relative_gap < 1e-3),
        })
    if full:
        case = load_case("chicago_sketch")
        start = time.perf_counter()
        r = solve_fw(network_from_case(case), max_rounds=12, tolerance=1e-4)
        wall = time.perf_counter() - start
        rows.append(dict(
            network="Chicago Sketch", nodes=case.network.n_nodes,
            links=case.network.n_links, od_pairs=len(case.demand),
            columns=r.n_columns, full_space_gap=r.relative_gap,
            feasibility=r.feasibility_residual, wall_s=wall,
        ))
        certificates.append({
            "name": "chicago_sketch_full_space_gap",
            "relative_gap": float(r.relative_gap),
            "feasibility_residual": float(r.feasibility_residual),
            "passed": bool(r.relative_gap < 1e-3),
        })
    return pd.DataFrame(rows), certificates


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    all_certificates: dict[str, list[dict]] = {}

    case = MobilityTensorCase()
    scenarios = [
        Scenario(name="Base measured land use"),
        Scenario(
            name="Employment shifted to W2",
            work_w1_supply=600.0, work_w2_supply=1400.0,
            work_w1_capacity=400.0, work_w2_capacity=800.0,
        ),
        Scenario(name="Central capacity -40%",
                 central_capacity_multiplier=0.60),
    ]

    previous_flow = None
    metrics = []
    solutions: dict[str, np.ndarray] = {}
    for scenario in scenarios:
        result = case.solve(scenario, initial_flow=previous_flow)
        previous_flow = result.x
        solutions[scenario.name] = result.x
        metrics.append(case.scenario_metrics(scenario, result))
        all_certificates[scenario.name] = case.certificates(result.x, scenario)

    summary = pd.DataFrame(metrics)
    summary.to_csv(OUT / "scenario_summary.csv", index=False)
    case.column_catalog().to_csv(OUT / "column_catalog.csv", index=False)

    base = scenarios[0]
    base_flow = solutions[base.name]
    U = case.opportunity_supply_tensor(base)
    projections = case.projections(base_flow)
    X = projections["mobility_tensor"]
    X_stage = case.tour_stage_tensor(base_flow)
    case.tensor_to_dataframe(U, "opportunity_supply").to_csv(
        OUT / "opportunity_supply_U_ijm.csv", index=False)
    case.tensor_to_dataframe(X, "activity_flow").to_csv(
        OUT / "mobility_tensor_X_ijm.csv", index=False)
    case.stage_tensor_to_dataframe(X_stage).to_csv(
        OUT / "mobility_tensor_X_ijmt.csv", index=False)
    pd.DataFrame({
        "projection": ["y = E f", "q = R f", "x = Delta f", "X = Gamma f"],
        "axis": [case.operators.tour_axis.name, case.operators.leg_od_axis.name,
                 case.operators.arc_axis.name, "cell_row x cell_col x activity"],
        "size": [case.operators.tour_axis.size, case.operators.leg_od_axis.size,
                 case.operators.arc_axis.size, X.data.size],
        "total": [projections["tour_flow"].total(),
                  projections["leg_od_demand"].total(),
                  projections["arc_flow"].total(), X.total()],
    }).to_csv(OUT / "projection_totals.csv", index=False)

    pricing_samples, pricing_summary = case.pricing_experiment(
        base, samples=args.pricing_samples, step=args.pricing_step)
    pricing_samples.to_csv(OUT / "pricing_samples.csv", index=False)
    pricing_summary.to_csv(OUT / "pricing_summary.csv", index=False)
    mae = pricing_summary.set_index("metric")

    # Cubic remainder decay: the same seeded moves priced at half the
    # step; |exact - second_order| should scale by ~(1/2)^3.
    half_samples, _ = case.pricing_experiment(
        base, samples=args.pricing_samples, step=args.pricing_step / 2.0)
    err_full = np.abs(pricing_samples["exact_change"]
                      - pricing_samples["second_order"])
    err_half = np.abs(half_samples["exact_change"]
                      - half_samples["second_order"])
    valid = err_half > 1e-9
    decay_ratios = (err_full[valid] / err_half[valid]).to_numpy()
    decay = {
        "step_full": float(args.pricing_step),
        "step_half": float(args.pricing_step / 2.0),
        "n_moves": int(valid.sum()),
        "median_remainder_ratio": float(np.median(decay_ratios)),
        "theoretical_cubic_ratio": 8.0,
    }
    pd.DataFrame([decay]).to_csv(OUT / "remainder_decay.csv", index=False)
    all_certificates["pricing"] = [{
        "name": "second_order_improves_first_order_mae",
        "first_order_mae": float(mae.loc["MAE", "first_order"]),
        "second_order_mae": float(mae.loc["MAE", "second_order"]),
        "passed": bool(mae.loc["MAE", "second_order"]
                       < 0.05 * mae.loc["MAE", "first_order"]),
    }, {
        "name": "remainder_decays_cubically",
        "median_ratio_on_step_doubling": decay["median_remainder_ratio"],
        "passed": bool(6.0 < decay["median_remainder_ratio"] < 10.0),
    }]

    reference = case.feasible_interior_flow()
    eigenvalues = case.hessian_eigenvalues(reference, base)
    pd.DataFrame({"eigenvalue": eigenvalues}).to_csv(
        OUT / "hessian_eigenvalues.csv", index=False)

    scaling: pd.DataFrame | None = None
    if not args.skip_scaling:
        scaling, scaling_certs = run_scaling(args.full)
        scaling.to_csv(OUT / "scaling.csv", index=False)
        all_certificates["scaling"] = scaling_certs

    figure_typed_tensor(case, X, X_stage, FIG / "fig2_typed_tensor.png")
    figure_grid_tours(case, base, base_flow, FIG / "fig3_grid_tours.png")
    figure_operator_hessian(case, case.hessian(reference, base),
                            FIG / "fig4_operator_hessian.png")
    figure_numerical_evidence(summary, pricing_samples, scaling,
                              FIG / "fig5_numerical_evidence.png")

    (OUT / "certificates.json").write_text(
        json.dumps(all_certificates, indent=2), encoding="utf-8")

    lines = [
        "# Mobility-tensor paper case — fresh certified run\n",
        "Generated by `python cases/run_mobility_tensor_paper.py`"
        + (" --full" if args.full else "")
        + f" (pricing samples: {args.pricing_samples},"
        f" step {args.pricing_step}).\n",
        f"Grid {case.config.rows}x{case.config.columns}; "
        f"{case.n_columns} complete activity-tour-path columns; "
        f"{case.operators.arc_axis.size} directed arcs; "
        f"{case.operators.state_axis.size} tensor states (i,j,m); "
        f"{case.operators.leg_od_axis.size} tour-leg ODs.\n",
        "## Scenario responses\n",
        summary.round(4).to_markdown(index=False), "",
        "## Joint second-order replacement pricing "
        f"({args.pricing_samples} feasible moves, step "
        f"{args.pricing_step})\n",
        pricing_summary.round(6).to_markdown(index=False), "",
        f"Remainder decay: median |exact - second-order| ratio "
        f"{decay['median_remainder_ratio']:.2f} when the step doubles "
        f"({decay['step_half']:g} -> {decay['step_full']:g}); the cubic "
        f"remainder predicts 8.\n",
        "## Certificates\n",
    ]
    for section, certs in all_certificates.items():
        status = "; ".join(
            f"{c['name']} {'PASS' if c['passed'] else 'FAIL'}" for c in certs)
        lines.append(f"- **{section}**: {status}")
    if scaling is not None:
        lines += ["", "## Certified scalability ladder (same settings as "
                  "run_demo_suite.py)\n",
                  scaling.round(6).to_markdown(index=False), "",
                  "External-data note: TRMG2 regional and IEEE corridor "
                  "rows require locally held datasets "
                  "(TENSORMOBILITY_TRMG2_DATA / TENSORMOBILITY_TFB_DATA) "
                  "and are quoted in the paper as optional evidence only."]
    lines += ["", "## Figure map\n",
              "- Figure 1: figures/framework_2.png (Represent -> Conserve "
              "-> Connect -> Stabilize -> Scale -> Transfer)",
              "- Figure 2: figures/fig2_typed_tensor.png",
              "- Figure 3: figures/fig3_grid_tours.png",
              "- Figure 4: figures/fig4_operator_hessian.png",
              "- Figure 5: figures/fig5_numerical_evidence.png"]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nall outputs in {OUT}")


if __name__ == "__main__":
    main()
