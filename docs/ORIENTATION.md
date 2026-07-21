# Canonical operator orientation

One rule everywhere: **y_target = M[target <- source] x_source**
(column-vector propagation), as implemented by `TypedOperator` and
`STBTensor.contract`.

Known legacy deviations (review P0.1), scheduled for migration in
v0.7 and until then quarantined behind the sanctioned adapter
`tensor_contracts.operator_from_source_target_table(...)`:

| module | current storage | status |
|---|---|---|
| `neural/tcg_graph.py` | source-by-target, row-vector propagation | migrate v0.7 |
| `core/flow_through.py` | zone->OD, OD->path stored source-by-target; path->link target-by-source | migrate v0.7 |

Rules until migration completes:
1. New application code MUST use the canonical orientation.
2. Imported row-stochastic tables (splits, proportions) MUST pass
   through the adapter -- raw `.T` in application code is a defect.
3. Every operator constructor documents its orientation in its
   docstring.
