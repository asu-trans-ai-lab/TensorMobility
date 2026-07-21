# Design update: answering the reproduction report (2026-07-21)

*An independent reproduction confirmed the package end-to-end (58/58 at
that snapshot; Chicago gap 9.76e-5 in ~11 s on a slower CPU; the
6-algorithm benchmark table reproduced exactly) and made five comments.
All five are accepted. This document is the update design; D1 and
D4-row-2 are implemented in this commit, D3/D5 are specified for the
next build cycle.*

## D1 — Certificate-honest names [DONE]

`exact_fw` landed 5.1e-5 ABOVE the `fw_gp` minimum: the row named
"exact" was not the certificate. Renamed:

| old | new | role |
|---|---|---|
| `fw_gp` | **`certified_fw`** | the certificate (full-space priced) |
| `exact_fw` | **`reference_fw`** | reference run, looser tolerance |

Rule generalized: *no algorithm may carry a name stronger than its
certificate.*

## D2 — Proposer honesty, made structural

The reproduction sharpened our own H1 verdict: the learned proposer is
~30x slower in wall clock, promotes the same single column the
un-learned reduced-gradient latent finds, and its ranking disagreed
with exact pricing on 23/50 audits (base) and 50/50 (shock). Design
response, permanent:

- the benchmark table gains two always-on columns:
  `proposer_false_negative_rate` and `pricing_cost_per_sweep`;
- a stated **value condition** in the benchmark docstring: a learned
  proposer can pay only when (i) pricing-per-sweep is expensive and
  (ii) proposer ranking correlates with reduced cost — neither holds on
  a 228-column toy;
- the claim in any table caption is "architecture demonstrated", never
  "acceleration achieved", until (i)+(ii) are measured true.

## D3 — THE rank-economy experiment [next cycle, centerpiece]

The thesis question — *how many ranks do we actually need* — is
currently unmeasured: every latent result sits at one operating point
(120 atoms, fixed route richness). Design:

```
cases/run_rank_economy.py
  for K in {2, 4, 8, 12, 16}:            # routes per OD (SF k_paths)
      build instance (build_sioux_falls_path_set(k_paths=K))
      certified = solve certified_fw            -> objective*
      for the atom ladder:
          static latent (major+minor)           -> bias(K) = obj - obj*
          adaptive latent                       -> promotions_to_certify(K)
      record: bias(K), promotions(K), atoms/columns(K),
              pricing calls, wall
outputs: rank_economy.csv + one two-panel figure
  (i)  pure-latent objective bias vs K          [expected: grows]
  (ii) promotions-to-certify vs K               [expected: sublinear]
```

Interpretation frame (the Newell rank economy): free-flow + congested
is ~rank-2; each condition-dependent parameter adds one mode. The
experiment turns "structure is all you need" from a README table into a
measured curve. Acceptance: the figure regenerates from one command
and the two expectations are stated as hypotheses IN the runner, so a
refutation would be reported, not hidden.

Extension of the same runner (D5 tie-in): `--case sioux_falls|grid`
now, `--case chicago_sketch` behind a flag (K̄≈2.2 predicts bias≈0 and
compression≈1 — the boundary datum re-measured by the same code).

## D4 — Correspondence table: tested vs structural, counted honestly

The README's 8-row identity table was load-bearing on ONE tested row.
Ladder:

| row | status |
|---|---|
| softmax router ≡ logit choice | **tested** (bitwise) |
| backprop ≡ discrete adjoint ≡ FTT Jacobian | **tested (this commit)**: hand adjoint == torch autograd to 1e-9, masked coords identically zero both ways |
| router score − duals ≡ reduced cost | tested (sign identity) |
| top-1 ≡ AON limit | tested (θ-ladder) |
| DEQ ≡ equilibrium fixed point | structural (engine ladder computes both; no cross-check yet) |
| masked-softmax stack ≡ conserved chain | tested (conservation + FD) |
| tanh residual ≡ admissible utility term | structural |
| atoms ≡ low-rank feasible coordinates | structural (D3 measures it) |

README language becomes: **"5 rows equality-tested, 3 structural"** and
the counter must be updated with the table.

## D5 — Scale-out of the latent/promotion table [next cycle]

The 6-algorithm table exists only on the toy. The sparse solvers
already run on SF and Chicago through `network_from_case`; design:
`cases/run_scale_table.py` emits the SAME table schema on
{toy, sioux_falls, chicago_sketch(top-N | full)} — one schema, three
scales, with the D2 honesty columns. Chicago's known outcome
(compression 1.28x, gap floor 6.1e-3) enters the table as the measured
boundary rather than a footnote.

## Sequencing

1. this commit: D1, D4-row-2, this design doc (suite 66 -> 67);
2. next: D3 runner + figure; D5 table runner;
3. then fold both into the paper as the two figures the reproduction
   asked for — at which point points 2–5 of the report become paper
   content, not caveats.
