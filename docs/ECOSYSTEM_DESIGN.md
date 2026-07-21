# TensorMobility ecosystem design (v0.7 target — FOR ITERATION)

*2026-07-21. How TensorMobility connects to TAPLite4MPO,
dynamic-odme-lab, and gui4gmns; how path flows, corridor data, ODME,
and notebook demonstrations fit one architecture. Principles first,
interfaces second, roadmap third — iterate on any of them.*

## 0. The one-picture design

Named workflows (rendered in `workflow.html`): **W1 generate · W2
assign · W3 import · W4 calibrate · W5 visualize · W6 teach ·
W7 profile**; pipelines quickstart/MPO/corridor/research/classroom
compose them.

```
                     ┌──────────────────────────────────┐
                     │        TensorMobility            │
                     │  axis calculus · typed columns · │
                     │  profiles · engines · CERTIFICATES│
                     └──────┬─────────┬─────────┬───────┘
        fast kernels        │         │         │   visualization
   ┌────────────────┐  ┌────▼────┐ ┌──▼───────┐ ┌▼──────────────┐
   │  TAPLite4MPO   │  │ DTALite │ │ dynamic-  │ │   gui4gmns    │
   │ C++ FW, 9 VDFs │  │  /DLSim │ │ odme-lab  │ │ HTML · Qt · GL│
   └────────────────┘  └─────────┘ └───────────┘ └───────────────┘
                 ═══ GMNS CSV folders are the ONLY contract ═══
```

TensorMobility is not another engine. It is the **tensor, column, and
certificate layer** that orchestrates the lab's engines and views, and
audits everything that crosses its boundary.

## 1. Design principles (the part to argue about)

**P1 — Kernels compute; TensorMobility certifies.** External engines
(TAPLite, DTALite) are treated as fast oracles. Every imported solution
is audited by our own full-space pricing (all-origin Dijkstra gap) and
conservation checks before it is used or displayed. A kernel result
without a certificate is a rumor.

**P2 — GMNS folders on disk are the only inter-tool contract.** No tool
imports another tool's Python objects. `node.csv / link.csv /
demand.csv / path.csv / link_performance.csv` + a `certificates.json`
sidecar. This is what already makes gui4gmns, TAPLite, and DTALite
composable; we conform rather than invent.

**P3 — Columns (path flows) are the lingua franca.** TAPLite outputs
link volumes and skims but no paths. Rule: *anything that arrives
without columns gets them re-priced.* `tensormobility` warm-starts its
column generator at the kernel's congested times and materializes
`path.csv` (GMNS route schema) in a handful of pricing rounds — cheap,
because the kernel already did the equilibrium work. Path generation is
therefore a first-class utility, not a solver internal:
`case.generate_paths(times) -> path.csv` works for the self-contained
grid, SF, Chicago, or any imported kernel run.

**P4 — Every demonstration is a notebook that CI executes.** The
teaching ladder (T1–T7) and every case become Jupyter notebooks run by
`nbmake` in CI: load → solve → certify → visualize (gui4gmns HTML
embedded inline). `pip install tensormobility[demo]` is the classroom
install.

**P5 — Dynamic ODME is a calibration profile of the same chain.** The
dynamic-odme-lab formulation is already ours in different notation:
its matrix-free `A = M·Δ·R` is the typed operator chain, its
departure-profile recovery φ(t) is a promotion of the departure axis
with observation feedback, its bounded ±10% low-rank OD adjustment is
exactly the spectator-axis compression doctrine, and its observability
gates are our identifiability gate G0. So ODME enters TensorMobility as
`profiles/odme.py` — not a fork, a profile — and dynamic-odme-lab's
stages 0–6 become its benchmark ladder.

**P6 — Corridor attribute data enters through adapters, never
hard-coded.** A data registry (`data/registry.yaml` + env overrides,
extending the current `TENSORMOBILITY_*_DATA` pattern) maps named
corridors (e.g., I-17, PeMS districts) to GMNS folders + sensor
attribute tables. Corridor-by-corridor testing = iterate the registry,
run the ODME profile per corridor, emit one certified dashboard each.

**P7 — Growth = axis extension + profile + certificate.** Established
mechanism (MAGE proved it): a new capability registers extension axes,
declares its fixed-point maps and engines, and ships its certificate
set. No new capability may touch the canonical core.

## 2. Interface designs

### I1 — Kernel interface (TAPLite first)

```python
class KernelAdapter(Protocol):
    name: str
    def assign(self, gmns_dir: Path, config: dict) -> KernelRun
        # KernelRun: link_flow, link_time, skims, run_manifest

class TAPLiteKernel(KernelAdapter):
    """pip install taplite4mpo; invokes `taplite run <config>` or
    pytaplite.assign(); parses link_performance.csv."""
```

Post-conditions enforced by TensorMobility on every `KernelRun`:
1. conservation audit (demand vs assigned);
2. **full-space gap audit** at the kernel's link times — the same
   35.7%-pool lesson applied to external engines;
3. column materialization (P3) → `path.csv`;
4. `certificates.json` written next to the kernel outputs.

This also gives kernel *cross-validation* for free: TAPLite vs
`tensormobility.dta.solve_fw` on the same case, gap-vs-gap and
flow-correlation reported (we already do exactly this against Chicago
`ref_volume`, correlation 1.0000 — the pattern generalizes).

### I2 — Visualization interface (gui4gmns)

```python
tensormobility.io.export_gmns(case, result, out_dir)
# writes: link_performance.csv (volume, v/c, speed, queue),
#         path.csv, od desire-line file, MOE files,
#         certificates.json  →  gui4gmns out_dir
```

One rule: we write only layers gui4gmns already renders (volume / v/c /
queue / speed maps, OD desire lines, space–time corridor contours from
the fluid-queue layer). Plus one addition to propose upstream: a
**certificate panel** (render `certificates.json` as a dashboard
table) so every published dashboard carries its audit.

### I3 — Dynamic ODME profile

```python
profiles/odme.py
  estimate(case, counts, speeds=None, *,
           adjust='departure_profile' | 'bounded_od' | 'both')
  -> ODMEResult(phi, delta_od, fit, certificates)
```

- Forward operator: the typed chain (SC3 machinery — analytic
  gradients FD-certified at 2e-7 already).
- Solvers: projected gradient / the engine ladder; observability gate
  (rank of H on the active tangent space) decides which parameters are
  even allowed to move — the G0 discipline.
- Benchmarks: adopt dynamic-odme-lab stages 0–6 as
  `cases/odme_stage*.ipynb`; corridor stages consume the P6 registry.
- Honest scope carried over from that repo: departure-profile recovery
  embedded in a corridor; queue layer diagnostic-only (C2 boundary).

### I4 — Notebook demonstration layer

```
notebooks/
  T1_ftt_chain.ipynb … T7_mixed_autonomy_engines.ipynb   (teaching)
  case_grid_ue.ipynb, case_chicago.ipynb, case_mage.ipynb,
  case_odme_corridor.ipynb                                (cases)
```
Each ends with `export_gmns(...)` + inline gui4gmns HTML. CI tier 2
runs them headless; tier 3 uploads the dashboards as artifacts.

### I5 — Grid self-containment (already true, made explicit)

The grid generator + path generation utility means the entire pipeline
— generate network → assign (internal or TAPLite) → materialize paths
→ perturb/ODME → visualize — runs with zero external data. Grid is the
CI-proof spine; SF/Chicago/corridors add realism through the registry.

## 3. Relation to the papers folder

Each paper maps to a pipeline slice: the FTT manuscript ↔ core chain +
I2 dashboards; the v0.6 wrap-up draft ↔ the whole architecture; the
ODME/TCG lineage ↔ I3; MAGE ↔ profiles; phase–time XYZ ↔ the declared
`.Phase` sub-name. Rule: every numerical claim in a paper points to a
case notebook that CI can re-run.

## 4. Roadmap (v0.7, order of work — iterate here)

| step | deliverable | depends on |
|---|---|---|
| 1 | `io/export_gmns.py` + gui4gmns hook on grid/SF/Chicago | nothing |
| 2 | `kernels/taplite.py` adapter + audit + column materialization | pip taplite4mpo |
| 3 | notebooks for T1–T7 + 3 cases, nbmake in CI | 1 |
| 4 | `profiles/odme.py` + stages 0–2 benchmarks | 1 |
| 5 | corridor registry + one real corridor end-to-end | 2, 4 |
| 6 | certificate panel proposal to gui4gmns | 1 |

## 5. Open questions for the author

1. TAPLite path output: adapter-side re-pricing (P3, no upstream
   change) vs adding a native `path.csv` dump to TAPLite4MPO — do both?
2. Should `certificates.json` become a lab-wide sidecar convention
   (TAPLite, DTALite, ODME all emitting it)?
3. Which corridor is the first registry entry — I-17 (SPR-790 line)
   or a PeMS district?
4. Notebook hosting: repo-only, or also nbviewer/Colab badges in the
   README?
