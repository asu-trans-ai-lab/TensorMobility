# Morning briefing — overnight agentic run (2026-07-22)

*Everything below is committed and pushed; the suite is at **82 tests,
all green**; the live site and book both rebuilt clean. Format: what
needs your decision first, then what was done, then FYI.*

## Needs your decision (3 items)

1. **LICENSE + CITATION.cff** — the planner reviewer called the
   missing license an adoption blocker ("agencies cannot legally use
   unlicensed code"). This is an author decision I did not make for
   you. MIT would match the ecosystem (TSE-CG source is MIT).
   One `LICENSE` file + a 10-line `CITATION.cff` closes it.
2. **TRMG2 count-based validation** — reviewers noted "certified will
   be read as validated." I added the scoping sentence (certificate =
   convergence/conservation, not calibration) but the convincing
   answer is a link-volume comparison against the model's native
   assignment. Wants your call on which TRMG2 reference output to
   compare against.
3. **Card 1–2 Python counterparts** — the behavior reviewer correctly
   flagged that the land-use and scheduling cards are browser-only
   (now labeled honestly). If you want them promoted to tested package
   code (`behavior/gravity.py`, `behavior/schedule_delay.py` +
   tests), it is a half-day task and removes the label.

## Done overnight

### Simulated review panels (4 personas) → fixes applied
Ran student / MPO-planner / behavior-researcher / TA-engineer panels
over the live explainer, README, and book Ch. 1. Twenty ranked
findings; all fixable ones fixed:
- **TA (clone-and-run audit, verified by execution):** README had no
  install step (added `pip install -e ".[dev]"` + 3-line first
  success); `tabulate` was an undeclared dependency that crashed the
  headline demo on clean installs (added to pyproject); "76 tests" was
  stale (82; 8 skip without local data — stated); phantom
  `tensormobility[demo]` extra in ECOSYSTEM_DESIGN (fixed); personal
  data paths as silent defaults (documented with env vars
  `TENSORMOBILITY_TFB_DATA` / `TENSORMOBILITY_TRMG2_DATA` in README;
  kept the local fallback so your runs are unaffected).
- **Behavior researcher:** cards 1–2 relabeled "illustrative,
  browser-only"; card 2 retitled from "Vickrey scheduling" to
  "schedule-delay (β–γ) preferences" with an honest note that
  congestion is fixed here and endogenous-queue equilibrium is card
  4's fixed point + book Ch. 9–10; card 1 now states gravity ≡ logit
  destination choice (V_j = ln e_j − βc_ij), singly-constrained, and
  that packaged OD demand is exogenous; `choice_graph.py` docstring
  now scopes the sequential≡flat identity (single global θ; nested
  logit breaks it; no overlap correction yet).
- **Student:** all off-by-N demo/panel cross-references renumbered;
  "behavioral column" and "certificate/relative gap/MSA/BPR" now
  defined in place on the cards; book Ch. 1's G-notation collision
  fixed (scales are now $G^{(0)}\ldots G^{(3)}$, grids $G_{3\times3}$
  etc.); the book's state set unified with the site and papers
  (𝒬→𝒟 demand, 𝒰 moved to operator argument) across Ch. 1/2/7 —
  book recompiled, 140 pp, 0 errors.
- **Planner:** "For practitioners" paragraph at the top of the README
  with honest capability scope (BPR-only VDF, stacked-OD multi-class,
  no turn penalties, CSV in / OMX planned, single-thread timings);
  certificate-vs-validation scoping sentence.

### Computational-graph TSE package (refenrences folder) → adopted
Studied the Lu (2021, MIT) TSE codebase end-to-end (Mobile Century
I-880 case; PINN with 6 trainable physical scalars; 9-term loss).
Adopted its best pattern immediately: **sensors as sparse linear
observation operators** on the (x,t) grid — now
`tensormobility/adapters/observation_operators.py` with 6 new tests
(bilinear point operators exact on bilinear fields; trapezoidal
segment integrals exact on linear densities). Full brief + the other
four planned adoptions (LWR-adjoint-is-backprop as book Ch. 15
witness; differentiating through the queue fixed point; ramp-masked
conservation priors for I405N; Mobile Century as a second corridor)
in `docs/TSE_COMPUTATIONAL_GRAPH.md`.

### poloclub
Done and linked: README explainer section and
`docs/EXPLAINER_DESIGN.md` now link poloclub.github.io and the four
studied explainers; the landing page footer credit was already live.

### Superhuman plugin — honest assessment
`.codex/.tmp/plugins/superhuman` is the **Superhuman Mail** MCP plugin
(inbox triage, meeting scheduling, morning briefings). It contains
nothing applicable to improving the book, visuals, or code, and no
mail account is connected in this environment. I borrowed exactly one
thing: this briefing's chief-of-staff format (decisions first, then
done, then FYI).

## FYI
- Live site verified after deploy: five cards run, values match the
  headless certifications (CBD share 4%→46%; mean departure 7:24 with
  γ>β, 8:02 balanced).
- TRB draft (`TM_Paper/TRB_TensorMobility.tex`, 10 pp, 3,670/7,500
  words) untouched tonight — awaiting your read.
- Reviewer findings **not** actioned (besides the 3 decision items):
  OMX demand import, W3 bring-your-own-GMNS worked example (wants a
  small API check first), TRMG2 class-count in the site table.
