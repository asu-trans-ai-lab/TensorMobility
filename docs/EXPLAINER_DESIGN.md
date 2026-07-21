# TensorMobility Explainer — design canon

Lessons extracted from a source-level study of the Polo Club explainer
suite (CNN Explainer, Transformer Explainer, GAN Lab, Diffusion
Explainer), reduced to rules for the TensorMobility / GridCity
explainer panels. Companion to the course book's 12-panel plan
(`website_plan.md`).

## The five rules

1. **Ship a tiny live solver, not screenshots.** Every panel runs its
   real computation in-browser: MSA/FW on a 5×5 GridCity, a point
   queue, a gravity kernel. Our solvers are far smaller than the
   models poloclub ships (Tiny-VGG in tf.js; quantized GPT-2 in
   onnxruntime); there is no excuse for static figures. Corollary
   (Transformer Explainer's temperature slider): cache the expensive
   result and let sliders **re-derive** cheaply — e.g. cache route
   costs, re-derive logit shares per θ without re-solving.
2. **One entity, every view.** Hovering an OD pair highlights its
   desire line on the map, its row in the demand table, and its cost
   curve simultaneously (a single shared highlight state). This is the
   deepest poloclub habit: cross-linked views bound to one entity.
3. **Click-to-expand into live math.** Clicking a link unfolds the BPR
   or queue equation *populated with that link's current numbers* —
   CNN Explainer's kernel-math pattern. Equations never live only in
   prose; they appear inside the expanded view with live values.
4. **Scrub the algorithm's time.** Iterations (equilibrium) and clock
   time (queues) both get slider + play + single-step + slow-motion,
   with numbered "what just happened" tooltips (GAN Lab's training-loop
   narration maps exactly onto MSA/CG iterations).
5. **Widget on top, anchored article below, presets first.** Full-view
   interactive scene; long-form article underneath with bidirectional
   anchors ("learn more" from widget into article, article links back
   into widget state). 3–5 preset scenarios so the first ten seconds
   work with zero user effort. One fixed color semantics reused in
   every panel — ours: **teal = flow/choice, orange = cost/employment,
   green = departures/served, magenta = control/incident**.

## Engineering posture

Poloclub uses Svelte+D3 (CNN, Transformer), framework-free TS (GAN
Lab), or pure vanilla JS + precomputation (Diffusion). For us:
vanilla JS + SVG single-file pages (the current landing page) until a
panel genuinely needs shared-state plumbing; then Svelte, statically
built to `public_gui/`. Precompute-everything (Diffusion's lattice of
JPGs keyed by every slider combination) is the fallback for panels
whose computation is too heavy for the browser — e.g. a Chicago-scale
assignment scrubber can ship as a precomputed iteration lattice.

## Panel roadmap (course book's 12, current status)

| # | panel (book) | live today | next per the rules |
|---|---|---|---|
| 1 | grid scale & topology | grid rows in demo suite | scale slider 3/10/50 with hover |
| 2 | typed axes & contraction | — | axis inspector; "contract an axis" shows what aggregation destroys |
| 3 | flow-through lineage | land-use card (ℒ→𝒟) | click a link-time cell → trace back to paths/OD/activities |
| 4 | agent choice & feasible sets | logit card | edit one agent's attributes → utilities update |
| 5 | activity schedule consistency | departure card | drag an activity on a timeline |
| 6 | fluid queue & cumulative counts | queue card | play/step time; click link → live queue equation |
| 7 | DTA fixed-point iteration | equilibrium card | slow-motion iteration narration (numbered tooltips) |
| 8–10 | tokenization, Q–K–V masks, MoE routing | — | build after the neural chapters' notebooks |
| 11 | projection & certificates | certificate badge | before/after projection diff view |
| 12 | adjoint influence | — | reverse-mode animation link→…→land use |

Current landing-page cards satisfy rules 1 and (partially) 4–5;
the gaps to close first are hover cross-linking (rule 2) and
click-to-expand live math (rule 3) on the land-use and equilibrium
cards, plus preset buttons on every card.
