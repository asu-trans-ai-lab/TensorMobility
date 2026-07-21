# Overnight agentic run — morning review (2026-07-21)

*Suite: 73 → **76 tests, all green**. Everything below is committed and
pushed; numbers are from tonight's certified runs.*

## 1. Rank-economy experiment — THE thesis measurement (D3) ✅

`cases/run_rank_economy.py` + `outputs/rank_economy/` (CSV, figure,
RESULTS.md). Route richness K ∈ {2,4,6,8,12,16} on congested Sioux
Falls (demand ×8), certified reference at gap 1e-7 for every K.

**Both pre-stated hypotheses SUPPORTED:**
- **H-A** (bias grows with support): static-latent relative bias
  2.0e-5 (K=2) → 1.07 (K=4) → **2.95 (K=16)**. The static major/minor
  decoder is catastrophically biased on rich congested supports.
- **H-B** (promotions sublinear): promotions 0 → 38 → … → 70 while
  columns 66 → 528; log-log slope < 1; promotions **per column
  decline** (0.29 → 0.13). Certified active set saturates at 46–62
  columns while support grows 8× — the measured rank economy.
- Metric honesty note: the first H-B verdict was an artifact
  (zero-baseline ratio); replaced with slope + per-column metrics,
  documented in the runner.

**Decision for you:** this is the two-panel figure the reproduction
report asked for — ready to drop into the paper as the E3 centerpiece.

## 2. TRMG2 regional case — E4 stage F1 DEMONSTRATED, F2 running ✅

`tensormobility/adapters/trmg2.py` + `cases/run_trmg2_f1.py` +
`outputs/trmg2/`:
- **F1 transfer validation (full AM bundle)**: 33,963 nodes / 75,939
  links / 3,247 zones / **1,039,117 OD pairs / 458,797 trips**
  (sov+hov2+hov3); loads in 4.6 s; machine-readable report: 0 self-ODs,
  0 unmapped zones, 4 duplicate links flagged.
- **F2-lite certified assignment**: top 300,000 ODs (81.1% of AM
  volume) → full-space-priced gap **2.4e-5 in 49 s** (network lightly
  congested at AM volumes — free-flow paths nearly optimal; the honest
  reading, stated in RESULTS).
- **F2 full (all 1,039,117 ODs)**: launched; result appends to
  `outputs/trmg2/RESULTS.md` when finished.

**Paper impact:** E4/F1 flipped from [architectural] to
[demonstrated] in the integrated draft (§12.5), recompiled clean.

## 3. Paper updated and recompiled ✅

`TM_Paper/TensorMobility_integrated_v1.pdf` (17 pp, 0 errors):
E3 now carries the measured rank-economy curve and verdicts; E4
carries the TRMG2 F1 numbers.

## 4. Small fixes en route
- `dta/latent.py`: accepts both column-cost schemas (toy +
  Sioux Falls path sets), enabling the sweep.
- Rank-economy + TRMG2 regression tests added (local-data skips in CI).

## Decisions waiting for you
1. Rank-economy figure → paper E3 figure slot (recommended yes).
2. Full-OD TRMG2 result: fold into paper §12.5 wording once you read
   the number.
3. Next agentic targets, in my recommended order: F2 super-zone warm
   starts (F2 proper), scale-table runner (D5), typed STBState master
   loop (review P0.3), corridor PINN training (JSQE face).
