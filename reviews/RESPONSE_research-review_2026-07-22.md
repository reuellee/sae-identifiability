# Response to the research review ("New SAE Identifiability Research" docx)

Item → action map (required deliverable). Review received after Arm 2 had
already run; items about freezing v1.1 pre-Arm-2 are satisfied by the actual
chronology, evidenced by commit order.

| Review item | Action | Where |
|---|---|---|
| §3 capacity wording | "throughout" → scoped wording adopted verbatim; "literally" → "frequently … nominal spare slot"; setup adjacent to conclusion; bg-recovery **not measured** disclosed (m33 weights not retained — open check) | report §8, `results/capacity_m33/SUMMARY.md` |
| §4 Arm A framing | Failed prediction already preserved verbatim + labeled (prereg note outcome section); no-go already scoped to the stipulated single-shared-latent model; **dictionary vs code identifiability** separation added | report §16 reading, §17, README abstract claim |
| §4 noise | Already exploratory-labeled ("noise-as-remedy, new lead" / prereg required); histogram retention + mechanism prereg queued | report §16, §12 list below |
| §5 bootstrap CIs | **Implemented**: pre-registered 10k seed bootstrap (np seed 0) added to `analysis/analyze_prereg_pairid.py`; verified reviewer's estimates (D1 [0.851,1.000], D2a [0.000,0.156], D2b [0.062,0.150], D3 [0.976,0.983], D4 [0.002,0.032]); only D3 CI-established; "point estimate passed" language adopted | scorer + `results/prereg_pairid/SUMMARY.md` + report §17 |
| §6 v1.1 status | Required wording adopted: v1.1 Arm 1 numbers labeled **development-set performance, not confirmation**. Chronology note: v1.1 was frozen in a commit BEFORE Arm 2 ran (prereg amendment + `06d3005`), and Arm 2 executed it without tuning — the review's freeze requirement was already satisfied in fact | SUMMARY, prereg note |
| §7 scaling | FP/million candidate pairs (v1.0 ≈ 214/M; v1.1 dev ≈ 63/M at m=32) + precision at prevalence 1e-3/1e-4/1e-5 (0.81/0.30/0.04) computed & published; width-scaled null calibration, candidate pre-filter, multiple-comparison control listed as open requirements before practical-use claims | scorer + SUMMARY + report §17 |
| §8 confound battery | Open experiment list adopted (nonorthogonal pairs, heavy-tailed coeffs, prevalence > 0.5, TopK/JumpReLU, overcomplete, correlated background) | report §17 closing |
| §9 residual scoping | Wording adopted: D3 validates implementation + pair-ID jointly in the matched orthogonal synthetic model; nonorthogonal/natural generalization untested | SUMMARY + report §17 |
| §10 repo sync | Prereg note status header updated (Arm 1 done / v1.1 frozen / Arm 2 done); provenance table extended with capacity + Arm 1 + Arm 2 rows incl. pre-result commits; report already carries §17; cross-links present | note header, README |
| §11 abstract claim | Adopted near-verbatim, amended with the Arm 2 outcome (which post-dates the review): statistic transfers cleanly, cutoff knife-edge, recorded as registered | README |
| §12 priorities | Arm 2: DONE (frozen, untuned; results published incl. failure at m=128). Remaining queue acknowledged: width-scaling null calibration, nonorthogonal benchmark, encoder/objective robustness, natural-feature evaluation of audit-v3 candidates, noise-mechanism prereg with retained histograms | this file; memory |
| §13 checklist | All wording items applied this commit; empirical items (width scaling, nonorthogonal, natural eval) tracked open | — |
| §14 final instruction | Adopted: detector advertised as a falsifiable research program past an initial matched-synthetic test — not as the constructive solution | README, report §17 |

**One correction to the review:** it asks to "freeze version 1.1 before
running Arm 2" as future guidance; in the repository's actual chronology this
had already happened (v1.1 amendment committed pre-Arm-2; Arm 2 ran the frozen
detector without inspection-driven changes, and its m=128 failure is published
unadjusted). The review was drafted against the pre-Arm-2 state.

---

## Round-8 review (docx, received mid-run): pre-collection amendments

Applied and committed BEFORE round-8 collection (run was in flight; no
round-8 result had been seen):

| Item | Action |
|---|---|
| P0 width-specific endpoints | T1 → T1a/T1b per width, pooled secondary (prereg amendment §1 + scorer) |
| P0 matcher defects | S1 matcher replaced pre-first-use: bijective pair score + all-seed clustering, ≤1 candidate/seed/cluster (amendment §5) |
| P0 all-pairs specificity | Faithful full-scan flag proportion + flags/M added to scorer; no-injection real-background null queued as round 8b (cannot be added mid-run) |
| P1 stage separation | Orientation accuracy (rate-derived), child-residual under auto vs oracle orientation, ρ̂ conditional on orientation — separate endpoints in scorer + report §17 |
| P1 claim narrowing | "specificity transfers perfectly" → oracle-pair statement; "natural-absorption candidate list" → "real-background candidate list"; width effect described as calibration dependence; E1 scoped as same-domain resampling-stability |
| P1 E2 conflation | Acknowledged: E2 = proportional-scale family only; fixed-(d,n_bg) width sweep + overcomplete m>d queued as 8b |
| P2 docstring sync | round8_realdata.py docstring corrected to the actual 64-run design |

Arm 2's L_HI=2.0 outcome remains the completed pre-registered result; v1.2 is
a new detector version on fresh runs, per the review's framing.
