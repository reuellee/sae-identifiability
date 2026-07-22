# Response to the fourth external review (per-item, requested format)

Context the review lacked: it was drafted against the pre-round-8 tree, but
round 8 had already executed with the pre-collection amendment `69ca642`
(committed before collection; collection at 21:54 UTC per `~/retry_round8.log`,
mirrored in `ops/`). Items are answered against actual chronology.

---

**Issue 1: Preserve prereg + dated amendment.**
Decision: accepted.
Files changed: `notes/prereg-round8-scaling-robustness.md` (Amendment 2 appended; original + Amendment 1 untouched).
Commit hash: `1efd18a`.
Exact change: dated Amendment 2 with per-item classes (a) pre-collection / (b) post-run-from-frozen-artifacts / (c) round-8b.
Pre-results status: Amendment 1 completed pre-collection (`69ca642`); Amendment 2 post-run by necessity, honestly labeled.

**Issue 2: Width-specific recall endpoints.**
Decision: accepted — already implemented pre-collection.
Files changed: prereg note §Amendment 1, `analysis/analyze_round8.py`.
Commit hash: `69ca642` (pre-collection).
Exact change: T1a recall(m=128) ≥ 0.90 AND T1b recall(m=256) ≥ 0.90, each confirmatory with per-width bootstrap; pooled secondary. Outcome: 24/24 at each width (`68f444c`). Unconditional yield = conditional here (formation 48/48).
Pre-results status: completed (pre-collection).

**Issue 3: Geometrically matched negative + shuffled-firing null.**
Decision: modified — shuffled null computed now from frozen artifacts; matched in-band negative requires new data.
Files changed: `analysis/r8_specificity_recompute.py`, `results/round8/SUMMARY.md`; prereg Amendment 2(c) registers the matched control for round 8b.
Commit hash: `1efd18a` (code) + this commit (results).
Exact result: **shuffled-firing null = 0.00 flags/SAE in all 64 SAEs** at both widths and conditions — every flag is dependence-driven.
Reason for deviation: an in-band non-absorbed pair on real activations needs a new trained condition (round 8b); the shuffled null was derivable immediately and is the cleaner dependence null per the review's own §5.
Pre-results status: shuffled null completed (post-hoc, disclosed); matched control pending (8b).

**Issue 4: Exact / oracle-touch / full-scan specificity fields.**
Decision: accepted; mismatch acknowledged.
Files changed: `analysis/r8_specificity_recompute.py`; docstring note in `experiments/round8_realdata.py`; `results/round8/SUMMARY.md`.
Commit hash: `1efd18a` + this commit.
Exact result: oracle-touch ≡ exact-pair in every SAE (absorbed: only flag touching an oracle latent is the true pair, 1.00/SAE; faithful: 0.00); full-scan 4.04/3.00 (m=128) and 10.83/9.25 (m=256) per SAE.
Reason for deviation: computed post-run from frozen weights + seeded eval streams rather than pre-run (the mismatch was found after execution); the coincidence of touch and exact metrics means the registered concept's verdict is unchanged.
Pre-results status: completed (post-hoc, disclosed).

**Issue 5: Operational candidate-burden endpoints.**
Decision: accepted — registered in Amendment 2(c) for round 8b (flags/M, proportion of control SAEs with ≥1 flag, precision in planted conditions, excess vs shuffled null); current values reported descriptively (`68f444c`, this commit).
Pre-results status: pending (8b), descriptive values published.

**Issue 6: Orientation + auto-vs-oracle recovery primary endpoints.**
Decision: accepted — implemented pre-collection; measured and published.
Commit hash: `69ca642` (registration), `68f444c` (results: orientation 0.88/0.75; child recovery 0.990±0.001 oracle vs 0.948/0.909 auto; ρ̂ 0.755/0.746).
Pre-results status: completed (pre-collection registration).

**Issue 7: E1/E2 scope labels.**
Decision: accepted — pre-collection labels "same-domain resampling stability" / "proportional scale family" (`69ca642`); reviewer's exact phrasings ("held-out seed-level replication", "proportional scale-family null calibration") adopted in SUMMARY/PAPER (`1efd18a`, this commit).
Pre-results status: completed.

**Issue 8: Stability matcher replacement.**
Decision: accepted.
8.1 seed-zero anchoring + 8.2 bijective score + 8.4 clustering: fixed pre-first-use in `69ca642` — no stability result was ever produced with the anchored matcher.
8.3 injection-touch exclusion: valid new catch — the v1 run (`841e8cd`) excluded only the exact planted pair. Fixed in `1efd18a`; **v2 rerun result: identical clusters at every seed and width** (zero contamination, empirically). v2 (`s1_stability_v2.log` + `s1_clusters_m*.json`, machine-readable per 8.4) supersedes v1, which is retained as a record.
Pre-results status: 8.1/8.2/8.4 pre-first-use; 8.3 post-first-use, superseding rerun completed.

**Issue 9: Headers, version labels, provenance hashes.**
Decision: accepted.
Files changed: `experiments/round8_realdata.py` (v1.2 labels; docstring corrected in `69ca642` for seed counts, mismatch note in `1efd18a`), `README.md` (exact hashes: locks `e586f02`/`1bbca24`/`a539c76`/`69ca642`; results `465a139`/`319fa1f`/`68f444c`; v1.1 freeze `465a139` strictly precedes Arm 2 `319fa1f`).
Commit hash: `1efd18a`.
Pre-results status: completed. Note: label edits to the executable are cosmetic post-run; the executed copy is the one pushed at `a539c76`+`69ca642`.

**Issue 10: Remove "perfect specificity" / "natural absorption" language.**
Decision: accepted.
Files changed: `report.md` §17, `PAPER.md`, `results/prereg_pairid/SUMMARY.md` (geometric-gate scoping sentence adopted near-verbatim; "width-dependent calibration failure"; "real-background candidate list" was already adopted in `69ca642`).
Commit hash: `1efd18a`.
Pre-results status: completed.
