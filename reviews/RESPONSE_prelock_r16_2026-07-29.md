# Response to the round-16 pre-lock reviews (Gemini + GPT, 2026-07-29)

Both reviewers returned REVISE-BEFORE-LOCK. Every finding was adjudicated
before lock; the lock commit contains the revised prereg, scorer, evaluator,
and driver. Verification state after revision: evaluator self-test 22/22
branches; scorer shared-key fidelity vs the frozen 13b scorer 0 mismatches on
a real two-SAE input; new counters and the matched-budget pass exercised on a
synthetic construction; `bash -n` on both drivers.

## Gemini findings

| # | Finding | Disposition |
|---|---|---|
| 1 | P1 verdict blind to the D-control (mechanical tautology) | **ADOPTED.** Registered D-gate: `rise(fam_fire_absent) ≤ 0.5 × fall(absorption)` or the verdict becomes CONFIRMED-BUT-MECHANICAL. Additionally superseded in strength by GPT P1.3's matched-budget D-sensitivity (below). |
| 2 | MC lacks the absolute L0 band | **ADOPTED.** MC now requires the ratio AND both cells within [0.75,1.25]×target; self-tested (ratio-passes/band-fails case). |
| 3 | P2 survivor bias (unpaired clean-letter sets) | **ADOPTED.** P2 is letter-paired within seed over letters clean in both cells; paired-letter counts reported. |
| 4 | Interior absence silently passes the SHA pin | **ADOPTED (as clarification).** The prereg now distinguishes mismatch (= contamination, fails conformance) from absence (= named deficiency, printed in the gate line, non-fatal — the cell is descriptive annotation, not an endpoint). |

## GPT findings

| # | Finding | Disposition |
|---|---|---|
| P1.1 | Gates 1/2/4 gate nothing | **ADOPTED.** `global_ok = g1∧g2∧g4`; failure suppresses P1/P2/P3 to UNINTERPRETABLE (self-tested). *Adapted:* the evaluator still exits 0 — the driver must ship artifacts after the verdict, and the verdict text is the deliverable; a nonzero exit under `set -e` would strand results on the box. |
| P1.2 | Cache identity unchecked (the r15 wrong-layer class) | **ADOPTED.** Scorer records `words_model`/`words_layer` from the words cache and `eval_src` from the trainer's stats; gate 1 fails rows on any mismatch or non-held-out eval. Driver verifies all three input caches against md5s pinned at lock time and aborts on mismatch. *Adapted:* training `steps` is not added to the weight blob — that would modify the registered round-12 trainer; steps remain driver provenance (STEPS env + shipped train.log). |
| P1.3 | D-control insufficient; mechanical favoring | **ADOPTED (the strong form).** Registered matched-budget D-sensitivity: every SAE re-scored under a common top-16 firing mask (native reconstruction/retention), same seed-clustered contrast; SURVIVES licenses representational language, DOES NOT SURVIVE restricts P1 to "L0 association". The proposed seed-level specificity contrast is not separately registered — the matched-budget analysis dominates it and metric sprawl has its own multiplicity cost. |
| P1.4 | Resume violates the registered wipe; N≠32 not fatal | **ADOPTED.** Lock-bound RUN_MANIFEST: wipe exactly once per lock, resume refused across locks, name gate aborts unless exactly 32 survive. Cached λ files are wiped with the manifest. |
| P1.5 | L1 dose bands/calibration not enforced | **ADOPTED.** Band is in MC (Gemini 2); λ equality vs the shipped calibration output is gate-checked when the file is present (absence = named deficiency); with bands enforced the effective minimum ratio is 56/18 ≈ 3.11 as noted. |
| P2.1 | P4 pooling includes MC-failed arch; CI labeling | **ADOPTED.** P4's interpretive means restrict to MC-passing arches; failed-arch CIs labeled uninterpretable. |
| P2.2 | "SHA256" is a 16-hex prefix | **ADOPTED (wording), REBUTTED (full digests).** All prints/prereg text now say "sha256 prefix". Full digests would break commensurability with the prefixes recorded in `round13b_results.json`, which are the interior pins' source of truth; 64-bit prefixes identify against accidents, which is the threat model here. |
| P2.3 | Blob/filename identity incomplete | **ADOPTED** for arch, seed, eval_src (gate 1). Steps: see P1.2 adaptation. |
| P2.4 | P3 vocabulary only in code | **ADOPTED.** Vocabulary registered in the prereg; P3 sign branches self-tested. |

## Verified-by-reviewer items carried forward

GPT independently verified: the scorer diff vs 13b contains only the
docstring, the additive counters, and the default output name; the counters
are probe-independent; the rename/anti-contamination sequence is sound;
gate 3 correctly scopes to P3.
