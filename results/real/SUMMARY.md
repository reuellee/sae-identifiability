# Round 11: real-model SAEs (Pythia-1.4B) — infrastructure + first absorption look

*Started 2026-07-24. Graduates the absorption program from the toy model +
semi-synthetic GPT-2-small injections to **real SAEs trained on a real
model's activations** — the credibility gap every reviewer has flagged, and
the natural next stage of the north star (SAE geometry → **identifiable
codes** → causal features → reusable abstractions → novel-task adaptation).*

## Why Pythia-1.4B on an L4 (not Gemma on an A100)

- **A100/H100 quota is 0** on this account across every region (on-demand and
  spot); the GPU ceiling is a single L4. An A100 needs a quota-increase
  request (owner-filed, slow, often denied for low-usage accounts).
- **The experiment doesn't need an A100.** SAE work = extract activations once
  (a 2B model fits on the 24 GB L4), cache, then train SAEs cheaply on the
  cached vectors. The A100's 80 GB only matters for 8B+ models or much faster
  iteration.
- **Model:** Gemma-2-2B (the SAEBench/Gemma-Scope absorption standard) is
  license-gated (needs an HF token on the box). **Pythia-1.4B** (EleutherAI,
  open, d_model = 2048, 24 layers) is a legitimate real model, a clean ~3×
  scale-up from the GPT-2-small (d = 768) used so far, and unblocked.

## Pipeline (committed)

- `experiments/real_extract.py` — model-agnostic residual-stream extraction
  (Pythia-1.4B layer 12; keeps token ids for later feature labeling).
- `experiments/real_train_sae.py` — standard wide SAE (pre-decoder bias,
  unit-norm decoder, ReLU, **L1 or TopK**, Anthropic-style dead-latent
  resampling); reports FVU / L0 / dead%. (fp16 activations, per-batch
  normalization — an early full-float32 copy OOM-killed the first attempt and
  was fixed.)
- `experiments/real_analyze.py` — scales the project's own rounds-8/9
  label-free pair detector (decoder-cosine band + two-sided co-firing lift +
  overlap veto) to large m by filtering to the rate window first; reports
  flagged-pair counts + interpretable examples (top-activating tokens).
- Ops: `ops/l4_real_pipeline.sh` / `ops/l4_retrain.sh` run self-contained on a
  spot L4 (git-clone the public repo, upload big artifacts to
  `gs://sae-identifiability-artifacts-ebd5a273/round11/`). Weights (~270 MB/SAE)
  and the ~5.7 GB activation cache exceed GitHub's 100 MB limit, so **only
  small stats/findings go to git; big binaries live in GCS.**

## Infrastructure milestone (achieved)

A real Pythia-1.4B layer-12 SAE (m = 16384, x8) trains to high quality on the
L4: **FVU ≈ 0.048** (95% of activation variance reconstructed), **L0 = 32**
(TopK budget), with dead-latent resampling recovering latents online. This
proves the real-model SAE infrastructure works end to end on the available
hardware. Activations (~1.4 M tokens) and weights are in GCS.

## First L1-vs-TopK absorption look (exploratory)

*Exploratory (not pre-registered): the first pass to see whether the detector
finds sane absorbed-pair structure on real SAEs and to calibrate a
confirmatory comparison. A pre-registered L1-vs-TopK comparison (matched
seeds, bars) follows once these numbers are in hand.*

Two matched Pythia-1.4B layer-12 SAEs (m = 16384, x8), same activations, same
detector:

| SAE | FVU | L0 | dead% | rate-window latents | cosine-band pairs | flagged pairs |
|---|---|---|---|---|---|---|
| **TopK** (k=32) | 0.043 | 32.0 | 5.8% | 9,518 | 1,072 | **936** |
| **L1** (λ=5) | 0.056 | 32.0 | 2.1% | 12,573 | 25,158 | **25,041** |

**The detector flags ~27× more candidate pairs in the L1 SAE than in the TopK
SAE** (and ~23× more decoder-cosine-band pairs). Inspecting the top-activating
tokens of the flagged pairs (`*_pairs.json`) shows *what* the difference is:

- **L1 pairs are dominated by feature-splitting** — pairs of latents firing on
  **near-identical** token sets with aligned decoders and high overlap
  (0.75–0.89): e.g. two latents both firing on `_bow/_in` (lift 273), two both
  on the `čĊč` newline, two both on the `âĢĻ` apostrophe byte-fragment. These
  are the same feature represented by multiple latents.
- **TopK pairs are far fewer and are mostly low-overlap (0.03–0.06)
  coincidental decoder alignments** — a function-word latent and a newline
  latent that share decoder geometry but fire on *different* tokens (not
  splitting).

**Interpretation (exploratory).** L1 SAEs exhibit far more redundant / split
feature structure than TopK SAEs on real Pythia-1.4B activations. This
**supports the original round-11 hypothesis — that TopK resists the
absorption/splitting L1 suffers — in exactly the regime round 10 said was the
meaningful one.** Round 10 (isolated, no background) *refuted* the hypothesis
and identified background competition as the missing ingredient; round 11
(real model = background-rich) finds the predicted direction: TopK's hard
budget yields cleaner, less-redundant, more orthogonal features than L1's
shrinkage. This is consistent with the field's move to TopK/BatchTopK and with
the north-star's *identifiable codes* stage — the sparsity mechanism strongly
affects code redundancy at scale.

**Honest caveats (why this is exploratory, not a confirmed count of
"absorption").**
1. **The toy-calibrated detector conflates splitting and absorption at real
   scale.** Its overlap veto (< 0.9) was tuned on the toy model to remove
   feature-splitting doublets, which there sat at overlap ≈ 1.0. Real L1 splits
   sit at overlap 0.75–0.89 and slip *under* the veto, so "25,041 flagged" is a
   **redundant-pair** count (splitting + absorption), not a clean absorption
   count. The detector needs recalibration for real-SAE scale.
2. **Absolute counts are noisy** (the TopK flags are largely coincidental
   low-overlap alignments); the robust signal is the **~27× ratio**, driven by
   L1's heavy splitting.
3. **One SAE per arch, one seed, one λ/k, one layer.** No multiplicity, no
   validation that flagged pairs are causal absorption.

**Confirmatory next steps (to pre-register).** (a) Recalibrate the detector to
separate splitting from absorption on real SAEs (tighter overlap, or an
explicit split-vs-absorb classifier); (b) validate flagged pairs with the
first-letter absorption task (Chanin/SAEBench); (c) multiple seeds and a
λ/k sweep so the L1-vs-TopK contrast has error bars; (d) the causal test —
does ablating a "recovered" child code actually remove the child feature's
effect (the bridge to the north-star's *causally valid features*).

Artifacts: SAE weights + activations in `gs://sae-identifiability-artifacts-ebd5a273/round11/`;
detector outputs `results/real/sae_pythia-1.4b_L12_{topk,l1}_x8_pairs.json`
(committed — small). Pipeline: `experiments/real_{extract,train_sae,analyze}.py`.
