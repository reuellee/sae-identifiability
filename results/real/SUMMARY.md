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

A real Pythia-1.4B layer-12 SAE (m = 16384, x8) trains on the L4 with
**FVU ≈ 0.043 (TopK) / 0.056 (L1)** and **L0 = 32**, with dead-latent
resampling online. These are **legitimate real SAEs in the minimal sense**
(wide SAEs trained on real Pythia-1.4B residual-stream activations), which is
the infrastructure milestone. They are **not** established as benchmark-quality
SAEs (whole-repo review §2): the reported FVU is an **in-cache Monte Carlo
evaluation, not held-out generalization** — the "held-out slab" in
`real_train_sae.py` is sampled from the same ~1.4 M-token cache used for
training (training draws ~123 M rows ≈ 88 cache-passes), and the corpus is a
narrow ≤4-novel Gutenberg set with no document-level train/test split. FVU
alone does not measure feature disentanglement, behavior preservation, or
absorption; a proper evaluation (doc-separated test set, loss-recovered /
KL after SAE insertion, an L0–reconstruction Pareto rather than one λ/one k)
is queued.

## First L1-vs-TopK detector pass (raw fact only — semantic headline WITHDRAWN)

*Exploratory, not pre-registered. The whole-repo review (GPT-5.6, 2026-07-24,
`reviews/WHOLE_REPO_REVIEW_GPT-5.6_2026-07-24.md`, finding #1) showed the
original "~27× more redundant/split pairs" interpretation is **not
scientifically defensible as written** — the analysis is architecture-asymmetric
and not reproducibly sampled or opportunity-normalized. The semantic claim is
**withdrawn** pending the corrected, pre-registered comparison below. What
survives is the raw fact.*

**Raw fact.** Under `experiments/real_analyze.py` as run (one seed per arch):

| SAE | FVU | L0 | rate-window latents | cosine-band pairs | flagged pairs |
|---|---|---|---|---|---|
| TopK (k=32) | 0.043 | 32.0 | 9,518 | 1,072 | **936** |
| L1 (λ=5) | 0.056 | 32.0 | 12,573 | 25,158 | **25,041** |

**Why the "27×" and "supports the hypothesis" reading is not established (all
correct, per the review; the analysis script is being fixed):**
1. **Architecture-asymmetric firing threshold.** The script sets θ = **0.0**,
   *not* the rounds-8/9 registered θ = 0.05. Under L1 every tiny positive ReLU
   output counts as "firing"; TopK emits exact zeros outside its selected set —
   so "same detector" is false, and the counts aren't comparable.
2. **Not reproducibly sampled.** `torch.randperm(N)` is unseeded, so the two
   architectures were scored on *different* 50k-token subsamples.
3. **Absolute-cosine geometry.** `|D·D|` lets *negatively* aligned decoder pairs
   enter a detector described as finding "aligned" features.
4. **Raw counts, not opportunity-normalized.** L1 has more eligible latents
   (12,573 → 79.0M possible pairs) than TopK (9,518 → 45.3M). Normalizing by
   opportunity turns 26.75× into **~15.3×**, and pair counts also over-count
   redundant *clusters* (a cluster of size r → r(r-1)/2 pairs — quadratic
   amplification).
5. **The co-firing test barely discriminates here.** 99.5% of L1 cosine-band
   pairs and 87.3% of TopK's are flagged, so **most of the difference already
   lives in the decoder-cosine geometry**, not the absorption-specific co-firing
   signal.
6. **Only the top-15 pairs were saved/inspected** — an extreme-value sample, not
   evidence about the other ~25,000.

Qualitatively the saved examples *do* show L1 feature-splitting (latents on
near-identical tokens — two on `_bow/_in`, two on the `čĊč` newline, two on the
`âĢĻ` apostrophe fragment), which is consistent with the well-documented L1
splitting problem and with the field's move to TopK/BatchTopK. But that is a
qualitative lead, **not** a validated absorption measurement or a robust
architecture-effect size.

**The corrected, pre-registered experiment (queued; the review's
highest-value next step, combined into one).** A matched-seed L1-vs-TopK
real-SAE comparison whose **primary endpoint is causal first-letter absorption
on a held-out SAEBench-style dataset**, with the label-free detector evaluated
only as a *secondary* predictor of those ground-truth outcomes. Minimum design:
≥5 seeds/arch on identical model/layer/corpus/token-order/init/width/budget; a
λ/k sweep compared at matched points on the L0–loss-recovered Pareto frontier
(not one λ vs one k); fixed persisted held-out eval tokens shared by every SAE;
first-letter absorption *with the causal ablation component* as the primary
metric; splitting measured separately (sparse-probe / connected-component
criteria); the detector scored blind (precision/recall/calibration vs the
first-letter labels); pair- *and* cluster-level redundancy reported; and, if
compute permits, a modern control (BatchTopK / Matryoshka / OrtSAE / C2R). This
answers the real question — *does the toy geometry predict a reproducible
difference in causally-validated absorption between real SAE objectives?* — and
validates or falsifies the detector on the phenomenon it is meant to measure.

Artifacts: SAE weights + activations in `gs://sae-identifiability-artifacts-ebd5a273/round11/`;
detector outputs `results/real/sae_pythia-1.4b_L12_{topk,l1}_x8_pairs.json`
(committed — small). Pipeline: `experiments/real_{extract,train_sae,analyze}.py`.
