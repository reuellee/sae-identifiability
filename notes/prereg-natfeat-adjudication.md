# Pre-registration: natural-feature adjudication of S1 seed-stable candidates

**Date:** 2026-07-23. **Status:** locked before generating any interpretation;
this note + `analysis/natfeat_adjudicate.py` are the pre-results commit.
Successor to the round-8 S1 result (`results/round8/SUMMARY.md` §S1) and queue
item #1 of `RESEARCH_PLAN.md`. **Type:** exploratory characterization with a
**pre-registered classification rule** — the point of locking it is to stop
post-hoc storytelling, a known failure mode of SAE-latent interpretability.

## Question

The S1 scan produced seed-stable pairs of latents that flag the pair-detector
on **real** GPT-2 layer-6 activations (no injected pair): 3 clusters at m=128
(two in 8/8 seeds), 12 at m=256 (several 8/8, incl. the mutual 4-latent clique
{51, 54, 107, 172}). Their status is unknown. Are they:

- **(A) natural absorption** — one latent is a *specialization* whose firing is
  contained in a broader parent latent, and whose semantics refine the parent's
  (the program's first wild-caught absorption pair); or
- **(B) exclusive / correlated feature family** — two genuinely distinct but
  co-firing real features (the disclosed CDX equivalence class the detector
  cannot separate from absorption by co-firing alone); or
- **(C) uninterpretable / artifact** — no coherent token structure.

## Data & procedure (locked)

- **Activations:** `experiments/activations_l6.pt` (500k GPT-2-small layer-6
  residual tokens, War & Peace + Sherlock Holmes), normalized **identically to
  training** (mean-subtract, scale so mean row-norm = √768). **Pure background,
  no injected pair** — we are characterizing the *real* latents.
- **Token strings:** regenerated from the exact same corpus fetch + chunking +
  500k truncation as `extract_activations.py` (deterministic), decoded with the
  HF GPT-2 tokenizer. Alignment is 1:1 with activation rows; verified by
  `len(strings) == 500_000` and by re-deriving first letters and matching
  `natural_absorption_eval.get_letters`.
- **SAE weights:** `results/prereg_pairid/weights_arm2_m{128,256}.pt`. For each
  cluster, the **representative SAE = the minimum seed index present in the
  cluster** (pre-specified, no cherry-picking). Latents (i, j) are that seed's
  entry in `results/round8/s1_clusters_m{M}.json`.
- **Firing:** f = ReLU(W_i · x + b_i) over all 500k tokens; binary fire =
  f > θ = 0.05 (detector θ). Compute for each candidate latent.

## Per-cluster measurements (locked)

For latents i, j (parent := higher fire-rate, child := lower):

1. **rate_i, rate_j** (fire fractions).
2. **Directional containment:** `C_par|child = P(parent fires | child fires)`
   and `C_child|par = P(child fires | parent fires)`.
3. **Decoder cosine** cos(D_i, D_j) (band-check, ∈ [0.45, 0.90] by construction).
4. **Top-activating tokens:** top-50 token *positions* by f for each latent →
   decoded strings; plus the most-common token strings among the top-200
   positions (with a one-token left-context sample for readability).
5. **Residual-direction semantics:** r = D_child − (D_child·D_parent)·D_parent,
   normalized; top-50 tokens by ⟨x, r⟩. Characterizes what the child adds.

## Classification rule (locked — applied per cluster)

Primary discriminator is **quantitative containment**; semantics is
confirmatory. A cluster is:

- **A (natural absorption)** iff **BOTH**:
  (i) **asymmetric nesting:** `C_par|child ≥ 0.80` AND `C_child|par < 0.80`
      (child's firing implies parent's, not vice versa); AND
  (ii) **semantic specialization:** the child's dominant top-tokens are a
       coherent *subset / refinement* of the parent's (parent = broad class,
       child = specific member).
- **B (correlated family)** iff high co-firing but **NOT** asymmetric
  (both conditionals ≥ 0.80, or mutual-exclusion), AND the two latents have
  **distinct** coherent semantics (no subset relation).
- **C (artifact)** iff neither latent has coherent top-tokens (no dominant
  token type; top tokens dominated by unrelated/positional/whitespace noise).
- **Unresolved** — anything that does not cleanly meet A, B, or C is reported
  as unresolved. Borderline cases are NOT forced into a bucket.

For the **4-latent clique** {51, 54, 107, 172} (m=256): compute the full 4×4
containment matrix. A hierarchy (nested containments) supports absorption;
a flat, symmetric co-firing block supports a single correlated family (B).

## Semantic-judgment protocol (honesty guardrail)

The subset-vs-distinct-vs-incoherent call (ii, and B/C semantics) is an
interpretability judgment on the top-token lists. To keep it disciplined:

- The **quantitative containment** metric (1–2) is the PRIMARY, pre-registered
  discriminator and is reported for every cluster regardless of the semantic
  read.
- Semantic reads are made by (a) the author and (b) **Gemini as an independent
  second adjudicator** (`gemx`), each given only the raw top-token lists and the
  A/B/C definitions, blind to the containment numbers. Disagreements → the
  cluster is **Unresolved**.
- **Evidence tier (locked):** single-latent interpretability on a 500k-token
  two-book literary corpus is **weak/suggestive** evidence, not proof. No
  cluster will be reported as a confirmed natural-absorption feature; the
  strongest available claim is "natural-absorption *candidate*, tier-2 evidence."
  A confirmed wild-caught pair would additionally require causal ablation and
  cross-corpus replication (out of scope here; queued if any A survives).

## Registered readouts

- **R1:** per-cluster containment table (all clusters, both widths).
- **R2:** A/B/C/Unresolved label per cluster, with the token evidence.
- **R3:** 4-clique containment matrix + its verdict.
- **R4 (descriptive):** count of A-candidates. If ≥1 survives A with author +
  Gemini agreement → queue causal/cross-corpus confirmation. If 0 → the S1
  candidates are (as the null expects) correlated real-feature families, and the
  detector's inability to separate them from absorption on co-firing alone is
  reported as the operative limitation (consistent with the dict-vs-code and
  CDX-equivalence-class disclosures already in the paper).

**Guardrails:** `reviews/EXTERNAL_REVIEW_GPT-5.6_2026-07-22.md` §5 language
constraints apply; do not upgrade "candidate" to "confirmed"; report failures.
