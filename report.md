# Feature Absorption in Sparse Autoencoders is a Phase Transition: an Exact Boundary, a Non-Identifiability Wall, and GPU-Verified Predictions

*Session report — 2026-07-21. Theory machine-verified symbolically; experiments run on GCP `dev-gpu` (NVIDIA L4).*

## Executive summary

Sparse autoencoders (SAEs) are the workhorse of mechanistic interpretability: they decompose neural activations into "features," and safety-relevant claims are increasingly built on top of those features. The known failure mode called **feature absorption** — the SAE merges a parent concept and a child concept into one latent, so the parent feature appears to "turn off" exactly where it matters — has so far been an empirical phenomenon. This report makes it a theorem:

1. **A non-identifiability wall (Theorems 1, 1b).** When a child feature *never* occurs without its parent, the hierarchical ("faithful") and merged ("absorbed") feature ontologies generate *identical* data distributions — no method, present or future, can distinguish them from activations alone. Moreover, at this wall the absorbed dictionary is the *unique global optimum of the SAE objective over all dictionaries of any size* (two-line proof via ‖f‖₁ ≥ ‖Df‖). Absorption at the boundary is not a bug; it is both the information-theoretic truth and the objective's exact preference.

2. **An exact phase boundary (Theorem 2, machine-checked).** Away from the wall, let ε be the probability the child occurs alone, q the probability parent and child co-occur, and λ the SAE's L1 coefficient. The SAE objective *globally prefers the wrong (absorbed) dictionary* precisely when

   **ε < ε\*(λ, q) = λq(8 − 4√2 − λ) / (2(1 − (2 − √2)λ)) ≈ 1.17 λq** (small λ).

   In the band 0 < ε < ε\*, the truth is identifiable in principle but the objective is *misaligned with it* — more compute, better optimizers, and more data provably cannot fix absorption there. Only changing the objective (or λ) can.

3. **Absorption is continuous, not binary (Result 3).** Global optimization over 2-latent dictionaries shows the optimum interpolates: the child latent's direction tilts smoothly from the composite (45°) toward the true child (90°) as ε grows through the transition region — an exact mechanism for the empirically reported "feature hedging" phenomenon.

4. **With spare capacity, the objective wants *composition* (Remark, §4).** For any ε > 0 the unconstrained optimum is the redundant triple {parent, child, composite}: even ideal SAE training on ideal data does not return the generative feature ontology — it adds frequent feature combinations as first-class latents. Capacity scarcity (real SAEs cannot afford a latent per combination) is what converts this into the absorption tradeoff of Theorem 2.

5. **GPU experiments agree — and falsified one theorem productively.** SGD-trained SAEs track the predicted transition across a 135-run grid plus a 1,040-run fine measurement (transition at 0.58–0.70·ε\* with the λq scaling collapse intact; the lead factor is consistent with encoder shrinkage). A 144-run oracle-init study splits classical recovery into three regimes: trainability-limited (k ≤ 8, fixed by dead-latent reinit), optimization-barrier (k = 16), and a genuine stability limit (k ≥ 24, where even the true dictionary decays).

6. **The pre-registered coherence-penalty prediction was refuted, and the refutation became the deepest result (§7.1b).** The predicted critical penalty β\* misses the true global optimum: an **anti-rotated absorbed pair** — the composite keeps absorbing while the parent latent rotates to zero the pair's coherence. This generalizes to a **no-go theorem** (general-(d, m) derivation with proofs: `theory/general_no_go.md`): every penalty that is a function of pairwise decoder inner products is *constant on the manifold of orthonormal frames*, so none — Frobenius, |cos|, OrtSAE-style max-cos², any future variant — can distinguish a faithful orthogonal frame from an anti-rotated absorbed one. Quantitatively (corrected, machine-verified): the penalty shrinks the absorption region at most ~4× (0.0486 → 0.0112 at β ≈ β\*), never eliminates it, worsens when overdosed (ε\*\*(β) increases to a penalty-form-independent limit ε\*\*(∞) = 0.0159), and the absorbed configuration survives as a *local trap* up to ε ≈ 0.05 — nearly the original unpenalized boundary. GPU-validated at 160 runs (basin fractions cross at ≈ 0.6·ε\*\*; multistability persists exactly where the trap analysis predicts). The general derivation adds a **domain boundary**: this failure regime is precisely p₀ < p₀\* ≈ √2·q (parent-solo rarer than √2× co-occurrence — the case for all experiments here); for p₀ > p₀\*, the anti frame's 2-sparse parent-solo codes cost it the competition and coherence penalties genuinely work, even at ε = 0.

**Practitioner rule of thumb that falls out of the theorem:** to resolve a child feature that occurs alone with probability ε against a parent it co-occurs with at rate q, the sparsity coefficient must satisfy **λ ≲ ε / (1.17 q)**. Hierarchical concept pairs with tiny ε are unresolvable at any practical λ — and at ε = 0, unresolvable at all.

---

## 1. Motivation: when is an SAE feature real?

Interpretability's load-bearing assumption is that SAE latents correspond to the model's "real" features — safety-relevant claims (this latent detects deception, this one mediates a capability) are built on top of that correspondence holding. Feature absorption is the sharpest known counterexample: a child concept silently merges into a parent latent, so the parent appears to "turn off" exactly on the inputs where the distinction matters most, and a practitioner reading the dictionary has no way to tell absorbed latents from faithful ones by inspection.

The phenomenon has been documented empirically (Chanin et al., §10) but not characterized quantitatively: *when*, as a function of the SAE's own hyperparameters and the data's co-occurrence statistics, does absorption happen — and can it be fixed by changing the objective rather than by hoping optimization avoids it? This report answers both questions exactly, in a toy model low-dimensional enough to solve in closed form and cheap enough to validate on a single GPU in minutes per experiment, then tests the answers against SGD-trained SAEs and pre-registered remedies.

## 2. Model

Work in the plane spanned by two orthonormal feature directions **a_p** (parent) and **a_c** (child); everything else in the activation is orthogonal background handled independently. Each sample activates:

- both features (coefficients 1) with probability **q** — "joint",
- the parent alone with probability **p₀** — "parent solo",
- the child alone with probability **ε** — "child solo",
- neither, otherwise.

An SAE with unit-norm decoder columns D and nonnegative codes f pays the population loss
L(D) = E[ min_{f ≥ 0} ‖x − Df‖² + λ‖f‖₁ ].
Two natural 2-latent dictionaries compete inside the pair plane:

- **Faithful:** {a_p, a_c} — the generative truth.
- **Absorbed:** {a_p, (a_p + a_c)/√2} — one latent for the *combination*, mirroring absorbed latents observed in real SAEs.

## 3. Theorem 1 — the non-identifiability wall (ε = 0)

**Claim.** At ε = 0 the following two generative models induce the *same* distribution over activations:

- **M1 (hierarchical, 2-sparse):** features {a_p, a_c}; both fire together (prob q); parent fires alone (prob p₀).
- **M2 (flat, 1-sparse):** features {a_p, a_m} with a_m = (a_p + a_c)/√2; a_m fires alone with coefficient √2 (prob q); a_p fires alone (prob p₀).

*Proof.* Both place mass q at the point a_p + a_c = √2·a_m and mass p₀ at a_p. ∎

Consequently no procedure operating on activations can decide whether the "child" is a real separate feature or whether the composite *is* the feature — the feature ontology itself is undetermined by the data. Note M2 is strictly sparser on average, so any sparsity-seeking objective selects the absorbed ontology; this is the seed of Theorem 2.

**Theorem 1b (at ε = 0, absorption is the global optimum over all dictionaries).** For any dictionary D with unit-norm columns (any number of latents) and any code f ≥ 0, the triangle inequality gives ‖Df‖ ≤ ‖f‖₁, so for a sample x with r = ‖x‖ ≥ λ/2:

‖x − Df‖² + λ‖f‖₁ ≥ min_{t ≥ 0} (r − t)² + λt = λr − λ²/4,

with equality iff the active latents all point exactly at x/r. Summing over the ε = 0 event distribution, L(D) ≥ p₀(λ − λ²/4) + q(√2λ − λ²/4), and this bound is attained exactly by the absorbed dictionary {a_p, (a_p + a_c)/√2} — and only by dictionaries containing those two directions. Absorption at the wall is not just preferred over the faithful dictionary; it is the unique optimum over *every* dictionary of every size. ∎

## 4. Theorem 2 — the exact phase boundary (machine-checked)

For each event, the optimal nonnegative code is a 2D nonnegative lasso; enumerating KKT cases gives closed-form event losses (verified symbolically by exhaustive case enumeration in `verify_absorption_theory.py`; all six forms confirmed to 1e-9 over λ ∈ [0.01, 0.5]):

| event | faithful | absorbed |
|---|---|---|
| joint (prob q) | 2λ − λ²/2 | √2λ − λ²/4 |
| parent solo (prob p₀) | λ − λ²/4 | λ − λ²/4 |
| child solo (prob ε) | λ − λ²/4 | 1/2 + √2λ/2 − λ²/4 |

The absorbed dictionary wins on joint events (one latent instead of two → lower L1) and loses on child-solo events (the child direction sits outside its nonnegative cone → irreducible ½ reconstruction error). Parent-solo terms cancel, so **p₀ plays no role**. Subtracting:

L(faithful) − L(absorbed) = q[(2 − √2)λ − λ²/4] − ε[1/2 − (2 − √2)λ/2],

which is positive (absorption preferred) iff

**ε < ε\*(λ, q) = λq(8 − 4√2 − λ) / (2(1 − (2 − √2)λ)).**

For small λ: ε\* ≈ (4 − 2√2)·λq ≈ 1.172·λq. The formula was confirmed symbolically (sympy `solve` reproduces it exactly) and numerically (the winner flips between ε = 0.9ε\* and 1.1ε\* on a global scan).

**Interpretation.** For 0 < ε < ε\* the two models of Theorem 1 now have *different* distributions — the truth is identifiable — yet among per-pair 2-latent dictionaries the objective prefers the absorbed one. This is an **objective-misalignment regime**: absorption there is not a local-minimum problem, an optimizer problem, or a data problem. Training longer makes it *more* wrong.

**Scope: capacity.** Theorem 2 is the *capacity-limited* statement — two latents available for the pair. That is the operative regime for real SAEs, which cannot afford a latent per feature *combination* (combinations grow exponentially). When capacity is spare, the lower-bound technique of Theorem 1b settles the unrestricted problem too:

**Remark (redundant triple).** For any ε > 0 and λ < 1/(2 − √2) ≈ 1.71, the global optimum over unconstrained dictionaries is the *redundant triple* {a_p, a_c, (a_p + a_c)/√2}: joint events fire the composite latent, solo events fire their own latents, and each event attains the per-sample lower bound λ‖x‖ − λ²/4. Two readings: (i) even with unlimited capacity and perfect optimization, the L1-SAE's optimal feature set is **not** the generative ontology — it adds frequent combinations as first-class latents ("composition"); (ii) in experiments with spare capacity, any absorption observed below ε\* reflects *optimization dynamics* rather than the unconstrained objective — a distinction the experiments below measure directly (the analysis logs whether a separate ~90° child latent coexists with the ~45° composite).

## 5. Result 3 — absorption is continuous ("feature hedging" mechanism)

Scanning *all* unit-norm 2-latent dictionaries (angles on a fine grid with local refinement):

- At ε = 0 the global optimum is exactly the absorbed dictionary (angles (0°, 45°)).
- For ε near and above ε\*, the global optimum is neither pure strategy: the child-side latent sits at an intermediate angle (e.g., 69° at ε ≈ 0.9ε\*, 80° at 2ε\*, 85° at 4ε\* for λ = 0.1, q = 0.2), converging to the faithful 90° as ε grows.

So the transition is a continuous tilt of the learned direction — the latent "hedges" between child and composite — with ε\* marking the exact point where the pure absorbed strategy loses to the pure faithful one. The full theoretical angle curve φ\*(ε; λ, q) is computed in `theory_curves.py` and overlaid on the experimental results below.

## 6. Proposition 4 — the far (sufficiency) side, for context

Classical sparse-coding theory gives the matching positive regime: if all features have mutual coherence μ and activations are k-sparse with k < (1 + 1/μ)/2, every sample's sparse code is unique (Donoho–Elad), and with sufficiently diverse supports the dictionary is identifiable up to permutation and scale (Hillar–Sommer). These are worst-case bounds; random (typical) dictionaries recover far beyond them, which Experiment A quantifies.

## 7. Remedies, with proofs — closing the loop

Characterizing the failure is half a solution; the other half is an objective that provably avoids it. Two remedies were analyzed exactly (`verify_remedies.py`, `matryoshka_multichild.py`), one simple and new-ish, one being the field's leading method.

### 7.1 Coherence penalty: an exact threshold — **[SUPERSEDED by §7.1b: the GPU experiment refuted this analysis and the corrected scan shows why]**

Add β·Σ_{i<j}⟨d_i, d_j⟩² to the SAE loss (decoder-Gram penalty). The faithful dictionary pays nothing (orthonormal truth); the absorbed dictionary pays β/2; the redundant triple pays β. Machine-verified thresholds *for the faithful-vs-absorbed comparison* (the flaw, exposed in §8: these are not the only competing configurations):

- **β > β\*(λ, q) = λq(8 − 4√2 − λ)/2 ≈ 1.17·λq** makes the faithful dictionary beat the absorbed one **for every ε ≥ 0** (ε = 0 is the worst case, and the derivative in ε is strictly negative), and β\*/2 already eliminates the redundant triple.
- The same continuity caveat applies as in Result 3: just above β\* the global optimum is a partially-tilted dictionary (73° at 1.2β\*, 83° at 3β\* for λ = q = 0.2); full faithfulness is approached asymptotically. β\* is the exact crossover of the pure strategies.
- Verified to transfer to the two-child model unchanged: at β ≥ β\* the faithful triple beats both absorbed configurations at every ε tested, including ε = 0.
- Scope caveat: the penalty is only unbiased when true features are (near-)orthogonal; genuinely correlated feature *directions* would be distorted by large β. The usable window is β\* < β ≪ the scale set by true-feature coherence.

Note what β\* does at the ε = 0 wall: the data cannot distinguish the hierarchical from the flat ontology (Theorem 1), and the penalized objective resolves the tie by preferring the orthogonal (hierarchical) reading — a modeling choice made explicit, not a violation of Theorem 1.

### 7.1b Corrected theory: the anti-rotation evasion (found by pre-registered falsification)

The GPU experiment (§8, C1) refuted the §7.1 prediction, and running the refutation to ground exposed the error: §7.1 compared only the *faithful* and *absorbed* pure strategies. Scanning the exact population loss over **all** unit-norm 1- and 2-latent in-plane dictionaries (`theory/theory_merged.py`; full angle range including negative angles) shows that for β ≥ β\* and small ε the true global optimum is neither — it is the **anti-rotated absorbed pair** {≈ −40°, +46°}: the composite latent continues to absorb the child at ≈ 45°, while the *parent* latent rotates to the far side of the parent direction, driving the pair's inner product toward zero. The penalty is evaded, absorption survives.

The corrected picture (λ = q = p₀ = 0.2; all values from the exact scan, independently re-derived adversarially in `theory/verify_anti_rotation.py`):

- **A no-go theorem, penalty-form independent (`theory/no_go_theorem.py`; general derivation with proofs in `theory/general_no_go.md`).** Let the penalty be *any* function of pairwise decoder inner products with orthogonality optimal — Frobenius Gram, |⟨d_i,d_j⟩|, OrtSAE-style max-cos², all published coherence penalties. On the manifold of exactly orthonormal frames every such penalty is *constant*, so as β → ∞ the optimum is decided purely by reconstruction + L1 cost among orthonormal frames: the penalty's functional form drops out. Since the faithful frame {0°, 90°} and the anti-rotated absorbed frame {−45°, 45°} are both orthonormal, **no pairwise-coherence penalty, of any form, at any strength, can distinguish them** — the β → ∞ boundary ε\*\*(∞) is a penalty-form-independent constant. The general-(d, m) derivation rests on an exact closed form on the orthonormal manifold, L₀(D) = E‖x‖² − Σᵢ E[(⟨dᵢ,x⟩ − λ/2)₊²], which also yields the mechanism in one line: the soft-threshold tax λ/2 is paid once per active coordinate, so concentrating a co-occurrence event in one latent beats splitting it — absorption lives in the codes, invisible to decoder geometry.
- **Domain of validity — a new critical occurrence ratio (from the general derivation).** The no-go's bite depends on the hierarchy's occurrence statistics: the anti-rotated frame reconstructs *parent-solo* events 2-sparsely, paying the λ/2 tax twice, so its advantage collapses when parent-solo events are frequent. Closed form: the evasion defeats the penalty iff **p₀ < p₀\*(λ, q) = q·[(2−√2) − λ/4]/[(√2−1) − λ/4]**, with p₀\*/q → √2 as λ → 0. Below p₀\* (co-occurrence-dominated hierarchies; all GPU configurations in this report used p₀ = q, ratio 1 < √2), everything in this section applies in full. Above p₀\*, coherence penalties *genuinely work* — the faithful frame wins the orthogonal competition even at ε = 0, and a finite-β check confirms the flip already at β = β\* (p₀ = 0.35, ε = 0.002). The blanket "never eliminates it" below is therefore scoped to p₀ ≤ √2·q; pre-registered GPU prediction for the other regime in `theory/general_no_go.md`. Machine-verified end to end (`theory/general_no_go_check.py`, checks A–E: closed form to 2×10⁻¹⁵; boundary 0.0159 reproduced via an independent code path; no escape into mixed plane/background configurations found; p₀\* located numerically at 0.29–0.31 vs predicted 0.294).
- **Corrected boundary ε\*\*(β)** (global 2D scans with valley-aware refinement): ε\*\*(β\*) ≈ 0.0112, ε\*\*(2β\*) ≈ 0.0128, ε\*\*(4β\*) ≈ 0.0140, ε\*\*(16β\*) ≈ 0.0152, saturating at **ε\*\*(∞) = 0.0159** — versus the vanilla ε\* ≈ 0.0486. *(Correction note: an earlier version of this section reported ε\*\*(16β\*) ≈ 0.0188 and ε\*\*(∞) ≈ 0.0224; those values were artifacts of a coordinate-descent search that gets stuck on the orthogonal valley at large β — single-coordinate moves break orthogonality and are blocked, so the search cannot slide along the valley to the true optimum. Found by re-deriving the β → ∞ limit independently via the constraint manifold and reconciling the mismatch; the same verification discipline that produced §7.1b in the first place.)*
- **The penalty helps, but is bounded:** at its best it shrinks the absorption region ~3–4× (0.0486 → 0.0112 at β ≈ β\*) and it **never eliminates it** — no β rescues ε < 0.0112, and pushing β → ∞ only relaxes the boundary back up to 0.0159. ε\*\* is *increasing* in β: overdosing the penalty makes absorption *harder* to remove, because the anti-rotated branch is more orthogonal than the tilted faithful branch, so a stronger orthogonality reward favors it more.
- **The trap outlasts the boundary.** On the orthogonal valley (the operative geometry at large β), the anti-rotated configuration remains a strict *local* optimum up to ε ≈ 0.05 — essentially the original unpenalized ε\* = 0.0486 — even though the global optimum flips to faithful at 0.0159. So even where an infinitely strong penalty makes faithfulness globally optimal, a gradient-trained SAE seeded in the absorbed basin can stay absorbed nearly as long as with no penalty at all. This is exactly the multistability observed on the GPU (round 3: unreliable faithful recovery until ε ≈ 0.03 at 4β\*).
- **Why, and what would work instead:** the discriminating information is not in the decoder Gram — it is in the **codes**: the anti config reconstructs parent-solo events 2-sparsely where faithful does it 1-sparsely. Remedies must therefore be data-dependent — weighting reconstruction/sparsity per event class (cf. weighted SAEs, arXiv:2506.15963; C²R's activation-energy condition, arXiv:2606.30609) rather than shaping dictionary geometry. Exact boundaries for that remedy family are derived in §12.

Methodologically: this correction exists because the prediction was pre-registered, the experiment was allowed to falsify it, and the falsification was chased into the exact model rather than absorbed into hand-waving. The machine-checked §7.1 algebra was *correct* — for the wrong configuration space.

### 7.2 Matryoshka SAEs: the mechanism found, exactly — and its failure modes

[Matryoshka SAEs](https://arxiv.org/abs/2503.17547) (nested dictionaries, smaller prefixes must reconstruct alone) are the field's empirical remedy for absorption. Analyzing them in the exactly-solvable models produced a three-act result:

1. **The naive account fails.** With per-prefix *optimal* coding (decoupled codes), the prefix terms are identical for faithful and absorbed dictionaries — the boundary ε\* does not move. Machine-checked.
2. **Ordering freedom makes it worse.** With the shared-code objective and the arrangement (ordering + prefix allocation) chosen optimally — which is what SGD ultimately optimizes over — the *composite* latents claim the prefix and reconstruct co-occurrence events perfectly at every scale. In both the single-child and two-child models, idealized Matryoshka then prefers absorption **more strongly than vanilla**, including at values of ε where vanilla already recovers the truth. (An early parent-first-only derivation suggested the opposite; enumerating orderings overturned it — the kind of error the machine-checking discipline exists to catch.)
3. **Prefix scarcity is the real mechanism.** Forcing the smallest prefix to a single slot (the realistic regime: in a real model, thousands of features compete for the smallest nested dictionary) flips the result completely: the faithful dictionary wins at **every** ε, including ε = 0 where vanilla provably absorbs. The parent survives because it is the only single latent useful for *all* the concept's events across *all* its children — reusability — and once the parent anchors the prefix, the shared code makes the faithful dictionary's roles consistent across scales while the absorbed dictionary tears itself between prefix and full reconstructions.

This yields sharp, falsifiable corollaries that the literature's intuitive account does not predict: **(i)** Matryoshka cannot rescue a *single-child* hierarchy (a composite beats the parent for the prefix slot — no breadth advantage); **(ii)** an over-provisioned prefix schedule (smallest prefix large enough to hold the composites) should *strengthen* absorption, not reduce it. Both are tested on the GPU below.

### 7.3 Survey: how these fit the field's toolkit

Methods proposed for the absorption/identifiability cluster, per the survey pass: [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) (ICML 2025) and [Tree SAEs](https://arxiv.org/html/2605.07922v1) restructure the dictionary hierarchically; [weighted SAEs](https://arxiv.org/abs/2506.15963) reweight reconstruction toward ground-truth features with a principled weight rule; the [feature-hedging paper](https://arxiv.org/pdf/2505.11756) documents the continuous variant of absorption in narrow SAEs (loss analysis, no closed-form boundary, remedies without proofs); [\"Which Sparse Code?\"](https://openreview.net/forum?id=JiyytGKbA9) shows inference-side non-uniqueness (different sparse-coding algorithms find disjoint but equally valid feature sets); and [attribution-based parameter decomposition](https://arxiv.org/pdf/2501.14926) sidesteps activation-space dictionaries entirely by decomposing in parameter space. Against this backdrop, the present contribution is the exact solvable-model layer: closed-form boundaries and thresholds (ε\*, β\*), proofs of when each mechanism can and cannot work, and identified failure modes for the leading remedy.

## 8. Experiments (NVIDIA L4, `dev-gpu`)

**Design.** Full-system SAEs (not the 2D toy): d = 64 ambient dimensions, 30 orthonormal background features firing independently (rate 0.08, coefficients U[0.8, 1.2]) plus the parent/child pair; m = 32 latents; untied ReLU encoder, unit-norm decoder, Adam, 15k steps of fresh 2048-sample batches (population regime). The learned child-side decoder direction is projected into the (a_p, a_c) plane and its angle φ recorded (45° = absorbed, 90° = faithful). Grid: λ ∈ {0.05, 0.1, 0.2, 0.3} × ε ∈ {0 … 0.18} × 3 seeds at q = 0.2, plus a q = 0.1 slice for the λq-scaling check — 135 runs. Controls: ε = 0 must absorb (predicted 45°); ε ≫ ε\* must not (predicted → 90°).

**Results (135/135 runs).**

![Absorption phase transition](fig1_absorption_transition.png)

- **The wall behaves exactly as proven.** At ε = 0, all 15 runs across every (λ, q) land on the absorbed dictionary: mean φ = 45.75°, with a separate parent latent present in each — Theorem 1b's unique global optimum, found by SGD every time.
- **The transition tracks the theory curve.** Mean learned angle by ε/ε\* bin: 46° (below 0.25) → 54° → 61° → 73° (at ε\*) → 78° → 82° → 87° → 88° (beyond 5ε\*). Under ε/ε\* rescaling, all five (λ, q) configurations — λ spanning 6×, q spanning 2× — collapse onto a single sigmoid with midpoint at ε/ε\* = 1 (Fig 2). The small systematic lead at low λ (SGD transitioning slightly before the theory curve) is the expected direction of the encoder-shrinkage effect, which lowers the effective λ.

![Scaling collapse](fig2_collapse.png)

- **The dynamics gap, demonstrated.** In all 135 latent inventories there is not a single "triple" solution (separate parent, child, *and* composite latents) — even though for every ε > 0 the triple is the unconstrained objective's true optimum (§4 Remark). SGD invariably lands on the 2-latent competition instead, which means the capacity-limited Theorem 2 — not the unconstrained optimum — is the correct predictor of trained SAE behavior. Absorption in practice is thus doubly robust: the objective prefers it under capacity scarcity, and optimization dynamics select it even without scarcity.

**Experiment A (classical recovery).** Random unit-column dictionaries in d = 64 with n ∈ {128, 256} features (coherence μ ≈ 0.45–0.52, so worst-case k\* ≈ 1.5–1.6), k-sparse data, TopK SAEs with oracle k, mean-max cosine similarity (MMCS) and fraction of features recovered above 0.9.

![Recovery phase diagram](fig3_recovery.png)

**Results (24/24 runs).** Recovery peaks at k = 8 — fraction 0.49 for both n = 128 and n = 256 — roughly 5× beyond the worst-case Donoho–Elad threshold k\* ≈ 1.5, confirming that typical-case (random-dictionary) identifiability far outruns worst-case bounds. But the diagram's other two features matter as much: recovery collapses by k = 24 (0.00–0.02, the superposition-interference limit), and it is *also* poor at very low k (0.08–0.26 at k = 2–4), where TopK training starves most latents of gradient — the dead-latent pathology, an optimization failure rather than an identifiability one (auxiliary losses exist in the literature precisely for this). Even at the sweet spot, plain TopK training recovers under half the features at m = n: in this regime the binding constraint on identifiability is training dynamics, not information — consistent with the spurious-minima analysis of the 2025 theory literature and with the dynamics gap observed in Experiment B.

**Experiment C (remedies — predictions pre-registered before results).** Same full-system setup, λ = 0.2, q = 0.2 (vanilla ε\* ≈ 0.049, β\* ≈ 0.043):

| sub-exp | design | pre-registered prediction |
|---|---|---|
| C1 | coherence penalty, single child, ε ∈ {0, .01, .02}, β ∈ {0, ½, 1, 2, 4}·β\* | φ transitions 45°→~90° as β crosses ~β\*, at every ε including 0 |
| C2 | real Matryoshka (prefixes 1,2,4,8,16,32), single child, ε ∈ {0, .01} | absorption **persists** (φ ≈ 45°) — single-child hierarchies unrescuable |
| C3 | two-child model, vanilla vs Matryoshka, ε ∈ {0, .01} | vanilla → absorbed (composites, no child latents); Matryoshka → faithful (children recovered) via prefix scarcity |

**Results and honest scoring (all pre-registered predictions, rounds 1–3; round-2/3 experiments used batched training — hundreds of SAEs per einsum program — enabling 16 seeds and fine grids; code `sae_round2.py`, `sae_round3.py`):**

| sub-exp | prediction | verdict |
|---|---|---|
| C1 | φ: 45°→~90° as β crosses β\*, at every ε incl. 0 | **REFUTED** — φ stuck at 34–46° for ε ≤ 0.01 at all β up to 4β\* |
| C2 | Matryoshka, single child: absorption persists | **Confirmed** (no faithful child latent; geometry is a *merged* single latent at ~33°, see below) |
| C3 | vanilla two-child: absorbed | **Confirmed** 6/6 (composites at cos 0.98–0.99, no child latents) |
| C3 | Matryoshka two-child: faithful via prefix scarcity | **Partial** — composites weakened (0.98→0.83–0.89), child-*dominant* latents appear (max cos up to 0.89), but they are hedged blends (e.g. a c₁-latent carrying +0.41 parent and −0.29 c₂), not clean recovery. The idealized prefix-scarcity account predicts the right direction and overpredicts its completeness. |

**The C1 refutation, run to ground (this became the main result of the session).** The logged latent geometry ruled out both initial hypotheses: latents stay fully in-plane at all β (ρ ≈ 0.96–0.99 — no escape into the 62 background dimensions), and the failure is not penalty diffusion over the 496 background Gram pairs. Re-scanning the exact 2D model over *all* 1- and 2-latent dictionaries — including negative angles the original derivation never considered — found the real mechanism: for β ≥ β\* and small ε the global optimum is an **anti-rotated absorbed pair** {≈ −40°, +46°}: the composite latent keeps absorbing while the *parent* latent rotates to the other side of it, making the pair near-orthogonal and the penalty ≈ 0. The original §7.1 threshold compared only the two named pure strategies and missed this branch. See §7.1b for the corrected theory; independent adversarial verification in `theory/verify_anti_rotation.py`.

SGD exhibits exactly the three basins the corrected landscape predicts: anti-rotated (latents at −31…−39° *and* +45–46°), a merged single latent at ~34° (exact loss 0.162–0.178 — a genuine optimization trap, worse than both optima), and faithful {≈5°, 85°} — with seed-splits precisely where the exact losses are near-degenerate (e.g. 0.1150 vs 0.1174 at ε = 0.02, 4β\*).

**Round-3 boundary validation (160 runs, 8 seeds × 10 ε × 2 β).** At β = β\*, the faithful-basin fraction crosses 50% between ε = 0.004 and 0.007 and reaches 7–8/8 above 0.01 — the corrected-boundary structure (anti below, faithful above), with the crossing leading the predicted ε\*\*(β\*) = 0.0112 by a factor ≈ 0.6. At β = 4β\* the picture degrades exactly as the corrected theory predicts (∂ε\*\*/∂β > 0): faithful majorities are unreliable until ε ≈ 0.03, with severe multistability (anti + merged basins persisting). Overdosing the penalty is empirically worse than β ≈ β\*, and the merged trap appears *only* at 4β\*.

**Round-2 fine transition measurement (1,040 runs, 16 seeds × 13 ε × 5 cells, functional child metric).** The per-seed transition ε_c (φ crossing 67.5°) sits at **0.58–0.70 · ε\*** uniformly across λ spanning 6× and q spanning 2× — the λq scaling collapse holds; the prefactor does not. The exact global-optimum tilt curve puts the 67.5° crossing at ≈ 0.88 ε\*, so SGD leads the *objective's own optimum* by ≈ 25%, and the same ≈ 0.6 lead factor reappears in the round-3 penalty boundary — consistent with a uniform effective-λ rescale from encoder shrinkage (both ε\* and ε\*\* are ∝ λ to first order). Round 1's "midpoint at ε/ε\* = 1" used the max-angle metric, which is biased toward 90° near the transition; the functional metric (child latent = argmax encoder response on a child-solo input) supersedes it.

**Experiment A, resolved (round-2 oracle controls, 144 runs).** Three conditions per (n, k): random init, random + dead-latent reinit, oracle init (decoder = true dictionary). Result — three sharply distinct regimes:

- **k ≤ 8: trainability-limited.** Dead-latent reinit alone lifts recovery from 0.09–0.46 to **0.95–0.97**; oracle init is perfectly stable (1.00). Round 1's low-k failure was pure optimization, and a two-line auxiliary fix recovers essentially everything.
- **k = 16: optimization-barrier regime.** Reinit helps partially (n=128: 0.60) or barely (n=256: 0.04), yet oracle init remains stable (0.98–1.00) — the truth is still an optimum, but SGD cannot reach it from random init.
- **k ≥ 24: identifiability/stability limit.** Even oracle init degrades (mmcs 0.59–0.70, frac ≈ 0) — the true dictionary is no longer a stable optimum under superposition interference. The genuine boundary lies between k = 16 and k = 24, ~10× beyond the worst-case Donoho–Elad k\* ≈ 1.5.

## 9. Implications for practice

- **λ is not a free knob.** ε\* ∝ λq converts the sparsity coefficient into a *resolution limit*: features whose solo rate is below ~1.17·λ·(co-occurrence rate) will be absorbed at the objective's optimum. Choosing λ is choosing which parts of the feature hierarchy to erase.
- **Hierarchical concepts are the worst case.** For strict hierarchies ("is a token" ⇒ "starts with S"), ε ≈ 0, and absorption is near-inevitable under any L1-type objective — consistent with where absorption is actually observed in LLM SAEs.
- **Remedies must change the objective, not the optimization.** In the misalignment band, reweighted losses, feature anchoring, or Matryoshka-style hierarchical objectives (per the 2025 literature) attack the right thing; bigger dictionaries, longer training, and better optimizers provably do not.
- **If you do use a decoder-orthogonality penalty: dose it at β ≈ β\* and expect a bounded effect.** The corrected theory (§7.1b) and the round-3 data agree: the penalty buys roughly a 4× smaller absorption region at β ≈ β\*, degrades beyond that (∂ε\*\*/∂β > 0, plus a merged-latent optimization trap that only appears at high β), and cannot remove absorption at any strength. Penalties that act on the *codes* per event class, not the dictionary Gram, are the open direction.
- **Interpretability audits should report co-occurrence statistics.** ε and q are measurable on real activation data; the theory says the ratio ε/(λq) predicts which feature pairs are trustworthy.

## 10. Relation to prior work

*(Novelty assessment below is from a multi-source verified literature sweep, 2026-07-21 — 7 primary papers full-text checked, claims adversarially 3-vote verified.)*

**The closed-form boundary ε\*(λ, q) appears to be unclaimed.** Absorption was coined and mechanistically explained by Chanin et al. ("A is for Absorption," arXiv:2409.14507), but their toy theory is qualitative: Appendix A.2 proves dL_sp/dδ = −p₁₁ (any nonzero co-occurrence strictly favors absorption) in a model where the child *never* appears alone — the solo rate ε is not even a parameter, and no threshold in (λ, q, ε) appears. C²R (Jin et al., arXiv:2606.30609, Lemma 4.1) proves per-sample sparsity objectives unconditionally prefer absorbed solutions at equal reconstruction — a preference ordering, not an ε-dependent phase boundary. The 2025–26 theory wave (Cui et al., arXiv:2506.15963 — closed-form but in the asymptotic S→1 limit with no L1 term; the GBA provable-recovery paper, arXiv:2506.14002 — guarantees for a *new* algorithm, not failure conditions for vanilla SAEs; Tang et al., arXiv:2512.05534 — spurious-minima structure) leaves the quantitative failure boundary unoccupied. No phase-diagram treatment of SAE recovery in occurrence-probability/penalty space was found.

**Coherence penalties as an absorption remedy are prior art — the critical-strength derivation and the anti-rotation obstruction are not.** OrtSAE (Korznikov et al., arXiv:2509.22033) adds a chunked max-pairwise-cosine decoder penalty explicitly against absorption and reports a 65% absorption reduction, with the weight (γ = 0.25) chosen empirically and no theorem or threshold anywhere. C²R's Eq. 13/14 is the nearest critical-condition analogue but is expressed in data-dependent latent energies. Against that backdrop, §7.1b contributes: the first *derived* critical strength β\*, the proof that it was the wrong question (anti-rotation evasion), the corrected bounded-and-non-monotone ε\*\*(β), and a falsifiable prediction about published methods — OrtSAE-style penalties should show a *bounded* absorption reduction that *degrades* when the penalty weight is pushed well past its sweet spot, because no Gram-type penalty can distinguish faithful orthogonal frames from anti-rotated absorbed ones.

**The Matryoshka characterization is unclaimed.** Bussmann et al. (arXiv:2503.17547) report ~10× absorption reduction empirically; Nabeshima's toy-model post notes runs that get "stuck in a bad state" — acknowledged but uncharacterized. The single-child-unrescuable / multi-child-partial mechanism (prefix scarcity + parent reusability, §7.2) and its GPU confirmation, including the hedged-blend geometry of the partial rescue, have no published counterpart.

The overall contribution: the *failure side made exact* — a closed-form phase boundary with a machine-checked proof, the objective-misalignment regime (absorption as global optimum despite identifiability), the corrected remedy theory, and SGD-level confirmation of each — plus one refuted pre-registration converted into the sharpest result.

## 11. Limitations

- The theorem compares population losses of dictionaries under *optimal* nonnegative coding; real 1-layer encoders add shrinkage that effectively rescales λ (the experiments absorb this into the empirical fit).
- Unit activation coefficients and an orthonormal pair; nonzero coherence μ perturbs the constants by O(μ).
- Vanilla L1 SAEs; TopK/JumpReLU variants replace the L1 mechanism with slot competition — same qualitative story, different constants (not derived here).
- Global optimality for ε > 0 is certified numerically (fine grid + refinement over all 2-latent dictionaries), analytically only pairwise; at ε = 0 the global claim is exact on the scan grid.
- Toy generative model: real activations are not exact sparse sums of fixed directions.

## 12. Constructive remedy: event-weighted losses (exact results)

The no-go theorem says dictionary-geometry penalties cannot work; the code/event structure can. The simplest remedy family that uses it — weight the per-sample loss by the event class — admits a complete exact analysis (`theory/remedy_weighted.py`):

**Weighted boundary (closed form).** With weights w_j on joint events and w_c on child-solo events, the loss difference scales term-by-term and the absorption boundary becomes **ε\*_w = (w_j/w_c) · ε\*(λ, q)** — the transition point is simply multiplied by the weight ratio.

**Inverse-density weighting eliminates the phase transition.** Setting w(event) = 1/P(event) gives w_j/w_c = ε/q, and ε cancels out of its own boundary condition: absorption is preferred iff

**C(λ) = λ(8 − 4√2 − λ) / (2(1 − (2 − √2)λ)) > 1, independent of ε and q.**

Solving C(λ) = 1: **λ_crit = [(12 − 6√2) − √((12 − 6√2)² − 8)]/2 ≈ 0.714**. For every λ < λ_crit — i.e., every practically relevant sparsity coefficient — the faithful dictionary beats the absorbed one **for every ε > 0, however small**. Verified by global scans over all 1- and 2-latent dictionaries down to ε = 10⁻⁴ (faithful global optimum at every ε; unweighted control reproduces the vanilla transition).

**The trap dies too.** The stronger result, invisible to the closed-form comparison: under inverse-density weighting the absorbed configuration is no longer even a *local* minimum — a local search seeded exactly at the absorbed dictionary walks to the faithful one, at every ε tested. Compare the coherence penalty, where the absorbed trap survives to ε ≈ 0.05: weighting fixes both the global optimum *and* the landscape. At ε = 0 exactly, Theorem 1 still applies (the ontologies are indistinguishable in data) — this remedy fixes everything that is information-theoretically fixable, which is the best any method can do.

**Why this is the principled version of existing practice.** Weighted SAEs (arXiv:2506.15963) reweight reconstruction empirically; C²R (arXiv:2606.30609) conditions on activation energies. The results here say data-dependent weighting is not merely *a* remedy option — by the no-go theorem it is the only *kind* of remedy (among dictionary-geometry vs data-weighting families) that can work, and inverse-density is the canonical instance with a clean optimality guarantee in the solvable model.

**GPU validation (round 4, pre-registered, 96 runs): all three predictions confirmed without exception.** P1 — the inverse-density-weighted SAE recovers the faithful child latent at every ε down to **5×10⁻⁴** (100× below the vanilla boundary ε\* = 0.0486, 20× below the best any coherence penalty achieves): **48/48 weighted runs faithful**, φ = 87–91°, ρ ≈ 0.99. P2 — recovery is *reliable*: zero multistability, seed spread under 2° — the trap-elimination prediction holds in SGD, not just in the exact landscape. P3 — unweighted controls at identical ε: 0/8 faithful everywhere below 0.02 (φ ≈ 45°, absorbed). Even the variance concern proved manageable at this scale: at ε = 5×10⁻⁴ a batch of 2048 contains ~1 child-solo sample carrying ~500× weight, and training still converged cleanly. (`experiments/sae_round4.py`, `results/round4/`.)

**Independent penalty-form check (OrtSAE-style chunked max-cos², 320 converged runs).** As the no-go theorem requires, the conclusion transfers to a structurally different coherence penalty: below the corrected boundary (ε ≤ 0.01), *no* strength γ ∈ [0.05, 20] — up to 80× OrtSAE's recommended γ = 0.25 — rescues the child (0–2/8 faithful across the entire sweep); at ε = 0.02 the effect is non-monotone (peak 5/8 at γ = 0.05, degrading to 1/8 at γ ≥ 10); at ε = 0.03, above the boundary, everything works and the penalty is superfluous. A design-specific finding: the chunked variant usually fails without visible anti-rotation because random chunk assignment co-penalizes the parent/composite pair only ~23% of the time — the penalty mostly never sees the offending pair, a second, more mundane evasion channel on top of the theoretical one. (Experiment code by Gemini, convergence fixes by Claude; `experiments/ortsae_style_test.py`, `results/round4/ortsae_style/`.)

**Honest caveats.** (i) Inverse-density weighting requires estimating event probabilities — the GPU validation uses oracle event labels (we generate the data); estimating them on real activations is the open problem, and the natural next step is an approximate-weighting proof-of-concept on a small real model; (ii) upweighting rare events by 1/P inflates SGD variance as ε → 0 and would amplify noise/outlier events in real data — a practical implementation needs clipping or annealing, whose theory is not done here; (iii) all results remain toy-model: d = 64, orthonormal features, vanilla L1 ReLU SAEs.

## 13. Reproducibility and cost

- `theory/verify_absorption_theory.py` — symbolic KKT enumeration + threshold + global scan (laptop, sympy).
- `theory/theory_curves.py` — global-optimum angle curves φ\*(ε; λ, q) (laptop, numpy).
- `theory/theory_merged.py` — corrected variable-latent-count + penalty scan (§7.1b); pure python, no dependencies.
- `theory/no_go_theorem.py` — penalty-form-independence theorem, corrected ε\*\*(β) curve (global scans with valley-aware refinement), trap-persistence analysis; pure python.
- `theory/general_no_go.md` + `theory/general_no_go_check.py` — general-(d, m) derivation with proofs (Lemmas 1–3, closed-form gain function, ε₀ anchor, critical ratio p₀\* ≈ √2·q) and its five-part verification suite; pure python.
- `theory/remedy_weighted.py` — event-weighted remedy: closed-form boundary scaling, λ_crit, inverse-density global scans + trap elimination (§12); pure python.
- `theory/verify_anti_rotation.py` — independent adversarial re-derivation of §7.1b (Gemini-authored).
- `experiments/sae_experiments.py`, `sae_remedies.py` — round-1 GPU experiments (torch 2.5.1+cu121, L4).
- `experiments/sae_round2.py`, `sae_round3.py` — batched suites (run dimension folded into the einsums: ~200 SAEs per training program, ~40× serial wall-clock); `SMOKE=1` for a pipeline check.
- Raw data: `results/` (one directory per round); analysis scripts in `analysis/`.
- VM: single g2-standard-8 (NVIDIA L4, ~$0.99/hr), total GPU session ≈ 2.4 h ≈ **$2.40** for all 1,483 trained SAEs across rounds 1–3 (the batched rounds trained 1,344 of them in ~25 GPU-minutes). Orchestration ran on an e2-standard-4. Both stopped at session end.

## 14. Real-activation results (POC + method eval + audit v1; honest scoring)

**Injected-pair POC on GPT-2-small layer-6 activations (36 runs, `experiments/poc_weighted_gpt2.py`); interpretations bounded by adversarial review (`results/round6/REVIEW_D12.md`).** The vanilla ε-trend is present (clean recovery at ε = 0.05, cos ≈ 0.90; degraded split geometry at ε = 0.002) but the low-ε data is too seed-noisy to *cleanly* distinguish absorption from undertraining — the transfer claim is supported, not proven; the reviewer's proposed activation-distribution metric is queued. **Residual weighting (oracle-free) rescues 3/4 seeds at ε = 0.002 (cos 0.84–0.85) and matches vanilla at large ε — P3 confirmed.** The oracle condition shows a *functional-metric anomaly* rather than a clean failure: its dictionary retains a child-aligned direction (max-cos 0.70–0.72) while the encoder-argmax probe metric collapses at ε ≥ 0.01 — most likely an encoder/probe artifact (adversarial review's verdict), so the headline is **not** "practical beats oracle" but "oracle needs decoder-side disambiguation; the practical estimator needs no such caveat." P2 is unresolved pending that check, and the §12 variance concern remains the candidate mechanism if the anomaly survives it.

**Natural-absorption eval (no injection, first-letter proxy metric).** Residual weighting gives a small consistent improvement: mean coverage gap 0.768 → 0.735 (largest: 'h' 0.74 → 0.61, 's' 0.89 → 0.81), feature-splitting reduced (6.4 → 5.6 probe-aligned latents). Adversarial-review caveat: the improvement could be confounded by a globally higher firing rate under residual weighting (the weights raise activation norms); the disambiguating control — global L0/L1 comparison plus per-feature selectivity on previously-uncovered tokens — is specified in `REVIEW_D12.md` and queued. Until then this is directional evidence only, small-scale (500k tokens).

**Weight-misestimation robustness (theory, `theory/remedy_robustness.py` + symbolic verification `theory/symbolic_verify.py`).** The remedy tolerates k-fold misestimation of event rarity up to a provable **k\* = 1/C(λ) = 4.12** at λ = 0.2 (actual tolerance ≈ 6.6× by global scan), with failure ε-independent as theory requires. Practical reading: density estimates need only be order-of-magnitude correct. All hand-derived formulas of §7.1b/§12 and the general derivation are now sympy-verified end to end.

**Absorption-risk audit v1 (`experiments/sae_audit.py`) — null result, reported as such.** Auditing a public GPT-2-small SAE (jbloom, layer 6) via co-activation statistics found zero hierarchical latent pairs under the v1 criteria (top-1024 latents, child-share > 0.7). Two candidate readings: absorbed pairs are invisible by construction (the child never obtains its own latent — consistent with the theory the audit is based on), or the criteria are insensitive. An audit v2 should detect absorption *signatures* (composite-decomposable latents, meta-SAE-style) rather than surviving pairs. No claim is made from this null.
