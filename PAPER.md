# A Solvable Model of Feature Absorption in $L_1$ Sparse Autoencoders

**Living paper draft.** This document is the formal distillation of `report.md`
(the session-log-style record) at the current repository state; claims follow
the guardrails established in `reviews/`. Experimental provenance: README
table (results CSV → commit). *Current through round 13b (2026-07-25).* The
theory sections (§1–§13) are the strongest content; the detector (§17), the
gating-corrected estimator (§18/round 9), the TopK note (round 10) and the
real-model rounds 11–13b (§8b; round 11 exploratory, rounds 12–13b
pre-registered) are scoped empirical results, flagged as such.
"External review" throughout means **LLM-assisted adversarial review**
(Gemini 2.5 Pro + GPT-5.6), which materially improved the work but is **not**
human peer review.

---

## Abstract

We study feature absorption — a sparse autoencoder (SAE) merging a parent
concept and a child concept into one latent — in an analytically solvable
model. At zero child-solo probability the hierarchical and merged ontologies
are observationally equivalent, and the merged configuration is the global
optimum of the SAE objective: the set of active dictionary *directions* is
uniquely determined, though columns and codes are not. Away from that wall we
derive the exact loss crossover between the pure faithful and pure absorbed
candidate dictionaries,
$\varepsilon^*(\lambda,q) = \frac{\lambda q\,(8-4\sqrt2-\lambda)}{2\,(1-(2-\sqrt2)\lambda)} \approx 1.17\,\lambda q,$
and show that the continuously optimized dictionary instead tilts smoothly
through intermediate angles, with its functional midpoint near
$0.88\,\varepsilon^*$ and SGD's near $0.58$–$0.70\,\varepsilon^*$. A
controlled width experiment shows capacity scarcity is the operative cause of
the two-latent transition in this model: with one nominal spare slot, SGD
frequently finds the redundant parent–child–composite triple instead of
absorbing. Pairwise-coherence (Gram) penalties have a proved rotation blind
spot on orthonormal frames, and — **within the restricted orthonormal-frame
class** ($m \le d$, stated column conditions; the overcomplete and
mixed-plane cases are open) — cannot remove absorption in the
co-occurrence-dominated regime $p_0 < p_0^*(\lambda,q)$ (with
$p_0^*/q \to \sqrt2$ only asymptotically as $\lambda\to0$);
inverse-density event weighting removes the transition exactly but requires
an oracle. In trained models, decoder-level absorption coexists with
code-level separation through encoder gating — dictionary identifiability and
code identifiability are distinct properties — and a detector combining
decoder geometry with code co-firing identifies planted parent/composite
pairs on matched synthetic data and partially transfers, with strong width
dependence, to semi-synthetic GPT-2 activations. The detector remains a
synthetic proof of concept: all-pairs specificity on real backgrounds,
orientation, scaling, and cutoff transfer are open, pre-registered problems.
At real scale (pre-registered eight-seed rounds on Pythia-1.4B layer 12,
matched $L_0 = 32$), the architecture prediction fails cleanly: L1 and TopK
SAEs show no difference in first-letter absorption (paired diff $+0.0030$,
95% CI $[-0.0010, +0.0067]$), a null that survives both a
splitting-corrected family endpoint and — refuting the last pre-registered
rescue — an $8\times$ capacity sweep. The sweep moreover *inverts* the toy
capacity story on this endpoint: measured absorption falls monotonically as
the dictionary shrinks (paired $-0.0445$, CI $[-0.0493, -0.0397]$), a small
L1$-$TopK gap opens only in the *low*-absorption scarce regime (interaction
$+0.0070$, CI $[+0.0014, +0.0135]$; the round's least-powered registered
test), and the endpoint co-moves with feature fragmentation — leaving the
construct validity of first-letter absorption metrics, rather than the
architecture comparison, as the central open question.

---

## 1. Introduction

Interpretability pipelines increasingly treat SAE latents as *the* features
of a model, and safety-relevant claims inherit that assumption. Feature
absorption [Chanin et al., 2024] is the sharpest known counterexample: a
child concept (e.g. *short*) merges into a latent that also carries its
parent (*starts with S*), so the parent latent appears to switch off exactly
where the distinction matters. Prior work documented the phenomenon
empirically and gave qualitative mechanisms; this project makes the failure
mode exact in a toy model, maps where remedies can and cannot work, and
develops (through pre-registered rounds, including refuted predictions) a
label-free detector for absorbed pairs.

Method note: every confirmatory experiment in this project was pre-registered
(hypotheses, thresholds, falsifiers committed before results), failures are
reported as first-class results, and post-hoc refinements are labeled
development-set work until they pass held-out data. Theory is computationally
verified (sympy symbolic enumeration and exact numeric scans — not a proof
assistant).

## 2. Model

Work in the plane spanned by orthonormal feature directions $a_p$ (parent)
and $a_c$ (child) in $\mathbb{R}^d$; background features are orthonormal to
the pair and handled independently. Each sample activates

- both features (coefficients 1) with probability $q$ ("joint"),
- the parent alone with probability $p_0$,
- the child alone with probability $\varepsilon$,
- neither, otherwise.

An SAE with unit-norm decoder columns $D$ and nonnegative code $f$ pays the
population loss
$$\mathcal{L}(D) \;=\; \mathbb{E}_x\Big[\min_{f\ge 0}\;\lVert x - Df\rVert^2 + \lambda \lVert f\rVert_1\Big].$$
Two pure candidate dictionaries compete in the pair plane: **faithful**
$\{a_p, a_c\}$ and **absorbed** $\{a_p, a_m\}$ with
$a_m = (a_p + a_c)/\sqrt2$.

## 3. The non-identifiability wall ($\varepsilon = 0$)

**Theorem 1 (observational equivalence).** At $\varepsilon = 0$ the
hierarchical model (features $\{a_p, a_c\}$, joint w.p. $q$, parent-solo
w.p. $p_0$) and the flat model (features $\{a_p, a_m\}$, $a_m$ alone with
coefficient $\sqrt2$ w.p. $q$, $a_p$ alone w.p. $p_0$) induce the same
distribution over activations. *Proof.* Both place mass $q$ at
$a_p + a_c = \sqrt2\, a_m$ and mass $p_0$ at $a_p$. $\square$

No procedure operating on activations alone can decide which ontology is
"real": the feature ontology is undetermined by the data.

**Theorem 1b (active directions at the wall).** Assume $p_0, q > 0$ and
$0 < \lambda < 2$ (so $r = \lVert x \rVert \ge \lambda/2$ for both event
types). For any dictionary $D$ with unit-norm columns and any $f \ge 0$,
$\lVert Df\rVert \le \lVert f\rVert_1$ gives, for a sample of norm $r$,
$$\lVert x - Df\rVert^2 + \lambda\lVert f\rVert_1 \;\ge\; \min_{t\ge 0}\,(r-t)^2 + \lambda t \;=\; \lambda r - \tfrac{\lambda^2}{4},$$
with equality iff all active atoms point at $x/r$. Summing over events,
$\mathcal{L}(D) \ge p_0(\lambda - \tfrac{\lambda^2}{4}) + q(\sqrt2\lambda - \tfrac{\lambda^2}{4})$,
attained **iff the set of directions used with positive probability is
exactly $\{a_p, a_m\}$**. The *set of active directions* is unique;
dictionary columns and codes are not — permutation, duplicate collinear
columns with arbitrarily split coefficients, and unused atoms all attain the
bound. $\square$

*(Two scope corrections in this statement — active directions rather than
"unique dictionary", and duplicates-may-be-active — are due to external
review; see `reviews/`.)*

## 4. The exact pure-strategy crossover

For each event the optimal nonnegative code is a 2-D nonnegative lasso;
enumerating KKT cases gives closed-form event losses (verified symbolically):

| event | faithful | absorbed |
|---|---|---|
| joint ($q$) | $2\lambda - \lambda^2/2$ | $\sqrt2\lambda - \lambda^2/4$ |
| parent solo ($p_0$) | $\lambda - \lambda^2/4$ | $\lambda - \lambda^2/4$ |
| child solo ($\varepsilon$) | $\lambda - \lambda^2/4$ | $\tfrac12 + \tfrac{\sqrt2\lambda}{2} - \tfrac{\lambda^2}{4}$ |

**Active-set domain (required).** The absorbed child-solo entry uses the
positive composite coefficient $1/\sqrt2 - \lambda/2$, so that active set — and
hence the displayed formula — is valid only for $\lambda < \sqrt2$; above it the
optimum changes active set and the formula must be extended piecewise. All
experiments and the machine check (which tests $\lambda \le 0.5$) sit safely
inside this range.

Parent-solo terms cancel, so $p_0$ plays no role. Subtracting,

$$\mathcal{L}_{\text{faithful}} - \mathcal{L}_{\text{absorbed}} \;=\; q\Big[(2-\sqrt2)\lambda - \tfrac{\lambda^2}{4}\Big] \;-\; \varepsilon\Big[\tfrac12 - \tfrac{(2-\sqrt2)\lambda}{2}\Big],$$

which is positive (absorption preferred among the two pure candidates) iff

$$\boxed{\;\varepsilon \;<\; \varepsilon^*(\lambda, q) \;=\; \frac{\lambda q\,(8 - 4\sqrt2 - \lambda)}{2\,\big(1 - (2-\sqrt2)\lambda\big)} \;\approx\; 1.172\,\lambda q \quad (\lambda \to 0).\;}$$

**Scope.** $\varepsilon^*$ is the exact crossover of the two *pure candidate
dictionaries* under the two-latent capacity constraint — not the transition
point of the continuously optimized dictionary:

**Result 3 (continuous tilt).** Scanning all unit-norm 2-latent dictionaries,
the global optimum interpolates: at $\varepsilon = 0$ it is exactly the
absorbed pair $(0^\circ, 45^\circ)$; near and above $\varepsilon^*$ the
child-side latent sits at intermediate angles (e.g. $69^\circ$ at
$0.9\,\varepsilon^*$, $80^\circ$ at $2\varepsilon^*$, $85^\circ$ at
$4\varepsilon^*$ for $\lambda{=}0.1, q{=}0.2$), approaching $90^\circ$
asymptotically. The functional $67.5^\circ$ crossing of the tilt curve sits
at $\approx 0.88\,\varepsilon^*$. This is a **solvable toy analogue / candidate
mechanism** for the empirically reported "feature hedging"
[Chanin, Dulka \& Garriga-Alonso, 2025] — not a claim to be the exact mechanism
of all reported hedging (a different model and phenomenon).

**Remark (redundant triple; spare capacity).** For $\varepsilon > 0$ and
$\lambda < 1/(2-\sqrt2)$, the unconstrained global optimum is the triple
$\{a_p, a_c, a_m\}$: joint events fire the composite, solo events their own
atoms, and every event attains the per-sample bound. Ideal SAE training with
unlimited capacity does *not* return the generative ontology — it adds
frequent combinations as first-class latents ("composition").

**Practitioner scale.** $1.17\,\lambda q$ is the characteristic scale of the
absorption transition, exact only as the pure-strategy crossover. Deep inside
it ($\varepsilon \ll 1.17\lambda q$), absorption is the population optimum of
this objective under the two-latent constraint — within that scope more
compute or data cannot help; changing the objective, $\lambda$, or capacity
can.

## 5. Remedies

### 5.1 Coherence (Gram) penalties: a rotation blind spot

Adding $\beta \sum_{i<j}\langle d_i, d_j\rangle^2$ suggests a critical
strength $\beta^*(\lambda,q) = \tfrac{\lambda q}{2}(8-4\sqrt2-\lambda)$ above
which the faithful pair beats the absorbed pair for all $\varepsilon \ge 0$.
This pre-registered prediction was **refuted** by GPU experiment, and the
refutation is the sharper result. The true optimum at $\beta \ge \beta^*$,
small $\varepsilon$, is an **anti-rotated absorbed pair**
$\{\approx -40^\circ, +46^\circ\}$: the composite keeps absorbing while the
parent rotates to zero the pair's Gram entry.

Evidence tiers, kept separate:

- *(proved)* every penalty that is a function of pairwise decoder inner
  products is constant on the manifold of orthonormal frames; hence no such
  penalty distinguishes the faithful frame $\{0^\circ, 90^\circ\}$ from the
  anti-rotated absorbed frame $\{-45^\circ, 45^\circ\}$. On that manifold the
  objective admits the closed form
  $$\mathcal{L}_0(D) \;=\; \mathbb{E}\lVert x\rVert^2 \;-\; \sum_i \mathbb{E}\big[(\langle d_i, x\rangle - \lambda/2)_+^2\big],$$
  verified to $2\times10^{-15}$; the $\lambda/2$ soft-threshold tax is paid
  once per active coordinate, so concentrating a co-occurrence in one latent
  beats splitting it — absorption lives in the codes, invisible to decoder
  geometry.
- *(analytic reduction)* the in-plane frame competition holds for $m \le d$
  with non-pair latents off the pair plane; the overcomplete case is open.
- *(numerical)* corrected boundaries: $\varepsilon^{**}(\beta^*) \approx
  0.0112$, increasing in $\beta$ to a penalty-form-independent
  $\varepsilon^{**}(\infty) \approx 0.0159$ (vs vanilla
  $\varepsilon^* \approx 0.0486$ at $\lambda{=}q{=}0.2$): the penalty shrinks
  the absorption region at most $\sim 4\times$, never removes it, and
  overdosing worsens it.

**Domain boundary (critical occurrence ratio).** The evasion defeats the
penalty iff
$$p_0 \;<\; p_0^*(\lambda, q) \;=\; q\,\frac{(2-\sqrt2) - \lambda/4}{(\sqrt2-1) - \lambda/4}, \qquad \frac{p_0^*}{q} \xrightarrow{\lambda\to 0} \sqrt2 .$$
Below $p_0^*$ (co-occurrence-dominated hierarchies) the no-go applies in
full; above it, coherence penalties genuinely work, even at $\varepsilon=0$
(finite-$\beta$ flip verified at $p_0 = 0.35$).

### 5.2 Inverse-density event weighting: exact removal, oracle-dependent

Weighting each event by $w(\text{event}) = 1/P(\text{event})$ makes
$w_j/w_c = \varepsilon/q$, and $\varepsilon$ cancels from its own boundary
condition: absorption is preferred iff $C(\lambda) > 1$, independent of
$\varepsilon$, with
$\lambda_{\text{crit}} = \tfrac12\big[(12-6\sqrt2) - \sqrt{(12-6\sqrt2)^2 - 8}\big] \approx 0.714$.
For every $\lambda < \lambda_{\text{crit}}$ the faithful dictionary wins for
all $\varepsilon > 0$ — the transition is eliminated (GPU-validated 48/48 at
$\varepsilon = 5\times10^{-4}$, no multistability). A robustness bound
$k^* = 1/C(\lambda) \approx 4.12$ at $\lambda = 0.2$ under-predicts the
observed tolerance ($6.6\times$).

**Status: a diagnostic existence result, not a practical method.** The
weights require event-class labels. Two label-free substitutes were refuted
with understood mechanisms (self-residual: signal drowned by background
error; background-relative novelty: detects events, not classes), and the
capacity-limited semi-synthetic test (§6.3) shows the oracle version works
exactly where absorption bites.

### 5.3 Matryoshka

Exact mechanism analysis (prefix scarcity + parent reusability): single-child
hierarchies are unrescuable (GPU-confirmed); the two-child rescue observed
under rich metrics is partial (child-dominant hedged latents).

## 6. Experiments (synthetic and semi-synthetic)

All GPU experiments: batched einsum programs training up to 208 SAEs
simultaneously on one NVIDIA L4; environment pinned in `ENVIRONMENT.md`.

### 6.1 Transition measurement

135-run grid + 1,040-run fine measurement: the empirical transition (per-seed
functional-child crossing) sits at $0.58$–$0.70\,\varepsilon^*$ across
$\lambda$ spanning $6\times$ and $q$ spanning $2\times$; the prefactor is
$\approx 0.7\times$ the continuous-optimum midpoint ($0.88\,\varepsilon^*$),
consistent with a uniform effective-$\lambda$ rescale from encoder shrinkage.
*(Scoped: this is consistency with $\lambda q$ scaling on a modest grid with
seed-level spread — not a certified "uniform scaling collapse." A proper test
regresses the transition estimate on $\lambda q$ and reports the exponent/
intercept uncertainty against separate-$\lambda$/separate-$q$ alternatives;
queued.)*

### 6.2 Capacity, controlled

At $m = 32$ with 30 background features + parent + child, the redundant
triple needs 33 columns — architecturally impossible. (An earlier "dynamics
gap" claim built on that grid was withdrawn after external review.) The
controlled rerun ($m \in \{32, 34, 40\}$, 288 runs, K1–K3 pre-registered):
$m{=}32$ → 0/96 triples; $m{=}34$ → triples in 69% of $\varepsilon>0$ runs;
$m{=}40$ → functional transition below $0.25\,\varepsilon^*$. **In this
generative model, nominal headroom moves the learned solution from absorption
toward redundant composition** — capacity scarcity is the operative cause of
the two-latent transition here. This suggests — a hypothesis following from
the model, not an evidenced claim — that for real-world models, where the
landscape of feature combinations is likely far richer than the number of
available latents, capacity scarcity may be a primary driver of the
absorption pathology. *(Round 13b tested the natural transfer of this
hypothesis at real scale on the first-letter family endpoint and found the
opposite monotone profile — measured absorption falls as capacity shrinks;
see §8b.4. The controlled synthetic result above stands; its extrapolation
to that real endpoint does not.)*

### 6.3 Semi-synthetic regime structure (synthetic pairs injected into real GPT-2 activations)

With $m = 1536$ (spare), every condition forms the triple — composition, no
harm, and weighting has nothing to fix. With $m \in \{128, 256\}$ (scarce),
true harmful absorption appears (composite cosine $0.99$, child reduced to
the composite's shadow $0.74$), and **oracle** inverse-density weighting
rescues the child ($\cos \ge 0.99$ in 7/8 seeds). The natural-feature audits
found no qualifying natural pairs; all real-activation evidence is
semi-synthetic.

## 7. Gated absorption: dictionary vs. code identifiability

A pre-registered test (Arm A) of a two-sided identifiability note — (i) a
no-go: binarized co-firing signatures are invariant to the child rate $\rho$
under absorption; (ii) an estimator: the within-composite activation
magnitude is bimodal ($1/\sqrt2$ vs $\sqrt2$) and a 2-component mixture
recovers $\rho$ — **inverted both hypotheses**. Trained absorbed SAEs are not
the stipulated single shared composite. They are the theory's own absorbed
branch: a parent-aligned latent plus an **encoder-gated** composite that
fires on $100\%$ of joint events (mean activation $\approx 1.28$) and
essentially never on host-only events ($3$–$20\%$ of them at
$\approx 0.01$). Consequently the binarized code separates the
sub-populations nearly perfectly (conditional TV $0.9999$), signature
*counting* recovers $\rho$ to $\le 0.02$ given the pair, and the magnitude
mixture has nothing to fit at $\sigma = 0$.

**The general lesson: dictionary identifiability (a child atom in the
decoder) and code identifiability (child events distinguishable in the
sparse code) are distinct.** Absorption here destroys the former while
encoder gating preserves the latter. The gating mechanism itself has prior
art: Chanin et al.'s toy-model work documented absorption's encoder/decoder
asymmetry — an encoder "hole" with an unchanged decoder — and suggested it
as a detection signal, and Chanin & Till's "Broken Latents" observed
multi-peak activation histograms in broken latents and proposed label-free
detection from them. Our delta is the quantification in trained untied SAEs
(mutual gating, conditional TV 0.9999), the identifiability consequences
(counting recovers ρ given the pair; the dictionary-vs-code distinction as
stated properties), and — notably — a pre-registered test showing the
histogram/bimodality route *fails at σ = 0 precisely because gating
suppresses the second mode*, qualifying the Broken Latents suggestion. The
binarized-signature no-go remains valid for its stipulated
single-shared-latent model; trained SAEs do not realize that model, even
when capacity-forced.

Exploratory: activation noise $\sigma \ge 0.2$ destroys absorption in favor
of faithful child latents; the noise mechanism is unmeasured (histograms not
retained) and queued for its own pre-registration.

## 8. Label-free pair detection

**Detector.** For each latent pair $(i, j)$ with firing rates in
$[5\times10^{-4}, 0.6]$ (binarize at activation $> \theta = 0.05$):

$$c_{ij} = \lvert\cos(D_i, D_j)\rvert, \qquad \ell_{ij} = \frac{P(i \wedge j)}{P(i)\,P(j)}, \qquad \omega_{ij} = \frac{P(i \wedge j)}{\min(P(i), P(j))}.$$

Flag iff $c_{ij} \in [0.45, 0.90]$ **and** ($\ell_{ij} \le 0.5$ or
$\ell_{ij} \ge L_{\text{HI}}$) **and** $\omega_{ij} < 0.9$ (v1.1;
$L_{\text{HI}} = 2.0$).

The two-sided lift rule is a calibration-pilot discovery: an absorbed pair's
latents are driven by one host event stream — exclusively (clean gating,
$\ell \approx 0$ at $\sigma{=}0$) or jointly (leak coupling,
$\ell \approx 3$ at $\sigma{=}0.1$) — never independently, whereas genuinely
correlated-but-independent features sit at $\ell \approx 1$. The overlap veto
removes feature-splitting doublets ($\omega \approx 1$; true pairs
$\le 0.81$).

**Arm 1 (matched synthetic, 176 SAEs, v1.0 confirmatory).** Recall $0.9315$
(10k-seed-bootstrap CI $[0.851, 1.000]$); correlated-independent control flag
rate $0.0625$ $[0.000, 0.156]$; FP/SAE $0.1062$ $[0.062, 0.150]$ (missed its
$0.10$ threshold by $0.006$; every null FP is a splitting doublet); child
recovery median $\cos = 0.979$ $[0.976, 0.983]$ (the only CI-established
endpoint); counting $\hat\rho$ error $0.0134$ $[0.002, 0.032]$ at
$\sigma = 0$. v1.1 (the overlap veto) is development-set performance on this
data. Scaling context: FP $\approx 214$/million candidate pairs at $m = 32$;
precision $0.81 / 0.30 / 0.04$ at assumed absorbed-pair prevalence
$10^{-3} / 10^{-4} / 10^{-5}$.

**Arm 2 (held-out, semi-synthetic GPT-2, frozen v1.1).** A detector frozen on
synthetic development data identified the planted absorbed oracle pair with
strong width dependence: $1/8$ detections at $m = 128$ and $8/8$ at
$m = 256$ under the pre-registered threshold — every true pair sat at
$\ell = 2.00 \pm 0.05$ against $L_{\text{HI}} = 2.0$, a width-dependent
*calibration* failure of the cutoff (the statistic itself concentrated tightly). The planted faithful
oracle pair was never flagged ($c \in [0.27, 0.31]$, outside the band). This
supports partial transfer of the pair-detection signal; it does **not**
establish all-pairs specificity on real backgrounds (full-scan flag counts
are similar in absorbed and faithful conditions), automatic orientation
(rarity-based orientation was right in $\sim 5/9$ detections; child recovery
$0.99$ when right, $0.66$ when wrong), or natural-feature absorption (the
full scan yields a *real-background candidate list* of unknown status).
$\hat\rho \approx 0.75$ vs true $0.5$ — the counting estimator assumes
gating, which real (leaky) activations violate.

**Round 8 (completed; pre-registered with a pre-collection amendment adopting
width-specific endpoints, stage-separated reporting, and a corrected
candidate-stability matcher).** v1.2 ($L_{\text{HI}} = 1.9$, calibrated on
Arm 2, otherwise frozen) passed its **width-specific** held-out endpoints on
fresh same-domain runs: recall $24/24 = 1.000$ at **each** of $m = 128$ and
$m = 256$ (48/48 formation; lift clusters $1.975 \pm 0.026$ and
$2.055 \pm 0.024$, both clear of the cutoff; exact binomial 95% lower bound
$\approx 0.86$ per width), faithful oracle-pair flags $0.000$. Scope:
same-domain resampling stability, not cross-domain transfer. Stage
separation: orientation accuracy by the rarity rule is $0.88 / 0.75$;
child-residual recovery is $0.990 \pm 0.001$ under **oracle** orientation at
both widths versus $0.948 / 0.909$ automatic — residualization is
near-perfect and *orientation is the pipeline's weakest stage* (it fails
completely, $0.00$, under a prevalence-inversion stress cell, as
pre-registered; a containment-based orientation rule is the queued fix).
$\hat\rho \approx 0.75$ remains leak-inflated. Proportional-scale null
calibration (run with v1.1, not v1.2: the $L_{\text{HI}}$ recalibration was
real-data-specific, and v1.1 preserves comparability with Arm 1's synthetic
calibration; observed lifts sit $\ge 2.26$ or $\approx 0$, so the outcome is
threshold-insensitive here): v1.1 produced **zero** null false positives at every scale from
$(d, m) = (64, 32)$ to $(512, 256)$ — the overlap veto fully suppresses the
splitting-doublet mode — with recall $1.000$ on formed runs throughout and
declining formation at the largest scale ($4/8$, disclosed). Robustness:
detection and recovery survive child–parent cosine $0.3$ (7/7) but detection
degrades at $0.5$ (3/6, pair cosine approaches the band edge); **the
detector signature and gated absorption survive TopK encoders** ($k{=}4$,
$\lambda{=}0$; 6/6 flagged, child recovery $0.999$). Every faithful-control
SAE still carries $\ge 1$ full-scan real-background flag (candidate counts
do not separate conditions), so all-pairs specificity on un-injected real
backgrounds — with a fixed-dimension width sweep and an overcomplete
$m > d$ setting — remains the round-8b gap. Post-run recomputes from frozen artifacts (post-hoc status disclosed):
oracle-*touch* specificity coincides with exact-pair specificity (the only
flag touching an oracle latent is the true pair; zero in faithful controls),
and a **shuffled-firing dependence null** — permute each latent's firing
column, preserving decoder geometry and marginal rates while destroying
dependence — yields **zero flags in all 64 SAEs**: every detector flag is
dependence-driven. Cross-seed clustering of the real-background candidates
(corrected bijective matcher, planted-latent-touching pairs excluded — the
exclusion changed nothing, confirming zero injection contamination) shows
they are **properties of the data, not of a particular SAE**: 3 stable
clusters at $m = 128$ (two in 8/8 seeds) and 12 at $m = 256$, including a
mutual 4-latent clique.

**Natural-feature adjudication (pre-registered, `notes/prereg-natfeat-adjudication.md`).**
We adjudicated all 15 seed-stable candidate clusters against a locked
asymmetric-nesting criterion — natural absorption requires
$C(\text{parent}\mid\text{child}) \ge 0.80$ with $C(\text{child}\mid\text{parent}) < 0.80$
(child's firing contained in the parent's, not vice versa) — computed on pure
background activations with top-activating-token semantics as a confirmatory
read. **None qualifies: the maximum child$\to$parent containment observed is
$0.46$, far below $0.80$.** The candidates split perfectly by the sign of the
decoder cosine into two non-absorption families: (i) **positive-cosine
typographic families** ($9$ clusters, incl. the 4-clique
$\{51,54,107,172\}$ = {possessive `'s`, opening `"`, contraction `'`, closing
`"`}), whose members co-fire because multi-byte curly-quote *byte fragments*
cluster in dialogue-dense passages — a tokenizer$\times$non-ASCII-typography
correlation with a flat, hierarchy-free containment matrix; and (ii)
**negative-cosine anti-correlated linguistic pairs** ($6$ clusters, cosine to
$-0.83$, co-firing $\le 0.014$), flagged only through the low branch of the
two-sided lift. Both are the **CDX equivalence class**: real, dependence-driven
co-firing structure that is *not* absorption. The wild hunt is thus a
pre-registered null, and it isolates the operative fix — a wild absorption
scan needs a positive-cosine **asymmetric-containment** gate (the exact metric
that rejects all 15 here) on an ASCII-clean corpus. Full result:
`results/round8/natfeat_SUMMARY.md`.

**Round 9 (pre-registered, lock `b0276cc`; dual pre-lock external review —
Gemini 2.5 Pro minor, GPT-5.6 major, all required changes applied before
lock): gating-corrected $\rho$ estimation.** The counting estimator's leak
inflation ($\hat\rho \approx 0.75$ vs $0.5$) is corrected by a token-level
rank rule rather than a constant. Model classes J (joint), S (parent-solo),
B (background) with fires $g_0 = P(b_{\text{par}}\mid S)$,
$g_1 = P(b_{\text{par}}\mid J)$, $a_1 = P(b_{\text{comp}}\mid J)$,
$a_0 = P(b_{\text{comp}}\mid S)$. Structurally $a_1, g_0 \ge 0.998$
everywhere measured, so exclusive firing patterns are class-pure, and the
binary pattern system is exactly one constraint short of identifying $\rho$
(background-free); the closing constraint is **tokenwise dominance** — on a
co-firing token the host class's latent carries the larger activation
(scale-valid because decoder columns are unit-renormalized every step).
The **dominance-partition estimator** ($01 \to J$, $10 \to S$, $11 \to J$
iff $a_{\text{comp}} > a_{\text{par}}$) is exact when the class-conditional
inversion rates $\delta_J, \delta_S$ vanish, with bias
$(1-\rho)a_0\delta_S - \rho g_1\delta_J$ plus a background mixture toward
$h_B = (q_{01} + q_{11}\pi_B)/(1 - q_{00})$. Confirmatory outcome (384
fresh-seed runs: $\rho \in \{0.1,0.3,0.5,0.7\}$, $\sigma \in \{0,0.05,0.1\}$
synthetic + four real-GPT-2 prevalence cells, 24 seeds/cell, RC formation
96/96): **mechanism endpoints passed every cell** — on parent-event tokens,
MAE $\le 0.0026$ in all 16 cells against locked bars of $0.03$, while the
leak being corrected varied from $a_0 = 0.014$ to $0.67$ and $g_1$ from
$0.005$ to $0.88$ (strongly $\rho$-dependent on real data), and measured
inversions stayed at $\delta_J \le 10^{-4}$, $\delta_S \le 0.004$ (P4
passed 16/16, with sparse class-11 contributor counts in the clean-gating
$\sigma = 0$ cells, disclosed). **Both operational predictions are
inconclusive overall**: 14 cell-level passes plus one inconclusive
$\rho = 0.1$ cell per harness ($0.066$–$0.070$; the real-data one disclosed
a-priori as at-risk). The post-run measured mixture
$(1-w_B)\hat\rho_{D,M} + w_B h_B$ accounts for every cell's operational
value with residual $\le 10^{-4}$ — a same-run diagnostic locating the
deviation in background-active tokens, not an a-priori prediction ($h_B$
measured $0.36$–$0.54$ on real data, $\approx 0.5$ in noisy synthetic
cells, NA where background-active mass is negligible). One registered
prediction **falsified**: the beats-the-baseline margin (P3) failed in the
two $\sigma = 0$ synthetic cells. Post-hoc diagnosis: eligibility was
predicted from Arm A's $\sigma = 0$ leak ($a_0 \approx 0.16$) while this
harness measured $a_0 \le 0.038$, so the baseline was only $0.025$–$0.035$
wrong and the $-0.05$ margin was arithmetically unattainable; the corrected
estimator remained more accurate in both cells ($0.0013$ vs $0.0350$;
$0.0021$ vs $0.0245$) but did not meet the registered comparative claim.
Leak magnitudes do not transfer across harnesses — round 8b's lesson,
reappearing on the eligibility side. Scope: oracle pair
location and orientation (estimation given the pair, Arm-A style);
background-corrected operational estimation and orientation remain open
(the estimator's swap-equivariance $\hat\rho_D \to 1 - \hat\rho_D$ is a
candidate orientation signal, untested). Prior art: the leak phenomenon is
documented ([@chanin2024absorption] App. A.3 [@chanin2025feature]); a verified
literature sweep found no existing statistical estimator correction on
binarized co-activation counts (novelty hedged; sweep archived). Full
scoring: `results/round9/SUMMARY.md`; prereg
`notes/prereg-gating-corrected-rho.md`; theory
`theory/gating_corrected_rho.md`.

## 8b. Real scale: the architecture prediction on Pythia-1.4B (rounds 11–13b)

The program's real-model arc tests whether the toy geometry predicts a
reproducible L1-vs-TopK difference in absorption in a real language model.
All confirmatory rounds were pre-registered with frozen scorers and
pre-results locks; full outputs and registered verdicts live in
`results/real/SUMMARY_round12.md`, `SUMMARY_round13a.md`,
`SUMMARY_round13b.md`, with locks and amendments recorded per round.
Common substrate: Pythia-1.4B layer 12 residual activations ($d = 2048$),
first-letter absorption task, SAEs matched at $L_0 = 32$ (TopK $k = 32$;
L1 $\lambda$ calibrated to $L_0 \approx 32$, blind to absorption).

### 8b.1 Round 11 (exploratory infrastructure) and round 12 (registered): NOT CONFIRMED

Round 11 (exploratory, no lock) built the extract–cache–train pipeline; its
semantic "$\sim 27\times$ redundancy" reading was **withdrawn** as
confounded (see `CLAIM_LEDGER.md`) and is not relied on anywhere below.
Round 12 (registered, lock `0722212`; 16 SAEs $=$ 8 seeds
$\times$ {L1 $\lambda{=}4.5$, TopK $k{=}32$} at $m = 16384$): **P1 NOT
CONFIRMED** — paired per-seed L1$-$TopK first-letter absorption diff
$+0.0030$, 95% CI $[-0.0010, +0.0067]$ ($n = 8$, 5/8 positive). A stale
out-of-config file contaminated the first collection; the conformance and
seed gates caught it, and the disclosed clean re-score (unmodified frozen
scorer, registered conformance rule) is the reported result. Secondary:
both architectures' absorption is letter-concentrated (P2), and the §8
detector's pair flags are enriched for absorption-implicated latents in
both (P3: recall 0.812 vs baseline 0.153 for L1; 0.333 vs 0.030 for TopK).
Post-hoc diagnosis (exploratory, hypothesis-generating only): the endpoint
is carried by $\sim 3$ of 24 letters and is largely a selectivity
re-expression ($R^2 = 0.673$); two registered rescue hypotheses followed —
H2, the endpoint conflates absorption with splitting; H1, $m = 16384$ at
46–57% dead latents is a spare-capacity regime where the toy predicts no
absorption pressure.

### 8b.2 Round 13a: the null is not a splitting artifact (H2 refuted)

Round 13a (registered, lock `9728663`, amendment + evaluator pre-results)
re-scored the frozen round-12 SAEs against each letter's whole split family
instead of one designated latent. Registered verdicts: absorption
**survives** the correction (family rate $0.0542$, CI $[0.0494, 0.0592]$;
the single-latent endpoint inflates absorption by $\sim 25\%$ — a validity
finding for SAEBench-style metrics in current use), the corrected endpoint
is meaningfully less a selectivity re-expression ($R^2 = 0.381$ vs
$0.673$; a narrow pass of the registered $0.40$ bar), letter concentration
is unchanged, and the architecture null **persists** under the corrected
endpoint ($-0.0012$, CI $[-0.0081, +0.0049]$). The sharp architecture
difference is in *splitting*, not absorption: mean split-family size
$|F_L|$ is L1 $2.61$ vs TopK $1.25$ (paired $+1.36$, CI $[+0.94, +1.88]$,
22/24 letters). H2 is refuted; H1 became the last registered rescue.

### 8b.3 Round 13b: capacity-sweep design, gates, manipulation check

Round 13b (prereg lock `c934d33`; Amendments 1–2 at `7501486`, both
pre-results and based on timing/$L_0$ measurements only: adaptive
$\lambda$ calibration, and dropping $m = 8192$ for budget) trained 48
fresh SAEs — 3 widths $m \in \{2048, 4096, 16384\}$ $\times$ {L1, TopK}
$\times$ 8 seeds — on round 12's cached activations, 15k steps, family
endpoint primary. Calibrated $\lambda = 3.784 / 4.1265 / 4.5$ by width.
**Gates 1–4 all pass** (conformance; exact seed/filename pattern —
load-bearing after round 12's contamination; matched $L_0$ at every width,
$|\Delta L_0| \le 1.5$; SHA256 provenance). **Manipulation check passes**:
dead latents fall from 53.1% to 6.3% (46.8pp against a 15pp bar) and live
latents are monotone in $m$ ($1919, 3256, 7682$) — scarcity genuinely
binds at $m = 2048$.

### 8b.4 Round 13b registered outcomes: P1 is not merely unconfirmed, it is backwards

**P1 (primary): FALSIFIED-DIRECTION.** The registered prediction was that
absorption rises as capacity falls. Instead, the pooled paired diff
$\mathrm{rate}_{\mathrm{fam}}(m{=}2048) - \mathrm{rate}_{\mathrm{fam}}(m{=}16384)
= -0.0445$, 95% CI $[-0.0493, -0.0397]$ ($n = 16$; within-arch: L1
$-0.0410$ $[-0.0468, -0.0356]$, TopK $-0.0480$ $[-0.0548, -0.0413]$). The
width profile is perfectly monotone in **both** architectures
(Spearman(live latents, rate) $= +1.00$):

| arch | $m{=}2048$ | $m{=}4096$ | $m{=}16384$ |
|---|---|---|---|
| L1 | 0.0133 | 0.0361 | 0.0542 |
| TopK | 0.0071 | 0.0145 | 0.0551 |

![Round 13b: measured absorption *rises* with dictionary width in both
architectures, the opposite of the registered prediction. Points are means
over 8 seeds; bars are standard errors. Generated from the frozen
`round13b_results.json` by `ops/mkfigs.py`.](figs/fig_width_sweep.pdf){width=72%}

Two artifact explanations were checked and excluded: (i) denominator
selection — all 48 SAEs score the identical 24 letters with identical
denominator $n = 17{,}981$, so the comparison is exactly matched; (ii) the
registered retention confound (P5) does **not** fire — outright loss
*also* falls with capacity ($0.0322 \to 0.0216$ pooled), and
absorbed$+$lost falls $\sim 2.7\times$ ($\approx 0.087 \to 0.032$) as the
dictionary shrinks $8\times$. **H1 is refuted**: round 12's null cannot be
explained by spare capacity; $m = 16384$ is the *high*-absorption regime
on this endpoint.

**P2 (primary): CONFIRMED as registered, with caveats stated in advance or
forced by the data.** Interaction
$[\mathrm{L1}-\mathrm{TopK}]_{m=2048} - [\mathrm{L1}-\mathrm{TopK}]_{m=16384}
= +0.0070$, CI $[+0.0014, +0.0135]$ ($n = 8$; gap $+0.0062$ at $m{=}2048$
vs $-0.0009$ at $m{=}16384$; per-seed 6/8 positive, one zero, one
negative). Three caveats: (i) the *registered power caveat* applies — a
difference-of-differences on 8 seeds, CI half-width $0.0061$, lower bound
barely clear of zero and sensitive to one seed ($+0.0263$); (ii) the gap
is confirmed in the **opposite regime** from the one H1 posited — where
absorption is $4$–$8\times$ *lower*, not higher ($+0.0062$ on a base of
$\sim 0.01$); (iii) because P1 inverted, the CONFIRMED branch's literal
wording ("round 12's null was regime-bound") is no longer the supported
reading. What survives is narrower: **at small $m$ the architectures
differ; at large $m$ they do not** — and given P3, the most likely
mechanism is splitting, not absorption, consistent with 13a's P5.
*(Exploratory, not registered: on a relative scale L1 absorbs
$1.87\times$ TopK at $m = 2048$ vs $0.98\times$ at $m = 16384$.)*

**P3 (secondary):** splitting shrinks under scarcity, and it is an L1
phenomenon — mean $|F_L|$ grows $1.84 \to 2.23 \to 2.61$ for L1 across
widths but stays flat $1.21 \to 1.22 \to 1.25$ for TopK. **P4
(secondary):** the single-latent metric's splitting inflation is 23–33%
at every width, replicating 13a's metric finding. **P5 (confound
control):** does not fire, as reported under P1.

![The SAEBench-style single-latent endpoint scores more "absorption" than the
splitting-corrected family endpoint at every width and in both architectures;
the gap is feature splitting counted as absorption. Means over 8
seeds.](figs/fig_endpoint_inflation.pdf){width=72%}

### 8b.5 Blind-committed theory notes and their scorecard

Two theory notes were derived by agents under an enforced blinding
instruction and committed before any 13b number was read
(`9adaea3`, `ac4b7ca`; results were auto-committed by the ops pipeline
$\sim 10$ minutes earlier, so the predictions are blind-to-results though
the results predate the commits on the clock — chronology auditable from
git plus agent transcripts, stated plainly in `SUMMARY_round13b.md`).

**`theory/matched_L0_invariance.md`** proves, inside the 2-D toy, a
*matched-$L_0$ agreement theorem*: under fixed-point (emergent) $L_0$
matching, the L1 and TopK absorption decisions coincide everywhere in
parameter space for all
$\lambda \le \lambda_c = 8 - 4\sqrt2 - \sqrt{92 - 64\sqrt2} \approx 1.1224$
(the unique root of $\varepsilon^*_{L1}(\lambda, q) = 2q$ in the validity
window), with a real disagreement window only for $\lambda > \lambda_c$
(machine-checked, 35/35). Its boxed 13b prediction — the P2 interaction
$\approx 0$ — was **FALSIFIED**: the CI $[+0.0014, +0.0135]$ excludes
zero. The note's pre-stated escape channel (L1 splitting consumes live
capacity $\Rightarrow$ positive sign; toy silent on sign if nonzero)
matches the observed sign, but that is an escape, not a hit: matched-$L_0$
single-pair agreement does not govern the scarce multi-pair regime. Its
conditional P1 prediction (absorption rises if scarcity binds) was
likewise **FALSIFIED-DIRECTION**. What stands is the agreement theorem as
toy mathematics and its account of the round-12/13a null in the
spare-capacity regime; what fails is any extension of toy
capacity-scarcity logic to this real endpoint.

**`theory/splitting_asymmetry.md`** derives the splitting asymmetry: at
matched per-token $L_0$ the objective's splitting preference is actually
*larger* for TopK, so the observed L1 excess is attributed to gating
dynamics — L1's **self-gated** nucleation gives a nascent sub-feature
latent first-order descent whenever the residual clears $\lambda/2$
($\lVert\rho\rVert^2 = \sin^2\varphi + \lambda^2/4 > (\lambda/2)^2$ for
every $\varphi > 0$), while TopK's **rank-gated** spare latent has exactly
zero gradient below the cutoff, making merged-plus-dead-spare a genuine
local minimum despite the split being globally better for TopK. Its blind
13b predictions largely **HIT**: families shrink as $m$ falls with the
sign persisting at every width (high confidence — hit); the paired
splitting gap shrinks monotonically ($+1.36 \to +1.01 \to +0.63$, medium
confidence — hit); quantitatively, TopK at $m{=}2048$ ($1.21 \in
[1.0, 1.25]$) and the gap bound ($+0.63 \le +0.7$) hit, while L1 at
$m{=}2048$ and $m{=}4096$ miss their predicted ranges at the high edge by
$0.04$ and $0.03$; the "gap tracks dead%, not $m$" exposure is consistent
with the data. Its mechanism is also the natural reading of P2's positive
sign. Scorecard symmetry, for the record: one note's primary boxed
prediction was falsified outright; the other's held with small edge
misses. Both outcomes are reported at equal prominence.

### 8b.6 Standing, and the construct-validity problem now at the center

- Round 12's **NOT CONFIRMED** stands and is now *doubly robust*: 13a
  removed the splitting-artifact explanation, 13b removed the
  spare-capacity explanation — the last pre-registered rescue.
- The toy capacity story does **not** transfer to this endpoint at real
  scale in the predicted direction; on this endpoint it is inverted.
- Since absorbed$+$lost is exactly "no sufficiently selective family
  latent fired", and that quantity falls $\sim 2.7\times$ as the
  dictionary shrinks $8\times$ while L1 families fragment with width (P3),
  the reading is that **the endpoint co-moves with fragmentation of the
  letter representation — a property of dictionary width — rather than
  with hierarchical merging.** The thesis-defense review raised the same
  concern independently (`reviews/DEFENSE_round13b_2026-07-25/`, Q2
  Finding 1): the metric flags "no family latent fired" without verifying
  that the missing mass is picked up by anything. The
  capacity profile makes that concern concrete rather than hypothetical.

  *Direction of the merge, stated carefully, because earlier drafts of this
  subsection had it backwards.* The letter feature ("starts with L") is the
  **general/parent** concept and a token such as *short* is the **child**;
  under the absorbed dictionary $\{a_p, a_m\}$ of §4 the joint event is coded
  by the composite $a_m$ alone, so it is the **parent latent that falls
  silent** while a token-specific composite carries the letter direction —
  exactly the "hole" in the *starts-with-S* latent reported by
  [@chanin2024absorption], and exactly what §1 describes. The family $F_L$ is
  built from letter *selectivity*, so it captures the general latents; a
  composite that fires on one token is not selective for the letter and is
  correctly outside $F_L$. The endpoint therefore fires precisely when the
  parent is silent, and the object to look for is a **composite**, not a
  parent. The round-14 scorer searches $\arg\max$ over latents *not* in
  $F_L$, which is the right target; only the prose here was inverted.
- The highest-value next experiment is therefore the **residual projection
  check** — on "absorbed" trials, does a parent latent's activation
  increase to carry the child's missing mass? It is post-hoc but cheap and
  decisive for the endpoint's meaning: all 48 round-13b weight files plus
  round 12's 16 are in hand. Until it is run, first-letter absorption
  rates (this project's and SAEBench-style metrics generally) should be
  read as composite fragmentation-sensitive quantities, not as validated
  measures of parent–child merging.

## 9. Related work

Chanin et al. [@chanin2024absorption] coined and mechanistically explained
absorption; their toy analysis proves any nonzero co-occurrence favors
absorption in a model without a child-solo rate — the closed-form boundary
$\varepsilon^*(\lambda, q)$ appears unclaimed. C$^2$R [@jin2026cross]
proves a per-sample sparsity preference ordering; Cui et al.
[@cui2025limits] give closed forms in an asymptotic limit without an $L_1$
term. Penalty-as-remedy is prior art (OrtSAE [@korznikov2025ortsae]; C$^2$R) —
the derived $\beta^*$, the anti-rotation obstruction, $\varepsilon^{**}(\beta)$,
and $p_0^*$ appear unclaimed. Matryoshka SAEs [@bussmann2025learning] reduce
absorption architecturally; feature hedging [@chanin2025feature] documents the
continuous tilt empirically. Classical sparse coding gives the positive
regime [@donoho2003optimally; @hillar2015uniqueness]. The
label-free frequency identifiability analysis draws on mixture/topic-model
identifiability [@arora2012learning; @anandkumar2014tensor;
@allman2009identifiability; @fu2016robust], whose common
linear-independence condition is exactly what full absorption degenerates.
For the detector arc: prior work groups SAE features into feature
families spanning levels of abstraction (O'Neill et al. [@oneill2024disentangling]),
clusters functional structure (Michaud et al. [@li2024geometry]), and
imposes hierarchy as a training constraint (Tree SAE [@cao2026tree];
HSAE [@luo2026from]); Chanin et al. measure absorption with NPMI given
probe-based labels; encoder holes and activation-histogram multimodality
were observed and proposed as detection signals in Chanin et al.'s
toy-model posts and Chanin & Till's "Broken Latents" [@chanin2024broken]. A label-free
post-hoc detector of absorbed *pairs* — geometry band + two-sided
co-firing lift + containment veto, with locked thresholds, held-out
validation, and a shuffled-firing null — appears unclaimed
(full sweep: `notes/novelty-round8-detector.md`). Seed-level feature
instability [@paulo2025sparse; @gerasimov2026unstable] contextualizes S1: the
seed-stable pair *clusters* we find persist even though individual
features generally do not. SynthSAEBench [@chanin2026synthsaebench] offers
ground-truth synthetic SAE evaluation without absorption metrics — a
natural future host for detector benchmarking.

The metric-criticism this paper belongs to is now an active line, and the
present contribution is a specific and complementary one. SAEBench
[@karvonen2025saebench] standardised the first-letter absorption score that
§8b measures. Chanin [@chanin2026sparse] audits SAEBench's metrics for
*reliability* — reseed noise, ground-truth correlation, discriminability —
and finds several unfit for use; Bal [@bal2026from] audits *recovery*
metrics for construct validity, showing that decoder-geometry alignment and
encoder-activation behaviour are separate claims and that up to 77% of
geometrically "recovered" features never fire. Neither asks the question in
§8b.7: whether the absorption endpoint, which infers a merge from the
*absence* of a firing selective latent, is accompanied by any carrier at
all. Leask et al. [@leask2025sparse] independently argue SAE features are
not canonical units, and hierarchical architectures
[@muchane2025incorporating] target the same pathology from the training
side.

## 10. Limitations and open problems

1. **Model scope.** The exact theory is nonnegative-$L_1$, orthonormal
   features, population loss; the no-go's overcomplete case ($m > d$) is open.
   **TopK** is now treated (round 10, `theory/topk_absorption.md`,
   `results/round10/SUMMARY.md`): for a *capacity-limited two-atom* dictionary
   there is an exact crossover $\varepsilon^*_{\rm TopK} = 2q$ with no $\lambda$
   (machine-checked, incl. the reviewer-supplied 3-atom zero-loss
   counterexample that scopes it to two atoms — an overcomplete dictionary
   escapes). The SGD test was a productive **negative** round: the $m{=}2$ arm
   is SGD-degenerate (so the two-atom crossover/collapse are inconclusive/
   falsified *at SGD level*, not as theorems); overcomplete TopK recovers the
   child but *less* cleanly than L1; and — the sharpest finding — in isolation
   *neither* architecture absorbs an $\varepsilon>0$ child, so this project's
   own L1 absorption is **background-competition-driven, not rarity alone**.
   The decisive open experiment — L1-vs-TopK *with* background — has since
   been run at real scale (rounds 12–13b, §8b): NOT CONFIRMED at matched
   $L_0$, robust to the splitting correction and to an $8\times$ capacity
   sweep, with the endpoint's construct validity now the binding limitation.
   JumpReLU remains untested.
2. **Semi-synthetic evidence only.** No positive natural-absorption
   observation exists in this project; real-activation results are injected
   pairs against real backgrounds. A pre-registered adjudication of the 15
   seed-stable un-injected candidate clusters (§8) returned a **null**: all are
   correlated/anti-correlated real-feature families (the CDX class), none an
   absorption hierarchy (max child$\to$parent containment $0.46 < 0.80$). A
   positive wild-catch would need a positive-cosine asymmetric-containment gate
   on an ASCII-clean corpus.
3. **Detector maturity.** Synthetic proof of concept. Open: cutoff transfer
   across widths/layers/models (round 8/8b), all-pairs specificity on
   un-injected real backgrounds, orientation (fails under prevalence
   inversion by construction; $\sim 5/9$ in Arm 2; round 9's
   swap-equivariance is an untested candidate signal), and
   production-scale false-positive control (multiple comparisons over
   $\sim m^2/2$ pairs). Frequency estimation given the pair passed its
   oracle-scoped mechanism endpoint decisively (round 9, same activation
   bank); its all-token version retains a measured background bias
   $w_B(h_B - \rho)$ — worst observed $0.07$ at $\rho = 0.1$, both
   operational predictions inconclusive as registered — and an
   $h_B$-corrected or background-excluded operational estimator is queued,
   not claimed.
4. **Statistical power.** 16-seed cells CI-establish only the strongest
   endpoint; confirmatory cells are sized $\ge 24$ seeds from round 8 onward.
5. **Oracle remedy.** Inverse-density weighting is an existence result;
   no validated label-free substitute yet (two refuted, one detector-based
   route in progress).

## References

<!-- Generated by pandoc --citeproc from refs.bib (arXiv works, produced by
     ops/mkbib.py straight from the arXiv API) and refs_manual.bib (journal and
     forum works). Do not hand-maintain a list here: the previous one drifted,
     and one entry cited a real but WRONG paper for over a round. -->

::: {#refs}
:::

---

*Repository map: theory (`theory/`), pre-registrations (`notes/prereg-*`),
experiment code (`experiments/`), raw results + summaries (`results/`),
review trail (`reviews/`), plan (`RESEARCH_PLAN.md`), environment
(`ENVIRONMENT.md`).*
