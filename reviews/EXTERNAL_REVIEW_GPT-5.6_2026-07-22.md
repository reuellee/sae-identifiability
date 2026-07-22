# External review — GPT 5.6 (via user, 2026-07-22, verbatim)

Review verdict: Major revision

I found the public repository reuellee/sae-identifiability and reviewed its report, theoretical verification code, experiment descriptions, results, no-go argument, remedy experiments, and internal referee material. I also independently checked the central two-dictionary loss calculation and numerically examined the optimized two-feature objective.

My overall assessment:

This is a promising and unusually transparent research note with a valid core toy-model result, but it currently overstates that result as an "exact phase transition" and as a broad solution to SAE identifiability.

The strongest contribution is the analytically solvable example showing that a standard L_1 SAE objective can prefer a composite or "absorbed" feature over the intended ground-truth features. The main weakness is that the exact expression called the phase boundary is only the equality point between two selected candidate dictionaries, not the true global transition of the continuously optimized dictionary.

I would consider it workshop-worthy after reframing, but not yet ready as a definitive research paper.

What holds up well

1. The basic non-identifiability construction is sound

At eps=0, the proposed hierarchical and composite latent stories generate the same observable activation distribution. No method operating only on those activations can determine which latent interpretation was "really" intended without additional assumptions.

That is a legitimate identifiability result, although it is fundamentally an observational-equivalence argument rather than something specific to SAE training. The report presents this distinction reasonably clearly.

2. The faithful-versus-absorbed loss calculation appears correct

For the simplified nonnegative L_1 sparse-coding objective, the event-level losses and the derived equality

eps_* = lam q (8 - 4 sqrt2 - lam) / (2 (1 - (2 - sqrt2) lam))

correctly identify where the stated faithful dictionary and 45-degree absorbed dictionary have equal expected loss. The symbolic verification script checks those formulas and reproduces the crossover.

This is useful because it gives a clean demonstration that reconstruction plus sparsity can favor a representation that conflicts with the assumed generative ontology.

3. The anti-rotation observation is insightful

A penalty based solely on pairwise decoder-vector inner products cannot distinguish between two orthonormal bases related by a rotation. Thus a faithful orthogonal frame and an absorbed-but-still-orthogonal frame can receive exactly the same coherence penalty.

That is a simple but valuable observation, especially as a warning against assuming that decoder incoherence alone guarantees semantic recovery.

4. The project documents negative results unusually well

The report records failed label-free weighting schemes, discrepancies between global optima and SGD outcomes, null findings in the natural-feature audit, and limitations of the real-activation experiments. That is substantially better than presenting only favorable runs.

The most important technical problem

The reported eps_* is not an exact global phase boundary

The manuscript repeatedly frames eps_* as an exact transition between absorption and faithful recovery. But the report's own global-angle scans show that the optimal second feature does not generally jump directly from 45 deg to 90 deg. It passes through intermediate tilted solutions—for example, angles around 69 deg, 80 deg, and 85 deg at different multiples of the claimed boundary.

The verification code also separates:

* the comparison L_absorbed < L_faithful, and
* the independently scanned global optimum angle.

It proves the first comparison but does not prove that global optimization switches between those two dictionaries at eps_*.

My independent numerical optimization agrees with the report's caveat: the global optimum starts tilting away from the exactly absorbed direction before eps_*, and remains tilted rather than becoming exactly faithful at eps_*.

So the mathematically defensible statement is:

eps_* is the exact loss crossover between the specified faithful and 45-degree composite candidate dictionaries.

It is not presently an exact global phase boundary.

There may still be a genuine bifurcation or phase-like change in the fully optimized landscape, but that transition would require a separate derivation over the continuous dictionary angle. The current theorem does not establish it.

Other major issues

1. The uniqueness claim is too strong

The report says that at eps=0 the absorbed dictionary is the unique global optimum over dictionaries of any size. But if a larger dictionary contains the required parent and composite directions, arbitrary additional atoms can remain unused and attain the same objective value.

Indeed, the report's own equality discussion effectively acknowledges that any dictionary containing the necessary directions can achieve the lower bound.

A correct formulation would be something like:

The active directions are uniquely determined, up to permutation and irrelevant unused atoms, within the stated model.

Or, more narrowly:

The absorbed solution is unique among minimal two-column dictionaries, modulo permutation.

2. The no-go result has a much narrower scope than the headline suggests

The formal argument applies to an undercomplete setting with m<=d, orthonormal frames, an L_1 objective, and a restricted two-dimensional reduction. The manuscript's search over mixtures involving background directions is numerical rather than a complete proof, and the overcomplete case remains open.

That matters because practical language-model SAEs are often:

* highly overcomplete,
* trained with TopK, JumpReLU, or related objectives,
* nonorthogonal,
* and subject to optimization and activation-distribution effects absent from the toy model.

The structural "Gram penalties are blind to rotations of an orthonormal frame" observation is broad. The stronger conclusion that coherence penalties cannot rescue practical SAEs is not established.

3. The "real-model" evidence is semi-synthetic

The GPT-2 experiments place a deliberately constructed synthetic parent/child pair into real model activations. This demonstrates behavior in a realistic background distribution, but it does not demonstrate that naturally occurring model features undergo the same mechanism.

The later natural audit reportedly found no qualifying hierarchical pairs, so the project currently has no positive natural-feature absorption result.

The paper should consistently call this:

Synthetic feature injection into real-model activations

rather than simply "real-data validation."

4. The proposed weighting remedy is oracle-dependent

Inverse-density or event-balanced weighting is theoretically informative: it shows that the failure can be eliminated if rare child-only events are given enough importance. But the method requires knowing which events are joint, child-only, and so forth—the latent structure the SAE is supposed to discover.

The report appropriately notes that:

* oracle weighting can help in the scarce-capacity injected setting,
* it can hurt in other regimes,
* two label-free estimators failed,
* and practical identification of the relevant event pairs remains open.

Therefore, this is best presented as a diagnostic existence result, not yet a practical remedy.

5. One capacity conclusion is not supported by the setup

In the experiment with 30 background features and two ground-truth pair features, an SAE width of m=32 has room for those 32 directions—but not for all 30 background directions plus parent, child, and composite, which would require 33 slots.

Consequently, the absence of the three-feature solution at m=32 cannot cleanly establish that optimization prefers absorption "without capacity scarcity." The architecture is still one slot short of the unconstrained triple solution identified by the theory.

That experiment should either be rerun at m>=33, with adequate excess capacity, or described explicitly as capacity-constrained.

Framing and presentation issues

"Machine-checked theory" is somewhat misleading. The repository contains symbolic and numerical verification scripts, not a proof in Lean, Coq, Isabelle, or another formal proof assistant. "Computationally verified derivations" would be more precise.

Similarly, the "independent referee review" was performed by a fresh-context Claude agent. It is useful adversarial checking, but it is not independent external peer review, and the referee itself identifies the undercomplete setting, oracle remedy, injected features, and lack of natural absorption as major limitations.

The repository's claim that the research program was completed in a single day also works against its credibility. The work is better positioned as an actively developed technical report with extensive exploratory experiments.

Recommended revised contribution

A more defensible title would be:

A Solvable Model of Feature Absorption in L_1 Sparse Autoencoders

The paper's main claims could then be:

1. At zero child-only probability, two different latent ontologies are observationally indistinguishable.
2. In a two-feature nonnegative L_1 sparse-coding model, there is an exact loss crossover between faithful and composite candidate dictionaries.
3. Continuous dictionary optimization produces intermediate tilted optima, yielding a richer numerical phase diagram.
4. Pairwise coherence penalties have a rotation blind spot in the undercomplete orthogonal-frame setting.
5. Oracle event reweighting can repair the toy-model failure, while practical label-free weighting remains unresolved.

Those claims are substantial and appear supportable without overstating the results.

Highest-priority revisions

1. Replace every use of "exact phase boundary" with "exact candidate-dictionary loss crossover," unless a full continuous-angle theorem is added.
2. Correct the arbitrary-dictionary-size uniqueness theorem.
3. Restrict the no-go claim explicitly to its proved undercomplete L_1 setting.
4. Separate three concepts throughout the paper:
    * statistical non-identifiability,
    * objective-induced representation preference,
    * optimization instability.
5. Rerun the capacity experiment with enough width for background, faithful features, and composite features simultaneously.
6. Label the GPT-2 work as semi-synthetic and add a benchmark with naturally occurring or externally annotated hierarchical features.
7. Add a pinned environment or lockfile, exact run commands, commit hashes for each result table, confidence intervals, and a clear accounting of independent versus shared-seed observations.

Bottom line

The work does not yet solve the general SAE identifiability problem, and it does not establish the advertised exact global phase transition. But beneath the overclaiming is a worthwhile result:

A simple sparse-coding objective can provably prefer a semantically absorbed representation, even when the assumed ground-truth features are available, and common geometric regularizers may be unable to distinguish the two in a restricted but informative setting.

That is a credible research contribution after major revisions. I did not rerun the complete GPU experiment suite, so my empirical assessment is based on the committed code, reported outputs, and consistency checks rather than a full independent replication.
