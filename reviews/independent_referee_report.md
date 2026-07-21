# Independent referee report (fresh-context Claude agent, 2026-07-21)

Commissioned as a pre-publication adversarial review; reviewer had no role in
producing any result. Method: reran all four theory scripts, re-derived Lemma 2,
eps0, p0*, Theorems 1b/2, beta*, lam_crit by hand, cross-checked ten CSVs
(rounds 1-6) against every quantitative claim, hunted internal inconsistencies.

VERDICT: publishable with fixes. Zero blocking defects. All headline constants
reproduce exactly from committed dependency-free scripts (eps*=0.0486,
eps** 0.0112->~0.0159, lam_crit=0.714, p0*=0.294 located at 0.29-0.31); all
hand re-derivations correct; raw CSVs support the major experimental claims
including 48/48 weighted rescue. Six should-fix defects (undocumented OrtSAE
faithfulness criterion; missing p0<=sqrt2*q domain qualifiers in secs 9/12 and
README; general-derivation scope limits not surfaced in report body; a wrong-
beta loss citation in sec 8; an overstated 7/8 oracle count in sec 15; 15b
mechanism numbers not committed) and eight minor issues (figure links, SAE
count, coefficient slip in general_no_go.md, seed-variance smoothing, grid
precision, tilt-crossing prefactor unexplained, rule-of-thumb caveat,
'replicate exactly' phrasing) were identified. ALL FIXED in commit 61a41b1.

Reviewer's statement of the weakest load-bearing claim (preserved verbatim in
spirit): the no-go theorem as strictly proven covers orthonormal in-plane
frames in an L1 toy model with an undercomplete dictionary; the extension
beyond that family is a systematic search, not a proof; the overcomplete case
(the regime of production SAEs) is open; TopK/JumpReLU are not covered by the
proof (the soft-threshold mechanism is L1-specific); real-data support uses
injected synthetic pairs, and no result yet touches naturally occurring
absorption; the remedy is oracle-only pending a label-free estimator. These
scope limits are now stated in the report body (exec summary pt 6, secs 9/12).
