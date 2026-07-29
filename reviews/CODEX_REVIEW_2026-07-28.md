# Codex whole-repo research review — 2026-07-28

## Overall verdict

**Major revision, but with a credible publishable theoretical core.** The exact
toy-model work is the strongest contribution; the real-model evidence is
interesting but currently supports narrower claims than the summaries sometimes
make.

## Prioritized findings

### 1. Round 15 does not establish model-level transfer

The only released SAE per configuration means the confidence interval is over
23 letters, not SAE training runs. The bootstrap therefore measures
letter-to-letter heterogeneity conditional on one model, one word sample, and one
SAE per width — not uncertainty over retraining or external suites
(`notes/prereg-round15-gemmascope-crossval.md:193-199`,
`analysis/analyze_round15.py:21-31`).

Consequently, the statement that the result establishes the width effect “is not
an artifact of our training recipe” is too strong
(`results/real/SUMMARY_round15.md:100-103`). What is supported is:

> The positive within-suite width association reproduced in one independently
> trained JumpReLU suite.

Multiple SAE seeds at each width — or replication across several released suites
— are needed for a transfer claim.

### 2. Round 14's “carrier” conclusion is not identified by its statistic

The exploratory diagnostic computes:

```text
largest positive latent contribution / total positive contribution
```

and excludes trials with no positive mass
(`analysis/round14_validity.py:71-83`). This measures concentration conditional
on some positive projection. It does not measure:

- how much letter-direction mass exists;
- whether it explains the reconstruction's probe margin;
- whether negative contributions cancel it;
- whether the selected latent causally mediates the letter information.

Thus 0.5935 versus 0.6322 supports “the positive projection is similarly
concentrated,” but not yet “the letter's mass is carried by a token-specific
composite,” as claimed in `PAPER.md:763-768`.

This distinction matters because the original absorption study used latent
ablation and projection removal to establish causal carriage, not concentration
alone:
https://arxiv.org/html/2409.14507v6

A successor should report signed and absolute contribution magnitude relative
to the raw probe margin and perform held-out carrier selection plus ablation.

### 3. Round 13b's pooled P1 confidence interval uses the wrong clustering unit

The preregistration says paired-by-seed contrasts pooled across architectures
and bootstrapped over seeds. Instead, the evaluator appends eight L1 and eight
TopK differences and treats the resulting 16 observations as independent
(`analysis/analyze_round13b.py:150-162`).

I recomputed the contrast after averaging architectures within each of the eight
seed clusters:

- reported: -0.0445, bootstrap CI [-0.0493, -0.0397];
- seed-clustered t interval: **-0.0445, CI [-0.0515, -0.0374]**.

The substantive falsified-direction verdict survives comfortably, but the
published inferential procedure should be corrected.

### 4. Round 15 validates implementation on one domain while scoring on a poorly reconstructed different domain

The registered word-domain FVU gate was replaced after the pilot with an
in-distribution corpus gate
(`notes/prereg-round15-gemmascope-crossval.md:158-178`). That correctly
demonstrates that the loader and SAE implementation work, but the actual isolated
BOS+word evaluation has FVU 0.40–0.46
(`results/real/results_round15.txt:2-4`).

Therefore the result is conditional on a strong domain shift. The similar
16k/262k FVUs make a trivial reconstruction-quality explanation less likely,
but a prompt- or sequence-based replication matching the SAE's training
distribution would materially strengthen the result.

### 5. The novelty is narrower than the current narrative sometimes implies

The literature already contains:

- an analytical proof that hierarchical co-occurrence encourages absorption,
  plus experiments with imperfect co-occurrence and TopK SAEs:
  https://arxiv.org/html/2409.14507v6
- theoretical and empirical treatment of correlated-feature merging under
  insufficient width: https://arxiv.org/abs/2505.11756
- closed-form SAE recovery theory and a reweighted remedy:
  https://arxiv.org/html/2506.15963v2
- a theoretical cross-sample account and regularizer for absorption and
  splitting: https://arxiv.org/html/2606.30609v1

The likely novel theoretical pieces remain valuable:

- the explicit \(\varepsilon^*(\lambda,q)\) pure-candidate boundary;
- the active-direction characterization at \(\varepsilon=0\);
- the coherence-penalty rotation blind spot and occurrence-ratio boundary;
- the detailed distinction between decoder and code identifiability.

But “appears unclaimed” should be backed by a theorem-by-theorem comparison
table, not a general literature paragraph.

### 6. The publication artifacts are out of sync

The formal paper says it is current through Round 14 (`PAPER.md:3-10`); the claim
ledger includes Round 15 (`CLAIM_LEDGER.md:21`); and the README still says eleven
rounds and revised through July 24 (`README.md:7-15`). The reproduction section
also primarily documents early rounds, while the artifact manifest covers Round
11 only.

Before submission, generate all public-facing documents from one claim ledger
and provide a single reproducibility entry point.

### 7. The crossover domain needs to accompany the headline formula

The displayed absorbed child-solo loss — and hence the formula — uses an active
branch valid only for \(\lambda<\sqrt2\) (`PAPER.md:150-155`). The body discloses
this correctly, but the abstract presents the formula without the condition.
Put the domain directly in the theorem and abstract.

## What held up well

- The symbolic formula checker passed every registered identity.
- The frozen Round 13b evaluator reproduced its committed output.
- The frozen Round 15 evaluator reproduced `results_round15.txt` exactly.
- The Round 14 synthetic validity self-test passed.
- Negative and inverted results are reported unusually transparently.
- The distinction between the pure-candidate crossover and the continuously
  tilted optimum is handled correctly.
- The \(\varepsilon=0\) observational-equivalence argument is clean and
  convincing.
- The repository is much more auditable than most exploratory SAE research.

## Verification scope

Reviewed at commit `a4a89c3`.

The review inspected the formal paper, README, claim ledger, preregistrations,
frozen evaluators, experiment/scoring code, committed raw JSON outputs, result
summaries, review trail, and relevant primary literature.

Checks performed:

- `theory/symbolic_verify.py`: passed all registered formula identities.
- `analysis/analyze_round13b.py` on committed `round13b_results.json`:
  reproduced the committed verdicts and numbers.
- `analysis/analyze_round15.py` on committed `round15_rows.json` and
  `indist.json`: reproduced `results_round15.txt` exactly.
- `analysis/round14_validity_selftest.py`: passed.
- Independent seed-clustered recomputation of Round 13b P1, reported above.

The GPU training and long global numerical scans were not rerun. No scientific
failure is inferred from that omission.
