# Round 14 — the absorbed set has no consistent carrier

**Frozen evaluator output: `results_round14.txt`. Raw: `round14_results.json`.
Prereg `notes/prereg-round14-carrier.md` LOCKED at 708211f with the scorer and
evaluator; Amendment 1 (d2f32fa) fixed a words-file loader crash before any
round-14 number existed. 32 SAEs re-analysed (m=16384 primary, m=2048 contrast),
no new training.**

> ## ⚠ CORRECTION (2026-07-26, before publication) — read this first
>
> **Every registered number below stands. One inference drawn from them does not,
> and it was the headline.** An adversarial review
> (`reviews/GEMINI_round14_2026-07-26.md`, adjudicated in
> `reviews/RESPONSE_round14-review.md`) pointed out that P4 applies the *global
> modal* carrier `κ` to every absorbed trial while the control side uses the best
> *family* latent. Since `κ` is the top contributor on only 14.1% of trials, its
> mean share is small **by construction** — so P4 cannot tell "the mass is spread
> thinly" apart from "the mass is concentrated, on a different latent each trial".
>
> A post-hoc diagnostic recomputing concentration **per trial**, using each trial's
> own top non-family latent (`analysis/round14_validity.py`,
> `results_round14_validity.txt`), gives at m=16384:
>
> | | per-trial top share | 95% CI |
> |---|---|---|
> | absorbed trials (A) | **0.5935** | [0.5839, 0.6031] |
> | control trials (C) | 0.6322 | [0.5852, 0.6768] |
>
> Absorbed trials are **93.9% as concentrated as normally represented trials**, with
> ~32 distinct carriers serving ~96 absorbed trials. That is the signature of
> absorption distributed across many token-specific composites — which is what the
> Chanin mechanism actually predicts — **not** of representational loss.
>
> **Therefore the conclusion "the absence looks like loss / threshold suppression"
> is WITHDRAWN.** What survives is strictly narrower and is stated in
> "What this does and does not establish" below: there is no *single, broad,
> recurring* latent carrying the letter across absorbed trials. Carriage is
> trial-specific, and P2/P3/P4 measure exactly that and nothing more.
>
> Two further diagnostics, reported whichever way they fell:
> - **D1 = 0.0000.** The review's [BLOCKING] claim — that the argmax silently falls
>   through to an *inactive* latent — is rejected on the real weights, as it was on
>   synthetic data and by the distribution of `κ` over the dictionary.
> - **D3 = +0.0001.** I suspected the P2 null was inflated because the family is
>   excluded for the letter direction but nothing is excluded for a random
>   direction. Making the exclusion symmetric moves the null from 0.3484 to 0.3483.
>   **My objection was wrong**; the registered P2 comparator is fine.

## Registered verdicts

| | m=2048 | m=16384 |
|---|---|---|
| Gate 1 (reproduce 13b `rate_family` ≤0.002) | **PASS** — max \|d\| = 0.0016 | |
| **P1 (primary) compensation** | CONFIRMED (+3.19, CI [+2.45,+3.97]) | CONFIRMED (+1.54, CI [+1.35,+1.73]) |
| **P2 (primary) consistency** | NOT CONFIRMED (+0.072, CI [−0.064,+0.212]) | **NOT CONFIRMED (−0.199, CI [−0.217,−0.181])** |

**The two registered primaries disagree.** Both are reported as the frozen
evaluator produced them. The tie is not broken by picking the congenial one; it is
broken below by the secondaries and by a defect in P1 that I did not anticipate
when locking.

## P1 is confirmed and should not be believed

P1 selects `κ` as the modal carrier **on set A**, then compares `κ`'s activation on
A versus C. That conditions the statistic on the outcome: `κ` is chosen precisely
because `f_κ·(d_κ·u_L)` is largest on A, so `f_κ` is high on A *by construction*.
The comparison set C carries no such conditioning. A positive difference is close
to guaranteed regardless of whether absorption is occurring.

This is a design error in my own pre-registration, stated rather than quietly
fixed. **The registered verdict stands as CONFIRMED** — it is not being reversed
post hoc — but it is uninformative about the question, and it should not be cited
as evidence for absorption. Any successor must select `κ` on held-out trials or on
C, and compare on disjoint A-trials.

P3 corroborates that the P1 signal is selection: the carrier fires *less* often
than the family latents at every cell (m=16384: 0.0235 vs 0.0837), i.e. `κ` is a
**rarely-firing latent that happens to be large on the trials it was picked from**
— the opposite of the broad parent/composite the absorption story requires.

## P2, P3, P4 all say: no carrier

**P2 (consistency).** At m=16384 the modal carrier is the top contributor on only
**14.1%** of absorbed trials, against a random-direction null of **34.0%** — a
paired difference of −0.199, CI [−0.217, −0.181]. The letter direction is *worse*
than an arbitrary direction at picking out a repeated latent. (A random projection
is dominated by a few dense, high-firing latents, which is consistent; the letter
direction on absorbed trials is not.) At m=2048 the difference straddles zero.

**P3 (breadth).** Carrier fires less than the family at all four cells. Absorption
predicts the opposite.

**P4 (concentration).** On absorbed trials the modal carrier holds **9.2%** of the
positive letter-direction mass at m=16384 (24.9% at m=2048). On control trials the
top family latent holds **57.2%** (67.7%). ~~Absorbed trials are diffuse; normally
represented trials are concentrated.~~ **← WITHDRAWN, see the correction at the
top.** The 9.2% is the *global modal* carrier's share, which is small by
construction because that latent wins on only 14.1% of trials. Per trial, using
each trial's own top non-family latent, the share is **0.5935** against **0.6322**
on control trials — absorbed trials are *not* diffuse.

Together: when the family is silent, no *single recurring* latent picks the letter
up — but on each individual trial the letter direction's mass **is** concentrated,
on a latent that changes from trial to trial (~32 distinct carriers over ~96
absorbed trials). **That is consistent with absorption distributed over many
token-specific composites, and it is not evidence of representational loss.** The
earlier reading of this paragraph inverted that conclusion and is retracted.

## Power — the serious limitation

The registered |A| ≥ 20 floor excludes most cells:

| cell | scored | median \|A\| |
|---|---|---|
| L1 m=2048 | 26/192 | 4 |
| TopK m=2048 | 7/192 | 2 |
| L1 m=16384 | 81/192 | 12 |
| TopK m=16384 | 66/192 | 12 |

**m=2048 is effectively unpowered** — 7 of 192 TopK cells survive, median |A| = 2.
No m=2048 claim here should be relied on, including P1's large +3.19. Even at
m=16384 fewer than half of cells clear the floor and the median cell has |A| = 12,
below the floor. The absorbed set is simply small once you require enough trials to
estimate a modal carrier. P5's capacity contrast is therefore not interpretable and
is not claimed.

## What this does and does not establish

- It does **not** show absorption never occurs — and after the correction it does
  not point that way either. What it shows is that the trials this endpoint labels
  "absorbed" have **no single, broad, recurring carrier**: no one latent plays the
  parent role across trials. Each trial *does* have a concentrated carrier; the
  identity varies.
- It does **not** touch the architecture conclusions. Rounds 12/13a/13b stand
  unchanged either way.
- It **does not** support the strong form of the thesis defense's Finding 1
  (`reviews/DEFENSE_round13b_2026-07-25/`). The defense proposed that the metric
  infers absorption from an absence; that much is true by definition of the
  endpoint. But its decisive test — is the missing mass carried? — comes back
  **yes, trial-specifically**. The claim that "the absence looks like loss" was mine
  to make and mine to retract, and it is retracted.
- What remains of the construct-validity concern is the *other* line, which is
  untouched: 13b's endpoint is monotone in dictionary width and co-moves with
  fragmentation, and 13a/13b show the single-latent form inflates absorption by
  23–33% via splitting. Those stand on their own evidence.
- Consequence for the paper: PAPER §8b should describe the endpoint as measuring
  *"the letter is present in the reconstruction and no letter-selective latent
  fires"*, note that the mass is then carried by a varying token-specific composite,
  and **not** claim that this pattern is representational loss. The relevant
  contrast is single-broad-parent absorption (not supported) versus distributed
  compositional absorption (consistent with these data).

## Next

A successor round should (a) fix P1's selection defect by **sample-splitting within
A** — pick `κ` on half of A, evaluate on the disjoint half against C (this is the
reviewer's suggestion and is better than the select-on-C fix originally proposed
here, which risks choosing latents irrelevant to A); (b) raise |A| by pooling trials
across seeds within a letter rather than scoring per (SAE, letter); and (c) go after
the question the correction actually opens, which is now the interesting one: the
per-trial carriers are token-specific composites, so **are they the *right* ones?**
Ablate a trial's carrier and check the letter's effect disappears — the
interventional half of the question round 14 answered correlationally. Note
arXiv:2607.12166 (2026-07-13) already builds a general causal-audit instrument for
SAE features; start from it rather than rebuilding one, and keep the
absorption-specific angle, which it does not cover.

## Cost / ops

CPU-only ephemeral `r14` (e2-standard-8), ~1h wall clock, well under $1. VM
**deleted**. Scorer crashed once on a self-written words-file loader
(`KeyError: 'X'`; the key is `acts`) before reading any weight — fixed as
Amendment 1, pre-results, loader now copied verbatim from the 13b scorer.
