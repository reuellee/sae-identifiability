# Round 14 — the absorbed set has no consistent carrier

**Frozen evaluator output: `results_round14.txt`. Raw: `round14_results.json`.
Prereg `notes/prereg-round14-carrier.md` LOCKED at 708211f with the scorer and
evaluator; Amendment 1 (d2f32fa) fixed a words-file loader crash before any
round-14 number existed. 32 SAEs re-analysed (m=16384 primary, m=2048 contrast),
no new training.**

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
top family latent holds **57.2%** (67.7%). Absorbed trials are diffuse; normally
represented trials are concentrated.

Together: when the family is silent, the letter's residual presence is spread
thinly across many idiosyncratic latents rather than picked up by any identifiable
parent. **That is the signature of representational loss / threshold suppression,
not of hierarchical absorption into a parent latent.**

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

- It does **not** show absorption never occurs. It shows that *the trials this
  program's endpoint labels "absorbed"* do not carry the carrier signature that
  hierarchical absorption requires, at the one width with usable power.
- It does **not** touch the architecture conclusions. Rounds 12/13a/13b stand
  unchanged either way.
- It **does** support the thesis defense's Finding 1
  (`reviews/DEFENSE_round13b_2026-07-25/`): the first-letter absorption metric
  infers absorption from an absence and, tested directly, the absence looks like
  loss. Combined with 13b (endpoint monotone in dictionary width, co-moving with
  fragmentation), the construct-validity concern is now supported by two
  independent lines of evidence.
- Consequence for the paper: language in PAPER §8b and anywhere the SAEBench-style
  first-letter metric is called "absorption" needs to be scoped to what is actually
  measured — *the letter is present in the reconstruction and no selective latent
  fires* — rather than asserting a merge into a parent.

## Next

A successor round should (a) fix P1's selection defect as above, (b) raise |A| by
pooling trials across seeds within a letter rather than scoring per (SAE, letter),
and (c) test the decoder-side claim directly: regress the child's residual mass on
the parent latent's activation, rather than inferring from an argmax.

## Cost / ops

CPU-only ephemeral `r14` (e2-standard-8), ~1h wall clock, well under $1. VM
**deleted**. Scorer crashed once on a self-written words-file loader
(`KeyError: 'X'`; the key is `acts`) before reading any weight — fixed as
Amendment 1, pre-results, loader now copied verbatim from the 13b scorer.
