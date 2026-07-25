#!/usr/bin/env bash
# THESIS DEFENSE — put the paper in front of a hostile examiner (Gemini 2.5 Pro)
# and make it answer for its weakest claims, one attack surface at a time.
#
# WHY A DEFENSE RATHER THAN A "REVIEW"
# This program has already had four GPT reviews and two Gemini reviews, and they
# earned their keep (the eps* pure-strategy correction, the oracle-touch/exact-pair
# mismatch, the E3 angle=0.5 diagnosis, seed-bootstrap CIs). But a review asks
# "is anything wrong here?", which invites a list of local nits. A defense asks
# "this specific claim is the load-bearing one -- justify it or drop it", which is
# the question that actually decides whether the paper survives peer review.
#
# WHY SINGLE-SHOT PER QUESTION
# gemx's multi-step agentic loop is known-flaky on this box ("Invalid stream:
# malformed tool call"), and in batch it has eaten stdin inside while-read loops.
# So: one self-contained prompt per question, full context inlined, `< /dev/null`,
# a timeout, and retries. Each answer is checkpointed to its own file immediately,
# so a hang costs one question and not the run.
#
# DISCIPLINE
# The examiner is asked to self-classify every finding as
#   [PRE-RESULTS-OK]   can be acted on without touching a locked analysis
#   [POST-HOC-ONLY]    would be a post-hoc reanalysis; may only ever be reported
#                      as exploratory, never as a registered result
#   [PREREG-VIOLATION] acting on it would break the lock (e.g. moving a threshold
#                      after seeing the data, restoring a withdrawn claim)
# because the single most dangerous thing an articulate critic can do to this
# project is talk it into an unregistered reanalysis that happens to look better.
# Findings are NOT auto-adopted. Claude adjudicates each one into a RESPONSE file
# with evidence, per the existing reviews/RESPONSE_* convention.
#
# Usage: ops/defense.sh <tag> [question-numbers...]
#        ops/defense.sh round13b          # all questions
#        ops/defense.sh round13b 3 5      # just those
set -uo pipefail

TAG=${1:?usage: defense.sh <tag> [question-numbers...]}; shift || true
REPO=${REPO:-$HOME/sae-identifiability}
DATE=$(date -u +%F)
OUT=$REPO/reviews/DEFENSE_${TAG}_${DATE}
TIMEOUT=${TIMEOUT:-420}
RETRIES=${RETRIES:-3}
mkdir -p "$OUT"
cd "$REPO" || exit 2

# ---------------------------------------------------------------- context
ctx() { for f in "$@"; do [ -f "$f" ] && { echo "===== FILE: $f ====="; cat "$f"; echo; }; done; }
PAPER=$(ctx PAPER.md)
PREREG=$(ctx notes/prereg-round13b-capacity.md notes/prereg-round13a-family-endpoint.md)
RESULTS=$(ctx results/real/SUMMARY_round13a.md results/real/results_round13b.txt \
             results/real/results_round12_clean.txt results/real/round12_posthoc_diagnosis.txt)

ROLE='You are the hostile external examiner at a thesis defense for a mechanistic-
interpretability paper on sparse-autoencoder feature absorption. You are a
published expert in dictionary learning, identifiability, and experimental
statistics. You are not here to be encouraging. Your job is to find the claim
that will not survive peer review and force the candidate to defend or withdraw it.

Rules of engagement:
- Attack the LOAD-BEARING claim, not typos or presentation.
- Quote the exact sentence or number you are attacking.
- For each finding give: (a) the claim, quoted; (b) why it fails or is unsupported;
  (c) the specific evidence or experiment that would settle it; (d) severity
  BLOCKING / MAJOR / MINOR.
- Then classify each finding as exactly one of:
  [PRE-RESULTS-OK] actionable without touching any locked analysis;
  [POST-HOC-ONLY] would require reanalysing already-collected data, so it could
    only ever be reported as exploratory;
  [PREREG-VIOLATION] acting on it would break a pre-registration lock (moving a
    threshold after seeing data, restoring a previously withdrawn claim, or
    conditioning on a function of the outcome).
- If a claim is actually well-supported, say so plainly and move on. Do not
  manufacture objections to appear rigorous. A short, correct answer beats a long
  hedged one.
- You may not propose that the authors re-tune a threshold to make a result pass.'

# ---------------------------------------------------------------- questions
q_name() { case $1 in
  1) echo "central-null-informative-or-underpowered";;
  2) echo "endpoint-validity-vs-selectivity";;
  3) echo "toy-to-real-transfer";;
  4) echo "prereg-integrity-and-amendments";;
  5) echo "statistical-inference-seeds-and-multiplicity";;
  6) echo "detector-precision-at-scale";;
  7) echo "generalisation-single-model-single-task";;
  8) echo "weakest-claim-and-what-would-sink-it";;
esac; }

q_body() { case $1 in
1) cat <<'Q'
THE CENTRAL NULL. Rounds 12 and 13a report no L1-vs-TopK absorption gap on
Pythia-1.4B L12 at matched L0 (clean paired diff +0.0030, CI [-0.0010,+0.0067];
family endpoint -0.0012, CI [-0.0081,+0.0049]).

Interrogate whether this is an INFORMATIVE null or merely an UNDERPOWERED one.
What is the minimum effect this design could have detected, and is that effect
small enough to be scientifically interesting? The paper leans on a seed-level CI
"excluding a gap >25% of the 0.055 base" -- is that reasoning sound? Where does
the paper slide from "we did not detect a difference" to "there is no
difference"? Quote any sentence that overclaims.
Q
;;
2) cat <<'Q'
ENDPOINT VALIDITY. The first-letter absorption endpoint was found to be
R^2 = 0.673 a re-expression of main-latent selectivity. The family-based
correction (round 13a) reduced this to R^2 = 0.381 -- still 38% shared variance --
while removing 25.2% of measured absorption as splitting.

Is this endpoint measuring ABSORPTION, or is it measuring SELECTIVITY under
another name? If the latter, what exactly does the paper's real-data arc
establish? Note that the authors computed a regression of absorption rate on
architecture CONTROLLING for selectivity, found it flipped the sign, and then
DISCARDED it as invalid because it conditions on a function of the outcome
(rate ~ (1-FPR) - sel). Was discarding it correct? If you think that analysis
should be resurrected, say so explicitly and justify why it is not collider
conditioning.
Q
;;
3) cat <<'Q'
TOY-TO-REAL TRANSFER. The paper has two halves: an exactly solvable toy model
yielding closed forms (absorption iff eps < eps*(lam,q); coherence-penalty
critical beta*; the anti-rotation no-go; the critical ratio p0*/q -> sqrt(2)),
and a real-model arc on Pythia-1.4B that returns nulls for the architecture
prediction.

Make the case that these two halves are stapled together rather than integrated:
that no quantitative prediction of the toy model is actually TESTED at real
scale, and that the real-model sections would read the same if the theory were
deleted. What is the strongest version of that criticism? Then state what the
authors could legitimately claim about the theory given the real-data results
they actually have. Is "a solvable model of feature absorption" an honest title
for this artifact?
Q
;;
4) cat <<'Q'
PRE-REGISTRATION INTEGRITY. The program pre-registers rounds, freezes scorers and
evaluators at a lock commit, and declares amendments "PRE-RESULTS". Round 13b
declared two amendments after the lock: adaptive lambda calibration replacing a
fixed grid, and dropping the m=8192 cell for budget reasons.

As a skeptic who does not trust the authors' self-report: how would you VERIFY
that the amendments preceded any outcome measurement, and what could a
motivated author do that this process would fail to catch? Is "calibration reads
only L0 and is blind to absorption" actually airtight, given that L0, FVU and
dead-fraction are all correlated with the endpoint? Is dropping a cell for budget
a legitimate amendment or a degree of freedom? Say concretely what additional
artifact would make the lock externally auditable.
Q
;;
5) cat <<'Q'
STATISTICAL INFERENCE. Effects are estimated over 8 seeds with 10k-resample
bootstrap CIs. An earlier review already forced the concession that only one
detector endpoint (D3) is CI-established and the rest are point estimates.

Attack the inferential basis: (a) the bootstrap resamples SEEDS, holding the
activation dataset fixed -- what does the resulting CI actually cover, and what
does it NOT cover? (b) across rounds the program has reported many endpoints
(P1-P5 per round, multiple rounds) with no multiplicity control -- how should
that change how any single "CI excludes 0" result is read? (c) is a paired
difference-of-differences on 8 seeds (round 13b P2) capable of supporting any
conclusion, and if not, should it have been run at all?
Q
;;
6) cat <<'Q'
DETECTOR UTILITY. The pair-level label-free detector is reported as positive at
real scale (round 12 P3): recall 0.333 vs 0.030 baseline for TopK (11.2x) and
0.812 vs 0.153 for L1 (5.3x), but precision 0.007-0.025. Scaling metrics
elsewhere give ~214 false positives per million pairs and precision 0.81/0.30/0.04
at prevalence 1e-3/1e-4/1e-5.

At those precisions, is the detector useful for anything a practitioner would
actually do? The paper hedges it as a "synthetic proof of concept" rather than a
constructive solution. Is even that framing supported by the real-data numbers,
or should the detector arc be demoted further? Distinguish clearly between
"enrichment over baseline is statistically real" and "this is a usable tool".
Q
;;
7) cat <<'Q'
GENERALISATION. The real-data arc is one model (Pythia-1.4B), one layer (12), one
task family (first-letter absorption, the Chanin/SAEBench metric), one activation
cache, two architectures, 8 seeds.

State the strongest case that nothing here generalises -- including any concerns
about the first-letter metric itself as a construct, and about layer-12 residual
activations being unrepresentative. Then, being fair: which of the paper's claims
ARE appropriately scoped to survive this criticism as stated, and which are
currently phrased more broadly than the evidence permits? Quote the offending
phrasings.
Q
;;
8) cat <<'Q'
THE DECISIVE QUESTION. Two parts, answer both directly.

(a) Of everything in this paper, name the SINGLE claim most likely to be wrong or
to be withdrawn within a year, and say what you would need to see to believe it.

(b) The program's headline real-scale results are nulls: no architecture gap, two
label-free estimators refuted, a natural-feature adjudication that returned a
pre-registered null, and a binarized-signature no-go. Is this body of work
publishable and worth a reader's time as it stands -- at a workshop, on
LessWrong/Alignment Forum, or as an arXiv note -- or is it a well-documented
record of a research direction that did not pan out? Give a venue recommendation
and the one change that would most increase its value to a reader. Be blunt.
Q
;;
esac; }

# ---------------------------------------------------------------- run
QS=${*:-1 2 3 4 5 6 7 8}
echo "DEFENSE $TAG -> $OUT"
for i in $QS; do
  NAME=$(q_name "$i"); F="$OUT/$(printf '%02d' "$i")-$NAME.md"
  if [ -s "$F" ]; then echo "  [$i] already answered, skipping"; continue; fi
  echo "  [$i] $NAME ..."
  PROMPT="$ROLE

You are given the paper and its supporting pre-registrations and result files.

$PAPER
$PREREG
$RESULTS

===== YOUR QUESTION =====
$(q_body "$i")"
  ok=0
  for try in $(seq 1 "$RETRIES"); do
    # `< /dev/null`: gemx eats stdin in loops. timeout: agentic mode can hang.
    if timeout "$TIMEOUT" gemx -p "$PROMPT" > "$F.tmp" 2>"$F.err" < /dev/null \
       && [ -s "$F.tmp" ]; then
      { echo "# Defense $TAG — Q$i: $NAME"; echo; echo "_Examiner: Gemini 2.5 Pro via gemx. Generated $(date -u +%FT%TZ). Single-shot._"; echo; cat "$F.tmp"; } > "$F"
      rm -f "$F.tmp" "$F.err"; ok=1; echo "      -> $(wc -l < "$F") lines"; break
    fi
    echo "      attempt $try failed$( [ -s "$F.err" ] && echo ": $(tail -1 "$F.err")")"
    sleep 10
  done
  [ "$ok" -eq 0 ] && echo "      GAVE UP on Q$i (known gemx flakiness; rerun: ops/defense.sh $TAG $i)"
done

# ---------------------------------------------------------------- index
{
  echo "# Thesis defense — $TAG ($DATE)"
  echo
  echo "Examiner: Gemini 2.5 Pro (\`gemx\`), single-shot per question, hostile-examiner framing."
  echo "Findings are NOT adopted as written. Each is adjudicated with evidence in the"
  echo "matching \`RESPONSE_DEFENSE_${TAG}_${DATE}.md\`, per the reviews/RESPONSE_* convention."
  echo
  echo "Every finding is self-classified by the examiner as [PRE-RESULTS-OK],"
  echo "[POST-HOC-ONLY] or [PREREG-VIOLATION]. Anything in the latter two categories"
  echo "may inform future pre-registrations but must never be retro-fitted to a locked round."
  echo
  for i in 1 2 3 4 5 6 7 8; do
    N=$(q_name "$i"); F="$OUT/$(printf '%02d' "$i")-$N.md"
    [ -s "$F" ] && echo "- [Q$i $N]($(basename "$F"))" || echo "- Q$i $N — NOT ANSWERED"
  done
} > "$OUT/README.md"
echo "index -> $OUT/README.md"
