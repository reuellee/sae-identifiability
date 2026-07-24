#!/bin/bash
# Runs from cron on the orchestrator (every 30 min). Fires a Claude session ONCE,
# after drive_r12 has completed the run and committed the faithful results, to do
# the JUDGMENT layer the deterministic driver can't: verify the gates, enrich the
# SUMMARY narrative, update the plan/ledger, and confirm cleanup. Self-disables.
cd ~/sae-identifiability 2>/dev/null || exit 0
[ -f ~/r12_finalized ] && exit 0            # already finalized
[ -f ~/drive_r12_done ] || exit 0           # drive not finished yet
touch ~/r12_finalized                       # claim first (avoid double-run)
git pull -q 2>/dev/null

read -r -d '' PROMPT <<'PROMPT_END'
You are the orchestrator Claude session that took charge of the SAE-identifiability
research (public repo reuellee/sae-identifiability) after the laptop session closed.
Round 12 (pre-registered, LOCKED at commit 0722212) has just completed under the
deterministic driver. Read, in order: HANDOFF_round12.md, results/real/results_round12.txt
(the FROZEN scorer's verdict), results/real/SUMMARY_round12.md.

Do exactly this, changing NO frozen artifact (experiments/real_firstletter.py,
analysis/analyze_round12.py, the prereg predictions) and re-running NOTHING:
1. Verify from results_round12.txt that the gates are sound: matched-L0
   (|ΔL0|≤3, both near 32), config-conformance, the registered 8 seeds, and the
   matched-letter intersection. State whether P1 is CONFIRM / FALSIFIED /
   NOT-CONFIRMED as the scorer reports it — do not soften or overstate.
2. Rewrite results/real/SUMMARY_round12.md as an honest narrative: keep the
   frozen verdict verbatim, add the per-arch absorption + loss_rate + dead% +
   FVU, the P2 concentration (by arch, descriptive), P3 recall-vs-baseline, and
   what it means for the north star (geometry → identifiable codes → CAUSALLY
   valid features). If a gate failed or the result is negative/inconclusive, that
   is the headline — report it plainly.
3. Update RESEARCH_PLAN.md (mark round 12 complete with its outcome) and
   CLAIM_LEDGER.md (add the round-12 claim → lock → scorer → verdict row). If P1
   held, queue the registered next step: the deferred Chanin MODEL-BEHAVIOR
   causal test (round-12 P2 was reconstruction-space only).
4. Confirm the L4 is gone: run `gcloud compute instances list` — if dev-gpu-2
   still exists, delete it (`gcloud compute instances delete dev-gpu-2
   --zone=us-east1-b --quiet`). Never leave a GPU box running.
5. Commit and push everything (git identity user.email=reuellee@gmail.com
   user.name=reuellee). End with a one-paragraph status of what round 12 found.
PROMPT_END

echo "=== finalize_r12 firing claude $(date -u) ==="
claude -p "$PROMPT" --dangerously-skip-permissions < /dev/null 2>&1 | tail -40
echo "=== finalize_r12 done ==="
