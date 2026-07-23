# Round-9 results-stage reviews: response (2026-07-23)

Reviews: `ROUND9_RESULTS_REVIEW_GEMINI-2.5-PRO_2026-07-23.md` (**ACCEPT**,
2 optional suggestions, both applied) and
`ROUND9_RESULTS_REVIEW_GPT-5.6_2026-07-23.md` (**MINOR REVISION → ACCEPT
after corrections**; the reviewer independently cloned the public repo,
verified the lock and the byte-identical frozen scorer, and reproduced all
six registered verdicts from the committed CSV). No verdict, bar, or
endpoint changed; all corrections are reporting/rhetoric.

## GPT-5.6 required changes → dispositions (all applied)

1. Formation arithmetic (SC "335/384" was wrong) → **fixed**: SC 254/288,
   RC 96/96, total 350/384; a revision note in SUMMARY.md discloses the
   original errors.
2. Baseline range → **fixed** to 0.083–0.494 across the leaky (RC + σ>0
   SC) cells.
3. P5 against the registered 0.10 bound with eligibility conditions →
   **fixed**: 8 eligible cells, 7 within ±0.10, SC ρ=0.1 σ=0.1 at +0.125;
   odds-formula predictions (≤0.006 from observed everywhere) in the
   appendix.
4. "exactly the predicted background bias and nothing else" → **replaced**
   with the diagnostic-vs-prediction distinction (a-priori estimate 0.045
   vs observed 0.0695; same-run decomposition residual ≤ 1e-4).
5. P3 reframed → **applied**: FALSIFIED leads; eligibility misprediction
   labeled a post-hoc diagnosis; the deterministic-host explanation labeled
   a hypothesis; "not by the estimator" removed.
6. Scope language ("fixed"/"solved"/unqualified "decisively") →
   **tightened** in SUMMARY, report §18 (title included), PAPER §8/§10 to
   oracle-scoped, preregistered-harness, same-bank statements.
7. "zero tuned constants" → **replaced** with "no new harness-tuned
   constant beyond the inherited θ = 0.05, under unit decoder
   normalization" everywhere.
8. h_B reporting → **fixed**: NA where background-active mass is
   negligible; w_B reported; range restated (RC 0.36–0.54; SC σ=0.1
   ≈ 0.49–0.51).
9. Reporting-only appendix → **added**:
   `analysis/round9_reporting_appendix.py` (dated, frozen-CSV input, no
   endpoint changes) + committed output `results/round9/
   REPORTING_APPENDIX.md` covering bootstrap CIs, signed bias, RMSE,
   θ-sensitivity, auto-orientation (directed + unordered), tie rates, P4
   contributor counts, P5 eligibility + predictions, MAE vs realized ρ,
   w_B/h_B, and the decomposition table.
10. Explicit overall operational verdicts → **stated** in SUMMARY, report
    §18, PAPER, and the plan table: P1O and P2O inconclusive overall.

Follow-up observations adopted: MAE-vs-realized-ρ transparency note
(mechanism endpoint chiefly validates the token-classification rule);
sparse σ=0 P4 contributor counts disclosed next to the P4 verdict;
θ-sensitivity spread reported with θ = 0.05 remaining primary; the
registered auto-orientation readout reported as *tested and failed as a
directed procedure above ρ = 0.5* (unordered estimate accurate) rather
than "untested"; same-bank limitation kept adjacent to the mechanism
claim.

Suggestions noted for future rounds: generate SUMMARY tables from a
script; preregister a second activation bank/layer/model (the natural
C-dom transfer test); harden P4 scoreability with minimum contributor
counts.

## Gemini suggestions → dispositions

1. Frame the P3 falsification as pre-registration working as intended →
   **applied** (SUMMARY).
2. Quantify "decisively" as >10× under the locked bar in all 16 cells →
   **applied** (SUMMARY).
