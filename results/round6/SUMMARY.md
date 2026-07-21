# Direction 1+2 SUMMARY (auto-generated on dev-gpu)

## POC (injected pair on real GPT-2 acts): func_cos_child by cond/eps
- vanilla eps=0.002: func_cos_child=[0.45, 0.01, 0.43, -0.03] max_cos_child_med=0.82
- vanilla eps=0.01: func_cos_child=[0.7, 0.69, 0.67, 0.71] max_cos_child_med=0.83
- vanilla eps=0.05: func_cos_child=[0.9, 0.9, 0.89, 0.89] max_cos_child_med=0.90
- oracle eps=0.002: func_cos_child=[0.71, 0.69, 0.7, 0.74] max_cos_child_med=0.71
- oracle eps=0.01: func_cos_child=[0.04, 0.01, 0.05, -0.02] max_cos_child_med=0.72
- oracle eps=0.05: func_cos_child=[0.04, 0.03, -0.05, -0.02] max_cos_child_med=0.70
- residual eps=0.002: func_cos_child=[0.0, 0.85, 0.84, 0.84] max_cos_child_med=0.84
- residual eps=0.01: func_cos_child=[0.84, 0.65, 0.71, 0.69] max_cos_child_med=0.84
- residual eps=0.05: func_cos_child=[0.89, 0.9, 0.87, 0.89] max_cos_child_med=0.89

## Natural absorption (coverage gap, lower=better; mean over letters+seeds)
- vanilla: cov_gap=0.768 align=0.565 n_split=6.4
- residual: cov_gap=0.735 align=0.560 n_split=5.6
per-letter gaps (vanilla vs residual):
  a: 0.780 -> 0.754
  b: 0.835 -> 0.831
  h: 0.742 -> 0.605
  i: 0.750 -> 0.752
  o: 0.732 -> 0.738
  s: 0.893 -> 0.807
  t: 0.662 -> 0.644
  w: 0.752 -> 0.750

## Audit
source: jbloom/GPT2-Small-SAEs-Reformatted/blocks.6.hook_resid_pre/sae_weights.safetensors
hierarchical pairs found: 0
CAVEAT: latent pairs are post-absorption objects, not ground-truth features; this maps risk structure, not confirmed absorption.
