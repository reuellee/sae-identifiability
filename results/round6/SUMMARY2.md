# Disambiguation SUMMARY2


## A. POC v2: realistic probes + routing metric
- vanilla eps=0.002 s0: dec[child=0.82 comp=0.96] route(best-child fires|csolo)=0.57 argmax_latent_child_cos=0.01
- vanilla eps=0.002 s1: dec[child=0.85 comp=0.91] route(best-child fires|csolo)=0.84 argmax_latent_child_cos=0.01
- vanilla eps=0.002 s2: dec[child=0.83 comp=0.97] route(best-child fires|csolo)=0.67 argmax_latent_child_cos=0.01
- vanilla eps=0.002 s3: dec[child=0.82 comp=0.96] route(best-child fires|csolo)=0.46 argmax_latent_child_cos=0.01
- vanilla eps=0.01 s0: dec[child=0.82 comp=0.97] route(best-child fires|csolo)=0.57 argmax_latent_child_cos=0.00
- vanilla eps=0.01 s1: dec[child=0.86 comp=0.94] route(best-child fires|csolo)=0.75 argmax_latent_child_cos=0.01
- vanilla eps=0.01 s2: dec[child=0.82 comp=0.96] route(best-child fires|csolo)=0.47 argmax_latent_child_cos=0.01
- vanilla eps=0.01 s3: dec[child=0.83 comp=0.96] route(best-child fires|csolo)=0.48 argmax_latent_child_cos=0.00
- vanilla eps=0.05 s0: dec[child=0.90 comp=0.96] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.01
- vanilla eps=0.05 s1: dec[child=0.90 comp=0.96] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.01
- vanilla eps=0.05 s2: dec[child=0.89 comp=0.92] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.02
- vanilla eps=0.05 s3: dec[child=0.87 comp=0.93] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.01
- oracle eps=0.002 s0: dec[child=0.79 comp=0.81] route(best-child fires|csolo)=0.46 argmax_latent_child_cos=0.04
- oracle eps=0.002 s1: dec[child=0.73 comp=0.92] route(best-child fires|csolo)=0.42 argmax_latent_child_cos=0.04
- oracle eps=0.002 s2: dec[child=0.74 comp=0.89] route(best-child fires|csolo)=0.45 argmax_latent_child_cos=0.05
- oracle eps=0.002 s3: dec[child=0.68 comp=0.92] route(best-child fires|csolo)=0.18 argmax_latent_child_cos=0.04
- oracle eps=0.01 s0: dec[child=0.75 comp=0.90] route(best-child fires|csolo)=0.51 argmax_latent_child_cos=0.05
- oracle eps=0.01 s1: dec[child=0.74 comp=0.96] route(best-child fires|csolo)=0.08 argmax_latent_child_cos=0.05
- oracle eps=0.01 s2: dec[child=0.72 comp=0.94] route(best-child fires|csolo)=0.76 argmax_latent_child_cos=0.05
- oracle eps=0.01 s3: dec[child=0.71 comp=0.96] route(best-child fires|csolo)=0.07 argmax_latent_child_cos=0.05
- oracle eps=0.05 s0: dec[child=0.69 comp=0.91] route(best-child fires|csolo)=0.45 argmax_latent_child_cos=0.05
- oracle eps=0.05 s1: dec[child=0.72 comp=0.90] route(best-child fires|csolo)=0.43 argmax_latent_child_cos=0.05
- oracle eps=0.05 s2: dec[child=0.71 comp=0.93] route(best-child fires|csolo)=0.57 argmax_latent_child_cos=0.05
- oracle eps=0.05 s3: dec[child=0.68 comp=0.89] route(best-child fires|csolo)=0.54 argmax_latent_child_cos=0.07
- residual eps=0.002 s0: dec[child=0.83 comp=0.97] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.00
- residual eps=0.002 s1: dec[child=0.83 comp=0.97] route(best-child fires|csolo)=0.52 argmax_latent_child_cos=0.00
- residual eps=0.002 s2: dec[child=0.81 comp=0.97] route(best-child fires|csolo)=0.60 argmax_latent_child_cos=0.01
- residual eps=0.002 s3: dec[child=0.81 comp=0.97] route(best-child fires|csolo)=0.68 argmax_latent_child_cos=0.01
- residual eps=0.01 s0: dec[child=0.81 comp=0.97] route(best-child fires|csolo)=0.52 argmax_latent_child_cos=0.00
- residual eps=0.01 s1: dec[child=0.84 comp=0.96] route(best-child fires|csolo)=0.58 argmax_latent_child_cos=0.01
- residual eps=0.01 s2: dec[child=0.82 comp=0.97] route(best-child fires|csolo)=0.61 argmax_latent_child_cos=0.01
- residual eps=0.01 s3: dec[child=0.82 comp=0.97] route(best-child fires|csolo)=0.53 argmax_latent_child_cos=0.01
- residual eps=0.05 s0: dec[child=0.89 comp=0.95] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.01
- residual eps=0.05 s1: dec[child=0.89 comp=0.97] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.00
- residual eps=0.05 s2: dec[child=0.88 comp=0.95] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.02
- residual eps=0.05 s3: dec[child=0.88 comp=0.95] route(best-child fires|csolo)=1.00 argmax_latent_child_cos=0.01

## B. natabs v2 with firing-rate controls
- vanilla s0: global_L0=64.3  cov_gap_mean=0.770  offletter_fire_mean=0.0092
- vanilla s1: global_L0=63.8  cov_gap_mean=0.767  offletter_fire_mean=0.0097
- residual s0: global_L0=66.1  cov_gap_mean=0.732  offletter_fire_mean=0.0115
- residual s1: global_L0=65.1  cov_gap_mean=0.738  offletter_fire_mean=0.0126

## C. audit v2 (statistical criteria) on public GPT-2 SAE
- hierarchical pairs found: 0

total 2297s