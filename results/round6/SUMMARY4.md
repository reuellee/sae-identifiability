# Capacity-limited test SUMMARY3


## m = 128
- vanilla eps=0.002 s0: child=0.747 comp=0.993 parent=0.985 route=1.0
- vanilla eps=0.002 s1: child=0.747 comp=0.995 parent=0.991 route=1.0
- vanilla eps=0.002 s2: child=0.744 comp=0.995 parent=0.99 route=1.0
- vanilla eps=0.002 s3: child=0.745 comp=0.994 parent=0.99 route=1.0
- vanilla eps=0.05 s0: child=0.955 comp=0.872 parent=0.986 route=1.0
- vanilla eps=0.05 s1: child=0.958 comp=0.869 parent=0.99 route=1.0
- vanilla eps=0.05 s2: child=0.953 comp=0.873 parent=0.991 route=1.0
- vanilla eps=0.05 s3: child=0.958 comp=0.863 parent=0.99 route=1.0
- oracle eps=0.002 s0: child=0.197 comp=0.726 parent=0.983 route=0.46
- oracle eps=0.002 s1: child=0.996 comp=0.734 parent=0.21 route=1.0
- oracle eps=0.002 s2: child=0.997 comp=0.706 parent=0.989 route=1.0
- oracle eps=0.002 s3: child=0.997 comp=0.71 parent=0.989 route=1.0
- oracle eps=0.05 s0: child=0.994 comp=0.704 parent=0.987 route=1.0
- oracle eps=0.05 s1: child=0.995 comp=0.708 parent=0.99 route=1.0
- oracle eps=0.05 s2: child=0.993 comp=0.707 parent=0.991 route=1.0
- oracle eps=0.05 s3: child=0.993 comp=0.701 parent=0.991 route=1.0
- bgrel eps=0.002 s0: child=0.755 comp=0.994 parent=0.986 route=1.0
- bgrel eps=0.002 s1: child=0.753 comp=0.995 parent=0.99 route=1.0
- bgrel eps=0.002 s2: child=0.75 comp=0.996 parent=0.99 route=1.0
- bgrel eps=0.002 s3: child=0.752 comp=0.995 parent=0.99 route=1.0
- bgrel eps=0.05 s0: child=0.952 comp=0.879 parent=0.988 route=1.0
- bgrel eps=0.05 s1: child=0.955 comp=0.878 parent=0.992 route=1.0
- bgrel eps=0.05 s2: child=0.95 comp=0.883 parent=0.993 route=1.0
- bgrel eps=0.05 s3: child=0.955 comp=0.873 parent=0.993 route=1.0

## 15b mechanism numbers (from run log bgrel_v2.log, committed here for auditability)
- background SAE median clean residual: 201.9 (normalized activations, m=128)
- injected event residual delta (analytic, AMP=5, novel directions): +25 (solo) to +50 (joint)
- resulting bg-relative weight leverage: ~(202+25..50)/202 ≈ 1.12–1.25x, vs oracle child-solo leverage 1/(4*eps) = 125x at eps=0.002
- and the weight cannot distinguish child-solo from parent-solo from joint (all similarly novel)
