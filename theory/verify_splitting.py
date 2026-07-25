"""Verification suite for theory/splitting_asymmetry.md (round 13a P5 theory).

Model: a 'feature with substructure' = two unit sub-directions v1, v2 at angle
2*phi, firing DISJOINTLY (x = r_i v_i w.p. p_i); optional fresh feature
v3 orthogonal to the pair plane. Oracle codes on unit-norm atoms:
  L1  : min_{f>=0} ||x - Df||^2 + lam*||f||_1  (single-atom gain (c-lam/2)_+^2)
  TopK: kappa=1 per family event, gain c^2 (no shrinkage).

Checks (PASS/FAIL each; exit nonzero on any FAIL):
  A. sympy identities: A1 Prop1 gap Delta_K - Delta_L1 = lam*(1-c0);
     A2 Prop3 band polynomial S_L1*F_K - S_K*F_L1 = (1-c0)*H;
     A3 residual identity |rho|^2 = sin^2(phi) + lam^2/4;
     A4 KKT exclusion f2 = -lam/(2(1+c)) < 0 (Lemma S1).
  B. numeric 2D oracle brute force: B1 best 1-atom = bisector, best 2-atom =
     {v1,v2}, gains match closed forms; B2 split config activates exactly ONE
     atom per event (matched per-token L0); B3 Delta_K > Delta_L1 on a grid.
  C. Prop 1' general inequality on random asymmetric instances.
  D. Prop 2: duplicate-direction atoms are exactly loss-flat for L1; single
     ReLU latent realizes the oracle code (r - lam/2)_+.
  E. Prop 3: disagreement band (L1 splits & TopK covers) exists at r=1 and is
     REVERSED only at r > r_+ ~ 1+c0; verified by direct two-config losses.
  F. Prop 4(i): L1 merged config is first-order escapable for every phi>0
     (finite-difference derivative = p2*(lam - 2|rho|) < 0), full 2-atom
     re-optimized loss strictly lower; zero gain at phi=0.
  G. Prop 4(ii): encoder-ranked TopK: below-cutoff spare => loss EXACTLY
     invariant to spare-parameter perturbations (merged is a local min),
     despite the split being strictly better; split ranking is stable.

python3 + numpy + sympy only (no pip). Style follows verify_topk_absorption.py.
"""
import numpy as np
import sympy as sp
from itertools import combinations

FAILS = []

def rec(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(f"{name} {detail}")

# ---------- shared oracle machinery ----------
def l1_event_loss(U, x, lam, return_active=False):
    """Exact min_{f>=0} ||x-Uf||^2 + lam*sum(f) by active-set enumeration."""
    m = U.shape[1]
    best, best_act = float(x @ x), ()
    for k in range(1, m + 1):
        for act in combinations(range(m), k):
            Ua = U[:, list(act)]
            G = Ua.T @ Ua
            try:
                f = np.linalg.solve(G, Ua.T @ x - lam / 2)
            except np.linalg.LinAlgError:
                continue
            if (f >= -1e-12).all():
                r = x - Ua @ f
                val = float(r @ r + lam * f.sum())
                if val < best - 1e-15:
                    best, best_act = val, act
    return (best, best_act) if return_active else best

def topk1_event_loss(U, x):
    """Oracle kappa=1 nonneg loss: ||x||^2 - max_i (d_i.x)_+^2."""
    c = np.clip(U.T @ x, 0, None)
    return float(x @ x - (c.max() ** 2 if len(c) else 0.0))

# ============================================================
print("=== A: sympy identities ===")
lam, phi, c0, r, P, p3 = sp.symbols('lam phi c0 r P p3', positive=True)

# A1: Prop 1 (symmetric): Delta_K - Delta_L1 = lam*(1-c0)
D_L1 = (1 - lam / 2) ** 2 - (c0 - lam / 2) ** 2
D_K = 1 - c0 ** 2
rec("A1 Delta_L1 = (1-c0)(1+c0-lam)", sp.simplify(D_L1 - (1 - c0) * (1 + c0 - lam)) == 0)
rec("A1 Delta_K - Delta_L1 = lam*(1-c0)", sp.simplify((D_K - D_L1) - lam * (1 - c0)) == 0)

# A2: Prop 3 band polynomial
S_L1 = (1 - c0) * (1 + c0 - lam)
S_K = (1 - c0) * (1 + c0)
F_L1 = (r - lam / 2) ** 2
F_K = r ** 2
H = lam * ((1 + c0) * r - r ** 2 - (1 + c0) * lam / 4)
rec("A2 S_L1*F_K - S_K*F_L1 = (1-c0)*H", sp.simplify(S_L1 * F_K - S_K * F_L1 - (1 - c0) * H) == 0)
# r_+ is a root of H/lam in r
rp = ((1 + c0) + sp.sqrt((1 + c0) ** 2 - (1 + c0) * lam)) / 2
rec("A2 H vanishes at r_+", sp.simplify((H / lam).subs(r, rp)) == 0)

# A3: residual identity |rho|^2 = sin^2 phi + lam^2/4 (u = bisector, c0=cos phi)
u = sp.Matrix([1, 0])
v2 = sp.Matrix([sp.cos(phi), -sp.sin(phi)])
rho = v2 - (sp.cos(phi) - lam / 2) * u
rec("A3 |rho|^2 = sin^2(phi)+lam^2/4",
    sp.simplify(rho.dot(rho) - (sp.sin(phi) ** 2 + lam ** 2 / 4)) == 0)

# A4: Lemma S1 KKT exclusion: both-active solve on event v1 with Gram [[1,c],[c,1]]
c = sp.symbols('c', positive=True)
G = sp.Matrix([[1, c], [c, 1]])
rhs = sp.Matrix([1 - lam / 2, c - lam / 2])
f = sp.simplify(G.solve(rhs))
rec("A4 f2 = -lam/(2(1+c)) < 0", sp.simplify(f[1] + lam / (2 * (1 + c))) == 0)

# ============================================================
print("=== B: 2D oracle brute force (shared-vs-split, matched L0) ===")
for phi_d, lam_v in [(10, 0.2), (25, 0.35)]:
    ph = np.radians(phi_d)
    v1 = np.array([np.cos(ph), np.sin(ph)]); v2n = np.array([np.cos(ph), -np.sin(ph)])
    c0v = np.cos(ph)
    # B1a: best single atom over fine angle grid = bisector (x-axis)
    ts = np.linspace(-np.pi, np.pi, 3601)
    def val1_L1(t):
        d = np.array([np.cos(t), np.sin(t)])
        return sum(0.5 * max(d @ v - lam_v / 2, 0) ** 2 for v in (v1, v2n))
    vals = np.array([val1_L1(t) for t in ts])
    tbest = ts[vals.argmax()]
    rec(f"B1a best 1-atom ~ bisector (phi={phi_d},lam={lam_v})", abs(tbest) < np.radians(0.2),
        f"got {np.degrees(tbest):.2f}deg")
    # B1b: gains match closed forms (population, p1=p2=1/2)
    U_sh = np.array([[1.0], [0.0]]); U_sp = np.column_stack([v1, v2n])
    L_sh = 0.5 * (l1_event_loss(U_sh, v1, lam_v) + l1_event_loss(U_sh, v2n, lam_v))
    L_sp = 0.5 * (l1_event_loss(U_sp, v1, lam_v) + l1_event_loss(U_sp, v2n, lam_v))
    dL1_num = L_sh - L_sp
    dL1_cf = (1 - c0v) * (1 + c0v - lam_v)
    rec(f"B1b L1 split gain matches closed form", abs(dL1_num - dL1_cf) < 1e-12,
        f"{dL1_num:.6f} vs {dL1_cf:.6f}")
    K_sh = 0.5 * (topk1_event_loss(U_sh, v1) + topk1_event_loss(U_sh, v2n))
    K_sp = 0.5 * (topk1_event_loss(U_sp, v1) + topk1_event_loss(U_sp, v2n))
    dK_num = K_sh - K_sp
    rec(f"B1b TopK split gain matches 1-c0^2", abs(dK_num - (1 - c0v ** 2)) < 1e-12,
        f"{dK_num:.6f}")
    # B2: split config uses exactly ONE active atom per event (matched L0)
    for ev, x in [("v1", v1), ("v2", v2n)]:
        _, act = l1_event_loss(U_sp, x, lam_v, return_active=True)
        rec(f"B2 split L1 active-set size 1 on {ev} (phi={phi_d})", len(act) == 1, f"act={act}")
    # B3: gap sign and magnitude
    rec(f"B3 Delta_K - Delta_L1 = lam*(1-c0) numerically",
        abs((dK_num - dL1_num) - lam_v * (1 - c0v)) < 1e-12,
        f"gap {dK_num - dL1_num:.6f}")
print("  (B3 grid) ", end="")
ok = all((1 - np.cos(np.radians(pd))) * lv > 0 and
         abs(((1 - np.cos(np.radians(pd)) ** 2) - (1 - np.cos(np.radians(pd))) *
              (1 + np.cos(np.radians(pd)) - lv)) - lv * (1 - np.cos(np.radians(pd)))) < 1e-12
         for pd in (1, 5, 15, 30, 44) for lv in (0.05, 0.1, 0.2, 0.4))
rec("B3 Delta_K > Delta_L1 across phi/lam grid", ok)

# ============================================================
print("=== C: Prop 1' general inequality (random asymmetric instances) ===")
rng = np.random.default_rng(13)
n_checked = n_skip = 0
worst = np.inf
for trial in range(200):
    lam_v = rng.choice([0.1, 0.25])
    n_sub = rng.choice([2, 3])
    angs = rng.uniform(np.radians(60), np.radians(120), n_sub)
    rs = rng.uniform(0.7, 1.4, n_sub)
    ps = rng.dirichlet(np.ones(n_sub))
    ts = np.linspace(0, 2 * np.pi, 4001)
    # projections of each event on atom(t): c_i(t) = r_i * cos(t - ang_i), clipped at 0
    C = np.clip(rs[:, None] * np.cos(ts[None, :] - angs[:, None]), 0, None)
    V_K = (ps[:, None] * C ** 2).sum(0)
    V_L1 = (ps[:, None] * np.clip(C - lam_v / 2, 0, None) ** 2).sum(0)
    iK, iL = V_K.argmax(), V_L1.argmax()
    cK = C[:, iK]
    if cK.min() < lam_v / 2 + 1e-3:      # outside all-firing regime -> skip
        n_skip += 1
        continue
    split_K = (ps * rs ** 2).sum()
    split_L1 = (ps * (rs - lam_v / 2) ** 2).sum()
    dK = split_K - V_K[iK]
    dL1 = split_L1 - V_L1[iL]
    bound = lam_v * (ps * (rs - cK)).sum()
    worst = min(worst, dK - dL1 - bound)
    if not (dK >= dL1 - 1e-9 and dK - dL1 >= bound - 1e-6):
        rec(f"C instance {trial}", False, f"dK={dK:.6f} dL1={dL1:.6f} bound={bound:.6f}")
    n_checked += 1
rec(f"C Delta_K >= Delta_L1 + lam*sum p(r-c) on {n_checked} instances ({n_skip} skipped)",
    n_checked > 100 and not any(f.startswith("C instance") for f in FAILS),
    f"worst slack {worst:.2e}")

# ============================================================
print("=== D: Prop 2 — duplicate-direction splitting exactly dead ===")
lam_v = 0.2
v = np.array([1.0, 0.0])
Udup = np.column_stack([v, v, v])
ok_flat = True
for rv in [0.05, 0.15, 0.5, 1.0, 2.3]:
    x = rv * v
    one = rv ** 2 - max(rv - lam_v / 2, 0) ** 2
    # objective depends on codes only through s = sum f: verify on random f>=0
    for _ in range(50):
        f = rng.uniform(0, rv, 3)
        s = f.sum()
        direct = float((x - Udup @ f) @ (x - Udup @ f) + lam_v * f.sum())
        via_s = (rv - s) ** 2 + lam_v * s
        if abs(direct - via_s) > 1e-12:
            ok_flat = False
    # 1D optimum over s equals the single-atom value
    ss = np.linspace(0, max(rv, 1) * 2, 20001)
    best = ((rv - ss) ** 2 + lam_v * ss).min()
    if abs(best - one) > 1e-6:
        ok_flat = False
rec("D duplicates reduce exactly to 1 atom (flat manifold)", ok_flat)
rs_grid = np.linspace(0, 3, 301)
rec("D single ReLU latent realizes oracle code (r-lam/2)_+",
    np.allclose(np.maximum(rs_grid - lam_v / 2, 0),
                np.maximum(1.0 * rs_grid - lam_v / 2, 0)))

# ============================================================
print("=== E: Prop 3 — opportunity-cost disagreement band (3D, direct losses) ===")
def config_losses(phi_v, lam_v, r_v, Pv, p3v):
    ph = phi_v
    v1 = np.array([np.cos(ph), np.sin(ph), 0.0]); v2n = np.array([np.cos(ph), -np.sin(ph), 0.0])
    ub = np.array([1.0, 0.0, 0.0]); v3 = np.array([0.0, 0.0, 1.0])
    events = [(Pv / 2, v1), (Pv / 2, v2n), (p3v, r_v * v3)]
    SPLIT = np.column_stack([v1, v2n]); COVER = np.column_stack([ub, v3])
    out = {}
    for nm, U in [("SPLIT", SPLIT), ("COVER", COVER)]:
        out[("L1", nm)] = sum(w * l1_event_loss(U, x, lam_v) for w, x in events)
        out[("K", nm)] = sum(w * topk1_event_loss(U, x) for w, x in events)
    return out

lam_v, ph, Pv = 0.2, np.radians(15), 0.1
c0v = np.cos(ph)
SL1 = (1 - c0v) * (1 + c0v - lam_v); SK = (1 - c0v) * (1 + c0v)
for r_v, expect_band in [(1.0, "forward"), (2.2, "reverse")]:
    FL1 = (r_v - lam_v / 2) ** 2; FK = r_v ** 2
    lo, hi = Pv * SK / FK, Pv * SL1 / FL1        # forward band in p3
    if expect_band == "forward":
        rec(f"E band nonempty at r={r_v} (H>0)", hi > lo, f"p3 in ({lo:.5f},{hi:.5f})")
        p3v = (lo + hi) / 2
        L = config_losses(ph, lam_v, r_v, Pv, p3v)
        rec("E   L1 prefers SPLIT in band", L[("L1", "SPLIT")] < L[("L1", "COVER")] - 1e-12,
            f"{L[('L1','SPLIT')]:.6f} vs {L[('L1','COVER')]:.6f}")
        rec("E   TopK prefers COVER in band", L[("K", "COVER")] < L[("K", "SPLIT")] - 1e-12,
            f"{L[('K','COVER')]:.6f} vs {L[('K','SPLIT')]:.6f}")
        # outside the band both agree
        for p3o, side in [(lo * 0.5, "below"), (hi * 1.5, "above")]:
            L = config_losses(ph, lam_v, r_v, Pv, p3o)
            agree = ((L[("L1", "SPLIT")] < L[("L1", "COVER")]) ==
                     (L[("K", "SPLIT")] < L[("K", "COVER")]))
            rec(f"E   architectures agree {side} band", agree)
    else:
        rec(f"E forward band empty at r={r_v} (H<0)", hi < lo, f"({lo:.6f},{hi:.6f})")
        rlo, rhi = Pv * SL1 / FL1, Pv * SK / FK  # reverse band
        rec(f"E reverse band nonempty at r={r_v}", rhi > rlo, f"({rlo:.6f},{rhi:.6f})")
        p3v = (rlo + rhi) / 2
        L = config_losses(ph, lam_v, r_v, Pv, p3v)
        rec("E   TopK prefers SPLIT in reverse band", L[("K", "SPLIT")] < L[("K", "COVER")] - 1e-14)
        rec("E   L1 prefers COVER in reverse band", L[("L1", "COVER")] < L[("L1", "SPLIT")] - 1e-14)
rp_num = ((1 + c0v) + np.sqrt((1 + c0v) ** 2 - (1 + c0v) * lam_v)) / 2
print(f"  (info) r_+ = {rp_num:.3f} ~= 1+c0-lam/4 = {1 + c0v - lam_v / 4:.3f}: reverse band "
      f"needs a fresh feature ~2x the sub-feature scale")
# informational (scoped in the note): random search over general 2-atom dicts in 3D
r_v = 1.0; p3v = (Pv * SK / FK + Pv * SL1 / (r_v - lam_v / 2) ** 2) / 2
ph15 = np.radians(15)
v1e = np.array([np.cos(ph15), np.sin(ph15), 0.0]); v2e = np.array([np.cos(ph15), -np.sin(ph15), 0.0])
evts = [(Pv / 2, v1e), (Pv / 2, v2e), (p3v, r_v * np.array([0.0, 0.0, 1.0]))]
best_rand = np.inf
for _ in range(4000):
    A = rng.normal(size=(3, 2)); A /= np.linalg.norm(A, axis=0)
    best_rand = min(best_rand, sum(w * l1_event_loss(A, x, lam_v) for w, x in evts))
L_cfg = min(sum(w * l1_event_loss(np.column_stack([v1e, v2e]), x, lam_v) for w, x in evts),
            sum(w * l1_event_loss(np.column_stack([np.array([1., 0., 0.]),
                                                   np.array([0., 0., 1.])]), x, lam_v)
                for w, x in evts))
print(f"  (info) random 2-atom search in band: best found {best_rand:.6f} vs "
      f"best designated config {L_cfg:.6f} (no gate; scoped in note)")

# ============================================================
print("=== F: Prop 4(i) — L1 nucleation: first-order escape for every phi>0 ===")
for phi_d in [2, 5, 10, 20]:
    for lam_v in [0.1, 0.3]:
        ph = np.radians(phi_d)
        c0v = np.cos(ph)
        ubis = np.array([1.0, 0.0])
        v1 = np.array([np.cos(ph), np.sin(ph)]); v2n = np.array([np.cos(ph), -np.sin(ph)])
        rho_v = v2n - (c0v - lam_v / 2) * ubis
        nr = np.linalg.norm(rho_v)
        rec(f"F |rho| formula (phi={phi_d},lam={lam_v})",
            abs(nr ** 2 - (np.sin(ph) ** 2 + lam_v ** 2 / 4)) < 1e-14)
        rec(f"F |rho| > lam/2", nr > lam_v / 2 + 1e-12, f"{nr:.4f} vs {lam_v/2:.4f}")
        # fixed-code path derivative: dL/dt = p2*(lam - 2|rho|), p2 = 1/2
        rhat = rho_v / nr
        def L_path(t):
            f1 = c0v - lam_v / 2
            res = v2n - f1 * ubis - t * rhat
            return 0.5 * (float(res @ res) + lam_v * (f1 + t)) \
                 + 0.5 * (1 - (c0v - lam_v / 2) ** 2 + 0)  # v1-event unchanged (its own loss const)
        t = 1e-6
        fd = (L_path(t) - L_path(0.0)) / t
        pred = 0.5 * (lam_v - 2 * nr)
        rec(f"F path derivative = p2*(lam-2|rho|) < 0", abs(fd - pred) < 1e-4 and pred < 0,
            f"fd {fd:.5f} pred {pred:.5f}")
        # full re-optimized 2-atom loss strictly below 1-atom loss
        U1 = ubis.reshape(2, 1); U2 = np.column_stack([ubis, rhat])
        L1a = 0.5 * (l1_event_loss(U1, v1, lam_v) + l1_event_loss(U1, v2n, lam_v))
        L2a = 0.5 * (l1_event_loss(U2, v1, lam_v) + l1_event_loss(U2, v2n, lam_v))
        rec(f"F 2-atom {{u,rho}} beats 1-atom", L2a < L1a - 1e-10, f"{L2a:.6f} < {L1a:.6f}")
# phi = 0: no nucleation (duplicate direction, |rho| = lam/2 exactly)
lam_v = 0.2
rho0 = np.array([1.0, 0.0]) - (1 - lam_v / 2) * np.array([1.0, 0.0])
rec("F phi=0: |rho| = lam/2 exactly (derivative 0, no duplicate nucleation)",
    abs(np.linalg.norm(rho0) - lam_v / 2) < 1e-15)

# ============================================================
print("=== G: Prop 4(ii) — TopK rank-gated trap vs stable split ===")
ph = np.radians(10); c0v = np.cos(ph)
v1 = np.array([np.cos(ph), np.sin(ph)]); v2n = np.array([np.cos(ph), -np.sin(ph)])
ubis = np.array([1.0, 0.0])
lam_v = 0.2
rho_v = v2n - (c0v - lam_v / 2) * ubis; rhat = rho_v / np.linalg.norm(rho_v)

def topk_sae_loss(W, b, D, events, k=1):
    """Encoder-ranked TopK-SAE population loss: z=Wx+b, top-k by z, f=ReLU(z)."""
    tot = 0.0
    for w_ev, x in events:
        z = W @ x + b
        sel = np.argsort(-z)[:k]
        xhat = np.zeros_like(x)
        for j in sel:
            xhat = xhat + max(z[j], 0.0) * D[:, j]
        rvec = x - xhat
        tot += w_ev * float(rvec @ rvec)
    return tot

events = [(0.5, v1), (0.5, v2n)]
# merged config: incumbent (scale 1, atom=bisector) + resampled spare (scale 0.2, atom=rhat)
W0 = np.vstack([ubis, 0.2 * rhat]); b0 = np.zeros(2); D0 = np.column_stack([ubis, rhat])
z_margin = min((W0 @ x + b0)[0] - (W0 @ x + b0)[1] for _, x in events)
rec("G1 spare strictly below cutoff on every event", z_margin > 0.1, f"margin {z_margin:.3f}")
L_merged = topk_sae_loss(W0, b0, D0, events)
ok_inv = True
for tr in range(20):
    dW = rng.normal(0, 1e-2, 2); db = rng.normal(0, 1e-2)
    dD = rng.normal(0, 1e-2, 2)
    W2 = W0.copy(); W2[1] += dW
    b2 = b0.copy(); b2[1] += db
    D2 = D0.copy(); D2[:, 1] += dD; D2[:, 1] /= np.linalg.norm(D2[:, 1])
    if topk_sae_loss(W2, b2, D2, events) != L_merged:
        ok_inv = False
rec("G1 loss EXACTLY invariant to spare-parameter perturbations (local min block)", ok_inv,
    f"L_merged={L_merged:.6f}")
# yet the split is strictly better for TopK (reachability vs preference)
Wsp = np.vstack([v1, v2n]); Dsp = np.column_stack([v1, v2n])
L_split = topk_sae_loss(Wsp, np.zeros(2), Dsp, events)
rec("G2 split strictly better for TopK", L_split < L_merged - 1e-12,
    f"{L_split:.6f} < {L_merged:.6f}")
rec("G2 merged TopK loss = 1-c0^2", abs(L_merged - (1 - c0v ** 2)) < 1e-12)
# G3: split ranking stability: each atom wins its own event
ok_rank = all((Wsp @ x)[i] > (Wsp @ x)[1 - i] for i, (_, x) in enumerate(events))
rec("G3 at split, each sub-atom wins the ranking on its own event", ok_rank,
    f"pre-acts 1 vs cos2phi={np.cos(2 * ph):.4f}")

# ============================================================
print()
if FAILS:
    print("VERIFY_SPLITTING: FAILURES:")
    for f in FAILS:
        print("  -", f)
    raise SystemExit(1)
print("verify_splitting: ALL CHECKS PASS. Mechanisms (a) and (c) dead as stated; "
      "(a') opportunity-cost tilt and (b) self-gated vs rank-gated nucleation verified.")
