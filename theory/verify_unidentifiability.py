#!/usr/bin/env python3
"""Finite-certificate verification of rho-unidentifiability under gated absorption.

Companion to notes/unidentifiability-certificate.md. Everything is sympy-exact:
distributions are finite sets of atoms with Rational/radical probabilities and
coordinates; equality of distributions is exact equality of the atom->probability
maps. Prints PASS/FAIL per check and exits nonzero on any failure.

Levels:
  L1  SAE-observable certificate: a fixed Arm-A-style gated encoder; two
      generative processes G1, G2 with different child-given-parent rates rho
      whose pushforward code distributions are identical.
  L2  Input-distribution certificate: one distribution over R^d; two explicit
      dictionary+event decompositions (both exactly generating it) with
      different rho. Certificate A: rho = 2/5 vs 0 (reified composite / CDX;
      G2 is a mixture reading, no child). Certificate B: rho = 2/5 vs 1/4
      (both admit a hierarchical generative reading; G2's is NOT a strict
      activation hierarchy and its dictionary is support-REDUCIBLE — conceded
      per adversarial review objection (a)).
  L2'' Certificate C (the irreducibility fork, resolved as CONSTRUCTION):
      interleaved-cone dictionaries, both support-IRREDUCIBLE, both entrywise
      NONNEGATIVE (strict-NMF class), both STRICT activation hierarchies,
      rho = 3/4 vs 1/2 (non-complementary).
  B*  Boundary propositions: which anchors break the certificates.
"""

import sys

import sympy as sp
from sympy import Rational as R, sqrt, Matrix, symbols, solve, Eq

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ----------------------------------------------------------------------------
# Distribution helpers: a distribution is a dict {atom(tuple of sympy exprs): prob}
# ----------------------------------------------------------------------------

def normalize_dist(pairs):
    """pairs: iterable of (atom_tuple, prob). Merge equal atoms exactly."""
    dist = {}
    for atom, p in pairs:
        atom = tuple(sp.nsimplify(sp.simplify(a)) for a in atom)
        # find an existing exactly-equal atom
        for k in dist:
            if all(sp.simplify(k[i] - atom[i]) == 0 for i in range(len(atom))):
                dist[k] = sp.simplify(dist[k] + p)
                break
        else:
            dist[atom] = sp.simplify(p)
    return {k: v for k, v in dist.items() if v != 0}


def dists_equal(d1, d2):
    if len(d1) != len(d2):
        return False
    for a1, p1 in d1.items():
        hit = False
        for a2, p2 in d2.items():
            if len(a1) == len(a2) and all(
                sp.simplify(a1[i] - a2[i]) == 0 for i in range(len(a1))
            ):
                if sp.simplify(p1 - p2) != 0:
                    return False
                hit = True
                break
        if not hit:
            return False
    return True


def total_mass(d):
    return sp.simplify(sum(d.values()))


# ----------------------------------------------------------------------------
# Shared geometry (d = 2 unless stated)
# ----------------------------------------------------------------------------

v_p = Matrix([1, 0])                     # parent direction
v_c = Matrix([0, 1])                     # child direction
u = Matrix([1, 1]) / sqrt(2)             # composite direction (unit norm)

check("geometry: |v_p|=|v_c|=|u|=1, v_p.v_c=0",
      sp.simplify(v_p.norm() - 1) == 0 and sp.simplify(v_c.norm() - 1) == 0
      and sp.simplify(u.norm() - 1) == 0 and sp.simplify(v_p.dot(v_c)) == 0)

# ----------------------------------------------------------------------------
# LEVEL 1: fixed gated SAE (Arm A mechanism, stylized)
#   parent latent: encoder w_par = v_p - 2 v_c ("hole": silent on joint), b=0
#   composite latent: encoder u with bias -1 (gate: silent on host-only)
#   decoders: d_par = v_p, d_comp = u  (unit-norm, S1 precondition)
# ----------------------------------------------------------------------------

print("\n== LEVEL 1: SAE-observable certificate (gated encoder) ==")

w_par = Matrix([1, -2])
b_par = 0
w_comp = u
b_comp = -1


def relu(t):
    return sp.Max(t, 0)


def code(x):
    """x: Matrix 2x1 -> latent tuple (z_par, z_comp)."""
    z1 = relu(w_par.dot(x) + b_par)
    z2 = relu(w_comp.dot(x) + b_comp)
    return (sp.simplify(z1), sp.simplify(z2))


# sanity: the gate behaves as Arm A found
x_host = v_p
x_joint = v_p + v_c
check("L1 gate: comp silent on host-only, fires on joint; par silent on joint",
      code(x_host)[1] == 0 and sp.simplify(code(x_joint)[1] - (sqrt(2) - 1)) == 0
      and code(x_joint)[0] == 0 and code(x_host)[0] == 1)

# Generative processes: lists of (x_atom, prob, event_class)
# event classes: 'S' host-only(parent solo), 'J' joint(parent+child), 'C' child-solo,
#                'U' composite-sibling solo, 'B' background
G1_L1 = [
    (v_p,          R(3, 10), "S"),
    (v_p + v_c,    R(2, 10), "J"),
    (Matrix([0, 0]), R(1, 2), "B"),
]
G2_L1 = [
    (v_p,          R(3, 10), "S"),
    (v_p + v_c,    R(1, 10), "J"),
    (2 * v_c,      R(1, 10), "C"),   # child-solo at magnitude 2
    (Matrix([0, 0]), R(1, 2), "B"),
]
G3_L1 = [
    (v_p,          R(3, 10), "S"),
    (sqrt(2) * u,  R(2, 10), "U"),   # reified composite feature, no child at all
    (Matrix([0, 0]), R(1, 2), "B"),
]


def rho_of(process):
    """rho = r_J / (r_J + r_S): child-given-parent rate among parent events."""
    r_J = sum((p for x, p, c in process if c == "J"), R(0))
    r_S = sum((p for x, p, c in process if c == "S"), R(0))
    return sp.simplify(r_J / (r_J + r_S))


def child_rate(process):
    return sum((p for x, p, c in process if c in ("J", "C")), R(0))


def push_codes(process):
    return normalize_dist([(tuple(code(x)), p) for x, p, c in process])


C1, C2, C3 = push_codes(G1_L1), push_codes(G2_L1), push_codes(G3_L1)
rho1, rho2, rho3 = rho_of(G1_L1), rho_of(G2_L1), rho_of(G3_L1)

check("L1: total masses = 1",
      all(total_mass(x) == 1 for x in (C1, C2, C3)))
check("L1 certificate (G1 vs G2): code distributions EXACTLY equal",
      dists_equal(C1, C2), f"atoms={len(C1)}")
check("L1 certificate (G1 vs G3): code distributions EXACTLY equal",
      dists_equal(C1, C3))
check("L1: rho differs: rho1=2/5, rho2=1/4, rho3=0",
      rho1 == R(2, 5) and rho2 == R(1, 4) and rho3 == 0,
      f"rho1={rho1}, rho2={rho2}, rho3={rho3}")
check("L1: child base rates: G1=1/5, G2=1/5, G3=0 (G3 also differs in base rate)",
      child_rate(G1_L1) == R(1, 5) and child_rate(G2_L1) == R(1, 5)
      and child_rate(G3_L1) == 0)
# note: G1 vs G2 have DIFFERENT x-distributions (2v_c != v_p+v_c) -> encoder collapse
X1 = normalize_dist([(tuple(x), p) for x, p, c in G1_L1])
X2 = normalize_dist([(tuple(x), p) for x, p, c in G2_L1])
check("L1: G1,G2 x-distributions DIFFER (so the encoder itself destroys the info)",
      not dists_equal(X1, X2))

# ----------------------------------------------------------------------------
# LEVEL 2: input-distribution certificate (feature-decomposition non-uniqueness)
# A decomposition event = (list of (direction, magnitude), prob, event_class).
# It must reconstruct its atom EXACTLY: sum(mag*dir) == atom.
# ----------------------------------------------------------------------------

print("\n== LEVEL 2: input-distribution certificate ==")


def event_x(ev):
    feats, p, c = ev
    x = Matrix([0] * feats[0][0].rows)
    for d, m in feats:
        x = x + m * d
    return sp.simplify(x)


def decomposition_xdist(events):
    return normalize_dist([(tuple(event_x(ev)), ev[1]) for ev in events])


def decomposition_valid(events, name):
    ok = True
    for d, m in [fm for ev in events for fm in ev[0]]:
        if sp.simplify(d.norm() - 1) != 0:
            ok = False
    check(f"L2 {name}: all feature directions unit-norm", ok)
    check(f"L2 {name}: probabilities sum to 1",
          sp.simplify(sum(ev[1] for ev in events)) == 1)


def rho_of_decomp(events):
    r_J = sum((p for feats, p, c in events if c == "J"), R(0))
    r_S = sum((p for feats, p, c in events if c == "S"), R(0))
    if r_J + r_S == 0:
        return sp.nan
    return sp.simplify(r_J / (r_J + r_S))


ZERO = Matrix([0, 0])

# --- Certificate A (minimal): 3 atoms, rho = 2/5 vs rho = 0 ------------------
# atoms: 0 @ 1/2, v_p @ 3/10, v_p+v_c @ 1/5
GA1 = [  # hierarchical parent/child reading
    ([(v_p, 1)],                 R(3, 10), "S"),
    ([(v_p, 1), (v_c, 1)],       R(1, 5),  "J"),
    ([],                         R(1, 2),  "B"),
]
GA2 = [  # reified-composite reading (the CDX equivalence class, exact form)
    ([(v_p, 1)],                 R(3, 10), "S"),
    ([(u, sqrt(2))],             R(1, 5),  "U"),
    ([],                         R(1, 2),  "B"),
]


def fix_empty(events):
    # background event: represent x=0 explicitly
    out = []
    for feats, p, c in events:
        if not feats:
            out.append(([(v_p, 0)], p, c))
        else:
            out.append((feats, p, c))
    return out


GA1, GA2 = fix_empty(GA1), fix_empty(GA2)
decomposition_valid(GA1, "certA/G1")
decomposition_valid(GA2, "certA/G2")
XA1, XA2 = decomposition_xdist(GA1), decomposition_xdist(GA2)
check("L2 certA: x-distributions EXACTLY equal", dists_equal(XA1, XA2),
      f"atoms={len(XA1)}")
rA1, rA2 = rho_of_decomp(GA1), rho_of_decomp(GA2)
check("L2 certA: rho = 2/5 (G1) vs 0 (G2, no child exists)",
      rA1 == R(2, 5) and rA2 == 0, f"rho={rA1} vs {rA2}")

# --- Certificate B: 4 atoms, BOTH decompositions hierarchical, rho 2/5 vs 1/4 -
# atoms: 0 @ 1/2, v_p @ 3/10, v_p+v_c @ 1/10, v_p+2v_c @ 1/10
GB1 = [  # dict {v_p, v_c}; child fires at magnitude 1 or 2
    ([(v_p, 1)],                 R(3, 10), "S"),
    ([(v_p, 1), (v_c, 1)],       R(1, 10), "J"),
    ([(v_p, 1), (v_c, 2)],       R(1, 10), "J"),
    ([(v_p, 0)],                 R(1, 2),  "B"),
]
GB2 = [  # dict {v_p, v_c, u}; the mid atom is a reified composite firing solo
    ([(v_p, 1)],                 R(3, 10), "S"),
    ([(u, sqrt(2))],             R(1, 10), "U"),
    ([(v_p, 1), (v_c, 2)],       R(1, 10), "J"),
    ([(v_p, 0)],                 R(1, 2),  "B"),
]
decomposition_valid(GB1, "certB/G1")
decomposition_valid(GB2, "certB/G2")
XB1, XB2 = decomposition_xdist(GB1), decomposition_xdist(GB2)
check("L2 certB: x-distributions EXACTLY equal", dists_equal(XB1, XB2),
      f"atoms={len(XB1)}")
rB1, rB2 = rho_of_decomp(GB1), rho_of_decomp(GB2)
check("L2 certB: rho = 2/5 vs 1/4, both nonzero (both readings hierarchical)",
      rB1 == R(2, 5) and rB2 == R(1, 4), f"rho={rB1} vs {rB2}")
cbB1 = sum((p for f, p, c in GB1 if c == "J"), R(0))
cbB2 = sum((p for f, p, c in GB2 if c == "J"), R(0))
check("L2 certB: child BASE rates also differ: 1/5 vs 1/10",
      cbB1 == R(1, 5) and cbB2 == R(1, 10))

# Level 2 => Level 1 automatically: push certB through the fixed encoder
CB1 = normalize_dist([(tuple(code(event_x(ev))), ev[1]) for ev in GB1])
CB2 = normalize_dist([(tuple(code(event_x(ev))), ev[1]) for ev in GB2])
check("L2=>L1: certB pushes to identical code distributions under the gated SAE",
      dists_equal(CB1, CB2))

# ----------------------------------------------------------------------------
# LEVEL 2'': SUPPORT-IRREDUCIBILITY FORK (review objection (a)), resolved (A)
#
# Definition (per review): dictionary D is support-REDUCIBLE for an atom set if
# some PROPER subset of D nonnegatively reconstructs every atom. We (i) CONCEDE
# certB's G2 dict {v_p,v_c,u} is reducible ({v_p,v_c} suffices); (ii) note
# certA's two dicts are both irreducible (but its G2 reading has no child);
# (iii) construct CERTIFICATE C: interleaved-cone dictionaries, both
# support-IRREDUCIBLE, both entrywise-NONNEGATIVE (strict NMF — a stronger
# frame than the semi-NMF class SAEs occupy), both STRICT activation
# hierarchies, rho = 3/4 vs 1/2, non-complementary (not the known {rho,1-rho}
# orientation ambiguity of theory/gating_corrected_rho.md §4).
# ----------------------------------------------------------------------------

print("\n== LEVEL 2'': support-irreducibility fork (Certificate C) ==")

from itertools import combinations


def expected_L0(events):
    return sp.simplify(sum(p * len([1 for d, m in feats if m != 0])
                           for feats, p, c in events))


def nonneg_recon_possible(dirs, atom):
    """Exact test: exists m_i >= 0 with sum m_i*dirs_i == atom.
    dirs has <= 2 members (singleton, or linearly independent pair in the
    plane of atom) — sufficient for every subset tested here."""
    assert len(dirs) <= 2
    ms = list(sp.symbols(f"m0:{len(dirs)}", real=True))
    eqs = [Eq(sum(ms[i] * dirs[i][j] for i in range(len(dirs))), atom[j])
           for j in range(atom.rows)]
    sols = solve(eqs, ms, dict=True)
    for s in sols:
        vals = [sp.simplify(s.get(m, sp.nan)) for m in ms]
        if all(v.is_number for v in vals) and all(v >= 0 for v in vals):
            return True
    return False


def support_irreducible(dirs, atoms):
    """No proper nonempty subset of dirs nonnegatively reconstructs ALL atoms."""
    for k in range(1, len(dirs)):
        for sub in combinations(range(len(dirs)), k):
            if all(nonneg_recon_possible([dirs[i] for i in sub], a)
                   for a in atoms):
                return False, sub
    return True, None


# (i) concession: certB's G2 dict is support-reducible (objection valid there)
atomsB = [v_p, v_p + v_c, v_p + 2 * v_c]
redB, subB = support_irreducible([v_p, v_c, u], atomsB)
check("IRR concession: certB G2 dict {v_p,v_c,u} IS support-reducible via "
      "{v_p,v_c} — review objection (a) is valid for certB",
      (not redB) and subB == (0, 1))
check("IRR: certB G1 dict {v_p,v_c} is support-irreducible",
      support_irreducible([v_p, v_c], atomsB)[0])

# (ii) certA: BOTH dicts already irreducible (irreducibility alone cannot
# reject certA's no-child reading; what rejects it is a rho>0 assumption)
atomsA = [v_p, v_p + v_c]
check("IRR: certA dicts {v_p,v_c} and {v_p,u} are BOTH support-irreducible",
      support_irreducible([v_p, v_c], atomsA)[0]
      and support_irreducible([v_p, u], atomsA)[0])

# (iii) CERTIFICATE C: interleaved cones (angular order c2 < pC1 < pC2 < c1)
pC1 = Matrix([2, 1]) / sqrt(5)   # G1 parent
cC1 = Matrix([0, 1])             # G1 child
pC2 = Matrix([1, 1]) / sqrt(2)   # G2 parent
cC2 = Matrix([1, 0])             # G2 child
az = Matrix([2, 1])              # atom z: parent-SOLO in G1, JOINT in G2
ay = Matrix([1, 1])              # atom y: JOINT in G1, parent-SOLO in G2
av = Matrix([3, 2])              # atom v: JOINT in both (breaks complementarity)

GC1 = [
    ([(pC1, sqrt(5))],                          R(1, 10), "S"),
    ([(pC1, sqrt(5) / 2), (cC1, R(1, 2))],      R(1, 5),  "J"),
    ([(pC1, 3 * sqrt(5) / 2), (cC1, R(1, 2))],  R(1, 10), "J"),
    ([(pC1, 0)],                                R(3, 5),  "B"),
]
GC2 = [
    ([(pC2, sqrt(2))],                          R(1, 5),  "S"),
    ([(pC2, sqrt(2)), (cC2, 1)],                R(1, 10), "J"),
    ([(pC2, 2 * sqrt(2)), (cC2, 1)],            R(1, 10), "J"),
    ([(pC2, 0)],                                R(3, 5),  "B"),
]

decomposition_valid(GC1, "certC/G1")
decomposition_valid(GC2, "certC/G2")
check("certC: G1 events reconstruct atoms {z,y,v} exactly",
      sp.simplify((event_x(GC1[0]) - az).norm()) == 0
      and sp.simplify((event_x(GC1[1]) - ay).norm()) == 0
      and sp.simplify((event_x(GC1[2]) - av).norm()) == 0)
check("certC: G2 events reconstruct atoms {y,z,v} exactly (solo event is y)",
      sp.simplify((event_x(GC2[0]) - ay).norm()) == 0
      and sp.simplify((event_x(GC2[1]) - az).norm()) == 0
      and sp.simplify((event_x(GC2[2]) - av).norm()) == 0)

XC1, XC2 = decomposition_xdist(GC1), decomposition_xdist(GC2)
check("certC: x-distributions EXACTLY equal", dists_equal(XC1, XC2),
      f"atoms={len(XC1)}")


def strict_hierarchy(events, parent):
    """Child fires => parent fires, in every event (no solo child, no mixture)."""
    for feats, p, cls in events:
        active = [(d, m) for d, m in feats if m != 0]
        par_mag = sum((m for d, m in active
                       if sp.simplify((d - parent).norm()) == 0), R(0))
        child_active = any(sp.simplify((d - parent).norm()) != 0
                           for d, m in active)
        if child_active and not (par_mag > 0):
            return False
    return True


check("certC: BOTH readings are STRICT activation hierarchies "
      "(child never fires without its parent; no mixture reading)",
      strict_hierarchy(GC1, pC1) and strict_hierarchy(GC2, pC2))
check("certC: all four dictionary directions entrywise NONNEGATIVE "
      "(certificate lives even in the strict-NMF dictionary class)",
      all(all(d[j] >= 0 for j in range(2)) for d in (pC1, cC1, pC2, cC2)))

atomsC = [az, ay, av]
check("certC: BOTH dictionaries support-IRREDUCIBLE — objection (a) DEFEATED",
      support_irreducible([pC1, cC1], atomsC)[0]
      and support_irreducible([pC2, cC2], atomsC)[0])
check("certC: dictionary sizes EQUAL (2 each = minimum: atoms span R^2)",
      Matrix.hstack(az, ay).rank() == 2)

rC1, rC2 = rho_of_decomp(GC1), rho_of_decomp(GC2)
check("certC: rho = 3/4 vs 1/2 under identical x-distributions",
      rC1 == R(3, 4) and rC2 == R(1, 2), f"rho={rC1} vs {rC2}")
check("certC: NON-complementary (rho2 != 1-rho1 and != rho1): not the known "
      "{rho,1-rho} orientation ambiguity",
      sp.simplify(rC2 - (1 - rC1)) != 0 and rC1 != rC2)

# sparsity-rho coupling (the corrected P5): for two readings in which every
# active atom is parent-active with at most one child active,
#   E[L0]_G1 - E[L0]_G2 = (rho1 - rho2) * P(parent events).
# So min-E[L0] selection IS min-rho selection within this class: sparsity is a
# deterministic selector biased toward the smaller-rho reading, not an
# identification principle.
T_par = R(2, 5)
check("certC coupling: E[L0]_G1 - E[L0]_G2 == (rho1-rho2)*P(parent-events)",
      sp.simplify((expected_L0(GC1) - expected_L0(GC2))
                  - (rC1 - rC2) * T_par) == 0,
      f"E[L0]: {expected_L0(GC1)} vs {expected_L0(GC2)}")

# conic-hull remark: both CHILD directions lie OUTSIDE the conic hull of the
# data (rays z..y). Restricting features to the data cone rejects BOTH
# hierarchical readings rather than selecting one: the only 2-dict inside the
# cone is {zhat, yhat}, under which z and y are each SOLO, so whichever ray is
# called parent, the other fires alone -> strict hierarchy impossible.
zhat, yhat = az / az.norm(), ay / ay.norm()
check("certC remark: c1,c2 outside data conic hull; cone-restricted dict "
      "{zhat,yhat} admits NO strict-hierarchical reading (both rays fire solo)",
      (not nonneg_recon_possible([zhat, yhat], cC1))
      and (not nonneg_recon_possible([zhat, yhat], cC2))
      and nonneg_recon_possible([zhat, yhat], av)
      and not nonneg_recon_possible([zhat], ay)
      and not nonneg_recon_possible([yhat], az))

# ----------------------------------------------------------------------------
# BOUNDARY MAP: anchors that do / do not break the certificates
# ----------------------------------------------------------------------------

print("\n== BOUNDARY: anchor propositions ==")

# ANCHOR 'parent direction known': breaks certC (pC1 != pC2: knowing the parent
# selects the reading) but NOT certA (both its readings share parent v_p).
# Whether some certificate survives irreducibility + strict hierarchy + known
# parent direction jointly is OPEN (no construction, no impossibility proof).
check("ANCHOR 'parent direction known': breaks certC (pC1 != pC2) but not "
      "certA (parent v_p shared across its readings)",
      sp.simplify((pC1 - pC2).norm()) != 0
      and sp.simplify((GA1[0][0][0][0] - GA2[0][0][0][0]).norm()) == 0)

# P3: KNOWN linearly independent dictionary -> coefficients unique -> rho identified.
a1, a2 = symbols("a1 a2", real=True)
sols = solve([Eq(a1 * v_p[i] + a2 * v_c[i], (v_p + v_c)[i]) for i in range(2)],
             [a1, a2], dict=True)
check("P3 anchor 'known full-rank dictionary {v_p,v_c}': unique coefficients "
      "for the ambiguous atom -> rho forced to 2/5 (IDENTIFIABLE)",
      len(sols) == 1 and sols[0][a1] == 1 and sols[0][a2] == 1)

# P4: known but OVERCOMPLETE dictionary {v_p, v_c, u}: two exact nonnegative
# sparse solutions for the same atom -> certificate survives.
sol_hier = (1, 1, 0)
sol_reif = (0, 0, sqrt(2))
D3 = [v_p, v_c, u]


def recon(coeffs):
    x = Matrix([0, 0])
    for c, d in zip(coeffs, D3):
        x = x + c * d
    return sp.simplify(x)


check("P4 anchor 'known overcomplete dictionary (contains u)': two exact "
      "nonneg decompositions of v_p+v_c -> STILL UNIDENTIFIABLE",
      sp.simplify(recon(sol_hier) - (v_p + v_c)).norm() == 0
      and sp.simplify(recon(sol_reif) - (v_p + v_c)).norm() == 0)

# P5 (REVISED after review): the original "L0 vs dictionary-size disagreement"
# framing on certB is RETRACTED as the headline (certB's G2 is reducible; a
# joint objective with irreducibility rejects it — reviewer is right). The
# surviving, sharper statement lives on certC, where irreducibility and
# dictionary size TIE and E[L0]-minimization deterministically selects the
# smaller-rho reading (coupling identity above): sparsity is a biased
# selection rule, not an identification principle. The certB facts below are
# kept as descriptive computations only.
l01, l02 = expected_L0(GB1), expected_L0(GB2)
check("P5 (descriptive, certB): E[L0](G2) < E[L0](G1) and dict sizes 3 vs 2 "
      "— but certB's G2 is REDUCIBLE, so this no longer carries the sparsity "
      "argument; certC's coupling identity does",
      sp.simplify(l02 - l01) < 0
      and len({tuple(d) for f, p, c in GB1 for d, m in f if m != 0}) == 2
      and len({tuple(d) for f, p, c in GB2 for d, m in f if m != 0}) == 3,
      f"E[L0]: G1={l01}, G2={l02}")

# P6: additive observation noise x + eps (eps independent of the event) does
# NOT break the certificate: both processes have the same atom set, so both
# noisy distributions are the same mixture (same convolution). Verified by the
# exact atom equality already established (XB1 == XB2 => XB1*K == XB2*K for any
# noise kernel K).
check("P6 anchor 'nondegenerate additive input noise': same atoms => same "
      "convolution; certificate SURVIVES (equality already exact pre-noise)",
      dists_equal(XB1, XB2))

# P7: per-feature INDEPENDENT magnitude jitter DOES break reification:
# jittered joint events (1+e1)v_p + (1+e2)v_c span a 2-D patch, while any
# single-feature (reified) event lies on a 1-D ray {t*w}. Two generic jittered
# points are linearly independent -> no single ray covers them.
e = R(1, 100)
p1 = (1 + e) * v_p + (1 - e) * v_c
p2 = (1 - e) * v_p + (1 + e) * v_c
Mjit = Matrix([[p1[0], p1[1]], [p2[0], p2[1]]])
check("P7 anchor 'independent per-feature magnitude jitter': two jittered "
      "joint atoms are linearly independent (rank 2) -> not on any single ray "
      "-> REIFICATION defeated (this step only; see P7-scope)",
      Mjit.rank() == 2, f"det={sp.simplify(Mjit.det())}")
# and jittered parent-solo events stay rank 1 (on the v_p ray): the local
# support dimension counts active features.
q1, q2 = (1 + e) * v_p, (1 - e) * v_p
check("P7b: jittered parent-solo atoms stay on the v_p ray (rank 1) -> local "
      "support dimension = number of simultaneously active features",
      Matrix([[q1[0], q1[1]], [q2[0], q2[1]]]).rank() == 1)
# P7-scope (review objection (d), accepted): standard ICA identifiability
# (Comon 1994) assumes mutually INDEPENDENT sources; hierarchical parent/child
# activation indicators are DEPENDENT by construction (child support subset of
# parent support), so ICA theorems do NOT apply. Full dictionary uniqueness
# under hierarchical supports with independent magnitude jitter is OPEN; only
# the support-dimension / reification-killing step above is proven.
print("       P7-scope: ICA theorems inapplicable (dependent indicators); "
      "full uniqueness under jitter = OPEN. Only P7/P7b are proven.")

# P8: 'unit magnitude grid' anchor (all feature magnitudes = 1).
# d=2: decompositions of v_p+v_c into two unit-norm unit-magnitude features are
# UNIQUE ({v_p, v_c}) and the 1-feature reading is impossible (|v_p+v_c|=sqrt2!=1)
x1, y1 = symbols("x1 y1", real=True)
u2x, u2y = 1 - x1, 1 - y1
sols2 = solve([Eq(x1**2 + y1**2, 1), Eq(u2x**2 + u2y**2, 1)], [x1, y1], dict=True)
pairs = {tuple(sorted([(s[x1], s[y1]), (1 - s[x1], 1 - s[y1])])) for s in sols2}
check("P8 anchor 'unit magnitude grid', d=2: two-feature decompositions of "
      "v_p+v_c are UNIQUE = {v_p, v_c}; one-feature impossible (norm sqrt(2)) "
      "-> rho IDENTIFIABLE in d=2",
      pairs == {(((0, 1)), ((1, 0)))} and sp.simplify((v_p + v_c).norm() - 1) != 0,
      f"{len(sols2)} solutions -> {len(pairs)} unordered pair(s)")
# d=3: the anchor is DEFEATED: an out-of-plane cancellation pair exists.
w1 = Matrix([R(1, 2), R(1, 2), 1 / sqrt(2)])
w2 = Matrix([R(1, 2), R(1, 2), -1 / sqrt(2)])
tgt = Matrix([1, 1, 0])
check("P8b d>=3: unit-magnitude anchor defeated ONLY by OFF-SPAN features: "
      "w1+w2 = v_p+v_c with |w1|=|w2|=1 requires nonzero 3rd coordinate; the "
      "data span is the e1-e2 plane",
      sp.simplify((w1 + w2 - tgt).norm()) == 0
      and sp.simplify(w1.norm() - 1) == 0 and sp.simplify(w2.norm() - 1) == 0
      and sp.simplify((w1 - Matrix([1, 0, 0])).norm()) != 0
      and w1[2] != 0 and w2[2] != 0)
# P8b caveat (review objection (e), accepted): restricting dictionary features
# to the linear span of the data — standard practice — reduces d>=3 to the d=2
# case, where P8 proved uniqueness. The P8b defeat is a loophole practitioners
# would close; under span-restriction the unit-magnitude-grid anchor RESTORES
# identifiability in all d.
print("       P8b-caveat: span-restriction (features in data span) reduces "
      "d>=3 to d=2 -> P8 uniqueness applies -> anchor restores identifiability.")

# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
print()
if FAILURES:
    print(f"OVERALL: FAIL ({len(FAILURES)} failed): {FAILURES}")
    sys.exit(1)
print("OVERALL: PASS — all certificates and boundary propositions verified exactly.")
sys.exit(0)
