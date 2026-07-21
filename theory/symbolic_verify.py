"""Symbolic verification (sympy) of ALL hand-derived formulas from today:
eps0(lam,q,p0), p0*(lam,q), C(lam), lam_crit, k* -- derived from first
principles (the Lemma-2 gain formula), then compared to the hand formulas."""
import sympy as sp

lam, q, p0, eps, k = sp.symbols('lam q p0 eps k', positive=True)
rt2 = sp.sqrt(2)

def gain(c):  # (c - lam/2)^2 assuming c > lam/2 (valid: all active projections
    return (c - lam/2)**2   # here are in {1, sqrt2, sqrt2/2} > lam/2 for lam<rt2/... noted

# T(theta) from Lemma 2 for the two pure frames.
# theta=0 (faithful): projections: joint->(1,1), psolo->(1,0->col2 proj 0), csolo->(0,1)
T0 = q*(2 - gain(1) - gain(1)) + p0*(1 - gain(1)) + eps*(1 - gain(1))
# theta=-45 (anti): joint->(0, rt2); psolo->(rt2/2, rt2/2); csolo->(-rt2/2 [no gain], rt2/2)
T45 = q*(2 - gain(rt2)) + p0*(1 - gain(rt2/2) - gain(rt2/2)) + eps*(1 - gain(rt2/2))

# 1) eps0: solve T45 - T0 = 0 for eps
eps0_derived = sp.solve(sp.expand(T45 - T0), eps)[0]
eps0_hand = lam*((2-rt2)*q - (rt2-1)*p0 + lam*(p0-q)/4) / (sp.Rational(1,2) - (2-rt2)*lam/2)
print("eps0 match:", sp.simplify(eps0_derived - eps0_hand) == 0)

# 2) p0*: solve numerator of eps0 = 0 for p0
p0s_derived = sp.solve(sp.numer(sp.together(eps0_derived)), p0)[0]
p0s_hand = q*((2-rt2) - lam/4) / ((rt2-1) - lam/4)
print("p0* match:", sp.simplify(p0s_derived - p0s_hand) == 0)
print("p0*/q limit lam->0:", sp.limit(p0s_derived/q, lam, 0), "(expect sqrt(2) =", sp.sqrt(2), ")")

# 3) vanilla eps* from the ORIGINAL event-loss table (independent re-derivation):
# absorbed-preferred iff q*[(2-rt2)lam - lam^2/4] - eps*[1/2 - (2-rt2)lam/2] > 0
eps_star_derived = sp.solve(q*((2-rt2)*lam - lam**2/4) - eps*(sp.Rational(1,2) - (2-rt2)*lam/2), eps)[0]
eps_star_hand = lam*q*(8 - 4*rt2 - lam) / (2*(1 - (2-rt2)*lam))
print("eps* match:", sp.simplify(eps_star_derived - eps_star_hand) == 0)

# 4) weighted remedy: inverse-density weights w_j=1/q, w_c=1/eps ->
# absorbed iff (1/q)*q*[(2-rt2)lam - lam^2/4] > (1/eps)*eps*[1/2 - (2-rt2)lam/2]
C_derived = ((2-rt2)*lam - lam**2/4) / (sp.Rational(1,2) - (2-rt2)*lam/2)
C_hand = lam*(8 - 4*rt2 - lam) / (2*(1 - (2-rt2)*lam))
print("C(lam) match:", sp.simplify(C_derived - C_hand) == 0)

# 5) lam_crit: solve C(lam)=1
sols = sp.solve(sp.Eq(C_derived, 1), lam)
lam_crit_hand = ((12 - 6*rt2) - sp.sqrt((12 - 6*rt2)**2 - 8)) / 2
match = any(sp.simplify(s - lam_crit_hand) == 0 for s in sols)
print("lam_crit match:", match, " numeric:", [sp.N(s, 6) for s in sols])

# 6) misestimation bound k*: absorbed iff C(lam)*k > 1 -> k* = 1/C(lam); at lam=0.2:
print("k*(0.2) =", sp.N(1/C_derived.subs(lam, sp.Rational(1,5)), 4), "(hand: 4.12)")

# validity condition for the gain assumption used above:
print("gain-branch validity: requires lam/2 < sqrt2/2, i.e. lam <", sp.N(rt2,4))
