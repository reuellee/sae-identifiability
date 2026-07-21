"""
Verification script for the adversarial review.
This script implements the loss calculation from theory_merged.py to verify
the claim that L(anti) < L(faithful) under the specified conditions.
"""
import math
import itertools

# Parameters from the claim
LAM = 0.2
Q = 0.2
P0 = 0.2
EPS = 0.01
BSTAR = LAM * Q * (8 - 4 * math.sqrt(2) - LAM) / 2  # beta*
BETA = BSTAR

# Dictionary configurations to test
ANTI_ANGLES = [-37, 45]
FAITHFUL_OPT_ANGLES = [5, 85]

def dirv(deg):
    """Creates a 2D direction vector from an angle in degrees."""
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))

def dot(a, b):
    """Computes the dot product of two 2D vectors."""
    return a[0] * b[0] + a[1] * b[1]

def event_loss(x, D, lam, config_name, event_name):
    """
    Calculates min over f>=0 of ||x - D f||^2 + lam*sum(f), by active-set enumeration.
    Also prints the optimal coefficients found.
    """
    best_loss = dot(x, x)  # Loss for f=0
    best_f = [0.0] * len(D)
    m = len(D)

    # Active set of size 1
    for i in range(m):
        d = D[i]
        f0 = dot(d, x) - lam / 2
        if f0 > 0:
            rx = (x[0] - f0 * d[0], x[1] - f0 * d[1])
            loss = dot(rx, rx) + lam * f0
            if loss < best_loss:
                best_loss = loss
                best_f = [0.0] * m
                best_f[i] = f0

    # Active set of size 2
    if m == 2:
        d1, d2 = D[0], D[1]
        g = dot(d1, d2)
        det = 1 - g * g
        if abs(det) > 1e-12:
            b1 = dot(d1, x) - lam / 2
            b2 = dot(d2, x) - lam / 2
            f1 = (b1 - g * b2) / det
            f2 = (b2 - g * b1) / det
            if f1 > 0 and f2 > 0:
                rx = (x[0] - f1 * d1[0] - f2 * d2[0], x[1] - f1 * d1[1] - f2 * d2[1])
                loss = dot(rx, rx) + lam * (f1 + f2)
                if loss < best_loss:
                    best_loss = loss
                    best_f = [f1, f2]
    
    print(f"  {config_name} - {event_name}: Optimal coeffs f={ [round(c, 3) for c in best_f] }")
    return best_loss

def total_loss(angles, eps, beta, config_name):
    """Calculates the total loss for a given dictionary configuration."""
    D = [dirv(a) for a in angles]
    print(f"Calculating loss for {config_name} config: {angles} degrees")
    
    # Event definitions: (probability, vector)
    events = {
        "joint":       (Q,   (1.0, 1.0)),
        "parent-solo": (P0,  (1.0, 0.0)),
        "child-solo":  (eps, (0.0, 1.0)),
    }

    L_reconstruction = 0
    for name, (prob, x) in events.items():
        L_reconstruction += prob * event_loss(x, D, LAM, config_name, name)

    L_penalty = 0
    if len(D) > 1:
        for i in range(len(D)):
            for j in range(i + 1, len(D)):
                L_penalty += beta * dot(D[i], D[j]) ** 2
    
    total = L_reconstruction + L_penalty
    print(f"  {config_name}: Reconstruction Loss = {L_reconstruction:.6f}")
    print(f"  {config_name}: Coherence Penalty   = {L_penalty:.6f}")
    print(f"  {config_name}: Total Loss          = {total:.6f}")
    return total

# --- Main Calculation ---
print(f"Parameters: lam={LAM}, q={Q}, p0={P0}, eps={EPS}, beta=beta*={BETA:.4f}")

L_anti = total_loss(ANTI_ANGLES, EPS, BETA, "Anti-Rotated")
L_faithful = total_loss(FAITHFUL_OPT_ANGLES, EPS, BETA, "Faithful-Opt")

print("\n--- Comparison ---")
print(f"L(anti {-30,46})     = {L_anti:.6f}")
print(f"L(faithful-opt {-1,73}) = {L_faithful:.6f}")

if L_anti < L_faithful:
    print("\nConclusion: The claim holds. L(anti) < L(faithful).")
    print(f"Difference: {L_faithful - L_anti:.6f}")
else:
    print("\nConclusion: The claim is refuted. L(anti) >= L(faithful).")

# Verification of non-negative coding for parent-solo and child-solo events
# This is done by inspecting the print output from the event_loss function.
# For the anti-rotated config, we need to check if x=a_p and x=a_c can be
# reconstructed with positive coefficients.
# x=a_p = (1,0)
# x=a_c = (0,1)
# d1 = d(-37) = (cos(-37), sin(-37)) = (0.7986, -0.6018)
# d2 = d(46)  = (cos(46),  sin(46))  = (0.6947,  0.7193)
# To reconstruct a_p, we need c1*d1 + c2*d2 = (1,0) with c1,c2 >= 0.
# This implies:
# c1*0.7986 + c2*0.6947 = 1
# c1*(-0.6018) + c2*0.7193 = 0  => c1 = (0.7193/0.6018)*c2 = 1.195*c2
# Substituting into the first equation:
# (1.195*c2)*0.7986 + c2*0.6947 = 1
# c2 * (0.954 + 0.6947) = 1 => c2 = 1 / 1.6487 = 0.606
# c1 = 1.195 * 0.606 = 0.724
# Since c1 and c2 are positive, reconstruction is possible. The `event_loss` function
# should find something similar (though it includes the lambda penalty).
