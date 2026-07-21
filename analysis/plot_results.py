"""Figures for the SAE absorption report (light mode, static PNG).

Fig 1: angle-vs-eps small multiples per lambda, theory overlay, eps* marker.
Fig 2: scaling collapse — all configs, x = eps/eps*, single hue, shape = config.
Fig 3: classical recovery vs k for n=128/256, worst-case k* marker.
"""
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": SEC, "axes.titlecolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

def eps_star(lam, q):
    s2 = math.sqrt(2)
    return lam * q * (8 - 4 * s2 - lam) / (2 * (1 - (2 - s2) * lam))

ab = pd.read_csv("results_absorption.csv")
th = pd.read_csv("theory_curves.csv")

# ---------------------------------------------------------------- Fig 1
lams = [0.05, 0.1, 0.2, 0.3]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, lam in zip(axes, lams):
    q = 0.2
    es = eps_star(lam, q)
    t = th[(th.lam == lam) & (th.q == q)].sort_values("eps")
    d = ab[(ab.lam == lam) & (ab.q == q)]
    for y, txt in [(45, "absorbed 45°"), (90, "faithful 90°")]:
        ax.axhline(y, color=GRID, lw=1, zorder=1)
        ax.text(0.178, y + 1.5, txt, color=MUT, fontsize=8, ha="right")
    ax.axvline(es, color=MUT, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(es, 101, "ε*", color=SEC, fontsize=9, ha="center")
    ax.plot(t.eps, t.phi_theory, color=BLUE, lw=2, zorder=3,
            label="theory (global optimum)")
    ax.scatter(d.eps, d.phi_child, s=42, color=ORANGE, zorder=4, alpha=0.85,
               edgecolors=SURF, linewidths=1, label="trained SAE (3 seeds)")
    ax.set_title(f"λ = {lam}", fontsize=10)
    ax.set_xlim(-0.006, 0.19); ax.set_ylim(38, 106)
    ax.set_xlabel("ε  (child-solo probability)")
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("learned child-latent angle φ (deg)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=9)
fig.suptitle("Feature absorption is a phase transition  (q = 0.2, p₀ = 0.2)",
             y=1.14, fontsize=12, color=INK)
fig.tight_layout()
fig.savefig("fig1_absorption_transition.png", dpi=150, bbox_inches="tight")
print("fig1 done")

# ---------------------------------------------------------------- Fig 2
fig, ax = plt.subplots(figsize=(6.4, 4))
marks = {(0.2, 0.05): "o", (0.2, 0.1): "s", (0.2, 0.2): "^",
         (0.2, 0.3): "D", (0.1, 0.2): "v"}
for (q, lam), mk in marks.items():
    d = ab[(ab.lam == lam) & (ab.q == q)]
    es = eps_star(lam, q)
    g = d.groupby("eps").phi_child.mean().reset_index()
    ax.scatter(g.eps / es, g.phi_child, marker=mk, s=46, color=BLUE,
               facecolors="none", linewidths=1.6, label=f"λ={lam}, q={q}")
ax.set_ylim(42, 94)
ax.axvline(1.0, color=MUT, lw=1.2, ls=(0, (4, 3)))
ax.text(0.94, 43.2, "ε = ε*", color=SEC, fontsize=9, ha="right")
ax.axhline(45, color=GRID, lw=1)
ax.axhline(90, color=GRID, lw=1)
ax.text(0.3, 43.4, "absorbed", color=MUT, fontsize=8, ha="left")
ax.text(0.068, 90.8, "faithful", color=MUT, fontsize=8, ha="left")
ax.set_xscale("log")
ax.set_xlabel("ε / ε*(λ, q)   (log scale)")
ax.set_ylabel("mean learned angle φ (deg)")
ax.set_title("Transition location collapses under ε* rescaling", color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig("fig2_collapse.png", dpi=150, bbox_inches="tight")
print("fig2 done")

# ---------------------------------------------------------------- Fig 3
rec = pd.read_csv("results_recovery.csv")
fig, ax = plt.subplots(figsize=(6.4, 4))
for n, col in [(128, BLUE), (256, ORANGE)]:
    d = rec[rec.n == n].groupby("k").frac_recovered.mean().reset_index()
    ax.plot(d.k, d.frac_recovered, color=col, lw=2, marker="o", markersize=6,
            markeredgecolor=SURF, markeredgewidth=1, label=f"n = {n}")
ks = rec.kstar_worstcase.mean()
ax.axvline(ks, color=MUT, lw=1.2, ls=(0, (4, 3)))
ax.text(ks + 0.5, 0.30, f"worst-case k* ≈ {ks:.1f}\n(Donoho–Elad)",
        color=SEC, fontsize=8)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_xlabel("k  (active features per sample)")
ax.set_ylabel("fraction of features recovered (cos > 0.9)")
ax.set_title("Recovery runs ~5× past the worst-case bound,\nbut is capped by training dynamics (d = 64)",
             color=INK)
ax.set_ylim(-0.03, 1.05)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig("fig3_recovery.png", dpi=150, bbox_inches="tight")
print("fig3 done")
