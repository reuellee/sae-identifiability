"""Generate the real-model figures for PAPER.md from frozen result JSON.

Pure stdlib + numpy, emitting pgfplots LaTeX compiled by xelatex (the same
toolchain ops/build_paper.sh already requires). No matplotlib, no pip: the
orchestrator has neither, and figures that regenerate from the frozen JSON on any
box with texlive are more reproducible than committed binaries.

Sources (all frozen, none recomputed here):
  results/real/round13b_results.json   48 SAEs, 3 widths x {L1,TopK} x 8 seeds
  results/real/round14_results.json    32 SAEs, carrier decomposition

Outputs figs/fig_*.pdf (gitignored like PAPER.pdf) from figs/fig_*.tex (tracked).
Run: python3 ops/mkfigs.py && ls figs/
"""
import json, os, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

WIDTHS = [2048, 4096, 16384]
ARCHES = [("l1", "$L_1$", "blue"), ("topk", "TopK", "red!70!black")]

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{amsmath}
\begin{document}
"""


def _load(name):
    with open(os.path.join(ROOT, "results", "real", name)) as f:
        return json.load(f)


def _agg(rows, field):
    """mean and standard error over the 8 seeds, per (width, arch)."""
    out = {}
    for r in rows:
        out.setdefault((r["m"], r["arch"]), []).append(r[field])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
            for k, v in out.items()}


def fig_width_sweep(r13b):
    """Round 13b P1: absorption FALLS as the dictionary shrinks -- H1 refuted."""
    fam = _agg(r13b, "rate_family")
    body = [r"""\begin{tikzpicture}
\begin{axis}[width=9cm,height=6.4cm,xmode=log,log basis x=2,
  xlabel={dictionary width $m$},ylabel={family absorption rate},
  xtick={2048,4096,16384},xticklabels={2048,4096,16384},
  legend pos=north west,legend cell align=left,grid=major,
  ymin=0,every axis plot/.append style={thick,mark size=2.2pt}]"""]
    for arch, label, colour in ARCHES:
        pts = " ".join(f"({m},{fam[(m,arch)][0]:.5f}) +- (0,{fam[(m,arch)][1]:.5f})"
                       for m in WIDTHS)
        body.append(f"\\addplot+[color={colour},mark=*,error bars/.cd,"
                    f"y dir=both,y explicit] coordinates {{{pts}}};")
        body.append(f"\\addlegendentry{{{label}}}")
    body.append(r"\end{axis}" "\n" r"\end{tikzpicture}")
    return "\n".join(body)


def fig_endpoint_inflation(r13b):
    """Rounds 13a/13b P4: the single-latent endpoint overstates absorption at every
    width, because part of what it counts is feature splitting."""
    single, fam = _agg(r13b, "rate_single"), _agg(r13b, "rate_family")
    body = [r"""\begin{tikzpicture}
\begin{axis}[width=9cm,height=6.4cm,ybar,bar width=7pt,
  symbolic x coords={2048-L1,2048-TopK,4096-L1,4096-TopK,16384-L1,16384-TopK},
  xtick=data,x tick label style={rotate=40,anchor=east,font=\small},
  ylabel={absorption rate},legend pos=north west,legend cell align=left,
  grid=major,ymin=0]"""]
    order = [(m, a) for m in WIDTHS for a, _, _ in ARCHES]
    lbl = {"l1": "L1", "topk": "TopK"}
    for field, agg, colour, name in (
            ("single", single, "gray!60", "single-latent (SAEBench-style)"),
            ("family", fam, "blue!70", "family endpoint (splitting-corrected)")):
        pts = " ".join(f"({m}-{lbl[a]},{agg[(m,a)][0]:.5f})" for m, a in order)
        body.append(f"\\addplot[fill={colour},draw=black!50] coordinates {{{pts}}};")
        body.append(f"\\addlegendentry{{{name}}}")
    body.append(r"\end{axis}" "\n" r"\end{tikzpicture}")
    return "\n".join(body)


def fig_carrier(r14, val):
    """Round 14, two panels telling the two halves of the result.

    LEFT (registered P2): the letter direction finds a REPEATED carrier less often
    than an arbitrary direction does -- no single broad parent latent.
    RIGHT (post-hoc D2): per trial, the letter's mass is nearly as concentrated on
    absorbed trials as on control trials -- carriage is real, just trial-specific.
    Plotting only the left panel was how the earlier, withdrawn "diffuse" reading
    looked convincing, so both go in the same figure.
    """
    cells = [(r["m"], rec["share"], rec["share_null"])
             for r in r14 for rec in r["per_letter"].values() if rec.get("scored")]

    def pmean(field):
        v = [c[field] for r in val for c in r["per_letter"].values()
             if field in c and np.isfinite(c[field])]
        return float(np.mean(v))

    body = [r"""\begin{tikzpicture}
\begin{axis}[name=left,width=7.4cm,height=6.2cm,ybar,bar width=13pt,
  symbolic x coords={m=2048,m=16384},xtick=data,
  ylabel={top-1 \emph{modal} carrier share},
  title={\small registered P2: no \emph{recurring} carrier},
  legend style={at={(0.03,0.97)},anchor=north west,font=\footnotesize},
  legend cell align=left,grid=major,ymin=0]"""]
    for j, (name, colour) in enumerate((("letter direction", "blue!70"),
                                        ("random-direction null", "gray!60"))):
        pts = " ".join(f"(m={m},{np.mean([c[1 + j] for c in cells if c[0] == m]):.5f})"
                       for m in (2048, 16384))
        body.append(f"\\addplot[fill={colour},draw=black!50] coordinates {{{pts}}};")
        body.append(f"\\addlegendentry{{{name}}}")
    body.append(r"\end{axis}")

    a, c = pmean("d2_pertrial_conc_A"), pmean("d2_pertrial_conc_C")
    body.append(r"""\begin{axis}[at={(left.east)},xshift=1.9cm,anchor=west,
  width=7.4cm,height=6.2cm,ybar,bar width=22pt,
  symbolic x coords={absorbed,control},xtick=data,
  ylabel={\emph{per-trial} top carrier share},
  title={\small post-hoc D2: carriage is trial-specific},
  grid=major,ymin=0,ymax=0.8,
  nodes near coords,every node near coord/.append style={font=\footnotesize}]""")
    body.append(f"\\addplot[fill=blue!45,draw=black!50] coordinates "
                f"{{(absorbed,{a:.4f}) (control,{c:.4f})}};")
    body.append(r"\end{axis}" "\n" r"\end{tikzpicture}")
    return "\n".join(body)


def emit(name, body):
    tex = os.path.join(FIGS, name + ".tex")
    with open(tex, "w") as f:
        f.write(PREAMBLE + body + "\n" + r"\end{document}" + "\n")
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", FIGS, tex],
                       capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(os.path.join(FIGS, name + ".pdf"))
    print(f"  {name}: {'OK' if ok else 'FAILED'}")
    if not ok:
        print("\n".join(r.stdout.splitlines()[-25:]))
    return ok


def main():
    r13b, r14 = _load("round13b_results.json"), _load("round14_results.json")
    val = _load("round14_validity.json")
    ok = True
    ok &= emit("fig_width_sweep", fig_width_sweep(r13b))
    ok &= emit("fig_endpoint_inflation", fig_endpoint_inflation(r13b))
    ok &= emit("fig_carrier", fig_carrier(r14, val))
    for f in os.listdir(FIGS):                      # xelatex litter
        if f.rsplit(".", 1)[-1] in ("aux", "log", "out"):
            os.remove(os.path.join(FIGS, f))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
