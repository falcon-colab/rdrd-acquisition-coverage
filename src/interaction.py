"""
Proper factorial interaction test for the quantisation ablation.

A reviewer objected, correctly, that observing an excess of 0.0225 over the
additive prediction is descriptive, not a test. For a two-factor design

    Y_ij = mu + alpha_i + beta_j + (alpha*beta)_ij + eps

the interaction contrast is

    I = Y_both - Y_input - Y_weights + Y_neither

and the question is whether I is distinguishable from zero. This bootstraps
a confidence interval on I from the per-seed runs already saved.

Usage:
    python interaction.py step4_ablate.json
"""

import json
import sys

import numpy as np


def main(path):
    d = json.load(open(path))
    need = ["neither", "input only", "weights only", "both"]
    for k in need:
        if k not in d or "runs" not in d[k]:
            raise SystemExit("missing per-seed runs for '%s' in %s" % (k, path))
    runs = {k: np.asarray(d[k]["runs"], float) for k in need}
    n = min(len(v) for v in runs.values())

    print("=" * 68)
    print("FACTORIAL INTERACTION TEST, quantisation axes")
    print("=" * 68)
    for k in need:
        print("  %-14s %.4f +/- %.4f   %s"
              % (k, runs[k].mean(), runs[k].std(ddof=1),
                 " ".join("%.3f" % x for x in runs[k])))

    I_point = (runs["both"].mean() - runs["input only"].mean()
               - runs["weights only"].mean() + runs["neither"].mean())
    print("\n  interaction contrast I = both - input - weights + neither")
    print("  point estimate  %+.4f" % I_point)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(20000):
        s = {k: rng.choice(runs[k], len(runs[k]), replace=True).mean()
             for k in need}
        boots.append(s["both"] - s["input only"] - s["weights only"]
                     + s["neither"])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("  bootstrap 95%% CI  [%+.4f, %+.4f]" % (lo, hi))

    se = np.sqrt(sum(runs[k].var(ddof=1) / len(runs[k]) for k in need))
    t = I_point / se if se > 0 else float("nan")
    print("  approximate t     %+.2f  (se %.4f)" % (t, se))

    print("")
    if lo > 0 or hi < 0:
        print("  The interaction is distinguishable from zero. The axes do NOT")
        print("  act additively, and a study varying one at a time would")
        print("  misstate the joint cost.")
    else:
        print("  The confidence interval includes zero. We cannot distinguish")
        print("  the interaction from zero at this sample size. State that,")
        print("  NOT 'the axes are additive' -- absence of evidence for an")
        print("  interaction is not evidence that none exists.")
        print("  Report: I = %+.4f, 95%% CI [%+.4f, %+.4f]." % (I_point, lo, hi))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "step4_ablate.json")
