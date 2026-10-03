"""
Proper factorial interaction test for the quantisation ablation.

A reviewer objected, correctly, that observing an excess of 0.0225 over the
additive prediction is descriptive, not a test. For a two-factor design

    Y_ij = mu + alpha_i + beta_j + (alpha*beta)_ij + eps

the interaction contrast is

    I = Y_both - Y_input - Y_weights + Y_neither

and the question is whether I is distinguishable from zero.

A second reviewer objected, also correctly, that the first version of this
script answered that question the wrong way. It resampled each of the four
cells independently and summed four independent variances, which treats the
four conditions as four unrelated samples. They are not. Reading step4.py:

    for qi, qw, name in (...):
        for s in args.seeds:
            tr, va, te = standard_masks(y, unit, s)
            r = run_cell(X, yi, tr, va, te, 4, 4, s, args.epochs, qi, qw)

every cell is run at the same seeds, and the seed fixes the partition and the
initialisation. The design is fully crossed on seed, so the four measurements
at seed s share a common run-to-run component. The contrast should be formed
WITHIN each seed and the spread taken across seeds:

    I_s = Y_both,s - Y_input,s - Y_weights,s + Y_neither,s

The point estimate does not move, because a mean of per-seed contrasts equals
the contrast of cell means; the contrast is linear. What moves is the
uncertainty, and it is the uncertainty the claim rests on.

Both analyses are printed below, the unpaired one labelled as inappropriate,
so that the change in the reported interval can be traced rather than taken
on trust.

Usage:
    python interaction.py step4_ablate.json
"""

import json
import sys

import numpy as np

NEED = ["neither", "input only", "weights only", "both"]


def contrast_per_seed(runs, n):
    """I_s for each seed, preserving the crossing of cells on seed."""
    return (runs["both"][:n] - runs["input only"][:n]
            - runs["weights only"][:n] + runs["neither"][:n])


def main(path):
    d = json.load(open(path))
    for k in NEED:
        if k not in d or "runs" not in d[k]:
            raise SystemExit("missing per-seed runs for '%s' in %s" % (k, path))
    runs = {k: np.asarray(d[k]["runs"], float) for k in NEED}
    n = min(len(v) for v in runs.values())

    print("=" * 68)
    print("FACTORIAL INTERACTION TEST, quantisation axes")
    print("=" * 68)
    for k in NEED:
        print("  %-14s %.4f +/- %.4f   %s"
              % (k, runs[k].mean(), runs[k].std(ddof=1),
                 " ".join("%.3f" % x for x in runs[k])))

    I_point = (runs["both"].mean() - runs["input only"].mean()
               - runs["weights only"].mean() + runs["neither"].mean())
    print("\n  interaction contrast I = both - input - weights + neither")
    print("  point estimate  %+.4f" % I_point)

    # ---------------------------------------------------------- paired
    I = contrast_per_seed(runs, n)
    print("\n  PAIRED analysis, the appropriate one: contrast within each")
    print("  seed, spread across seeds (%d seeds)" % n)
    print("  per-seed I   %s" % " ".join("%+.4f" % x for x in I))
    print("  mean         %+.4f   (equals the point estimate above, since a"
          % I.mean())
    print("               mean of contrasts is the contrast of means)")
    sd = I.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else float("nan")
    t = I.mean() / se if se > 0 else float("nan")
    print("  sd %.4f   se %.4f   one-sample t %+.3f on %d df"
          % (sd, se, t, n - 1))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(20000, n))
    boots = I[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("  bootstrap 95%% CI, resampling SEEDS  [%+.4f, %+.4f]" % (lo, hi))

    # ------------------------------------------------- unpaired, for contrast
    rng2 = np.random.default_rng(0)
    ub = []
    for _ in range(20000):
        s = {k: rng2.choice(runs[k], len(runs[k]), replace=True).mean()
             for k in NEED}
        ub.append(s["both"] - s["input only"] - s["weights only"]
                  + s["neither"])
    ulo, uhi = np.percentile(np.array(ub), [2.5, 97.5])
    use = np.sqrt(sum(runs[k].var(ddof=1) / len(runs[k]) for k in NEED))
    print("\n  for comparison only, NOT appropriate here: treating the four")
    print("  cells as independent samples gives CI [%+.4f, %+.4f] and t %+.2f."
          % (ulo, uhi, I_point / use if use > 0 else float("nan")))
    print("  It discards the seed crossing and is reported solely so the")
    print("  change from the earlier version of this script is visible.")

    print("")
    if lo > 0 or hi < 0:
        print("  The interaction is distinguishable from zero. The axes do NOT")
        print("  act additively, and a study varying one at a time would")
        print("  misstate the joint cost.")
    else:
        print("  The confidence interval includes zero. We cannot distinguish")
        print("  the interaction from zero at this sample size. State that,")
        print("  NOT 'the axes are additive' -- absence of evidence for an")
        print("  interaction is not evidence that none exists. The interval is")
        print("  wide, so a moderate interaction would not have been detected;")
        print("  the honest summary is that the experiment is inconclusive")
        print("  about interaction rather than that there is none.")
        print("  Report: I = %+.4f, 95%% CI [%+.4f, %+.4f], t = %+.2f."
              % (I_point, lo, hi, t))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "step4_ablate.json")
