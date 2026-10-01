"""
Re-analysis of Step 4 from saved JSON. No retraining.

Question: is the 4x unseen-regime rescue a reliable shift, or a widened
distribution in which some seeds succeed and others do not? The mean rose
but the standard deviation rose four to five times more, and a mean is a
poor summary of a bimodal sample.
"""

import json
import os
import sys

import numpy as np


def mannwhitney_u(a, b):
    """Rank-sum statistic and a normal-approximation z. No SciPy needed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    return float(u1), float((u1 - mu) / sd) if sd > 0 else float("nan")


def bootstrap_ci(a, b, n=20000, seed=0):
    """CI on the difference of means, resampling seeds."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a), replace=True).mean()
         - rng.choice(b, len(b), replace=True).mean() for _ in range(n)]
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def gap_statistic(v):
    """Largest gap between consecutive sorted values, relative to the range."""
    s = np.sort(np.asarray(v, float))
    if len(s) < 3 or s[-1] == s[0]:
        return 0.0, None
    gaps = np.diff(s)
    i = int(gaps.argmax())
    return float(gaps[i] / (s[-1] - s[0])), (float(s[i]), float(s[i + 1]))


def main(path):
    d = json.load(open(path))
    print("=" * 72)
    print("PER-SEED VALUES, unseen-regime recall")
    print("=" * 72)
    for bits in (32, 8):
        print("\n  %d-bit" % bits)
        for f in (1, 2, 4, 8):
            k = "%dx_%dbit" % (f, bits)
            if k not in d:
                continue
            v = d[k]["unseen_runs"]
            g, where = gap_statistic(v)
            note = ""
            if g > 0.45 and where:
                note = "   <== large gap between %.2f and %.2f" % where
            print("    %-4s %s   mean %.4f  sd %.4f  gapstat %.2f%s"
                  % ("%dx" % f, " ".join("%.3f" % x for x in sorted(v)),
                     np.mean(v), np.std(v, ddof=1), g, note))

    print("\n" + "=" * 72)
    print("4x AGAINST 1x  (rank test and bootstrap, not just a t)")
    print("=" * 72)
    for bits in (32, 8):
        a = d["4x_%dbit" % bits]["unseen_runs"]
        b = d["1x_%dbit" % bits]["unseen_runs"]
        u, z = mannwhitney_u(a, b)
        lo, hi = bootstrap_ci(a, b)
        sep = min(a) > max(b)
        print("\n  %d-bit" % bits)
        print("    means            %.4f vs %.4f" % (np.mean(a), np.mean(b)))
        print("    Mann-Whitney U   %.1f of %d   z = %+.2f"
              % (u, len(a) * len(b), z))
        print("    bootstrap 95%% CI on the difference: [%+.4f, %+.4f]" % (lo, hi))
        print("    every 4x seed above every 1x seed: %s" % sep)
        if lo > 0:
            print("    CI excludes zero -> the shift is real")
        else:
            print("    CI includes zero -> not established at 5 seeds")

    print("\n" + "=" * 72)
    print("READING")
    print("=" * 72)
    print("  A high gap statistic with a wide sd means the sample is split:")
    print("  some seeds find a solution that transfers to the unseen regime")
    print("  and others do not. That is a different claim from 'aggregation")
    print("  improves transfer', and it should be reported as such.")
    print("")
    print("  If the bootstrap CI excludes zero, the effect is real but noisy,")
    print("  and the honest sentence is that 4x aggregation raises unseen")
    print("  recall on average while leaving it highly seed-dependent.")
    print("")
    print("  If the CI includes zero, five seeds do not establish it. Either")
    print("  run ten, or report the grid descriptively with no claim.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "step4_seeds.json")
