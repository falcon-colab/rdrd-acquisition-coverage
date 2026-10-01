"""
Step 5: does the 4x aggregation benefit generalise beyond one acquisition?

Reviewers correctly identified the paper's largest weakness: both findings
rest on the same single acquisition. The coverage result concerns 13-48
directly, and the aggregation result is measured by transfer TO 13-48. The
effective sample size at the level of acquisitions is one.

This repeats the aggregation comparison with SEVERAL acquisitions held out,
one at a time. For each held-out acquisition we train at full resolution and
at 4x Doppler aggregation and measure recall on the held-out acquisition.

If aggregation helps across acquisitions, the claim becomes "aggregation
improves transfer to unseen acquisitions", which is general. If it helps
only 13-48, the claim stays specific to that acquisition -- which is still
reportable, and honest.

Sessions are chosen by out-of-fold recall from Step 2c so that the set spans
easy and hard cases, not only the pathological one.

Usage:
    python step5.py --data D --sessions drone/13-48 drone/12-34 person/11-23 \\
                    car/15-37 drone/15-21 --seeds 0 1 2
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES
from step2b import normalise
from step3 import reduce_doppler, fit, evaluate

VAL_FRAC = 0.15


def masks_for(y, unit, target, seed):
    rng = np.random.default_rng(700 + seed)
    te = unit == target
    pool = np.where(~te)[0]
    rng.shuffle(pool)
    n_val = int(VAL_FRAC * len(pool))
    va = np.zeros(len(y), bool); va[pool[:n_val]] = True
    tr = np.zeros(len(y), bool); tr[pool[n_val:]] = True
    return tr, va, te


def recall_on(model, Xn, yi, te, cls):
    r = evaluate(model, Xn, yi, te)
    return r[cls]["recall"], r


def paired_t(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])

    missing = [s for s in args.sessions if not (unit == s).any()]
    if missing:
        raise SystemExit("not found: %s\navailable: %s"
                         % (missing, sorted(set(unit.tolist()))))

    # pre-compute the reduced inputs once
    Xn = {f: normalise(reduce_doppler(X, f), "offset") for f in args.factors}

    print("=" * 76)
    print("LEAVE-ONE-ACQUISITION-OUT, %d acquisitions x %d seeds"
          % (len(args.sessions), len(args.seeds)))
    print("=" * 76)

    results = {}
    for target in args.sessions:
        cls = target.split("/")[0]
        n = int((unit == target).sum())
        per_factor = {}
        for f in args.factors:
            recs = []
            for s in args.seeds:
                tr, va, te = masks_for(y, unit, target, s)
                m = fit(Xn[f], yi, tr, va, s, args.epochs)
                rec, _ = recall_on(m, Xn[f], yi, te, cls)
                recs.append(rec)
            per_factor[f] = recs
        results[target] = {"class": cls, "n": n,
                           "by_factor": {str(k): v for k, v in per_factor.items()}}
        line = "  %-16s (%-6s n=%4d)" % (target.split("/")[1], cls, n)
        for f in args.factors:
            line += "   %dx %.3f+/-%.3f" % (f, np.mean(per_factor[f]),
                                            np.std(per_factor[f]))
        if len(args.factors) == 2:
            a, b = args.factors
            line += "   delta %+.3f" % (np.mean(per_factor[b])
                                        - np.mean(per_factor[a]))
        print(line)

    if len(args.factors) != 2:
        if args.out:
            json.dump(results, open(args.out, "w"), indent=2)
        return

    lo, hi = args.factors
    print("\n" + "=" * 76)
    print("DOES %dx HELP ACROSS ACQUISITIONS?" % hi)
    print("=" * 76)
    deltas, names = [], []
    for t, r in results.items():
        dlt = np.mean(r["by_factor"][str(hi)]) - np.mean(r["by_factor"][str(lo)])
        deltas.append(dlt); names.append(t.split("/")[1])
    deltas = np.array(deltas)
    print("  per-acquisition change in recall from %dx to %dx:" % (lo, hi))
    for nm, dl in zip(names, deltas):
        print("    %-16s %+.4f" % (nm, dl))
    t = paired_t(deltas)
    n_pos = int((deltas > 0).sum())
    print("\n  mean %+0.4f   sd %.4f   paired t %+.2f over %d acquisitions"
          % (deltas.mean(), deltas.std(ddof=1) if len(deltas) > 1 else 0.0,
             t, len(deltas)))
    print("  improved in %d of %d acquisitions" % (n_pos, len(deltas)))

    # sign test, exact, two-tailed
    from math import comb
    n = len(deltas)
    k = n_pos
    p = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    print("  one-sided sign test p = %.3f" % p)

    print("")
    if n_pos == len(deltas) and len(deltas) >= 4:
        print("  Aggregation helped EVERY acquisition tested. The claim")
        print("  generalises: report it as improving transfer to unseen")
        print("  acquisitions, not only to 13-48.")
    elif n_pos > len(deltas) / 2 and abs(t) > 2.5:
        print("  Aggregation helps on average across acquisitions. Report as a")
        print("  general effect with the per-acquisition spread shown.")
    elif deltas[names.index("13-48")] > 0.1 if "13-48" in names else False:
        print("  Aggregation helps 13-48 but not consistently elsewhere. The")
        print("  claim must stay specific to that acquisition, and the paper")
        print("  should say the effect did not generalise in this test.")
    else:
        print("  No consistent benefit across acquisitions. Report the")
        print("  original 13-48 result as acquisition-specific.")

    if args.out:
        json.dump({"per_session": results,
                   "deltas": {n_: float(v) for n_, v in zip(names, deltas)},
                   "mean_delta": float(deltas.mean()),
                   "paired_t": t, "n_improved": n_pos,
                   "sign_test_p": p}, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
