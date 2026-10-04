"""The coverage experiment on every acquisition, not just three.

This answers the objection that has survived every review of this paper: the
coverage result rests on one acquisition, with two controls chosen after
seeing which acquisitions scored worst. Selecting controls on the dependent
variable is a real weakness, and no amount of careful wording repairs it.
The repair is to stop selecting. Run the identical experiment on all 52
units and report the distribution.

For each unit, at each seed: hold the unit out of training entirely and
score a held-back half of it; then add the other half to training and score
the same held-back half. Recovery is the difference. That is exactly the
procedure step2d applies to 13-48, reused here rather than reimplemented, so
the numbers are comparable to the ones already published.

What the result can settle, either way:

  If 13-48's recovery is an outlier against 52 units, the paper stops
  needing post hoc controls. The claim becomes distributional: recovery is
  small for almost every acquisition and very large for this one, and the
  scatter of baseline recall against recovery shows whether low recall
  predicts recovery in general or whether this acquisition is separate.

  If several units show large recovery, the finding is more common than the
  paper claims and the framing must change to say so. That would be a more
  interesting paper, not a failed experiment.

  If recovery tracks baseline recall closely, then recovery is mostly a
  regression-to-the-mean artefact of starting low, and the 13-48 result is
  weaker than it looks. This is the outcome that would most damage the
  paper's argument, which is why it is worth running.

COST. 52 units by 2 conditions by the seed count, so 312 trainings at the
default of three seeds. Expect several hours. The script therefore writes
its output after every unit and skips units already present on restart, so
a disconnected session loses at most one unit's work. Rerun the same
command and it resumes.

    python src/coverage_all.py --data DIR/dataset.npz \\
        --out DIR/reports/coverage_all.json --seeds 0 1 2

Small units are kept rather than filtered, because filtering on size would
reintroduce a selection rule. Their held-back halves are small and their
recall is correspondingly noisy, so the report records n for each and the
summary flags units below 40 samples.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2d import coverage_test                            # noqa: E402

SMALL = 40


def load_existing(path):
    if path and os.path.exists(path):
        try:
            d = json.load(open(path))
            if isinstance(d, dict) and "units" in d:
                return d
        except ValueError:
            pass
    return {"units": {}, "seeds": None, "epochs": None}


def summarise(out):
    rows = []
    for name, r in out["units"].items():
        rows.append((name, r["class"], r["n"], r["held_out"],
                     r["half_included"], r["recovery"]))
    if not rows:
        return
    rows.sort(key=lambda r: -r[5])
    print("\n" + "=" * 78)
    print("RECOVERY ACROSS ALL %d UNITS, largest first" % len(rows))
    print("=" * 78)
    print("  %-18s %-7s %6s %9s %9s %9s" %
          ("unit", "class", "n", "held out", "half in", "recovery"))
    for name, cls, n, a, b, d in rows:
        flag = "  small" if n < SMALL else ""
        print("  %-18s %-7s %6d %9.4f %9.4f %+9.4f%s"
              % (name, cls, n, a, b, d, flag))

    rec = np.array([r[5] for r in rows], float)
    base = np.array([r[3] for r in rows], float)
    print("\n  recovery: median %+.4f, mean %+.4f, sd %.4f, max %+.4f"
          % (np.median(rec), rec.mean(), rec.std(ddof=1), rec.max()))

    tgt = [r for r in rows if r[0].endswith("13-48")]
    if tgt:
        d = tgt[0][5]
        above = int((rec > d).sum())
        print("  13-48 recovery %+.4f ranks %d of %d (%d units above it)"
              % (d, above + 1, len(rows), above))
        others = rec[[i for i, r in enumerate(rows) if not r[0].endswith("13-48")]]
        if len(others) > 1:
            print("  without 13-48: median %+.4f, max %+.4f, sd %.4f"
                  % (np.median(others), others.max(), others.std(ddof=1)))
            z = (d - others.mean()) / (others.std(ddof=1) + 1e-12)
            print("  13-48 sits %.1f standard deviations above the rest" % z)

    if len(rows) > 3:
        # does starting low predict recovering? the regression-to-the-mean check
        bc = base - base.mean(); rc = rec - rec.mean()
        denom = np.sqrt((bc ** 2).sum() * (rc ** 2).sum())
        r_all = float((bc * rc).sum() / denom) if denom > 0 else float("nan")
        keep = [i for i, r in enumerate(rows) if not r[0].endswith("13-48")]
        b2, r2 = base[keep], rec[keep]
        bc2 = b2 - b2.mean(); rc2 = r2 - r2.mean()
        d2 = np.sqrt((bc2 ** 2).sum() * (rc2 ** 2).sum())
        r_excl = float((bc2 * rc2).sum() / d2) if d2 > 0 else float("nan")
        out["baseline_recovery_pearson"] = r_all
        out["baseline_recovery_pearson_excl_13_48"] = r_excl
        print("\n  Pearson(baseline recall, recovery) = %+.3f over all units,"
              % r_all)
        print("  %+.3f with 13-48 excluded. A strongly negative value with" % r_excl)
        print("  13-48 removed would mean low baseline recall predicts recovery")
        print("  in general, and that 13-48 is an extreme of a trend rather")
        print("  than a separate case. A weak value means it is separate.")

    out["summary"] = {
        "n_units": len(rows),
        "recovery_median": float(np.median(rec)),
        "recovery_mean": float(rec.mean()),
        "recovery_sd": float(rec.std(ddof=1)),
        "recovery_max": float(rec.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only-class", default=None,
                    help="restrict to one class, for a short trial run")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    units = sorted(set(unit.tolist()))
    if args.only_class:
        units = [u for u in units if u.startswith(args.only_class + "/")]

    out = load_existing(args.out)
    out["seeds"] = list(args.seeds)
    out["epochs"] = args.epochs
    done = set(out["units"])
    todo = [u for u in units if u not in done]

    print("=" * 78)
    print("COVERAGE ON EVERY UNIT: %d total, %d already done, %d to run"
          % (len(units), len(done), len(todo)))
    print("%d trainings remain at %d seeds" % (2 * len(todo) * len(args.seeds),
                                               len(args.seeds)))
    print("Output is written after each unit; rerun this command to resume.")
    print("=" * 78)

    for i, u in enumerate(todo, 1):
        n = int((unit == u).sum())
        print("\n[%d/%d] %s  (n=%d)" % (i, len(todo), u, n))
        cov = coverage_test(X, y, unit, u, args.seeds, args.epochs)
        out["units"][u] = {
            "class": u.split("/")[0], "n": n,
            "held_out": cov["held_out"],
            "half_included": cov["half_included"],
            "recovery": cov["recovery"],
            "runs": cov.get("runs", {})}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("  saved %d of %d units" % (len(out["units"]), len(units)))

    summarise(out)
    json.dump(out, open(args.out, "w"), indent=2)
    print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
