"""Per-acquisition Doppler centroid: mean and spread, by acquisition.

Section IV of the manuscript states that every drone acquisition other than
13-48 places its target energy at a Doppler centroid near 30, with a
within-acquisition standard deviation between 0.4 and 6.0 bins, and that
13-48 places it elsewhere with a much larger spread. Those figures were not
recorded by any stage, because step2d's profile stores only the mean of each
property over the acquisition and the z score against the pooled others. A
reader could not check them.

This script computes them and writes them to a report, so they can be. It
trains nothing and needs no GPU: it reads the cached array and reduces it.

    python src/centroids.py --data DIR/dataset.npz \\
        --out DIR/reports/step2d_centroids.json

Doppler centroid is the same quantity step2d uses, and it is imported from
there rather than reimplemented, so the two cannot drift apart.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2d import sample_features                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="drone/13-48")
    ap.add_argument("--cls", default="drone")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]

    cent = sample_features(X)["doppler_centroid"]
    mask = y == args.cls

    rows = []
    for u in sorted(set(unit[mask].tolist())):
        v = cent[unit == u]
        rows.append({"acquisition": u, "n": int(v.size),
                     "centroid_mean": float(v.mean()),
                     "centroid_sd": float(v.std(ddof=1)) if v.size > 1
                     else 0.0})

    tgt = [r for r in rows if r["acquisition"] == args.target]
    others = [r for r in rows if r["acquisition"] != args.target]
    if not tgt:
        sys.exit("acquisition %s not present; found %s"
                 % (args.target, ", ".join(r["acquisition"] for r in rows)))
    tgt = tgt[0]

    print("=" * 66)
    print("DOPPLER CENTROID BY ACQUISITION  (class %s, %d acquisitions)"
          % (args.cls, len(rows)))
    print("=" * 66)
    print("  %-22s %6s %12s %10s" % ("acquisition", "n", "mean", "sd"))
    for r in sorted(rows, key=lambda r: r["centroid_mean"]):
        mark = "  <== target" if r["acquisition"] == args.target else ""
        print("  %-22s %6d %12.2f %10.2f%s"
              % (r["acquisition"], r["n"], r["centroid_mean"],
                 r["centroid_sd"], mark))

    o_mean = [r["centroid_mean"] for r in others]
    o_sd = [r["centroid_sd"] for r in others]
    rep = {
        "target": args.target,
        "target_centroid_mean": tgt["centroid_mean"],
        "target_centroid_sd": tgt["centroid_sd"],
        "others_centroid_mean_min": min(o_mean),
        "others_centroid_mean_max": max(o_mean),
        "others_centroid_sd_min": min(o_sd),
        "others_centroid_sd_max": max(o_sd),
        "n_acquisitions": len(rows),
        "by_acquisition": rows,
    }

    print("\n  %s        mean %.2f   sd %.2f"
          % (args.target, tgt["centroid_mean"], tgt["centroid_sd"]))
    print("  every other %-5s   mean %.2f to %.2f   sd %.2f to %.2f"
          % (args.cls, min(o_mean), max(o_mean), min(o_sd), max(o_sd)))

    if tgt["centroid_sd"] > max(o_sd):
        print("\n  The target's spread exceeds every other acquisition's.")
    else:
        print("\n  The target's spread does NOT exceed every other "
              "acquisition's.\n  The manuscript's sentence about spread "
              "needs revisiting.")

    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
