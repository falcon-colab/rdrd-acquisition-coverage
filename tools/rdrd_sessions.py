"""
RDRD Step 1c: session-aware audit and grouped split builder.

Step 1b revealed that RDRD's acquisition structure is in the DIRECTORY
hierarchy, not the filenames:

    data/<Class>/<session>/<n>.csv

where <session> looks like an acquisition timestamp. Each folder is one
recording. That is the grouping unit both reviewers asked for.

This script:

  A  Enumerates every session folder, per class, with file counts and
     numbering gaps (gaps indicate dropped no-detection frames, which is
     evidence of a continuous recording).

  B  THE LEAKAGE TEST. Compares within-session similarity against
     cross-session similarity. If within >> cross, a random split places
     near-identical frames on both sides and every published accuracy on
     this dataset is inflated. This is the measurement the paper needs.

  C  Per-session signal statistics, so we can see whether the drone
     headroom deficit is a property of the class or of particular
     recordings.

  D  Builds and saves a session-grouped split, plus a matched random split
     for the comparison, as split_grouped.json / split_random.json.

Usage:
    python rdrd_sessions.py /path/to/rdrd
    python rdrd_sessions.py /path/to/rdrd --per-session 200 --test-frac 0.3
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdrd_inventory import load_matrix, guess_class_from_path

NUM_RE = re.compile(r"(\d+)")
DATA_EXT = {".csv", ".txt", ".dat"}


# ------------------------------------------------------------------ A

def find_sessions(root):
    """
    Return {class_label: {session_name: [abs file paths]}}.

    A session is the deepest directory that directly contains data files.
    The class is taken from any ancestor directory name that maps to a class.
    """
    sessions = defaultdict(dict)
    for dirpath, _dirs, files in os.walk(root):
        data = sorted(f for f in files
                      if os.path.splitext(f)[1].lower() in DATA_EXT)
        if not data:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)

        label = None
        for p in parts:                      # nearest ancestor wins
            g = guess_class_from_path(p)
            if g:
                label = g
        if label is None:
            label = guess_class_from_path(rel)
        if label is None:
            print("  skipping unmapped: %s" % rel, file=sys.stderr)
            continue

        name = parts[-1] if len(parts) > 1 else rel
        # disambiguate identical session names under different parents
        key = name if name not in sessions[label] else rel.replace(os.sep, "|")
        sessions[label][key] = [os.path.join(dirpath, f) for f in data]
    return dict(sessions)


def numbering_report(paths):
    nums = []
    for p in paths:
        f = NUM_RE.findall(os.path.basename(p))
        if f:
            nums.append(int(f[-1]))
    if not nums:
        return {"n": len(paths), "contiguous": None}
    a = np.array(sorted(nums))
    span = int(a.max() - a.min() + 1)
    return {"n": len(paths), "min": int(a.min()), "max": int(a.max()),
            "span": span, "missing": int(span - a.size),
            "contiguous": bool(span == a.size)}


# ------------------------------------------------------------------ helpers

def load_session(paths, limit, mask_dB=15.0):
    take = paths[:limit] if limit else paths
    mats = []
    for p in take:
        try:
            mats.append(load_matrix(p))
        except ValueError:
            pass
    if not mats:
        return None, None
    shapes = {}
    for m in mats:
        shapes[m.shape] = shapes.get(m.shape, 0) + 1
    dom = max(shapes, key=shapes.get)
    stack = np.stack([m for m in mats if m.shape == dom])

    flat = stack.reshape(len(stack), -1).astype(np.float32).copy()
    peak = flat.max(axis=1, keepdims=True)
    keep = flat >= peak - mask_dB
    flat = np.where(keep, flat, 0.0)
    cnt = keep.sum(axis=1, keepdims=True).astype(np.float32)
    cnt[cnt == 0] = 1.0
    mu = flat.sum(axis=1, keepdims=True) / cnt
    flat = np.where(keep, flat - mu, 0.0)
    nrm = np.linalg.norm(flat, axis=1)
    nrm[nrm == 0] = 1.0
    return stack, flat / nrm[:, None]


def signal_stats(stack):
    x = stack.reshape(len(stack), -1).astype(np.float64)
    lin = 10.0 ** ((x - x.max(axis=1, keepdims=True)) / 10.0)
    floor = np.median(lin, axis=1)
    ptf = -10 * np.log10(np.maximum(floor, 1e-30))
    within10 = (lin > 10 ** -1.0).sum(axis=1)
    return {"peak_to_floor_median": float(np.median(ptf)),
            "peak_to_floor_p10": float(np.percentile(ptf, 10)),
            "cells_within_10dB_median": float(np.median(within10))}


# ------------------------------------------------------------------ B

def leakage_test(per_session_flat, rng, n_pairs=4000):
    """
    Within-session similarity versus cross-session similarity.

    A large gap means a random split leaks: frames of one recording end up
    on both sides, and the model can score by recognising the recording
    rather than the target.
    """
    names = [k for k, v in per_session_flat.items() if v is not None and len(v) > 1]
    if len(names) < 2:
        return None

    within = []
    for k in names:
        f = per_session_flat[k]
        a = rng.integers(0, len(f), min(n_pairs // len(names) + 1, 2000))
        b = rng.integers(0, len(f), len(a))
        m = a != b
        if m.any():
            within.append(np.einsum("ij,ij->i", f[a[m]], f[b[m]]))
    within = np.concatenate(within)

    cross = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(names), 2, replace=False)
        fi, fj = per_session_flat[names[i]], per_session_flat[names[j]]
        cross.append(float(fi[rng.integers(len(fi))] @ fj[rng.integers(len(fj))]))
    cross = np.array(cross)

    return {"within_mean": float(within.mean()),
            "within_median": float(np.median(within)),
            "cross_mean": float(cross.mean()),
            "cross_median": float(np.median(cross)),
            "gap": float(within.mean() - cross.mean()),
            "frac_within_above_cross_p95": float(
                (within > np.percentile(cross, 95)).mean())}


# ------------------------------------------------------------------ D

def build_splits(sessions, rng, test_frac):
    """Session-grouped split, plus a matched random split for comparison."""
    grouped = {"train": [], "test": [], "unit": "session"}
    for label, sess in sorted(sessions.items()):
        names = sorted(sess)
        rng.shuffle(names)
        n_test = max(1, int(round(test_frac * len(names))))
        for k in names[:n_test]:
            grouped["test"] += sess[k]
        for k in names[n_test:]:
            grouped["train"] += sess[k]

    allf = []
    for label, sess in sorted(sessions.items()):
        for k in sorted(sess):
            allf += sess[k]
    idx = rng.permutation(len(allf))
    cut = int(round(test_frac * len(allf)))
    random_split = {"train": [allf[i] for i in idx[cut:]],
                    "test": [allf[i] for i in idx[:cut]],
                    "unit": "frame"}
    return grouped, random_split


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-session", type=int, default=150,
                    help="samples loaded per session for the audit")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print("RDRD SESSION AUDIT")
    print("=" * 74)

    sessions = find_sessions(args.root)
    if not sessions:
        sys.exit("no sessions found — check the root path")

    report = {}

    # ---- A
    print("\n[A] Session inventory")
    for label in sorted(sessions):
        sess = sessions[label]
        total = sum(len(v) for v in sess.values())
        print("\n  %s: %d sessions, %d files" % (label.upper(), len(sess), total))
        rows = []
        for k in sorted(sess):
            nr = numbering_report(sess[k])
            rows.append((k, nr))
            flag = "" if nr.get("contiguous") else "  <- %d missing" % nr.get("missing", 0)
            print("    %-16s %5d files   numbered %s..%s%s"
                  % (k[:16], nr["n"], nr.get("min", "?"), nr.get("max", "?"), flag))
        report[label] = {"n_sessions": len(sess), "n_files": total,
                         "sessions": {k: v for k, v in rows}}

    if all(report[l]["n_sessions"] < 2 for l in report):
        print("\n  WARNING: fewer than two sessions per class. Grouped splitting")
        print("  is not possible. Stop and re-check the directory layout.")

    # ---- load once, reuse
    print("\n[C] Per-session signal statistics")
    per_class_flat = {}
    for label in sorted(sessions):
        print("\n  %s" % label.upper())
        print("    %-16s %10s %10s %10s"
              % ("session", "n", "ptf median", "ptf p10"))
        flats = {}
        for k in sorted(sessions[label]):
            stack, flat = load_session(sessions[label][k], args.per_session)
            if stack is None:
                continue
            st = signal_stats(stack)
            flats[k] = flat
            report[label]["sessions"][k].update(st)
            print("    %-16s %10d %9.1f dB %9.1f dB"
                  % (k[:16], len(stack), st["peak_to_floor_median"],
                     st["peak_to_floor_p10"]))
        per_class_flat[label] = flats
        vals = [report[label]["sessions"][k]["peak_to_floor_median"]
                for k in flats]
        if vals:
            print("    class median across sessions: %.1f dB  (spread %.1f dB)"
                  % (float(np.median(vals)), float(np.max(vals) - np.min(vals))))
            report[label]["ptf_across_sessions"] = {
                "median": float(np.median(vals)),
                "spread": float(np.max(vals) - np.min(vals))}

    # ---- B
    print("\n" + "=" * 74)
    print("[B] LEAKAGE TEST  (the headline measurement)")
    print("=" * 74)
    print("\n  %-8s %12s %12s %10s %14s"
          % ("class", "within", "cross", "gap", "above cross p95"))
    any_leak = False
    for label in sorted(per_class_flat):
        lt = leakage_test(per_class_flat[label], rng)
        if lt is None:
            print("  %-8s  (needs at least two sessions)" % label)
            continue
        report[label]["leakage"] = lt
        print("  %-8s %12.3f %12.3f %+10.3f %13.2f"
              % (label, lt["within_mean"], lt["cross_mean"], lt["gap"],
                 lt["frac_within_above_cross_p95"]))
        if lt["gap"] > 0.10:
            any_leak = True

    print("")
    if any_leak:
        print("  Within-session pairs are markedly more similar than")
        print("  cross-session pairs. A random split therefore leaks, and the")
        print("  published accuracies on this dataset are optimistic by an")
        print("  amount the grouped-split experiment will quantify.")
    else:
        print("  No strong within-session advantage. Random splitting may be")
        print("  closer to honest than expected. Report this either way; a")
        print("  null leakage result is still a result worth publishing.")

    # ---- D
    grouped, randomd = build_splits(sessions, rng, args.test_frac)
    os.makedirs(args.outdir, exist_ok=True)
    for name, sp in [("split_grouped.json", grouped),
                     ("split_random.json", randomd)]:
        with open(os.path.join(args.outdir, name), "w") as fh:
            json.dump({"train": sp["train"], "test": sp["test"],
                       "unit": sp["unit"], "seed": args.seed,
                       "test_frac": args.test_frac}, fh)
    print("\n[D] Splits written")
    print("    grouped: %d train / %d test  (whole sessions held out)"
          % (len(grouped["train"]), len(grouped["test"])))
    print("    random : %d train / %d test  (frames shuffled)"
          % (len(randomd["train"]), len(randomd["test"])))
    print("    -> split_grouped.json, split_random.json in %s"
          % os.path.abspath(args.outdir))

    with open(os.path.join(args.outdir, "rdrd_session_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("    -> rdrd_session_report.json")


if __name__ == "__main__":
    main()
