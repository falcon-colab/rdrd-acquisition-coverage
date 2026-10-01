"""
RDRD Step 1d: deduplication, duplicate-session merging, clean leakage test.

Step 1c revealed two problems that invalidate its own output:

  1. The extracted dataset contains TWO copies of the whole tree (one under
     data/, one at the archive root). File counts were exactly double the
     published figures.

  2. Several session folders appear to be duplicates or reprocessings of
     each other (identical file counts and identical signal statistics to
     one decimal place, e.g. Cars/17-09 and Cars/17-09p).

Both corrupt the leakage test, because a session's twin lands in the
"cross-session" pool and inflates the baseline. Both also corrupt a
folder-grouped split, since twins live in different folders and can end up
on opposite sides.

This script:

  A  Detects and removes the duplicated tree by content hashing.
  B  Detects duplicate/near-duplicate SESSIONS and merges them into single
     grouping units.
  C  Re-runs the leakage test on the cleaned, merged data, using a
     rank-based representation so the comparison is not biased by class
     signal level.
  D  Rebuilds the splits over merged units and verifies them by content,
     not by path.

Usage:
    python rdrd_dedup.py /path/to/rdrd --outdir /path/to/splits
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdrd_inventory import load_matrix
from rdrd_sessions import find_sessions, numbering_report


# ------------------------------------------------------------------ A

def file_digest(path, nbytes=65536):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def session_digest(paths, n_probe=12):
    """Content fingerprint of a session: hashes of evenly spaced files."""
    if not paths:
        return None
    idx = np.linspace(0, len(paths) - 1, min(n_probe, len(paths))).astype(int)
    parts = [file_digest(paths[i]) for i in idx]
    return hashlib.sha1(("|".join(parts) + "|%d" % len(paths)).encode()).hexdigest()


def dedupe_tree(sessions):
    """
    Collapse sessions with identical content fingerprints, keeping the one
    with the shortest path (the canonical location).
    """
    kept, dropped = {}, []
    for label, sess in sessions.items():
        by_digest = defaultdict(list)
        for name, paths in sess.items():
            by_digest[session_digest(paths)].append((name, paths))
        keep = {}
        for dig, entries in by_digest.items():
            entries.sort(key=lambda e: (len(e[1][0]) if e[1] else 0, e[0]))
            name, paths = entries[0]
            keep[name] = paths
            for other_name, _ in entries[1:]:
                dropped.append((label, other_name, name))
        kept[label] = keep
    return kept, dropped


# ------------------------------------------------------------------ helpers

def load_session_features(paths, limit, top_k=12):
    """
    Rank-based feature vector: the top_k highest cells of each sample,
    as (index, relative level) encoded into a fixed-length sparse vector.

    Using a fixed CELL COUNT rather than a fixed dB mask keeps the
    representation comparable across classes. A dB mask retains fewer cells
    for a high-SNR class, which by itself changes correlation magnitudes and
    makes cross-class comparison meaningless.
    """
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

    flat = stack.reshape(len(stack), -1).astype(np.float32)
    n_cells = flat.shape[1]
    order = np.argsort(flat, axis=1)[:, ::-1][:, :top_k]
    feat = np.zeros((len(flat), n_cells), dtype=np.float32)
    rows = np.arange(len(flat))[:, None]
    vals = np.take_along_axis(flat, order, axis=1)
    vals = vals - vals[:, :1]                      # relative to the peak
    feat[rows, order] = vals
    feat -= feat.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(feat, axis=1)
    nrm[nrm == 0] = 1.0
    return stack, feat / nrm[:, None]


def signal_stats(stack):
    x = stack.reshape(len(stack), -1).astype(np.float64)
    lin = 10.0 ** ((x - x.max(axis=1, keepdims=True)) / 10.0)
    floor = np.median(lin, axis=1)
    ptf = -10 * np.log10(np.maximum(floor, 1e-30))
    return {"ptf_median": float(np.median(ptf)),
            "ptf_p10": float(np.percentile(ptf, 10))}


# ------------------------------------------------------------------ B

def merge_duplicate_sessions(feats, stats, sim_thresh=0.90, ptf_tol=0.15):
    """
    Merge sessions whose mean feature vectors are near-identical AND whose
    signal statistics match. Two independent signals must agree, so an
    honest pair of similar recordings is not merged by accident.
    """
    names = [k for k in feats if feats[k] is not None]
    if len(names) < 2:
        return {k: k for k in names}, []

    cent = np.stack([feats[k].mean(axis=0) for k in names])
    nrm = np.linalg.norm(cent, axis=1)
    nrm[nrm == 0] = 1.0
    cent = cent / nrm[:, None]
    sim = cent @ cent.T

    parent = {k: k for k in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sim[i, j] < sim_thresh:
                continue
            a, b = names[i], names[j]
            pa, pb = stats[a]["ptf_median"], stats[b]["ptf_median"]
            if abs(pa - pb) > ptf_tol * max(1.0, abs(pa)):
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                merges.append((a, b, float(sim[i, j]), pa, pb))
    return {k: find(k) for k in names}, merges


# ------------------------------------------------------------------ C

def leakage_test(feats, unit_of, rng, n_pairs=6000):
    """Within-unit versus cross-unit similarity, over MERGED units."""
    by_unit = defaultdict(list)
    for name, f in feats.items():
        if f is not None and len(f) > 1:
            by_unit[unit_of.get(name, name)].append(f)
    units = {u: np.concatenate(v) for u, v in by_unit.items() if len(v) > 0}
    units = {u: v for u, v in units.items() if len(v) > 1}
    if len(units) < 2:
        return None

    keys = list(units)
    within = []
    per = max(50, n_pairs // len(keys))
    for u in keys:
        f = units[u]
        a = rng.integers(0, len(f), per)
        b = rng.integers(0, len(f), per)
        m = a != b
        if m.any():
            within.append(np.einsum("ij,ij->i", f[a[m]], f[b[m]]))
    within = np.concatenate(within)

    cross = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(keys), 2, replace=False)
        fi, fj = units[keys[i]], units[keys[j]]
        cross.append(float(fi[rng.integers(len(fi))] @ fj[rng.integers(len(fj))]))
    cross = np.array(cross)

    return {"n_units": len(keys),
            "within_mean": float(within.mean()),
            "cross_mean": float(cross.mean()),
            "gap": float(within.mean() - cross.mean()),
            "within_p50": float(np.median(within)),
            "cross_p95": float(np.percentile(cross, 95)),
            "frac_within_above_cross_p95": float(
                (within > np.percentile(cross, 95)).mean())}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-session", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=12,
                    help="cells kept per sample for the similarity features. "
                         "Roughly the number of cells within 10 dB of the "
                         "peak. Too large and a weak class's features fill "
                         "with noise, understating its leakage gap.")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--sim-thresh", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print("RDRD DEDUPLICATION AND CLEAN LEAKAGE TEST")
    print("=" * 74)

    raw = find_sessions(args.root)
    print("\n[A] Tree deduplication by content hash")
    before = {l: (len(s), sum(len(v) for v in s.values())) for l, s in raw.items()}
    sessions, dropped = dedupe_tree(raw)
    after = {l: (len(s), sum(len(v) for v in s.values())) for l, s in sessions.items()}
    print("    %-8s %22s %22s" % ("class", "before (sess/files)", "after (sess/files)"))
    for l in sorted(before):
        print("    %-8s %11d /%9d %11d /%9d"
              % (l, before[l][0], before[l][1], after[l][0], after[l][1]))
    print("    dropped %d duplicate session folders" % len(dropped))
    total_after = sum(v[1] for v in after.values())
    print("    total files after dedup: %d   (published figure is 17485)" % total_after)
    if abs(total_after - 17485) > 50:
        print("    WARNING: still not matching the published count. Inspect manually.")

    # ---- load
    print("\n    loading up to %d samples per session..." % args.per_session)
    feats, stats, report = {}, {}, {}
    for label in sorted(sessions):
        feats[label], stats[label] = {}, {}
        for name, paths in sessions[label].items():
            stack, f = load_session_features(paths, args.per_session,
                                             top_k=args.top_k)
            if stack is None:
                continue
            feats[label][name] = f
            stats[label][name] = signal_stats(stack)
        print("      %-8s %d sessions loaded" % (label, len(feats[label])))

    # ---- B
    print("\n[B] Duplicate SESSION detection (content + signal statistics)")
    unit_of = {}
    for label in sorted(sessions):
        u, merges = merge_duplicate_sessions(feats[label], stats[label],
                                             args.sim_thresh)
        unit_of[label] = u
        n_units = len(set(u.values()))
        print("\n    %s: %d sessions -> %d independent units"
              % (label.upper(), len(u), n_units))
        for a, b, s, pa, pb in merges[:12]:
            print("      merged %-14s + %-14s  sim %.3f  ptf %.1f / %.1f dB"
                  % (a[:14], b[:14], s, pa, pb))
        if len(merges) > 12:
            print("      ... and %d more" % (len(merges) - 12))
        report[label] = {"n_sessions": len(u), "n_units": n_units,
                         "n_merges": len(merges)}

    # ---- C
    print("\n" + "=" * 74)
    print("[C] CLEAN LEAKAGE TEST  (deduplicated, merged units)")
    print("=" * 74)
    print("\n  %-8s %7s %10s %10s %9s %14s"
          % ("class", "units", "within", "cross", "gap", "above cross p95"))
    for label in sorted(sessions):
        lt = leakage_test(feats[label], unit_of[label], rng)
        if lt is None:
            print("  %-8s  insufficient units" % label)
            continue
        report[label]["leakage"] = lt
        print("  %-8s %7d %10.3f %10.3f %+9.3f %13.2f"
              % (label, lt["n_units"], lt["within_mean"], lt["cross_mean"],
                 lt["gap"], lt["frac_within_above_cross_p95"]))

    gaps = [report[l]["leakage"]["gap"] for l in report if "leakage" in report[l]]
    print("")
    if gaps and max(gaps) > 0.10:
        print("  Within-unit pairs are clearly more similar than cross-unit")
        print("  pairs. A random split leaks. The grouped-split experiment")
        print("  will quantify by how much.")
    elif gaps:
        print("  Gap is small even after cleaning. If this holds, random")
        print("  splitting on RDRD is closer to honest than expected, and")
        print("  that is itself the reportable result.")

    # ---- D
    print("\n[D] Rebuilding splits over merged units")
    grouped = {"train": [], "test": []}
    for label in sorted(sessions):
        units = defaultdict(list)
        for name, paths in sessions[label].items():
            units[unit_of[label].get(name, name)] += paths
        keys = sorted(units)
        rng.shuffle(keys)
        n_test = max(1, int(round(args.test_frac * len(keys))))
        for k in keys[:n_test]:
            grouped["test"] += units[k]
        for k in keys[n_test:]:
            grouped["train"] += units[k]

    allf = grouped["train"] + grouped["test"]
    idx = rng.permutation(len(allf))
    cut = int(round(args.test_frac * len(allf)))
    randomd = {"train": [allf[i] for i in idx[cut:]],
               "test": [allf[i] for i in idx[:cut]]}

    # verify by CONTENT, not path
    print("    verifying by content hash...")
    probe = rng.choice(len(grouped["test"]), min(600, len(grouped["test"])),
                       replace=False)
    test_hashes = {file_digest(grouped["test"][i]) for i in probe}
    probe2 = rng.choice(len(grouped["train"]), min(4000, len(grouped["train"])),
                        replace=False)
    collisions = sum(1 for i in probe2
                     if file_digest(grouped["train"][i]) in test_hashes)
    print("    identical files found on both sides: %d (of %d probed)"
          % (collisions, len(probe2)))
    if collisions:
        print("    SPLIT IS CONTAMINATED. Do not proceed.")
    else:
        print("    grouped split is clean by content")

    os.makedirs(args.outdir, exist_ok=True)
    for nm, sp in [("split_grouped.json", grouped), ("split_random.json", randomd)]:
        with open(os.path.join(args.outdir, nm), "w") as fh:
            json.dump({"train": sp["train"], "test": sp["test"],
                       "seed": args.seed, "test_frac": args.test_frac,
                       "deduplicated": True}, fh)
    print("    grouped: %d train / %d test" % (len(grouped["train"]), len(grouped["test"])))
    print("    random : %d train / %d test" % (len(randomd["train"]), len(randomd["test"])))

    with open(os.path.join(args.outdir, "rdrd_dedup_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("    -> written to %s" % os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
