"""
RDRD Step 1b: deeper structural probe.

Run this after rdrd_inventory.py returns an unclear grouping verdict.

It does four things the first script did not:

  A  Full filename anatomy. Every numeric field is extracted separately and
     checked for whether it looks like a session index, a frame index, or
     just a running counter. If the publishers encoded acquisition structure
     anywhere, it is here.

  B  Adjacency under ALTERNATIVE orderings. The first script sorted by the
     trailing number only. If a different numeric field is the true frame
     counter, sorting by it will make adjacency correlation jump.

  C  Near-duplicate graph. Independent of any filename order, find pairs of
     samples that are near-identical and take connected components. This
     recovers groups from content alone.

  D  Corrected concentration. Measures spread of the excess over the noise
     floor, and reports peak-to-floor ratio separately, so low SNR is no
     longer mistaken for low concentration.

Usage:
    python rdrd_probe.py /path/to/rdrd
    python rdrd_probe.py /path/to/rdrd --max-per-class 1500 --dup 0.97
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdrd_inventory import (walk_dataset, guess_class_from_path, load_matrix,
                            filename_numbers)

NUM_RE = re.compile(r"(\d+)")


# ------------------------------------------------------------------ A

def filename_anatomy(filenames):
    """Split every name into its numeric fields and characterise each field."""
    parsed = [filename_numbers(f) for f in filenames]
    widths = Counter(len(p) for p in parsed)
    n_fields = widths.most_common(1)[0][0]
    keep = [p for p in parsed if len(p) == n_fields]

    fields = []
    for k in range(n_fields):
        vals = np.array([p[k] for p in keep])
        uniq = np.unique(vals)
        contiguous = (uniq.size > 1 and
                      uniq.max() - uniq.min() + 1 == uniq.size)
        fields.append({
            "index": k,
            "n_distinct": int(uniq.size),
            "min": int(vals.min()),
            "max": int(vals.max()),
            "contiguous": bool(contiguous),
            "counts_per_value_median": float(np.median(
                np.bincount(np.searchsorted(uniq, vals)))),
        })

    # non-numeric skeleton, e.g. "Drone_####.csv" -> "Drone_#.csv"
    skeletons = Counter(NUM_RE.sub("#", f) for f in filenames)
    return {"field_count_distribution": dict(widths),
            "fields": fields,
            "skeletons": dict(skeletons.most_common(5))}


def interpret_fields(anat):
    """Human-readable guess at what each numeric field is."""
    notes = []
    for f in anat["fields"]:
        n, lo, hi = f["n_distinct"], f["min"], f["max"]
        if n <= 1:
            notes.append("field %d: constant (%d), carries nothing" % (f["index"], lo))
        elif n <= 30 and f["counts_per_value_median"] > 20:
            notes.append("field %d: %d distinct values, many samples each -> "
                         "CANDIDATE SESSION / CAMPAIGN INDEX" % (f["index"], n))
        elif f["contiguous"] and f["counts_per_value_median"] <= 1.5:
            notes.append("field %d: contiguous 1-per-value counter (%d..%d) -> "
                         "running sample id, not a group" % (f["index"], lo, hi))
        else:
            notes.append("field %d: %d distinct values in [%d, %d]"
                         % (f["index"], n, lo, hi))
    return notes


# ------------------------------------------------------------------ B

def normalise(stack, mask_dB=None):
    """
    Unit-norm, mean-removed feature vectors.

    mask_dB: if given, cells more than this many dB below the sample peak are
    zeroed before normalising. Without it, a low-SNR class produces vectors
    dominated by noise, and no similarity threshold can separate same-pass
    from different-pass pairs.
    """
    flat = stack.reshape(len(stack), -1).astype(np.float32).copy()
    if mask_dB is not None:
        peak = flat.max(axis=1, keepdims=True)
        flat[flat < peak - mask_dB] = np.nan
        col_ok = ~np.isnan(flat)
        flat = np.where(col_ok, flat, 0.0)
        # re-centre using only the retained cells
        cnt = col_ok.sum(axis=1, keepdims=True).astype(np.float32)
        cnt[cnt == 0] = 1.0
        mu = flat.sum(axis=1, keepdims=True) / cnt
        flat = np.where(col_ok, flat - mu, 0.0)
    else:
        flat -= flat.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(flat, axis=1)
    nrm[nrm == 0] = 1.0
    return flat / nrm[:, None]


def adjacency_stats(flat, order):
    f = flat[order]
    adj = np.einsum("ij,ij->i", f[:-1], f[1:])
    rng = np.random.default_rng(0)
    n = min(5000, len(f) * 3)
    a, b = rng.integers(0, len(f), n), rng.integers(0, len(f), n)
    m = a != b
    rnd = np.einsum("ij,ij->i", flat[a[m]], flat[b[m]])
    return {"adjacent_mean": float(adj.mean()),
            "random_mean": float(rnd.mean()),
            "gap": float(adj.mean() - rnd.mean())}


def try_orderings(filenames, flat):
    """Score every plausible ordering by how much adjacency it produces."""
    parsed = [filename_numbers(f) for f in filenames]
    widths = Counter(len(p) for p in parsed)
    n_fields = widths.most_common(1)[0][0]
    idx_ok = [i for i, p in enumerate(parsed) if len(p) == n_fields]

    results = {}
    results["as_listed"] = adjacency_stats(flat, np.arange(len(flat)))
    results["lexicographic"] = adjacency_stats(
        flat, np.array(sorted(range(len(filenames)), key=lambda i: filenames[i])))

    for k in range(n_fields):
        key = np.array([parsed[i][k] for i in idx_ok])
        order = np.array(idx_ok)[np.argsort(key, kind="stable")]
        results["by_numeric_field_%d" % k] = adjacency_stats(flat, order)

    if n_fields >= 2:
        for k in range(n_fields):
            others = [j for j in range(n_fields) if j != k]
            keys = tuple(np.array([parsed[i][j] for i in idx_ok])
                         for j in [k] + others)
            order = np.array(idx_ok)[np.lexsort(keys[::-1])]
            results["grouped_by_field_%d" % k] = adjacency_stats(flat, order)
    return results


# ------------------------------------------------------------------ C

def duplicate_components(flat, thresh, chunk=1024):
    """
    Union-find over pairs whose correlation exceeds `thresh`.
    Components are near-duplicate clusters, usable as split groups.
    """
    n = len(flat)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_edges = 0
    for s in range(0, n, chunk):
        block = flat[s:s + chunk]
        sim = block @ flat.T                       # (chunk, n)
        rows, cols = np.where(sim > thresh)
        for r, c in zip(rows, cols):
            gi = s + r
            if gi < c:
                union(gi, c)
                n_edges += 1

    roots = np.array([find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    sizes = np.bincount(comp)
    return {
        "threshold": thresh,
        "n_edges": int(n_edges),
        "n_components": int(sizes.size),
        "largest": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "singletons": int((sizes == 1).sum()),
        "frac_in_multi": float((sizes[comp] > 1).mean()),
    }, comp


# ------------------------------------------------------------------ D

def corrected_concentration(stack, values_are_dB=True):
    """
    Concentration of the EXCESS over the per-sample noise floor, plus a
    separate peak-to-floor figure.

    The naive version measured total power, which for a weak target is
    dominated by noise cells. That reports low SNR as low concentration.
    """
    x = stack.reshape(len(stack), -1).astype(np.float64)
    if values_are_dB:
        lin = 10.0 ** ((x - x.max(axis=1, keepdims=True)) / 10.0)
    else:
        lin = np.maximum(x, 0)

    # robust per-sample floor: median cell is noise for a sparse target
    floor = np.median(lin, axis=1, keepdims=True)
    excess = np.maximum(lin - floor, 0.0)

    tot = excess.sum(axis=1, keepdims=True)
    ok = (tot[:, 0] > 0)
    e = excess[ok] / tot[ok]
    srt = np.sort(e, axis=1)[:, ::-1]
    cum = np.cumsum(srt, axis=1)
    n90 = (cum < 0.90).sum(axis=1) + 1

    peak = lin.max(axis=1)
    ptf = 10 * np.log10(np.maximum(peak, 1e-30) /
                        np.maximum(floor[:, 0], 1e-30))

    # peak-relative extent: how many cells sit within 10 dB of the peak.
    # This is the SNR-robust measure and is the one to trust.
    within10 = (lin > 10.0 ** (-10.0 / 10.0)).sum(axis=1)   # lin is peak-normalised
    within20 = (lin > 10.0 ** (-20.0 / 10.0)).sum(axis=1)

    n_cells = x.shape[1]
    return {
        "n_cells": int(n_cells),
        "usable_samples": int(ok.sum()),
        "cells_within_10dB_of_peak_median": float(np.median(within10)),
        "cells_within_20dB_of_peak_median": float(np.median(within20)),
        "excess_cells_for_90pct_median": float(np.median(n90)),
        "excess_cells_for_90pct_p25": float(np.percentile(n90, 25)),
        "excess_cells_for_90pct_p75": float(np.percentile(n90, 75)),
        "peak_to_floor_dB_median": float(np.median(ptf)),
        "peak_to_floor_dB_p10": float(np.percentile(ptf, 10)),
        "frac_cells_above_floor_median": float(
            np.median((lin > floor * 2).mean(axis=1))),
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--mask-dB", type=float, default=15.0,
                    help="mask cells this many dB below the sample peak "
                         "before computing similarity")
    ap.add_argument("--dup-sweep", type=float, nargs="+",
                    default=[0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.87,
                             0.84, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                             0.45, 0.40, 0.35, 0.30],
                    help="similarity thresholds to sweep for grouping")
    args = ap.parse_args()

    layout = walk_dataset(args.root)
    by_class = defaultdict(list)
    for rel, files in layout.items():
        lab = guess_class_from_path(rel)
        if lab:
            by_class[lab].append((rel, files))

    report = {}
    print("=" * 74)
    print("RDRD STRUCTURAL PROBE")
    print("=" * 74)

    for label, entries in sorted(by_class.items()):
        rel, files = max(entries, key=lambda e: len(e[1]))
        print("\n" + "=" * 74)
        print("%s   dir=%s   %d files" % (label.upper(), rel, len(files)))
        print("=" * 74)
        rep = {}

        # ---- A
        anat = filename_anatomy(files)
        rep["anatomy"] = anat
        print("\n[A] Filename anatomy")
        for sk, n in anat["skeletons"].items():
            print("    pattern %-32s %d files" % (sk, n))
        for note in interpret_fields(anat):
            print("    " + note)

        # ---- load
        take = sorted(files)[:args.max_per_class]
        mats, names = [], []
        for f in take:
            try:
                mats.append(load_matrix(os.path.join(args.root, rel, f)))
                names.append(f)
            except ValueError:
                pass
        shapes = Counter(m.shape for m in mats)
        dom = shapes.most_common(1)[0][0]
        pair = [(m, n) for m, n in zip(mats, names) if m.shape == dom]
        stack = np.stack([m for m, _ in pair])
        names = [n for _, n in pair]
        flat = normalise(stack)
        flat_masked = normalise(stack, mask_dB=args.mask_dB)
        print("\n    loaded %d samples of shape %s" % (len(stack), dom))

        # ---- B
        print("\n[B] Adjacency under alternative orderings")
        orders = try_orderings(names, flat)
        rep["orderings"] = orders
        best = max(orders, key=lambda k: orders[k]["gap"])
        for k in sorted(orders, key=lambda k: -orders[k]["gap"]):
            v = orders[k]
            mark = "  <== best" if k == best else ""
            print("    %-26s adj %.3f  rand %.3f  gap %+.3f%s"
                  % (k, v["adjacent_mean"], v["random_mean"], v["gap"], mark))
        if orders[best]["gap"] < 0.15:
            print("    No ordering recovers acquisition structure.")
        else:
            print("    '%s' looks like acquisition order." % best)

        # ---- C
        print("\n[C] Similarity components, threshold sweep")
        print("    (cells more than %.0f dB below peak are masked out first)"
              % args.mask_dB)
        print("    %6s %9s %9s %9s %9s" %
              ("thresh", "comps", "largest", "median", "clustered"))
        sweep, scored = {}, []
        n = len(flat)
        for t in args.dup_sweep:
            cs, cm = duplicate_components(flat_masked, t)
            sweep["%.2f" % t] = cs
            # good grouping: most samples in multi-sample groups, but no
            # single group swallowing the class
            score = cs["frac_in_multi"] * (1.0 - cs["largest"] / n)
            cs["score"] = float(score)
            scored.append((score, t, cs))
            print("    %6.2f %9d %9d %9.0f %8.0f%%   score %.3f"
                  % (t, cs["n_components"], cs["largest"],
                     cs["median_size"], 100 * cs["frac_in_multi"], score))
        best_score, chosen, _ = max(scored, key=lambda z: z[0])
        if best_score < 0.15:
            chosen = None
        rep["duplicate_sweep"] = sweep
        rep["chosen_threshold"] = chosen
        comp_stats = sweep["%.2f" % (chosen if chosen else args.dup_sweep[-1])]
        rep["duplicates"] = comp_stats
        if chosen:
            print("    usable threshold: %.2f  ->  %d groups"
                  % (chosen, comp_stats["n_components"]))
        else:
            print("    No threshold yields usable groups: either everything")
            print("    collapses into one component or nothing clusters.")

        # ---- D
        print("\n[D] Corrected concentration")
        cc = corrected_concentration(stack)
        rep["concentration"] = cc
        print("    cells within 10 dB of peak: median %.0f   (PRIMARY measure)"
              % cc["cells_within_10dB_of_peak_median"])
        print("    cells within 20 dB of peak: median %.0f"
              % cc["cells_within_20dB_of_peak_median"])
        print("    cells for 90%% of excess power: median %.0f  (IQR %.0f to %.0f)"
              % (cc["excess_cells_for_90pct_median"],
                 cc["excess_cells_for_90pct_p25"], cc["excess_cells_for_90pct_p75"]))
        print("    peak-to-floor ratio: median %.1f dB  (p10 %.1f dB)"
              % (cc["peak_to_floor_dB_median"], cc["peak_to_floor_dB_p10"]))
        print("    fraction of cells more than 3 dB above floor: %.3f"
              % cc["frac_cells_above_floor_median"])

        report[label] = rep

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)

    print("\nGrouping")
    for label in sorted(report):
        o = report[label]["orderings"]
        b = max(o, key=lambda k: o[k]["gap"])
        d = report[label]["duplicates"]
        ch = report[label]["chosen_threshold"]
        print("  %-8s ordering gap %+.3f (%s) | groups %d at thresh %s"
              % (label, o[b]["gap"], b, d["n_components"],
                 ("%.2f" % ch) if ch else "none usable"))

    print("\nConcentration and signal level")
    print("  %-8s %12s %12s %14s"
          % ("class", "cells<10dB", "cells@90%", "peak-to-floor"))
    for label in sorted(report):
        c = report[label]["concentration"]
        print("  %-8s %12.0f %12.0f %11.1f dB"
              % (label, c["cells_within_10dB_of_peak_median"],
                 c["excess_cells_for_90pct_median"],
                 c["peak_to_floor_dB_median"]))

    order = sorted(report, key=lambda l:
                   report[l]["concentration"]["cells_within_10dB_of_peak_median"])
    print("\n  most concentrated first: " + "  <  ".join(order))
    snr_order = sorted(report, key=lambda l:
                       report[l]["concentration"]["peak_to_floor_dB_median"])
    print("  weakest signal first:    " + "  <  ".join(snr_order))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rdrd_probe_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n  written: %s" % out)


if __name__ == "__main__":
    main()
