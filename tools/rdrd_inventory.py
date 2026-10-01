"""
RDRD Step 1: dataset inventory and session structure audit.

Run this before writing any model code. It answers four questions that
determine whether the study as proposed is viable:

  Q1  What is the actual on-disk layout, file naming, and matrix shape?
  Q2  How many samples per class, and are the classes balanced?
  Q3  Do consecutive files come from the same continuous recording pass?
      (If yes, grouped splitting is possible. If no, the whole statistical
      plan has to change.)
  Q4  Is there low-amplitude structure in the samples, or did the CFAR
      stage already threshold it away? (This is the threat to the
      redundancy mechanism.)

Usage:
    python rdrd_inventory.py /path/to/rdrd_root
    python rdrd_inventory.py /path/to/rdrd_root --max-per-class 400

Outputs a printed report plus rdrd_inventory_report.json next to the script.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

CSV_EXT = {".csv", ".txt", ".dat"}


# --------------------------------------------------------------- discovery

def walk_dataset(root):
    """Return {relative_dir: [filenames]} for every dir containing data files."""
    found = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(root):
        keep = [f for f in filenames if os.path.splitext(f)[1].lower() in CSV_EXT]
        if keep:
            rel = os.path.relpath(dirpath, root)
            found[rel] = sorted(keep)
    return dict(found)


def guess_class_from_path(rel_dir):
    """Map a directory name onto a canonical class label, or None."""
    low = rel_dir.lower()
    for key, label in [("drone", "drone"), ("uav", "drone"), ("dron", "drone"),
                       ("car", "car"), ("vehic", "car"), ("coche", "car"),
                       ("people", "person"), ("pedestr", "person"),
                       ("person", "person"), ("human", "person")]:
        if key in low:
            return label
    return None


NUM_RE = re.compile(r"(\d+)")


def filename_numbers(name):
    return [int(n) for n in NUM_RE.findall(name)]


def describe_naming(filenames, n_show=6):
    """Report the filename pattern and whether it carries a usable index."""
    sample = filenames[:n_show]
    counts = Counter(len(filename_numbers(f)) for f in filenames)
    return {
        "examples": sample,
        "n_files": len(filenames),
        "numeric_fields_per_name": dict(counts),
    }


# --------------------------------------------------------------- loading

def load_matrix(path):
    """Load one sample. Tries comma, then whitespace, then semicolon."""
    for kwargs in ({"delimiter": ","}, {}, {"delimiter": ";"}):
        try:
            m = np.loadtxt(path, **kwargs)
            if m.ndim == 2 and m.size > 1:
                return m
        except Exception:
            continue
    raise ValueError("could not parse %s" % path)


def load_class_samples(root, rel_dir, filenames, limit):
    """Load up to `limit` samples in filename order. Returns (stack, names)."""
    ordered = sort_by_index(filenames)
    take = ordered[:limit] if limit else ordered
    mats, names = [], []
    for f in take:
        try:
            mats.append(load_matrix(os.path.join(root, rel_dir, f)))
            names.append(f)
        except ValueError as e:
            print("  warning: %s" % e, file=sys.stderr)
    if not mats:
        return None, []
    shapes = Counter(m.shape for m in mats)
    if len(shapes) > 1:
        print("  warning: mixed shapes in %s -> %s" % (rel_dir, dict(shapes)),
              file=sys.stderr)
        dominant = shapes.most_common(1)[0][0]
        keep = [(m, n) for m, n in zip(mats, names) if m.shape == dominant]
        mats = [m for m, _ in keep]
        names = [n for _, n in keep]
    return np.stack(mats), names


def sort_by_index(filenames):
    """Sort by trailing numeric field so 'x_2' precedes 'x_10'."""
    def key(f):
        nums = filename_numbers(f)
        return (nums[-1] if nums else 0, f)
    return sorted(filenames, key=key)


# --------------------------------------------------------------- Q3 audit

def adjacency_correlation(stack):
    """
    Correlation between each sample and its filename-order neighbour, versus
    correlation between random pairs.

    A large gap means adjacent files are temporally adjacent frames of one
    recording pass. That makes grouped splitting both necessary and feasible.
    """
    flat = stack.reshape(len(stack), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1)
    norm[norm == 0] = 1.0
    flat = flat / norm[:, None]

    adj = np.einsum("ij,ij->i", flat[:-1], flat[1:])

    rng = np.random.default_rng(0)
    n_pairs = min(4000, len(flat) * 4)
    a = rng.integers(0, len(flat), n_pairs)
    b = rng.integers(0, len(flat), n_pairs)
    mask = a != b
    rand = np.einsum("ij,ij->i", flat[a[mask]], flat[b[mask]])

    return {
        "adjacent_mean": float(adj.mean()),
        "adjacent_median": float(np.median(adj)),
        "random_mean": float(rand.mean()),
        "random_median": float(np.median(rand)),
        "gap": float(adj.mean() - rand.mean()),
        "frac_adjacent_above_0p9": float((adj > 0.9).mean()),
    }


def adaptive_threshold(adj_stats):
    """
    A fixed correlation cut fails across classes with different signal levels:
    a weak class has lower adjacent correlation everywhere, so a global cut
    shatters it into runs of length one. Place the cut midway between the
    random-pair baseline and the adjacent-pair median instead.
    """
    lo = adj_stats["random_median"]
    hi = adj_stats["adjacent_median"]
    return lo + 0.5 * (hi - lo)


def segment_runs(stack, threshold):
    """
    Cut the filename-ordered sequence wherever adjacent correlation drops
    below `threshold`. Each resulting run is a candidate recording pass.
    """
    flat = stack.reshape(len(stack), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1)
    norm[norm == 0] = 1.0
    flat = flat / norm[:, None]
    adj = np.einsum("ij,ij->i", flat[:-1], flat[1:])

    runs, start = [], 0
    for i, c in enumerate(adj):
        if c < threshold:
            runs.append((start, i + 1))
            start = i + 1
    runs.append((start, len(stack)))
    lengths = [b - a for a, b in runs]
    return {
        "threshold": threshold,
        "n_runs": len(runs),
        "run_length_min": int(min(lengths)),
        "run_length_median": float(np.median(lengths)),
        "run_length_max": int(max(lengths)),
    }


# --------------------------------------------------------------- Q4 audit

def dynamic_range_report(stack):
    """
    Is there low-amplitude structure left after the CFAR stage?

    If the distribution is hard-floored (a large spike of identical values at
    the minimum), the detector already clipped the weak content, and the
    redundancy mechanism has less to work with than assumed.
    """
    x = stack.reshape(len(stack), -1).astype(np.float64)
    per_sample_min = x.min(axis=1)
    per_sample_max = x.max(axis=1)
    span = per_sample_max - per_sample_min

    allv = x.ravel()
    vmin = allv.min()
    at_floor = float(np.isclose(allv, vmin).mean())

    # unique value count is a direct signal of prior quantisation/clipping
    sub = allv if allv.size < 2_000_000 else allv[:2_000_000]
    n_unique = int(np.unique(np.round(sub, 6)).size)

    return {
        "global_min": float(vmin),
        "global_max": float(allv.max()),
        "value_mean": float(allv.mean()),
        "value_std": float(allv.std()),
        "per_sample_span_median_dB": float(np.median(span)),
        "per_sample_span_p10_dB": float(np.percentile(span, 10)),
        "frac_cells_exactly_at_global_min": at_floor,
        "approx_unique_values": n_unique,
        "looks_hard_floored": bool(at_floor > 0.05),
    }


def energy_concentration(stack, values_are_dB=True):
    """
    Fraction of total power held by the top cells. The redundancy mechanism
    predicts drones concentrate energy in fewer cells than vehicles.
    This is the first direct test of that prediction.

    IMPORTANT: the samples are stored in dB. Summing dB values is meaningless,
    so they are converted to linear power first. Skipping this step makes every
    class look identical, because the log floor dominates the sum.
    """
    x = stack.reshape(len(stack), -1).astype(np.float64)
    if values_are_dB:
        x = 10.0 ** ((x - x.max(axis=1, keepdims=True)) / 10.0)
    else:
        x = x - x.min(axis=1, keepdims=True)
    total = x.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    p = np.sort(x / total, axis=1)[:, ::-1]
    cum = np.cumsum(p, axis=1)
    n_cells = x.shape[1]
    k5 = max(1, int(round(0.05 * n_cells)))
    k10 = max(1, int(round(0.10 * n_cells)))
    # cells needed to reach 90 percent of power
    n90 = (cum < 0.90).sum(axis=1) + 1
    return {
        "n_cells": int(n_cells),
        "power_in_top_5pct_cells": float(np.median(cum[:, k5 - 1])),
        "power_in_top_10pct_cells": float(np.median(cum[:, k10 - 1])),
        "cells_to_reach_90pct_power_median": float(np.median(n90)),
    }


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to the extracted RDRD dataset")
    ap.add_argument("--max-per-class", type=int, default=600,
                    help="samples loaded per class for the audit (0 = all)")
    ap.add_argument("--run-threshold", type=float, default=0.0,
                    help="adjacent-correlation cut for runs; 0 = adaptive "
                         "per class (recommended)")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit("not a directory: %s" % args.root)

    print("=" * 72)
    print("RDRD INVENTORY AND SESSION AUDIT")
    print("=" * 72)

    layout = walk_dataset(args.root)
    if not layout:
        sys.exit("no .csv/.txt/.dat files found under %s" % args.root)

    report = {"root": os.path.abspath(args.root), "layout": {}, "classes": {}}

    print("\n--- Q1  Layout -------------------------------------------------")
    for rel, files in sorted(layout.items()):
        label = guess_class_from_path(rel)
        info = describe_naming(files)
        info["guessed_class"] = label
        report["layout"][rel] = info
        print("  %-34s %6d files   class=%s"
              % (rel[:34], len(files), label or "UNMAPPED"))
        print("       e.g. %s" % ", ".join(info["examples"][:3]))

    unmapped = [r for r in layout if guess_class_from_path(r) is None]
    if unmapped:
        print("\n  NOTE: %d directories did not map to a class. Inspect the"
              % len(unmapped))
        print("        names above and extend guess_class_from_path().")

    # group directories by class
    by_class = defaultdict(list)
    for rel, files in layout.items():
        label = guess_class_from_path(rel)
        if label:
            by_class[label].append((rel, files))

    print("\n--- Q2  Class counts -------------------------------------------")
    for label, entries in sorted(by_class.items()):
        n = sum(len(f) for _, f in entries)
        print("  %-10s %6d files across %d directories" % (label, n, len(entries)))
        report["classes"][label] = {"n_files": n, "n_dirs": len(entries)}

    limit = args.max_per_class or None

    for label, entries in sorted(by_class.items()):
        rel, files = max(entries, key=lambda e: len(e[1]))
        print("\n--- %s  (auditing %s) ---------------------------------"
              % (label.upper(), rel))
        stack, names = load_class_samples(args.root, rel, files, limit)
        if stack is None:
            print("  could not load any samples")
            continue

        print("  loaded %d samples, matrix shape %s"
              % (len(stack), stack.shape[1:]))
        rep = report["classes"][label]
        rep["audited_dir"] = rel
        rep["n_audited"] = int(len(stack))
        rep["matrix_shape"] = list(stack.shape[1:])

        print("\n  Q3  session structure")
        adj = adjacency_correlation(stack)
        rep["adjacency"] = adj
        print("      adjacent-pair correlation   mean %.3f  median %.3f"
              % (adj["adjacent_mean"], adj["adjacent_median"]))
        print("      random-pair correlation     mean %.3f  median %.3f"
              % (adj["random_mean"], adj["random_median"]))
        print("      gap                         %.3f" % adj["gap"])
        print("      fraction of adjacent pairs above 0.9:  %.2f"
              % adj["frac_adjacent_above_0p9"])

        thr = (args.run_threshold if args.run_threshold > 0
               else adaptive_threshold(adj))
        runs = segment_runs(stack, thr)
        rep["runs"] = runs
        print("      candidate runs at threshold %.2f: %d"
              % (runs["threshold"], runs["n_runs"]))
        print("      run length  min %d  median %.0f  max %d"
              % (runs["run_length_min"], runs["run_length_median"],
                 runs["run_length_max"]))

        print("\n  Q4  dynamic range and concentration")
        dr = dynamic_range_report(stack)
        rep["dynamic_range"] = dr
        print("      value range        %.2f to %.2f  (mean %.2f, sd %.2f)"
              % (dr["global_min"], dr["global_max"],
                 dr["value_mean"], dr["value_std"]))
        print("      per-sample span    median %.2f  p10 %.2f"
              % (dr["per_sample_span_median_dB"], dr["per_sample_span_p10_dB"]))
        print("      cells exactly at global minimum: %.3f"
              % dr["frac_cells_exactly_at_global_min"])
        print("      approx distinct values: %d" % dr["approx_unique_values"])
        if dr["looks_hard_floored"]:
            print("      FLAG: distribution looks hard-floored by the detector")

        ec = energy_concentration(stack)
        rep["energy"] = ec
        print("      power in top 5%% of cells:  %.3f" % ec["power_in_top_5pct_cells"])
        print("      power in top 10%% of cells: %.3f" % ec["power_in_top_10pct_cells"])
        print("      cells to reach 90%% power:  %.0f of %d"
              % (ec["cells_to_reach_90pct_power_median"], ec["n_cells"]))

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    verdict = {}

    gaps = [report["classes"][c].get("adjacency", {}).get("gap")
            for c in report["classes"]]
    gaps = [g for g in gaps if g is not None]
    if gaps and min(gaps) > 0.20:
        verdict["grouping"] = "feasible"
        print("  GROUPING: adjacent files are strongly correlated in every")
        print("            class. Filename order tracks acquisition order, so")
        print("            contiguous runs are usable as grouping units.")
        print("            Proceed with grouped splitting as proposed.")
    elif gaps:
        verdict["grouping"] = "unclear"
        print("  GROUPING: adjacent-pair correlation is not clearly above the")
        print("            random baseline. Filename order may have been")
        print("            shuffled by the publishers. Do NOT assume contiguous")
        print("            runs. Options: look for session identifiers in the")
        print("            filenames, or cluster samples by content and treat")
        print("            clusters as groups, reporting this as a limitation.")

    conc = {c: report["classes"][c].get("energy", {}).get(
        "cells_to_reach_90pct_power_median") for c in report["classes"]}
    conc = {k: v for k, v in conc.items() if v is not None}
    if len(conc) >= 2:
        order = sorted(conc, key=conc.get)
        verdict["concentration_order"] = order
        print("\n  CONCENTRATION: cells needed for 90% of power, fewest first:")
        print("            " + "  <  ".join("%s (%.0f)" % (k, conc[k]) for k in order))
        if order[0] == "drone":
            print("            Drone is the most concentrated class, which is")
            print("            what the redundancy mechanism predicts.")
        else:
            print("            Drone is NOT the most concentrated class. The")
            print("            redundancy mechanism needs revisiting before the")
            print("            compression grid is run.")

    floored = [c for c in report["classes"]
               if report["classes"][c].get("dynamic_range", {}).get("looks_hard_floored")]
    verdict["hard_floored_classes"] = floored
    if floored:
        print("\n  FLOOR: %s appear hard-floored by the detector." % ", ".join(floored))
        print("            Low-amplitude content may already be gone, which")
        print("            limits what compression can additionally destroy.")
    else:
        print("\n  FLOOR: no hard floor detected. Low-amplitude structure")
        print("            survives in the published samples.")

    report["verdict"] = verdict
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rdrd_inventory_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n  written: %s" % out)


if __name__ == "__main__":
    main()
