#!/usr/bin/env python3
"""
Step 1 of the radar mini project: audit the Real Doppler RAD-DAR dataset.

This script answers the questions that everything downstream depends on:

  Q1  What is the actual on-disk structure, and how many samples per class?
  Q2  Are the sample matrices a consistent shape, and is it 11 x 61?
  Q3  Do filenames encode any grouping information (recording pass, run,
      session, timestamp)? This decides whether grouped splitting is possible
      from metadata, or whether it has to be inferred.
  Q4  What is the amplitude distribution per class? This is the first direct
      test of the redundancy mechanism: the drone class should show lower
      peak amplitude and fewer strong cells than the vehicle class.
  Q5  Are consecutive files correlated? Strong adjacency correlation is
      evidence of continuous recording passes and therefore of leakage risk
      under a random split.

It writes a manifest CSV with one row per sample. Every later stage of the
project reads that manifest instead of walking the directory again.

Nothing here assumes a folder layout. The script discovers it and reports
what it finds. If an assumption fails it says so rather than guessing.

Usage:
    python rdrd_audit.py --root /path/to/rdrd
    python rdrd_audit.py --root /path/to/rdrd --full     # stats on all files
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------- discovery

DATA_EXT = {".csv", ".txt", ".dat", ".npy"}


def walk_tree(root, max_depth=4):
    """Return {relative_dir: [filenames]} and a printable tree summary."""
    layout = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        data_files = [f for f in filenames
                      if os.path.splitext(f)[1].lower() in DATA_EXT]
        if data_files:
            layout[rel] = sorted(data_files)
    return layout


def report_layout(layout):
    print("\n" + "=" * 72)
    print("Q1  DIRECTORY STRUCTURE")
    print("=" * 72)
    if not layout:
        print("  No data files found. Check --root, and confirm the archive")
        print("  was extracted rather than left zipped.")
        return
    total = 0
    for rel in sorted(layout):
        n = len(layout[rel])
        total += n
        example = layout[rel][0]
        print("  %-38s %6d files   e.g. %s" % (rel, n, example))
    print("  %-38s %6d" % ("TOTAL", total))
    print("\n  Expected from the literature: 17,485 samples total,")
    print("  5,720 vehicle / 5,065 drone / 6,700 pedestrian.")
    print("  A mismatch is not necessarily an error, but explain it before")
    print("  quoting any published baseline number as comparable.")


# ---------------------------------------------------------------- filenames

# Candidate patterns for grouping metadata. Order matters only for reporting.
TOKEN_PATTERNS = [
    ("integer_runs", re.compile(r"(\d+)")),
    ("date_like", re.compile(r"(20\d{2}[-_]?\d{2}[-_]?\d{2})")),
    ("time_like", re.compile(r"(\d{2}[-_:]\d{2}[-_:]\d{2})")),
    ("keyword_run", re.compile(r"(?i)\b(run|pass|seq|session|track|meas|flight|trial)[-_]?(\d+)")),
]


def analyse_filenames(layout):
    print("\n" + "=" * 72)
    print("Q3  FILENAME STRUCTURE  (does metadata support grouped splitting?)")
    print("=" * 72)
    findings = {}
    for rel in sorted(layout):
        names = layout[rel]
        print("\n  [%s]  %d files" % (rel, len(names)))
        print("    first three : %s" % ", ".join(names[:3]))
        print("    last  three : %s" % ", ".join(names[-3:]))

        for label, pat in TOKEN_PATTERNS:
            hits = [pat.findall(n) for n in names]
            n_with = sum(1 for h in hits if h)
            if n_with == 0:
                continue
            flat = [tuple(h[0]) if isinstance(h[0], tuple) else (h[0],)
                    for h in hits if h]
            uniq = len(set(flat))
            print("    %-13s present in %d/%d files, %d distinct values"
                  % (label, n_with, len(names), uniq))
            findings[(rel, label)] = uniq

        # How many integer fields does a typical name carry?
        counts = Counter(len(re.findall(r"\d+", n)) for n in names)
        print("    integer fields per name: %s"
              % ", ".join("%d fields x%d" % (k, v)
                          for k, v in sorted(counts.items())))

    print("\n  HOW TO READ THIS:")
    print("  If a field has a small number of distinct values (say 3 to 60)")
    print("  while the file count is in the thousands, that field is a strong")
    print("  candidate for the recording pass identifier. If every integer")
    print("  field is simply a unique counter running 1..N, then the pass")
    print("  structure is NOT in the filenames and must be inferred from the")
    print("  data itself. Step 2 of the project handles that case.")
    return findings


# ------------------------------------------------------------------ loading

def load_matrix(path):
    """Load one sample. Tries the plausible formats and reports failure."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    for kwargs in ({"delimiter": ","}, {"delimiter": ";"}, {}):
        try:
            a = np.loadtxt(path, **kwargs)
            if a.ndim == 2 and a.size > 1:
                return a
        except Exception:
            continue
    try:
        import pandas as pd
        return pd.read_csv(path, header=None).to_numpy(dtype=float)
    except Exception as exc:
        raise RuntimeError("could not parse %s (%s)" % (path, exc))


def check_shapes(root, layout, n_probe=40):
    print("\n" + "=" * 72)
    print("Q2  SAMPLE SHAPE AND DTYPE")
    print("=" * 72)
    shapes = Counter()
    failures = []
    for rel in sorted(layout):
        names = layout[rel]
        idx = np.linspace(0, len(names) - 1, min(n_probe, len(names)))
        for i in idx.astype(int):
            p = os.path.join(root, rel, names[i])
            try:
                a = load_matrix(p)
                shapes[a.shape] += 1
            except Exception as exc:
                failures.append((p, str(exc)))
    for shp, n in shapes.most_common():
        print("  shape %-14s seen %d times" % (str(shp), n))
    if failures:
        print("\n  %d files failed to parse. First three:" % len(failures))
        for p, e in failures[:3]:
            print("    %s\n      %s" % (p, e))
    if len(shapes) > 1:
        print("\n  WARNING: shapes are not uniform. Resolve this before")
        print("  building any tensor pipeline.")
    print("\n  Expected 11 x 61 (or its transpose) after CFAR cropping.")
    print("  If the Doppler axis is 61 bins at 0.34 km/h resolution, the")
    print("  retained Doppler window is about 21 km/h. Confirm this, because")
    print("  it is the basis for the mechanism argument in the proposal.")
    return shapes


# -------------------------------------------------------------- amplitudes

def sample_stats(a):
    """Per-sample statistics used by the redundancy argument."""
    a = a.astype(float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    peak = float(finite.max())
    floor = float(np.percentile(finite, 10))
    rng = peak - floor
    # Two different concentration measures, because they answer different
    # questions and can disagree.
    #
    # Peak-referenced: how many cells sit within X dB of this sample's own
    # peak. This measures the SHAPE of the signature but is confounded by
    # dynamic range, since a weak target whose peak is close to the noise
    # floor will appear to have many cells near its peak simply because the
    # floor is included.
    n3 = int(np.sum(finite >= peak - 3))
    n6 = int(np.sum(finite >= peak - 6))
    n10 = int(np.sum(finite >= peak - 10))
    #
    # Floor-referenced: how many cells rise a given margin ABOVE the local
    # noise floor. This is the measure the redundancy argument actually
    # needs, because it counts how many cells carry usable evidence rather
    # than how many sit near an arbitrary peak. Use these columns, not the
    # peak-referenced ones, when testing the mechanism.
    a6 = int(np.sum(finite >= floor + 6))
    a12 = int(np.sum(finite >= floor + 12))
    a20 = int(np.sum(finite >= floor + 20))
    return dict(peak=peak, floor=floor, dyn_range=rng,
                mean=float(finite.mean()), std=float(finite.std()),
                cells_within_3db=n3, cells_within_6db=n6,
                cells_within_10db=n10,
                cells_above_floor_6db=a6, cells_above_floor_12db=a12,
                cells_above_floor_20db=a20, n_cells=int(finite.size))


def report_amplitudes(rows):
    print("\n" + "=" * 72)
    print("Q4  AMPLITUDE AND CONCENTRATION BY CLASS")
    print("=" * 72)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_dir"]].append(r)
    if not by_class:
        print("  No statistics collected.")
        return
    hdr = ("class", "n", "peak med", "floor med", "dyn rng",
           ">floor+6", ">floor+12", ">floor+20")
    print("  %-12s %6s %9s %10s %9s %10s %11s %11s" % hdr)
    for cls in sorted(by_class):
        rs = by_class[cls]
        med = lambda k: np.median([float(r[k]) for r in rs])
        print("  %-12s %6d %9.2f %10.2f %9.2f %10.1f %11.1f %11.1f"
              % (cls, len(rs), med("peak"), med("floor"), med("dyn_range"),
                 med("cells_above_floor_6db"),
                 med("cells_above_floor_12db"),
                 med("cells_above_floor_20db")))
    print("\n  WHAT TO LOOK FOR:")
    print("  Read the floor-referenced columns, not the peak-referenced ones.")
    print("  The redundancy mechanism predicts that the drone class shows a")
    print("  LOWER peak and FEWER cells standing clear of the noise floor")
    print("  than the vehicle class, because its energy is confined to few")
    print("  cells at low amplitude. The vehicle class should show the most")
    print("  cells above floor, since a large target disperses energy across")
    print("  many range cells.")
    print("\n  If the drone class carries as many cells above floor as the")
    print("  vehicle class, the mechanism is in trouble and the framing needs")
    print("  revisiting BEFORE the compression grid is run. Report the number")
    print("  either way: this table is a paper figure, not just a check.")
    print("\n  Also check the floor column against the peak. If the floor is")
    print("  flat and identical across samples, the detector thresholding has")
    print("  already removed the weak structure, which is the risk named in")
    print("  section 6 of the proposal.")


# ------------------------------------------------------------ adjacency

def adjacency_probe(root, layout, n_probe=200):
    """Correlation between files that are adjacent in sorted order.

    High adjacency correlation relative to a random pairing is evidence of
    continuous recording passes, and therefore of leakage under a random
    split. This is provisional: sorted order is not guaranteed to be
    acquisition order.
    """
    print("\n" + "=" * 72)
    print("Q5  ADJACENCY CORRELATION  (leakage risk probe)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    for rel in sorted(layout):
        names = layout[rel]
        if len(names) < 20:
            continue
        k = min(n_probe, len(names) - 1)
        starts = rng.choice(len(names) - 1, size=k, replace=False)
        adj, rnd = [], []
        for s in starts:
            try:
                a = load_matrix(os.path.join(root, rel, names[s])).ravel()
                b = load_matrix(os.path.join(root, rel, names[s + 1])).ravel()
                j = rng.integers(0, len(names))
                c = load_matrix(os.path.join(root, rel, names[j])).ravel()
            except Exception:
                continue
            if a.shape != b.shape or a.shape != c.shape:
                continue
            adj.append(np.corrcoef(a, b)[0, 1])
            rnd.append(np.corrcoef(a, c)[0, 1])
        if not adj:
            continue
        print("  [%s]  adjacent r = %.3f    random pair r = %.3f   (n=%d)"
              % (rel, float(np.nanmedian(adj)), float(np.nanmedian(rnd)),
                 len(adj)))
    print("\n  WHAT TO LOOK FOR:")
    print("  If adjacent correlation is much higher than random-pair")
    print("  correlation, neighbouring files come from the same pass and a")
    print("  random split leaks. That is the expected outcome and it is what")
    print("  justifies the grouped protocol. Record these numbers: they go")
    print("  into the paper as the motivation for grouped splitting.")


# ---------------------------------------------------------------- manifest

def build_manifest(root, layout, out_path, full=False, n_probe=60):
    rows = []
    for rel in sorted(layout):
        names = layout[rel]
        if full:
            chosen = list(range(len(names)))
        else:
            chosen = np.linspace(0, len(names) - 1,
                                 min(n_probe, len(names))).astype(int).tolist()
        for i in chosen:
            name = names[i]
            p = os.path.join(root, rel, name)
            row = {"path": os.path.relpath(p, root),
                   "class_dir": rel,
                   "filename": name,
                   "order_index": i,
                   "ints": "|".join(re.findall(r"\d+", name))}
            try:
                a = load_matrix(p)
                row["rows"], row["cols"] = a.shape
                st = sample_stats(a)
                if st:
                    row.update(st)
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    if not rows:
        return rows
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n  Manifest written to %s  (%d rows)" % (out_path, len(rows)))
    return rows


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory containing the extracted RDRD dataset")
    ap.add_argument("--full", action="store_true",
                    help="compute statistics on every file, not a subsample")
    ap.add_argument("--manifest", default="rdrd_manifest.csv")
    ap.add_argument("--skip-adjacency", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit("Not a directory: %s" % args.root)

    print("Auditing: %s" % os.path.abspath(args.root))
    layout = walk_tree(args.root)
    report_layout(layout)
    if not layout:
        return

    check_shapes(args.root, layout)
    analyse_filenames(layout)
    rows = build_manifest(args.root, layout, args.manifest, full=args.full)
    report_amplitudes([r for r in rows if "peak" in r])
    if not args.skip_adjacency:
        adjacency_probe(args.root, layout)

    print("\n" + "=" * 72)
    print("NEXT STEP")
    print("=" * 72)
    print("  Send me the printed output above and the manifest CSV.")
    print("  The two decisions that follow from it are:")
    print("    1. whether recording passes are recoverable from filenames,")
    print("       or must be inferred by clustering, and")
    print("    2. whether the amplitude statistics support the redundancy")
    print("       mechanism or force a change of framing.")
    print("  Do not start building the model until both are settled.")


if __name__ == "__main__":
    main()
