"""
Step 1 pipeline: audit, deduplicate, measure leakage, build splits.

Usage:
    python run_step1.py --root /content/rdrd/raw --out /content/drive/.../
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import rdrd


def short_name_safe(k):
    return rdrd.short_name(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-session", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--sim-thresh", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet-sessions", action="store_true",
                    help="suppress the per-session table")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    report = {"config": vars(args)}

    print("=" * 72)
    print("STEP 1  audit, dedup, leakage, splits")
    print("=" * 72)

    # ---------------------------------------------------------- 1  discover
    raw = rdrd.find_sessions(args.root)
    if not raw:
        raise SystemExit("no data found under %s" % args.root)
    print("\n[1] Raw discovery")
    raw_total = 0
    for lab in sorted(raw):
        n = sum(len(v) for v in raw[lab].values())
        raw_total += n
        print("    %-8s %3d folders %7d files" % (lab, len(raw[lab]), n))
    print("    total %d files" % raw_total)

    # ---------------------------------------------------------- 2  dedup tree
    sessions, dropped = rdrd.dedupe_tree(raw)
    print("\n[2] Tree deduplication (content hash)")
    total = 0
    for lab in sorted(sessions):
        n = sum(len(v) for v in sessions[lab].values())
        total += n
        print("    %-8s %3d folders %7d files" % (lab, len(sessions[lab]), n))
    n_ident = sum(1 for d in dropped if d[3] == "identical")
    n_part = len(dropped) - n_ident
    print("    dropped %d duplicate folders (%d identical, %d partial)"
          % (len(dropped), n_ident, n_part))
    for lab, k, keep_k, why in dropped:
        if why != "identical":
            print("      %-7s %-28s -> kept %-14s  %s"
                  % (lab, k, short_name_safe(keep_k), why))
    print("    total %d files   (published figure: 17485)" % total)
    report["dedup"] = {"raw_total": raw_total, "total": total,
                       "dropped": len(dropped)}
    if total != 17485:
        print("    NOTE: does not match the published count. Worth a look,")
        print("          but not necessarily wrong -- check [1] vs [2].")

    # ---------------------------------------------------------- 3  features
    print("\n[3] Loading features (%d per session, top_k=%d)"
          % (args.per_session, args.top_k))
    feats, stats = {}, {}
    for lab in sorted(sessions):
        feats[lab], stats[lab] = {}, {}
        for key, paths in sessions[lab].items():
            stack, f = rdrd.session_features(paths, args.per_session, args.top_k)
            if stack is None:
                continue
            feats[lab][key] = f
            stats[lab][key] = rdrd.signal_stats(stack)
            stats[lab][key].update(rdrd.numbering(paths))
        print("    %-8s %d sessions" % (lab, len(feats[lab])))

    # ---------------------------------------------------------- 4  units
    print("\n[4] Grouping units (acquisition timestamp)")
    unit_of = rdrd.group_by_timestamp(sessions)
    for lab in sorted(sessions):
        units = sorted(set(unit_of[lab].values()))
        multi = {}
        for key, u in unit_of[lab].items():
            multi.setdefault(u, []).append(rdrd.short_name(key))
        merged = {u: v for u, v in multi.items() if len(v) > 1}
        print("    %-8s %d folders -> %d units" % (lab, len(unit_of[lab]), len(units)))
        for u in sorted(merged)[:8]:
            print("        %-8s <- %s" % (u, ", ".join(sorted(merged[u]))))
        if len(merged) > 8:
            print("        ... %d more multi-folder units" % (len(merged) - 8))
        report.setdefault("units", {})[lab] = {
            "folders": len(unit_of[lab]), "units": len(units),
            "multi_folder_units": len(merged)}

    print("\n    Sessions with identical file counts AND identical signal")
    print("    statistics (candidate reprocessings, reported not merged):")
    any_susp = False
    for lab in sorted(sessions):
        for a, b, n, p in rdrd.suspicious_pairs(stats[lab]):
            ua, ub = unit_of[lab][a], unit_of[lab][b]
            note = "same unit" if ua == ub else "DIFFERENT UNITS -- check"
            print("      %-6s %-12s + %-12s  n=%d  ptf=%.1f dB  (%s)"
                  % (lab, rdrd.short_name(a), rdrd.short_name(b), n, p, note))
            any_susp = True
    if not any_susp:
        print("      none")

    # ---------------------------------------------------------- 5  signal
    print("\n[5] Signal statistics (peak above noise floor)")
    if not args.quiet_sessions:
        for lab in sorted(sessions):
            print("\n    %s" % lab.upper())
            print("      %-14s %6s %11s %10s" % ("session", "n", "ptf med", "ptf p10"))
            for key in sorted(stats[lab], key=lambda k: rdrd.short_name(k)):
                s = stats[lab][key]
                flag = "" if s.get("contiguous", True) else "  (%d gaps)" % s.get("missing", 0)
                print("      %-14s %6d %8.1f dB %7.1f dB%s"
                      % (rdrd.short_name(key)[:14], s["n"],
                         s["ptf_median"], s["ptf_p10"], flag))
    print("\n    %-8s %12s %10s %12s" % ("class", "ptf median", "spread", "cells<10dB"))
    for lab in sorted(sessions):
        v = [stats[lab][k]["ptf_median"] for k in stats[lab]]
        c = [stats[lab][k]["cells_within_10dB"] for k in stats[lab]]
        print("    %-8s %9.1f dB %7.1f dB %12.0f"
              % (lab, float(np.median(v)), float(np.max(v) - np.min(v)),
                 float(np.median(c))))
        report.setdefault("signal", {})[lab] = {
            "ptf_median": float(np.median(v)),
            "ptf_spread": float(np.max(v) - np.min(v)),
            "cells_within_10dB": float(np.median(c))}

    # ---------------------------------------------------------- 6  leakage
    print("\n[6] LEAKAGE TEST")
    print("    %-8s %7s %9s %9s %9s %13s"
          % ("class", "units", "within", "cross", "gap", "above p95"))
    gaps = []
    for lab in sorted(sessions):
        lt = rdrd.leakage_test(feats[lab], unit_of[lab], rng)
        if lt is None:
            print("    %-8s insufficient units" % lab)
            continue
        gaps.append(lt["gap"])
        report.setdefault("leakage", {})[lab] = lt
        print("    %-8s %7d %9.3f %9.3f %+9.3f %12.2f"
              % (lab, lt["n_units"], lt["within_mean"], lt["cross_mean"],
                 lt["gap"], lt["frac_within_above_cross_p95"]))
    if gaps:
        print("")
        if max(gaps) > 0.10:
            print("    Within-unit pairs are clearly more alike than cross-unit")
            print("    pairs: a random split leaks.")
        else:
            print("    Small gap even after dedup. If this holds, random")
            print("    splitting here is closer to honest than expected --")
            print("    which is itself a reportable result.")

    # ---------------------------------------------------------- 7  splits
    print("\n[7] Splits")
    grouped, rand = rdrd.build_splits(sessions, unit_of, rng, args.test_frac)
    ver = rdrd.verify_split_by_content(grouped, rng)
    print("    grouped  %6d train / %6d test" % (len(grouped["train"]), len(grouped["test"])))
    print("    random   %6d train / %6d test" % (len(rand["train"]), len(rand["test"])))
    print("    content check: %d collisions in %d probed -> %s"
          % (ver["collisions"], ver["probed_train"],
             "CLEAN" if ver["clean"] else "CONTAMINATED"))
    report["split_verification"] = ver
    if not ver["clean"]:
        print("    DO NOT PROCEED. Identical files appear on both sides.")

    for name, sp in [("split_grouped.json", grouped), ("split_random.json", rand)]:
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump({"train": sp["train"], "test": sp["test"],
                       "seed": args.seed, "test_frac": args.test_frac}, fh)
    with open(os.path.join(args.out, "step1_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n    written to %s" % args.out)

    # ---------------------------------------------------------- 8  cache
    print("\n[8] Building array cache")
    X, y, sess, unit = [], [], [], []
    for lab in sorted(sessions):
        for key, paths in sessions[lab].items():
            for p in paths:
                try:
                    X.append(rdrd.load_matrix(p))
                except ValueError:
                    continue
                y.append(lab)
                sess.append(rdrd.short_name(key))
                unit.append(rdrd.short_name(unit_of[lab].get(key, key)))
    shapes = defaultdict(int)
    for m in X:
        shapes[m.shape] += 1
    dom = max(shapes, key=shapes.get)
    keep = [i for i, m in enumerate(X) if m.shape == dom]
    print("    shapes: %s  keeping %d of %d"
          % (dict(shapes), len(keep), len(X)))
    np.savez_compressed(
        os.path.join(args.out, "rdrd_cache.npz"),
        X=np.stack([X[i] for i in keep]).astype(np.float32),
        y=np.array([y[i] for i in keep]),
        session=np.array([sess[i] for i in keep]),
        unit=np.array([unit[i] for i in keep]))
    print("    cache: %d samples of shape %s" % (len(keep), dom))
    print("\ndone.")


if __name__ == "__main__":
    main()
