"""
Step 2c: does per-session performance track per-session headroom?

Step 2b found that which drone sessions are held out swings drone F1 by
about +/-0.10, while the same swing for cars is about +/-0.01 -- a variance
ratio near 18x on drone recall. The mean difference between grouped and
random splitting was NOT significant, so the leakage claim is a null. The
variance asymmetry is the real finding.

This asks the obvious next question: is that variance explained by signal
level? Session peak-to-floor ranges from 16.1 dB (drone 12-41) to 52.8 dB
(drone 13-48). If a session's recall tracks its headroom, the mechanism the
compression study rests on is demonstrated on real data BEFORE any bit is
quantised.

Method: run grouped k-fold. Every unit is in the test fold exactly once, so
one pass yields an out-of-fold recall for every session. Join that against
the session's measured headroom and correlate.

Usage:
    python step2c.py --data DIR/dataset.npz --folds 5 --seeds 0 1
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model
from step2b import normalise, make_grouped_folds, split_for_fold


# ------------------------------------------------------------ headroom

def unit_headroom(X, unit):
    """Median peak-to-floor, in dB, per unit."""
    lin = 10.0 ** ((X - X.reshape(len(X), -1).max(axis=1)[:, None, None]) / 10.0)
    floor = np.median(lin.reshape(len(X), -1), axis=1)
    ptf = -10 * np.log10(np.maximum(floor, 1e-30))
    out = {}
    for u in sorted(set(unit.tolist())):
        m = unit == u
        out[u] = {"ptf_median": float(np.median(ptf[m])),
                  "ptf_p10": float(np.percentile(ptf[m], 10)),
                  "n": int(m.sum())}
    return out


# ------------------------------------------------------------ out-of-fold

def out_of_fold_predictions(Xn, yi, y, unit, gfold, k, seeds, epochs):
    """Every unit is tested exactly once per seed. Returns per-unit recall."""
    import torch
    import torch.nn as nn

    per_unit = defaultdict(list)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        for f in range(k):
            tr, va, te = split_for_fold(gfold, f, k, y, unit, True, seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
            Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])
            Xte = torch.tensor(Xn[te])

            model = make_model(torch, nn).to(dev)
            if Xn.shape[1] != 1:
                old = model.stem[0]
                model.stem[0] = nn.Conv2d(Xn.shape[1], old.out_channels,
                                          3, 1, 1, bias=False).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                    weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=3e-3, total_steps=epochs * max(1, len(Xtr) // 128 + 1))
            lossf = nn.CrossEntropyLoss()

            best, best_state, bad = -1.0, None, 0
            for _ep in range(epochs):
                model.train()
                perm = torch.randperm(len(Xtr))
                for i in range(0, len(perm), 128):
                    j = perm[i:i + 128]
                    opt.zero_grad()
                    lossf(model(Xtr[j].to(dev)), ytr[j].to(dev)).backward()
                    opt.step()
                    try:
                        sched.step()
                    except Exception:
                        pass
                model.eval()
                with torch.no_grad():
                    pv = [model(Xva[i:i + 512].to(dev)).argmax(1).cpu()
                          for i in range(0, len(Xva), 512)]
                    acc = (torch.cat(pv) == yva).float().mean().item()
                if acc > best:
                    best, bad = acc, 0
                    best_state = {kk: v.detach().cpu().clone()
                                  for kk, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10:
                        break
            model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                pt = [model(Xte[i:i + 512].to(dev)).argmax(1).cpu()
                      for i in range(0, len(Xte), 512)]
                pred = torch.cat(pt).numpy()
            idx = np.where(te)[0]
            for u in sorted(set(unit[idx].tolist())):
                m = unit[idx] == u
                per_unit[u].append(float((pred[m] == yi[idx][m]).mean()))
            print("    seed %d fold %d done" % (seed, f))
    return per_unit


# ------------------------------------------------------------ correlation

def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def spearman(a, b):
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, float)))
        return o.astype(float)
    return pearson(rank(a), rank(b))


def t_from_r(r, n):
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return float("nan")
    return float(r * np.sqrt((n - 2) / (1 - r * r)))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--scheme", default="offset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])

    head = unit_headroom(X, unit)
    gfold = make_grouped_folds(y, unit, args.folds)
    Xn = normalise(X, args.scheme)

    print("running %d folds x %d seeds (scheme=%s)"
          % (args.folds, len(args.seeds), args.scheme))
    per_unit = out_of_fold_predictions(Xn, yi, y, unit, gfold,
                                       args.folds, args.seeds, args.epochs)

    print("\n" + "=" * 74)
    print("PER-SESSION RECALL vs HEADROOM")
    print("=" * 74)

    report = {}
    for c in CLASSES:
        rows = []
        for u in sorted(per_unit):
            if not u.startswith(c + "/"):
                continue
            rec = float(np.mean(per_unit[u]))
            sd = float(np.std(per_unit[u]))
            rows.append((u.split("/", 1)[1], head[u]["ptf_median"],
                         head[u]["ptf_p10"], head[u]["n"], rec, sd))
        if len(rows) < 3:
            continue
        rows.sort(key=lambda r: r[1])
        print("\n  %s  (%d sessions, sorted by headroom)" % (c.upper(), len(rows)))
        print("    %-10s %10s %10s %7s %9s %8s"
              % ("session", "ptf med", "ptf p10", "n", "recall", "sd"))
        for nm, pm, pp, n, rec, sd in rows:
            print("    %-10s %8.1f dB %8.1f dB %7d %9.4f %8.4f"
                  % (nm, pm, pp, n, rec, sd))

        ptf = [r[1] for r in rows]
        rec = [r[4] for r in rows]
        rp, rs = pearson(ptf, rec), spearman(ptf, rec)
        n = len(rows)
        print("    correlation of recall with headroom:")
        print("      Pearson  r = %+.3f  (t = %+.2f, n = %d)"
              % (rp, t_from_r(rp, n), n))
        print("      Spearman r = %+.3f" % rs)
        print("    recall spread across sessions: %.4f (sd), min %.4f max %.4f"
              % (float(np.std(rec)), min(rec), max(rec)))
        report[c] = {"n_sessions": n, "pearson": rp, "spearman": rs,
                     "t": t_from_r(rp, n),
                     "recall_sd": float(np.std(rec)),
                     "sessions": [{"name": r[0], "ptf_median": r[1],
                                   "ptf_p10": r[2], "n": r[3],
                                   "recall": r[4], "sd": r[5]} for r in rows]}

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("  %-8s %10s %12s %12s %12s"
          % ("class", "sessions", "recall sd", "Pearson r", "Spearman r"))
    for c in CLASSES:
        if c in report:
            r = report[c]
            print("  %-8s %10d %12.4f %12.3f %12.3f"
                  % (c, r["n_sessions"], r["recall_sd"],
                     r["pearson"], r["spearman"]))
    print("")
    if "drone" in report and "car" in report:
        rd, rc = report["drone"], report["car"]
        if rc["recall_sd"] > 1e-6:
            print("  Recall varies %.1fx more across drone sessions than car"
                  % (rd["recall_sd"] / rc["recall_sd"]))
        else:
            print("  Car recall is constant across sessions; ratio undefined.")
        tc = 2.09 if rd["n_sessions"] > 18 else 2.20
        if not np.isfinite(rd["pearson"]):
            print("  Correlation undefined (recall or headroom has no spread).")
        elif abs(rd["t"]) > tc and rd["pearson"] > 0:
            print("  Drone recall rises with session headroom (r = %+.3f)."
                  % rd["pearson"])
            print("  The headroom mechanism is demonstrated on real data with")
            print("  no compression applied. That is the strongest form this")
            print("  argument can take, and it should lead the paper.")
        else:
            print("  Drone recall does NOT track session headroom (r = %+.3f)."
                  % rd["pearson"])
            print("  The cross-session variance is real but headroom does not")
            print("  explain it. Something else distinguishes the hard")
            print("  sessions -- look at the table above for what the low-recall")
            print("  sessions have in common before assuming the mechanism.")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
