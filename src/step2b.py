"""
Step 2b: grouped k-fold and normalisation ablation.

Two gaps in Step 2 that must close before the compression grid.

  1. Five seeds varied the INITIALISATION, not the partition. The grouped
     split was a single fixed partition, so a 2.1-point drone F1 drop could
     equally be an accident of which units landed in test. With 21 drone
     units and about 6 held out, that is a live risk. Grouped k-fold rotates
     the held-out units and separates split variance from seed variance.

  2. The baseline sits ~5 points below the published 98%. One suspect is
     peak-referenced normalisation: it preserves the headroom structure the
     mechanism depends on, but discards absolute return level, which is a
     strong class cue given person 43.5 / car 34.7 / drone 27.3 dB. That
     trade may be correct for this study, but it should be measured.

Usage:
    python step2b.py kfold --data DIR/dataset.npz --folds 5 --seeds 0 1 2
    python step2b.py norm  --data DIR/dataset.npz --folds 5
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model

VAL_FRAC = 0.15

# two-tailed t critical values at p=0.05, indexed by degrees of freedom
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
          7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


# ------------------------------------------------------------ normalisation

def normalise(X, scheme):
    """
    peak       peak -> 0 dB, scaled by 60 dB. Keeps headroom shape,
               DISCARDS absolute return level.
    offset     fixed shift only. Keeps absolute level.
    global     standardised by global mean/sd. Keeps absolute level.
    peak_level two channels: peak-referenced, plus a constant plane holding
               the peak itself. Keeps headroom shape AND absolute level.
    """
    X = X.astype(np.float32)
    peak = X.reshape(len(X), -1).max(axis=1)[:, None, None]
    if scheme == "peak":
        return ((X - peak) / 60.0)[:, None]
    if scheme == "offset":
        return ((X + 60.0) / 60.0)[:, None]
    if scheme == "global":
        return ((X - X.mean()) / (X.std() + 1e-6))[:, None]
    if scheme == "peak_level":
        a = (X - peak) / 60.0
        b = np.broadcast_to((peak + 60.0) / 60.0, X.shape).astype(np.float32)
        return np.stack([a, b], axis=1)
    raise ValueError(scheme)


# ------------------------------------------------------------------ folds

def make_grouped_folds(y, unit, k, seed=777):
    """
    Assign each class's UNITS to k folds, balancing sample counts rather
    than unit counts, since unit sizes vary from 31 to 660 files.
    """
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, dtype=int)
    for c in CLASSES:
        us = sorted({unit[i] for i in range(len(y)) if y[i] == c})
        sizes = {u: int(np.sum((unit == u))) for u in us}
        order = sorted(us, key=lambda u: -sizes[u])
        load = np.zeros(k, dtype=int)
        assign = {}
        for u in order:
            f = int(np.argmin(load))       # greedy: fill the emptiest fold
            assign[u] = f
            load[f] += sizes[u]
        for i in range(len(y)):
            if y[i] == c:
                fold[i] = assign[unit[i]]
        rng.random()
    return fold


def make_random_folds(y, k, sizes_per_fold, seed=777):
    """Frame-level folds matched in size to the grouped folds, per class."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, dtype=int)
    for c in CLASSES:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        pos = 0
        for f in range(k):
            n = sizes_per_fold[c][f]
            fold[idx[pos:pos + n]] = f
            pos += n
        fold[idx[pos:]] = k - 1
    return fold


def split_for_fold(fold, f, k, y, unit, grouped, seed):
    """test = fold f; val carved from the remaining folds (units if grouped)."""
    rng = np.random.default_rng(1000 + seed)
    test = fold == f
    pool = ~test
    val = np.zeros(len(y), dtype=bool)
    if grouped:
        for c in CLASSES:
            us = sorted({unit[i] for i in range(len(y)) if pool[i] and y[i] == c})
            rng.shuffle(us)
            n = max(1, int(round(VAL_FRAC * len(us))))
            chosen = set(us[:n])
            for i in range(len(y)):
                if pool[i] and y[i] == c and unit[i] in chosen:
                    val[i] = True
    else:
        idx = np.where(pool)[0]
        rng.shuffle(idx)
        val[idx[:int(round(VAL_FRAC * len(idx)))]] = True
    train = pool & ~val
    return train, val, test


# ------------------------------------------------------------------ train

def train_once(Xn, yi, train, val, test, seed, epochs=40, bs=128, lr=3e-3):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(Xn[train]); ytr = torch.tensor(yi[train])
    Xva = torch.tensor(Xn[val]);   yva = torch.tensor(yi[val])
    Xte = torch.tensor(Xn[test]);  yte = torch.tensor(yi[test])

    model = make_model(torch, nn).to(dev)
    if Xn.shape[1] != 1:                      # widen the stem for 2 channels
        old = model.stem[0]
        model.stem[0] = nn.Conv2d(Xn.shape[1], old.out_channels, 3, 1, 1,
                                  bias=False).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss()

    best, best_state, bad = -1.0, None, 0
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
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
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
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
    true = yte.numpy()

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    out = {"accuracy": float((pred == true).mean()), "confusion": cm.tolist()}
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / max(1, cm[i].sum())
        pre = cm[i, i] / max(1, cm[:, i].sum())
        out[c] = {"recall": float(rec), "precision": float(pre),
                  "f1": float(2 * rec * pre / max(1e-9, rec + pre))}
    return out


# ------------------------------------------------------------------ stats

def decompose(runs, metric):
    """Separate between-fold (split) variance from within-fold (seed) variance."""
    by_fold = defaultdict(list)
    for r in runs:
        by_fold[r["fold"]].append(r["result"][metric] if metric == "accuracy"
                                  else r["result"][metric[0]][metric[1]])
    fold_means = np.array([np.mean(v) for _f, v in sorted(by_fold.items())])
    within = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0
                       for _f, v in sorted(by_fold.items())])
    return {"mean": float(fold_means.mean()),
            "between_fold_sd": float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0,
            "within_fold_sd": float(within.mean()),
            "fold_means": [float(x) for x in fold_means]}


def paired(g_runs, r_runs, metric):
    """Per-fold paired difference, grouped minus random."""
    def fold_means(runs):
        d = defaultdict(list)
        for r in runs:
            d[r["fold"]].append(r["result"][metric] if metric == "accuracy"
                                else r["result"][metric[0]][metric[1]])
        return np.array([np.mean(v) for _f, v in sorted(d.items())])
    g, r = fold_means(g_runs), fold_means(r_runs)
    d = g - r
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan")
    return {"diffs": [float(x) for x in d], "mean_diff": float(d.mean()),
            "sd": float(sd), "t": t, "n": int(len(d))}


def label(metric):
    return metric if isinstance(metric, str) else "%s %s" % metric


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("kfold", "norm"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--out", default=None)
        if name == "kfold":
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
            p.add_argument("--scheme", default="peak")
        else:
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--schemes", nargs="+",
                           default=["peak", "offset", "global", "peak_level"])
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])
    k = args.folds

    gfold = make_grouped_folds(y, unit, k)
    sizes = {c: [int(np.sum((y == c) & (gfold == f))) for f in range(k)]
             for c in CLASSES}
    rfold = make_random_folds(y, k, sizes)

    print("=" * 70)
    print("fold sizes per class (grouped)")
    for c in CLASSES:
        print("  %-8s %s" % (c, sizes[c]))
    print("=" * 70)

    if args.cmd == "kfold":
        Xn = normalise(X, args.scheme)
        allruns = {}
        for gname, fold, grouped in (("grouped", gfold, True),
                                     ("random", rfold, False)):
            runs = []
            for f in range(k):
                for s in args.seeds:
                    tr, va, te = split_for_fold(fold, f, k, y, unit, grouped, s)
                    r = train_once(Xn, yi, tr, va, te, s, args.epochs)
                    runs.append({"fold": f, "seed": s, "result": r})
                    print("  %-8s fold %d seed %d   acc %.4f  drone F1 %.4f"
                          % (gname, f, s, r["accuracy"], r["drone"]["f1"]))
            allruns[gname] = runs

        print("\n" + "=" * 70)
        print("VARIANCE DECOMPOSITION  (%d folds x %d seeds)"
              % (k, len(args.seeds)))
        print("=" * 70)
        metrics = ["accuracy", ("drone", "f1"), ("drone", "precision"),
                   ("drone", "recall"), ("car", "f1")]
        print("  %-18s %-9s %8s %12s %12s"
              % ("metric", "split", "mean", "fold sd", "seed sd"))
        summary = {}
        for m in metrics:
            for gname in ("random", "grouped"):
                st = decompose(allruns[gname], m)
                summary.setdefault(label(m), {})[gname] = st
                print("  %-18s %-9s %8.4f %12.4f %12.4f"
                      % (label(m), gname, st["mean"],
                         st["between_fold_sd"], st["within_fold_sd"]))

        print("\n" + "=" * 70)
        print("PAIRED PER-FOLD DIFFERENCE  (grouped minus random)")
        print("=" * 70)
        print("  %-18s %10s %10s %8s   %s"
              % ("metric", "mean diff", "sd", "t", "per fold"))
        pairs = {}
        for m in metrics:
            pr = paired(allruns["grouped"], allruns["random"], m)
            pairs[label(m)] = pr
            print("  %-18s %+10.4f %10.4f %8.2f   %s"
                  % (label(m), pr["mean_diff"], pr["sd"], pr["t"],
                     " ".join("%+.3f" % x for x in pr["diffs"])))

        print("")
        acc = pairs["accuracy"]
        dr = pairs["drone f1"]
        tc = T_CRIT.get(k - 1, 2.0)
        print("  With %d folds (df=%d), |t| above %.2f is significant at the"
              % (k, k - 1, tc))
        print("  5 percent level. Compare the aggregate accuracy row against")
        print("  the drone rows: if accuracy is flat while drone F1 drops, the")
        print("  leakage is concentrated in one class and the headline number")
        print("  conceals it.")
        for nm, pr in (("accuracy", acc), ("drone F1", dr)):
            verdict = ("significant" if abs(pr["t"]) > tc else "not significant")
            print("    %-10s %+.4f  t=%6.2f  %s"
                  % (nm, pr["mean_diff"], pr["t"], verdict))

        if args.out:
            json.dump({"summary": summary, "paired": pairs,
                       "fold_sizes": sizes}, open(args.out, "w"), indent=2)
            print("\n  written %s" % args.out)
        return

    # ---------------------------------------------------------- norm
    print("\nNORMALISATION ABLATION (grouped folds, seed %d)" % args.seed)
    res = {}
    for scheme in args.schemes:
        Xn = normalise(X, scheme)
        runs = []
        for f in range(k):
            tr, va, te = split_for_fold(gfold, f, k, y, unit, True, args.seed)
            r = train_once(Xn, yi, tr, va, te, args.seed, args.epochs)
            runs.append({"fold": f, "seed": args.seed, "result": r})
            print("  %-11s (%dch) fold %d   acc %.4f  drone F1 %.4f"
                  % (scheme, Xn.shape[1], f, r["accuracy"], r["drone"]["f1"]))
        res[scheme] = runs

    print("\n" + "=" * 70)
    print("  %-12s %10s %10s %12s %12s"
          % ("scheme", "accuracy", "fold sd", "drone F1", "drone prec"))
    out = {}
    for scheme in args.schemes:
        a = decompose(res[scheme], "accuracy")
        f1 = decompose(res[scheme], ("drone", "f1"))
        pr = decompose(res[scheme], ("drone", "precision"))
        out[scheme] = {"accuracy": a, "drone_f1": f1, "drone_precision": pr}
        print("  %-12s %10.4f %10.4f %12.4f %12.4f"
              % (scheme, a["mean"], a["between_fold_sd"], f1["mean"], pr["mean"]))

    base = out.get("peak", {}).get("accuracy", {}).get("mean")
    best = max(out, key=lambda s: out[s]["accuracy"]["mean"])
    print("")
    if base is not None and out[best]["accuracy"]["mean"] - base > 0.01:
        print("  '%s' beats peak-referencing by %+.4f accuracy."
              % (best, out[best]["accuracy"]["mean"] - base))
        print("  Absolute return level is carrying real class information.")
        print("  Note the tension: that level is exactly what the headroom")
        print("  mechanism is about, so a scheme that hands the network the")
        print("  level directly may mask the effect the study wants to see.")
        print("  'peak_level' keeps both and is the compromise to prefer if")
        print("  it is the winner.")
    else:
        print("  No scheme beats peak-referencing by more than 1 point.")
        print("  The gap to the published 98 percent is not a normalisation")
        print("  artefact and needs another explanation.")

    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
