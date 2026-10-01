"""
Step 2d: what is different about drone session 13-48?

Context. Session 13-48 sits at 42.5 dB headroom, nearly 10 dB above any
other drone recording, and scores 0.2494 out-of-fold recall against 0.90 to
0.99 everywhere else. It alone produced the fold-3 collapse in Step 2b and
the apparent headroom correlation in Step 2c. Remove it and drone recall sd
falls from 0.1504 to 0.0262, which is LOWER than car (0.0505) or person
(0.0549).

Deliberately descriptive first. Three mechanisms have already been proposed
for this project and all three failed against data, so this measures a broad
set of properties and reports which ones actually separate 13-48, rather
than testing one story I find appealing.

  A  Descriptive profile. Twelve measurable properties per session, with
     13-48's z-score against the rest of the drone distribution. Whatever
     is different should announce itself.

  B  Where the errors go. Confusion for 13-48 specifically, plus which
     class centroid its samples sit nearest.

  C  Coverage test. THE DECISIVE ONE. Put half of 13-48 into training and
     test on the other half. If recall recovers, the model simply had no
     comparable training data and this is a dataset coverage finding. If it
     stays low, the recording is intrinsically hard or mislabelled, which
     is a different and less interesting story.

Usage:
    python step2d.py --data DIR/dataset.npz --target drone/13-48
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model
from step2b import normalise


# ------------------------------------------------------------------ A

def sample_features(X):
    """Twelve descriptive properties per sample, in physical terms."""
    n, H, W = X.shape
    flat = X.reshape(n, -1)
    peak = flat.max(axis=1)
    lin = 10.0 ** ((X - peak[:, None, None]) / 10.0)
    linf = lin.reshape(n, -1)
    floor = np.median(linf, axis=1)

    arg = flat.argmax(axis=1)
    prow, pcol = arg // W, arg % W

    excess = np.maximum(linf - floor[:, None], 0.0)
    tot = excess.sum(axis=1)
    tot[tot == 0] = 1e-12
    e2 = excess.reshape(n, H, W)
    rows = np.arange(H)[None, :]
    cols = np.arange(W)[None, :]
    rmass = e2.sum(axis=2)
    cmass = e2.sum(axis=1)
    rcent = (rmass * rows).sum(1) / np.maximum(rmass.sum(1), 1e-12)
    ccent = (cmass * cols).sum(1) / np.maximum(cmass.sum(1), 1e-12)
    rspread = np.sqrt((rmass * (rows - rcent[:, None]) ** 2).sum(1)
                      / np.maximum(rmass.sum(1), 1e-12))
    cspread = np.sqrt((cmass * (cols - ccent[:, None]) ** 2).sum(1)
                      / np.maximum(cmass.sum(1), 1e-12))

    return {
        "peak_dB": peak,
        "headroom_dB": -10 * np.log10(np.maximum(floor, 1e-30)),
        "peak_range_bin": prow.astype(float),
        "peak_doppler_bin": pcol.astype(float),
        "range_centroid": rcent,
        "doppler_centroid": ccent,
        "range_spread": rspread,
        "doppler_spread": cspread,
        "cells_within_10dB": (linf > 10 ** -1.0).sum(axis=1).astype(float),
        "cells_within_20dB": (linf > 10 ** -2.0).sum(axis=1).astype(float),
        "dynamic_range_dB": peak - flat.min(axis=1),
        "mean_dB": flat.mean(axis=1),
    }


def profile(X, y, unit, target):
    feats = sample_features(X)
    tcls = target.split("/")[0]
    tmask = unit == target
    omask = (y == tcls) & ~tmask
    print("=" * 76)
    print("[A] DESCRIPTIVE PROFILE  %s  (n=%d) vs other %s sessions (n=%d)"
          % (target, int(tmask.sum()), tcls, int(omask.sum())))
    print("=" * 76)
    print("  %-20s %12s %12s %10s   %s"
          % ("property", target.split("/")[-1], "others", "z", ""))
    out = {}
    for name, v in feats.items():
        a, b = v[tmask], v[omask]
        z = (a.mean() - b.mean()) / (b.std() + 1e-9)
        flag = "  <== OUTLIER" if abs(z) > 2.0 else ""
        print("  %-20s %12.2f %12.2f %10.2f%s"
              % (name, a.mean(), b.mean(), z, flag))
        out[name] = {"target": float(a.mean()), "others": float(b.mean()),
                     "z": float(z)}
    big = sorted(out, key=lambda k: -abs(out[k]["z"]))[:3]
    print("\n  Most distinguishing: %s" % ", ".join(
        "%s (z=%+.1f)" % (k, out[k]["z"]) for k in big))
    return out


# ------------------------------------------------------------------ B

def nearest_class(X, y, unit, target):
    """Which class centroid do the target's samples sit nearest?"""
    Xn = normalise(X, "offset")[:, 0].reshape(len(X), -1)
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
    tmask = unit == target
    cents = {}
    for c in CLASSES:
        m = (y == c) & ~tmask
        v = Xn[m].mean(axis=0)
        cents[c] = v / (np.linalg.norm(v) + 1e-9)
    M = np.stack([cents[c] for c in CLASSES])
    sim = Xn[tmask] @ M.T
    win = sim.argmax(axis=1)
    print("\n" + "=" * 76)
    print("[B] NEAREST CLASS CENTROID for %s samples" % target)
    print("=" * 76)
    for i, c in enumerate(CLASSES):
        print("    %-8s mean similarity %+.4f   nearest for %5.1f%% of samples"
              % (c, sim[:, i].mean(), 100.0 * (win == i).mean()))
    tc = target.split("/")[0]
    if CLASSES[int(np.bincount(win, minlength=3).argmax())] != tc:
        print("    The samples sit closer to a DIFFERENT class centroid than")
        print("    their own label. Worth checking the labelling of this")
        print("    recording before building any story on it.")
    return {CLASSES[i]: {"mean_sim": float(sim[:, i].mean()),
                         "frac_nearest": float((win == i).mean())}
            for i in range(3)}


# ------------------------------------------------------------------ C

def train_eval(Xn, yi, tr, va, te, seed, epochs):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])
    Xte = torch.tensor(Xn[te])
    model = make_model(torch, nn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=epochs * max(1, len(Xtr) // 128 + 1))
    lossf = nn.CrossEntropyLoss()
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
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
            state = {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pt = [model(Xte[i:i + 512].to(dev)).argmax(1).cpu()
              for i in range(0, len(Xte), 512)]
    return torch.cat(pt).numpy()


def coverage_test(X, y, unit, target, seeds, epochs):
    """
    Held-out: target entirely excluded from training (the k-fold situation).
    Included: half of target in training, evaluate on the other half.

    A large recovery means the model lacked comparable training data, which
    is a coverage finding. Little recovery means the recording is
    intrinsically hard or mislabelled.
    """
    Xn = normalise(X, "offset")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])
    tidx = np.where(unit == target)[0]

    print("\n" + "=" * 76)
    print("[C] COVERAGE TEST for %s" % target)
    print("=" * 76)
    res = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tidx)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]

        others = np.where(unit != target)[0]
        rng.shuffle(others)
        n_val = int(0.15 * len(others))
        val_o, tr_o = others[:n_val], others[n_val:]

        for name, extra in (("held out", np.array([], dtype=int)),
                            ("half included", half_a)):
            tr = np.zeros(len(y), dtype=bool)
            tr[tr_o] = True
            tr[extra] = True
            va = np.zeros(len(y), dtype=bool)
            va[val_o] = True
            te = np.zeros(len(y), dtype=bool)
            te[half_b] = True

            pred = train_eval(Xn, yi, tr, va, te, seed, epochs)
            true = yi[te]
            rec = float((pred == true).mean())
            dist = {CLASSES[i]: int((pred == i).sum()) for i in range(3)}
            res.setdefault(name, []).append(rec)
            print("    seed %d  %-14s recall on held-back half: %.4f   %s"
                  % (seed, name, rec, dist))

    a = float(np.mean(res["held out"]))
    b = float(np.mean(res["half included"]))
    print("\n    held out      %.4f" % a)
    print("    half included %.4f   (recovery %+.4f)" % (b, b - a))
    print("")
    if b - a > 0.30:
        print("    Large recovery. The model simply had no comparable training")
        print("    data for this recording. This is a DATASET COVERAGE")
        print("    finding: one flight regime out of twenty-one is effectively")
        print("    unrepresented, and standard evaluation hides it entirely.")
    elif b - a > 0.10:
        print("    Partial recovery. Coverage explains some of it, but the")
        print("    recording is also harder than the rest on its own terms.")
    else:
        print("    Little recovery. Not a coverage problem. The recording is")
        print("    intrinsically hard or mislabelled -- check [A] and [B]")
        print("    before drawing any conclusion from it.")
    return {"held_out": a, "half_included": b, "recovery": b - a,
            "runs": {k: [float(x) for x in v] for k, v in res.items()}}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="drone/13-48")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    if not (unit == args.target).any():
        avail = sorted({u for u in unit.tolist()
                        if u.startswith(args.target.split("/")[0])})
        raise SystemExit("target not found. available: %s" % avail)

    rep = {"target": args.target}
    rep["profile"] = profile(X, y, unit, args.target)
    rep["nearest"] = nearest_class(X, y, unit, args.target)
    rep["coverage"] = coverage_test(X, y, unit, args.target,
                                    args.seeds, args.epochs)

    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
