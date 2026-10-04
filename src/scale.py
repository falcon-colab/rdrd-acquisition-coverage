"""Does the 13-48 failure survive a classifier the size of the published one?

This is the last standing objection to the paper, and it is a fair one. The
acquisition-level failure has been demonstrated for a compact network of
about 100k parameters. DopplerNet, which reports 0.9948 on this benchmark,
has 3,818,755 parameters and a genuine three-frame temporal input. A
reviewer can reasonably ask whether the failure is a property of the
benchmark or of a model thirty-seven times smaller than the published one.

The obvious answer would be to reproduce DopplerNet. We do not, and the
reason is worth stating plainly rather than burying. The published
description does not pin down every architectural detail, and the public
release does not preserve the 400 ms frame timing its input depends on, so
anything we built and labelled "DopplerNet" would be a guess wearing someone
else's name. That is the same error this paper criticises elsewhere, and we
decline to commit it.

What can be done honestly is to vary the one quantity the objection is
about. This sweeps model capacity across the range that separates our
network from the published one, roughly 0.1 to 3.8 million parameters, using
the three-frame input throughout, and measures two things at each size:

  random split    does aggregate accuracy climb into the published 0.98 to
                  0.99 band as capacity grows?

  13-48 held out  does the acquisition-level failure go away?

Both outcomes are reportable and neither is assumed.

  If accuracy climbs to the published band while 13-48 recall stays low, the
  objection is answered by measurement: the failure is not explained by
  classifier strength anywhere in the range we could test, including at the
  published model's own parameter count.

  If 13-48 recall rises with capacity, the paper's central claim needs
  qualifying, and the honest report is that the failure is partly a property
  of the compact model. We would then say so and narrow the claim.

  If accuracy plateaus well below 0.98, capacity is not what separates us
  from the published figure either, and the gap lies in preprocessing or in
  protocol details that the publications do not record.

Note what this does NOT establish even in the first case. Capacity is one
axis. A different architecture at the same parameter count could behave
differently, and this sweep cannot exclude that. The claim it supports is
about capacity, not about architecture in general, and the manuscript states
it at that width.

    python src/scale.py --data DIR/dataset.npz \\
        --out DIR/reports/scale.json --seeds 0 1 2 3 4

Runtime grows with the square of the width multiplier. The x6.3 cell is
roughly forty times the cost of the x1 cell, so budget for it: on a single
GPU the full sweep is a few hours rather than a few minutes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model                       # noqa: E402
from step2b import normalise                                # noqa: E402
from step3 import TARGET, evaluate                          # noqa: E402
from temporal import (triplet_rows, companion_rows,         # noqa: E402
                      clean_companion_rows, build_arm,
                      unseen_masks_for, stat)

BASE = (64, 128, 208, 288)
DOPPLERNET_PARAMS = 3818755


def fit_eval(Xn, yi, tr, va, te, seed, epochs, widths, lr=3e-3, bs=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])

    model = make_model(torch, nn, widths=widths).to(dev)
    if Xn.shape[1] != 1:
        old = model.stem[0]
        model.stem[0] = nn.Conv2d(Xn.shape[1], old.out_channels, 3, 1, 1,
                                  bias=False).to(dev)
    n_param = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss()

    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(perm), bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            lossf(model(Xtr[idx].to(dev)), ytr[idx].to(dev)).backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
        model.eval()
        with torch.no_grad():
            pv = [model(Xva[k:k + 512].to(dev)).argmax(1).cpu()
                  for k in range(0, len(Xva), 512)]
            acc = (torch.cat(pv) == yva).float().mean().item()
        if acc > best:
            best, bad = acc, 0
            state = {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    if state is not None:
        model.load_state_dict(state)
    return evaluate(model, Xn, yi, te), n_param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--random-seeds", type=int, default=3,
                    help="how many of --seeds to use for the random split")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--scale", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 6.3],
                    help="width multipliers on (64,128,208,288); 6.3 lands "
                         "within a per cent of DopplerNet's parameter count")
    ap.add_argument("--arm", default="index", choices=("index", "single"))
    ap.add_argument("--scheme", default="offset")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit, rnd = d["X"], d["y"], d["unit"], d["random"]
    if "path" not in d:
        raise SystemExit("this archive has no 'path' array; rebuild with "
                         "step2.py build")
    paths = [str(p) for p in d["path"]]

    Xn1 = normalise(X, args.scheme)
    trips = triplet_rows(paths)
    comps, _ = companion_rows(paths, trips)
    clean, _, _ = clean_companion_rows(paths, trips, rnd)
    centre = np.array([t[1] for t in trips])
    yt, ut, rt = y[centre], unit[centre], rnd[centre]
    yi = np.array([CLASSES.index(c) for c in yt])
    Xa = build_arm(Xn1, trips, comps, args.arm, clean)

    tr_r = np.where(rt == "train")[0]
    va_r = np.where(rt == "val")[0]
    te_r = np.where(rt == "test")[0]

    print("=" * 78)
    print("CAPACITY SWEEP at the published parameter scale, '%s' input"
          % args.arm)
    print("=" * 78)
    print("  DopplerNet reports 0.9948 with %s parameters and a genuine"
          % format(DOPPLERNET_PARAMS, ","))
    print("  three-frame temporal input. We do not reproduce that network;")
    print("  we vary the one axis the objection concerns, which is capacity.")
    print("  triplets %d   random split %d/%d/%d   13-48 triplets %d\n"
          % (len(trips), len(tr_r), len(va_r), len(te_r),
             int((ut == TARGET).sum())))
    print("  %-6s %12s %8s   %-22s %-22s"
          % ("scale", "params", "/Dopp.", "random split", "13-48 held out"))

    out = {"arm": args.arm, "epochs": args.epochs, "seeds": list(args.seeds),
           "doppler_net_params": DOPPLERNET_PARAMS, "cells": {}}

    for m in args.scale:
        widths = tuple(int(round(w * m)) for w in BASE)
        accs, npar = [], 0
        for s in args.seeds[:args.random_seeds]:
            r, npar = fit_eval(Xa, yi, tr_r, va_r, te_r, s, args.epochs, widths)
            accs.append(r["accuracy"])
        am, asd = stat(accs)

        uns = []
        for s in args.seeds:
            tr, va, te = unseen_masks_for(yt, ut, s)
            r, _ = fit_eval(Xa, yi, tr, va, te, s, args.epochs, widths)
            uns.append(r["drone"]["recall"])
        um, usd = stat(uns)

        out["cells"]["x%g" % m] = {
            "widths": list(widths), "params": int(npar),
            "params_ratio_to_dopplernet": npar / float(DOPPLERNET_PARAMS),
            "random_accuracy": am, "random_accuracy_sd": asd,
            "random_runs": accs,
            "unseen_recall": um, "unseen_recall_sd": usd, "unseen_runs": uns}
        print("  x%-5g %12s %7.2fx   %.4f +/- %.4f      %.4f +/- %.4f"
              % (m, format(npar, ","), npar / DOPPLERNET_PARAMS,
                 am, asd, um, usd))

    cells = out["cells"]
    keys = list(cells)
    if len(keys) >= 2:
        lo, hi = cells[keys[0]], cells[keys[-1]]
        print("\n  across %.0fx of capacity, from %s to %s parameters:"
              % (hi["params"] / lo["params"], format(lo["params"], ","),
                 format(hi["params"], ",")))
        print("    random-split accuracy %.4f -> %.4f" %
              (lo["random_accuracy"], hi["random_accuracy"]))
        print("    13-48 held-out recall %.4f -> %.4f" %
              (lo["unseen_recall"], hi["unseen_recall"]))
        best = max(cells.values(), key=lambda c: c["unseen_recall"])
        out["best_unseen_recall"] = best["unseen_recall"]
        out["best_random_accuracy"] = max(c["random_accuracy"]
                                          for c in cells.values())
        print("    best 13-48 recall at any size %.4f" % best["unseen_recall"])
        print("")
        if best["unseen_recall"] < 0.6:
            print("  The failure persists at every capacity tested, including")
            print("  at the published model's own parameter count. The claim")
            print("  the manuscript may then make is about capacity, not")
            print("  about architecture in general.")
        else:
            print("  13-48 recall improves substantially with capacity. The")
            print("  manuscript's claim must be narrowed accordingly, and this")
            print("  is the outcome that would require it.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
