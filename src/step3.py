"""
Step 3: the compression grid.

Framing note. This grid was originally motivated by a physical mechanism
predicting that drones would degrade first. Three such mechanisms were
proposed and all three failed against data (rotor sidebands, cell-count
redundancy, dynamic-range headroom). The grid is therefore run as an
EMPIRICAL AUDIT with no predicted outcome, plus one motivated question that
does come from a supported finding:

    Does compression widen the gap on an UNSEEN FLIGHT REGIME?

Step 2d established that session drone/13-48 is a regime absent from the
other twenty drone recordings: held out, recall is 0.157; given half of it
in training, 0.967. If compression widens that gap, an embedded deployment
is worse on exactly the case standard evaluation already hides.

Two axes:

  resolution  Doppler-axis aggregation at 1x, 2x, 4x, 8x. Aggregation is
              INCOHERENT (power summed across adjacent magnitude bins). It
              is NOT a shorter coherent processing interval -- the dataset
              publishes magnitude only, so the phase needed for that does
              not exist. Reported as "input tensor resolution", never as CPI.

  precision   Uniform quantisation of input and weights to 32/16/8/4 bits.
              Normalisation is 'offset', a fixed affine map, so quantising
              before or after normalisation is equivalent up to scale. That
              is why 'offset' was chosen in Step 2b.

Two evaluations:

  standard    the frozen grouped split
  unseen      drone/13-48 held out of training entirely, evaluated on it

Usage:
    python step3.py grid    --data DIR/dataset.npz --seeds 0 1
    python step3.py ablate  --data DIR/dataset.npz --seeds 0 1
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

TARGET = "drone/13-48"


# ------------------------------------------------------------ compression

def reduce_doppler(X, factor):
    """
    Aggregate `factor` adjacent Doppler bins by summing LINEAR power, then
    return to dB. Summing dB values would be meaningless.

    This is incoherent aggregation. It reduces the size of the input tensor,
    which is a real memory cost on an embedded target. It is not equivalent
    to shortening the coherent processing interval, and must not be
    described as such.
    """
    if factor == 1:
        return X
    n, H, W = X.shape
    keep = (W // factor) * factor
    lin = 10.0 ** (X[:, :, :keep] / 10.0)
    lin = lin.reshape(n, H, keep // factor, factor).sum(axis=3)
    return (10.0 * np.log10(np.maximum(lin, 1e-30))).astype(np.float32)


def quantise(a, bits, lo=None, hi=None):
    """Uniform symmetric quantisation to `bits`, returned dequantised."""
    if bits >= 32:
        return a
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    if hi <= lo:
        return a
    levels = 2 ** bits - 1
    step = (hi - lo) / levels
    return (np.round((np.clip(a, lo, hi) - lo) / step) * step + lo).astype(a.dtype)


def quantise_weights(model, bits):
    """Post-training fake-quant of conv and linear weights, per tensor."""
    import torch
    if bits >= 32:
        return
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                w = m.weight.detach().cpu().numpy()
                m.weight.copy_(torch.tensor(
                    quantise(w, bits, -np.abs(w).max(), np.abs(w).max()),
                    device=m.weight.device))


# ------------------------------------------------------------------ train

def fit(Xn, yi, tr, va, seed, epochs, lr=3e-3, bs=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])
    model = make_model(torch, nn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss()
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
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
            state = {k: v.detach().cpu().clone()
                     for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    return model


def evaluate(model, Xn, yi, mask):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    Xte = torch.tensor(Xn[mask])
    with torch.no_grad():
        pt = [model(Xte[i:i + 512].to(dev)).argmax(1).cpu()
              for i in range(0, len(Xte), 512)]
    pred = torch.cat(pt).numpy()
    true = yi[mask]
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


def run_cell(X, yi, tr, va, te, factor, bits, seed, epochs,
             quant_input=True, quant_weights=True):
    Xr = reduce_doppler(X, factor)
    if quant_input and bits < 32:
        Xr = quantise(Xr, bits)
    Xn = normalise(Xr, "offset")
    model = fit(Xn, yi, tr, va, seed, epochs)
    if quant_weights:
        quantise_weights(model, bits)
    r = evaluate(model, Xn, yi, te)
    r["input_cells"] = int(Xr.shape[1] * Xr.shape[2])
    r["input_bytes"] = float(r["input_cells"] * bits / 8.0)
    return r


# ------------------------------------------------------------------ masks

def standard_masks(y, unit, seed):
    gfold = make_grouped_folds(y, unit, 5)
    return split_for_fold(gfold, 0, 5, y, unit, True, seed)


def unseen_masks(y, unit, seed):
    """13-48 excluded from training and validation; evaluated on it alone."""
    rng = np.random.default_rng(500 + seed)
    te = unit == TARGET
    pool = np.where(~te)[0]
    rng.shuffle(pool)
    n_val = int(0.15 * len(pool))
    va = np.zeros(len(y), dtype=bool); va[pool[:n_val]] = True
    tr = np.zeros(len(y), dtype=bool); tr[pool[n_val:]] = True
    return tr, va, te


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("grid", "ablate"):
        p = sub.add_parser(nm)
        p.add_argument("--data", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--out", default=None)
        p.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4, 8])
        p.add_argument("--bits", type=int, nargs="+", default=[32, 16, 8, 4])
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])

    if args.cmd == "ablate":
        print("=" * 74)
        print("ABLATION at 4 bits, 4x reduction: which axis does the damage?")
        print("=" * 74)
        rows = []
        for qi, qw, name in ((False, False, "neither"),
                             (True, False, "input only"),
                             (False, True, "weights only"),
                             (True, True, "both")):
            accs, dro = [], []
            for s in args.seeds:
                tr, va, te = standard_masks(y, unit, s)
                r = run_cell(X, yi, tr, va, te, 4, 4, s, args.epochs, qi, qw)
                accs.append(r["accuracy"]); dro.append(r["drone"]["f1"])
            rows.append((name, float(np.mean(accs)), float(np.mean(dro))))
            print("  %-14s accuracy %.4f   drone F1 %.4f"
                  % (name, np.mean(accs), np.mean(dro)))
        base = rows[0][1]
        print("\n  Relative to no quantisation (%.4f):" % base)
        for name, a, _f in rows[1:]:
            print("    %-14s %+.4f" % (name, a - base))
        if args.out:
            json.dump([{"setting": r[0], "accuracy": r[1], "drone_f1": r[2]}
                       for r in rows], open(args.out, "w"), indent=2)
        return

    results = defaultdict(dict)
    for setting, maskfn in (("standard", standard_masks),
                            ("unseen", unseen_masks)):
        print("\n" + "=" * 74)
        print("SETTING: %s" % setting.upper())
        if setting == "unseen":
            print("  %s excluded from training; evaluated on it alone." % TARGET)
        print("=" * 74)
        for factor in args.factors:
            for bits in args.bits:
                accs, dro, cells, byts = [], [], None, None
                for s in args.seeds:
                    tr, va, te = maskfn(y, unit, s)
                    r = run_cell(X, yi, tr, va, te, factor, bits, s, args.epochs)
                    accs.append(r["accuracy"]); dro.append(r["drone"]["recall"])
                    cells, byts = r["input_cells"], r["input_bytes"]
                key = "%dx_%dbit" % (factor, bits)
                results[setting][key] = {
                    "factor": factor, "bits": bits,
                    "accuracy": float(np.mean(accs)),
                    "accuracy_sd": float(np.std(accs)),
                    "drone_recall": float(np.mean(dro)),
                    "drone_recall_sd": float(np.std(dro)),
                    "input_cells": cells, "input_bytes": byts}
                print("  %2dx %2d-bit  cells %4d  bytes %7.0f   acc %.4f   drone recall %.4f"
                      % (factor, bits, cells, byts,
                         np.mean(accs), np.mean(dro)))

    print("\n" + "=" * 74)
    print("DRONE RECALL GRID")
    print("=" * 74)
    for setting in ("standard", "unseen"):
        print("\n  %s" % setting)
        print("      %s" % "".join("%9d-bit" % b for b in args.bits))
        for f in args.factors:
            row = "  %2dx" % f
            for b in args.bits:
                row += "%13.4f" % results[setting]["%dx_%dbit" % (f, b)]["drone_recall"]
            print(row)

    print("\n" + "=" * 74)
    print("DOES COMPRESSION WIDEN THE UNSEEN-REGIME GAP?")
    print("=" * 74)
    print("      %s" % "".join("%9d-bit" % b for b in args.bits))
    gaps = {}
    for f in args.factors:
        row = "  %2dx" % f
        for b in args.bits:
            k = "%dx_%dbit" % (f, b)
            g = (results["standard"][k]["drone_recall"]
                 - results["unseen"][k]["drone_recall"])
            gaps[k] = g
            row += "%13.4f" % g
        print(row)
    base = gaps["%dx_%dbit" % (args.factors[0], args.bits[0])]
    worst = gaps["%dx_%dbit" % (args.factors[-1], args.bits[-1])]
    print("\n  gap at %dx/%d-bit (uncompressed): %.4f"
          % (args.factors[0], args.bits[0], base))
    print("  gap at %dx/%d-bit (most compressed): %.4f"
          % (args.factors[-1], args.bits[-1], worst))
    print("  change: %+.4f" % (worst - base))
    print("")
    if worst - base > 0.10:
        print("  Compression widens the unseen-regime gap. An embedded")
        print("  deployment is worse on exactly the case standard evaluation")
        print("  already hides. That is the result to lead this section with.")
    elif worst - base < -0.10:
        print("  Compression NARROWS the gap, which is counterintuitive and")
        print("  most likely means compression is hurting the standard case")
        print("  more than the unseen one. Check the absolute numbers before")
        print("  reading anything into it.")
    else:
        print("  Compression does not measurably change the gap. The coverage")
        print("  failure is independent of compression: report it as a null")
        print("  and keep the coverage finding as the paper's contribution.")

    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
