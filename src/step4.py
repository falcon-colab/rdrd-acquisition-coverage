"""
Step 4: confirm the two Step 3 findings, and test the mechanism directly.

Step 3 produced two candidate results on two seeds each:

  1. Moderate Doppler aggregation partially rescues the unseen regime.
     Mean unseen-regime recall by aggregation (excluding the unstable
     4-bit column): 1x 0.20, 2x 0.35, 4x 0.53, 8x 0.19. An inverted U with
     a clear peak at 4x, and the gap halving from 0.64 to 0.35.

  2. Input and weight quantisation interact superadditively. At 4 bits and
     4x: input only -0.024, weights only -0.035, both -0.203.

Two seeds is not enough for either. Visible noise: 2x/8-bit unseen recall
0.44 against 2x/16-bit 0.25, an ordering with no reason to be real.

This does three things:

  seeds     Re-runs the aggregation axis at stable bit widths with more
            seeds, so we learn which effects survive.

  augment   THE MECHANISM TEST. If 4x aggregation helps because it makes
            the classifier less sensitive to Doppler POSITION, then
            augmenting training with Doppler shifts should produce the same
            rescue at full resolution -- and at no cost in resolution. If it
            does, the mechanism is confirmed AND you have a practical
            mitigation, which is what the proposal reviewers asked for. If
            it does not, the aggregation effect is something else.

  ablate    The quantisation interaction with more seeds.

Usage:
    python step4.py seeds   --data D --seeds 0 1 2 3 4
    python step4.py augment --data D --seeds 0 1 2 3 4
    python step4.py ablate  --data D --seeds 0 1 2 3 4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES
from step2b import normalise
from step3 import (reduce_doppler, quantise, quantise_weights, fit, evaluate,
                   standard_masks, unseen_masks, TARGET)


# ------------------------------------------------------- augmentation

def doppler_shift(X, max_shift, rng):
    """
    Shift each sample along the Doppler axis by a random amount, filling
    vacated cells with that sample's own noise floor.

    Deliberately NOT a circular roll: wrapping would move target energy from
    one Doppler edge to the other, which is not a physical operation and
    would teach the network something false.
    """
    n, H, W = X.shape
    out = np.empty_like(X)
    floor = np.median(X.reshape(n, -1), axis=1)
    shifts = rng.integers(-max_shift, max_shift + 1, n)
    for i in range(n):
        s = int(shifts[i])
        out[i] = floor[i]
        if s == 0:
            out[i] = X[i]
        elif s > 0:
            out[i, :, s:] = X[i, :, :W - s]
        else:
            out[i, :, :W + s] = X[i, :, -s:]
    return out


def fit_augmented(Xn, yi, tr, va, seed, epochs, X_raw, max_shift,
                  scheme="offset", bs=128, lr=3e-3):
    """Retrain with fresh Doppler shifts each epoch."""
    import torch
    import torch.nn as nn
    from step2 import make_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    idx_tr = np.where(tr)[0]
    ytr = torch.tensor(yi[idx_tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])

    model = make_model(torch, nn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(idx_tr) // bs + 1))
    lossf = nn.CrossEntropyLoss()

    best, state, bad = -1.0, None, 0
    for _ep in range(epochs):
        Xa = normalise(doppler_shift(X_raw[idx_tr], max_shift, rng), scheme)
        Xt = torch.tensor(Xa)
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xt[j].to(dev)), ytr[j].to(dev)).backward()
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


# ------------------------------------------------------------------ helpers

def stat(v):
    a = np.asarray(v, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def welch(a, b):
    """Kept for the factorial block below, where it is labelled as such.

    It is NOT the right test for the aggregation comparison: every condition
    there is trained at the same seeds, and the seed fixes the partition and
    the initialisation together, so the two samples are paired rather than
    independent. Use paired() for that. An earlier version of this script
    reported Welch for the paired comparison and the manuscript quoted it.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def paired(a, b):
    """Per-seed differences and the one-sample t on them, a minus b.

    Returns (mean difference, t, number of seeds where a exceeds b, n).
    The count is reported because it needs no distributional assumption and
    is the form the manuscript leads with.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    if n < 2:
        return float("nan"), float("nan"), 0, n
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")
    return float(d.mean()), t, int((d > 0).sum()), n


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("seeds", "augment", "ablate"):
        p = sub.add_parser(nm)
        p.add_argument("--data", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--out", default=None)
        if nm == "augment":
            p.add_argument("--shifts", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])

    # ------------------------------------------------------------ seeds
    if args.cmd == "seeds":
        print("=" * 74)
        print("AGGREGATION AXIS at 32 and 8 bits, %d seeds" % len(args.seeds))
        print("(the 4-bit column was unstable in Step 3 and is excluded)")
        print("=" * 74)
        out = {}
        for bits in (32, 8):
            print("\n  %d-bit" % bits)
            print("    %-6s %22s %22s %10s"
                  % ("factor", "standard drone recall", "unseen recall", "gap"))
            for f in (1, 2, 4, 8):
                st, un = [], []
                for s in args.seeds:
                    Xr = reduce_doppler(X, f)
                    if bits < 32:
                        Xr = quantise(Xr, bits)
                    Xn = normalise(Xr, "offset")
                    tr, va, te = standard_masks(y, unit, s)
                    m = fit(Xn, yi, tr, va, s, args.epochs)
                    quantise_weights(m, bits)
                    st.append(evaluate(m, Xn, yi, te)["drone"]["recall"])
                    tr, va, te = unseen_masks(y, unit, s)
                    m = fit(Xn, yi, tr, va, s, args.epochs)
                    quantise_weights(m, bits)
                    un.append(evaluate(m, Xn, yi, te)["drone"]["recall"])
                sm, ss = stat(st); um, us = stat(un)
                out["%dx_%dbit" % (f, bits)] = {
                    "standard": [sm, ss], "unseen": [um, us], "gap": sm - um,
                    "standard_runs": st, "unseen_runs": un}
                print("    %-6s %11.4f +/-%.4f %11.4f +/-%.4f %10.4f"
                      % ("%dx" % f, sm, ss, um, us, sm - um))
        print("\n  Does the 4x peak survive? Compare 4x unseen against 1x.")
        print("  Paired by seed: each seed fixes the split and the init, so")
        print("  the two conditions differ only in the input representation.")
        for bits in (32, 8):
            a = out["4x_%dbit" % bits]["unseen_runs"]
            b = out["1x_%dbit" % bits]["unseen_runs"]
            md, t, npos, n = paired(a, b)
            out["4x_%dbit" % bits]["paired_vs_1x_unseen"] = {
                "mean_diff": md, "paired_t": t, "n_positive": npos, "n": n}
            print("    %d-bit: %.4f vs %.4f   mean diff %+.4f   paired t %+.2f"
                  "   positive on %d of %d seeds"
                  % (bits, np.mean(a), np.mean(b), md, t, npos, n))
        for bits in (32, 8):
            a = out["4x_%dbit" % bits]["standard_runs"]
            b = out["1x_%dbit" % bits]["standard_runs"]
            md, t, npos, n = paired(a, b)
            out["4x_%dbit" % bits]["paired_vs_1x_standard"] = {
                "mean_diff": md, "paired_t": t, "n_positive": npos, "n": n}
            print("    %d-bit standard split: mean diff %+.4f  paired t %+.2f"
                  % (bits, md, t))
        print("    With ten seeds the two-sided critical value is 2.262 on")
        print("    nine degrees of freedom.")
        if args.out:
            json.dump(out, open(args.out, "w"), indent=2)
        return

    # ------------------------------------------------------------ augment
    if args.cmd == "augment":
        print("=" * 74)
        print("MECHANISM TEST: does Doppler-shift augmentation reproduce the")
        print("rescue that 4x aggregation gave, at FULL resolution?")
        print("=" * 74)
        print("\nIf yes, the 4x effect is about positional insensitivity and")
        print("you have a mitigation that costs no resolution. If no, the")
        print("aggregation effect is something else.\n")

        rows, out = [], {}

        # baselines
        for name, factor in (("1x, no augmentation", 1),
                             ("4x aggregation", 4)):
            un = []
            for s in args.seeds:
                Xn = normalise(reduce_doppler(X, factor), "offset")
                tr, va, te = unseen_masks(y, unit, s)
                m = fit(Xn, yi, tr, va, s, args.epochs)
                un.append(evaluate(m, Xn, yi, te)["drone"]["recall"])
            m_, sd_ = stat(un)
            rows.append((name, m_, sd_, un))
            print("  %-28s unseen recall %.4f +/- %.4f" % (name, m_, sd_))
            out[name] = {"mean": m_, "sd": sd_, "runs": un}

        # augmented, full resolution
        for k in args.shifts:
            un, stdr = [], []
            for s in args.seeds:
                Xn = normalise(X, "offset")
                tr, va, te = unseen_masks(y, unit, s)
                m = fit_augmented(Xn, yi, tr, va, s, args.epochs, X, k)
                un.append(evaluate(m, Xn, yi, te)["drone"]["recall"])
                tr, va, te = standard_masks(y, unit, s)
                m = fit_augmented(Xn, yi, tr, va, s, args.epochs, X, k)
                stdr.append(evaluate(m, Xn, yi, te)["drone"]["recall"])
            m_, sd_ = stat(un); sm, ss = stat(stdr)
            name = "1x + shift +/-%d bins" % k
            rows.append((name, m_, sd_, un))
            out[name] = {"mean": m_, "sd": sd_, "runs": un,
                         "standard_mean": sm, "standard_sd": ss}
            print("  %-28s unseen recall %.4f +/- %.4f   (standard %.4f)"
                  % (name, m_, sd_, sm))

        base = rows[0][3]
        agg = rows[1][3]
        print("\n  Against the 1x baseline:")
        for name, m_, sd_, runs in rows[1:]:
            print("    %-28s %+.4f   Welch t = %+.2f"
                  % (name, m_ - np.mean(base), welch(runs, base)))
        best = max(rows[2:], key=lambda r: r[1]) if len(rows) > 2 else None
        print("")
        if best and best[1] > np.mean(base) + 0.10:
            print("  Augmentation rescues the unseen regime at full resolution.")
            print("  The 4x aggregation effect is about Doppler positional")
            print("  insensitivity, and shift augmentation is the better")
            print("  mitigation because it costs no resolution. This is the")
            print("  result to lead the paper's mitigation section with.")
            if best[1] > np.mean(agg):
                print("  It also BEATS 4x aggregation (%.4f vs %.4f)."
                      % (best[1], np.mean(agg)))
        else:
            print("  Augmentation does not reproduce the rescue. Whatever 4x")
            print("  aggregation is doing, it is not simply making the model")
            print("  insensitive to Doppler position. Report the aggregation")
            print("  effect empirically and do not claim the mechanism.")
        if args.out:
            json.dump(out, open(args.out, "w"), indent=2)
        return

    # ------------------------------------------------------------ ablate
    print("=" * 74)
    print("QUANTISATION INTERACTION at 4 bits, 4x, %d seeds" % len(args.seeds))
    print("=" * 74)
    from step3 import run_cell
    out = {}
    for qi, qw, name in ((False, False, "neither"), (True, False, "input only"),
                         (False, True, "weights only"), (True, True, "both")):
        accs, f1s = [], []
        for s in args.seeds:
            tr, va, te = standard_masks(y, unit, s)
            r = run_cell(X, yi, tr, va, te, 4, 4, s, args.epochs, qi, qw)
            accs.append(r["accuracy"]); f1s.append(r["drone"]["f1"])
        am, asd = stat(accs)
        out[name] = {"accuracy": am, "sd": asd, "runs": accs,
                     "drone_f1": float(np.mean(f1s))}
        print("  %-14s accuracy %.4f +/- %.4f   drone F1 %.4f"
              % (name, am, asd, np.mean(f1s)))
    n = out["neither"]["accuracy"]
    di = n - out["input only"]["accuracy"]
    dw = n - out["weights only"]["accuracy"]
    db = n - out["both"]["accuracy"]
    print("\n  drop from input only      %.4f" % di)
    print("  drop from weights only    %.4f" % dw)
    print("  sum if independent        %.4f" % (di + dw))
    print("  drop from both            %.4f" % db)
    print("  excess over additive      %+.4f" % (db - di - dw))
    print("  Welch t, both vs neither  %+.2f"
          % welch(out["both"]["runs"], out["neither"]["runs"]))
    print("")
    if di > 0 and dw > 0 and db > 1.8 * (di + dw):
        print("  The two axes interact strongly. Neither alone predicts the")
        print("  joint effect, so a compression study that varies one at a")
        print("  time will understate the cost. That is a reportable result")
        print("  and it is the joint-compression question the proposal set")
        print("  out to answer.")
    elif di <= 0 or dw <= 0:
        print("  One axis did not hurt at all on its own, so the additive")
        print("  baseline is not meaningful here. Report the raw numbers and")
        print("  do not claim an interaction.")
    else:
        print("  The axes are close to additive. No interaction to report.")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
