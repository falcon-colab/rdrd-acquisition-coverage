"""Why is our accuracy six points below the published baseline?

The manuscript reports 0.9335 accuracy under random partitioning. The
published baselines on this dataset report 99.48 and 98.08 percent under the
same protocol. The Limitations section names that gap and says we cannot
attribute it. A reviewer's obvious reply is that we may have found a failure
mode of a weaker model rather than a property of the benchmark, and until the
gap is explained that reply stands.

This sweeps the two cheapest explanations, model capacity and training
length, under the random protocol the baselines used. It answers one
question: can this architecture reach published accuracy at all?

  If accuracy climbs toward 0.98 with more capacity or more epochs, the gap
  is ours and the next step is to rerun the coverage experiment with the
  better configuration. A classifier at 0.98 that still fails on 13-48 would
  remove the objection entirely, which is worth more to this paper than any
  other experiment available.

  If accuracy plateaus near 0.93 regardless, the gap is not capacity or
  schedule. It then lies in preprocessing, in the published protocol, or in
  the architecture itself, and the honest report is that we measured it and
  could not close it. That is a stronger Limitations paragraph than the
  current one, which only concedes the gap.

Either outcome is reportable. Neither is assumed here.

    python src/capacity.py --data DIR/dataset.npz \\
        --out DIR/reports/capacity.json

Runtime is the product of the grid: six configurations at two seeds and 80
epochs is roughly twelve times one training run in the main pipeline.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model                      # noqa: E402
from step2b import normalise                               # noqa: E402


def fit_eval(Xn, yi, tr, va, te, widths, epochs, seed, lr=3e-3, bs=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])
    Xte = torch.tensor(Xn[te]); yte = torch.tensor(yi[te])

    model = make_model(torch, nn, widths=widths).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss()

    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(Xtr), bs):
            idx = perm[k:k + bs]
            xb = Xtr[idx].to(dev); yb = ytr[idx].to(dev)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for k in range(0, len(Xva), 512):
                pv.append(model(Xva[k:k + 512].to(dev)).argmax(1).cpu())
            acc = (torch.cat(pv) == yva).float().mean().item()
        if acc > best:
            best = acc
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pt = []
        for k in range(0, len(Xte), 512):
            pt.append(model(Xte[k:k + 512].to(dev)).argmax(1).cpu())
        pred = torch.cat(pt)
    return (pred == yte).float().mean().item(), best, n_param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, nargs="+", default=[40, 80])
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="width multipliers on the paper's (64,128,208,288)")
    ap.add_argument("--scheme", default="offset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, rnd = d["X"], d["y"], d["random"]
    Xn = normalise(X, args.scheme)
    yi = np.array([CLASSES.index(c) for c in y])

    tr = np.where(rnd == "train")[0]
    va = np.where(rnd == "val")[0]
    te = np.where(rnd == "test")[0]
    print("random protocol: %d train, %d val, %d test" % (len(tr), len(va),
                                                          len(te)))
    print("the published baselines report 0.9948 and 0.9808 on this data; the")
    print("manuscript reports 0.9335 under this same protocol\n")

    base = (64, 128, 208, 288)
    rows = []
    for scale in args.scale:
        widths = tuple(int(round(w * scale)) for w in base)
        for epochs in args.epochs:
            accs, params = [], None
            for s in args.seeds:
                acc, vacc, n_param = fit_eval(Xn, yi, tr, va, te, widths,
                                              epochs, s)
                accs.append(acc); params = n_param
            m = float(np.mean(accs))
            sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            rows.append({"scale": scale, "widths": list(widths),
                         "epochs": epochs, "params": params,
                         "accuracy": m, "accuracy_sd": sd,
                         "runs": [float(a) for a in accs]})
            print("  scale %.1f  widths %-22s %7d par  %3d ep   acc %.4f +/- %.4f"
                  % (scale, str(widths), params, epochs, m, sd))

    best = max(rows, key=lambda r: r["accuracy"])
    print("\n" + "=" * 70)
    print("best %.4f at scale %.1f, %d epochs, %d parameters"
          % (best["accuracy"], best["scale"], best["epochs"], best["params"]))
    print("manuscript                        0.9335")
    print("published baselines               0.9948 and 0.9808")
    print("=" * 70)

    if best["accuracy"] >= 0.97:
        print("\n  The gap closes with capacity or schedule. The next step is to")
        print("  rerun the coverage experiment at this configuration. If the")
        print("  failure on 13-48 survives at published accuracy, the strongest")
        print("  objection to this paper disappears.")
    elif best["accuracy"] >= 0.9335 + 0.02:
        print("\n  Partial. Accuracy improves but does not reach the published")
        print("  figures, so capacity and schedule explain some of the gap and")
        print("  not all of it. Report the measured ceiling rather than")
        print("  speculating about the remainder.")
    else:
        print("\n  The gap is not capacity or schedule: more of both does not")
        print("  help. It lies in preprocessing, in the published protocol, or")
        print("  in the architecture. Report that this was measured and not")
        print("  closed, which is a stronger statement than conceding the gap")
        print("  untested.")

    if args.out:
        json.dump({"rows": rows, "best": best,
                   "manuscript_accuracy": 0.9335,
                   "published": [0.9948, 0.9808]},
                  open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
