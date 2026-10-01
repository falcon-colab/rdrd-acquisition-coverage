"""
Step 2: build the model dataset, train the compact CNN, reproduce the
published error structure, and compare grouped against random splitting.

Two questions this answers, in order:

  Q1  Does a compact CNN on RDRD reproduce the published error structure?
      Specifically: is drone PRECISION below drone RECALL, and is
      vehicle-called-drone the dominant confusion? If not, the framing
      built on that structure needs revisiting before anything else.

  Q2  How much accuracy is lost moving from a random split to a
      unit-grouped split? The similarity screen in Step 1 said the gap
      should be small. A network can exploit cues that screen cannot see,
      so this is the measurement that decides it.

Usage:
    python step2.py build   --root RAW --splits DIR --out DIR
    python step2.py train   --data DIR/dataset.npz --split grouped --seed 0
    python step2.py compare --data DIR/dataset.npz --seeds 0 1 2 3 4
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdrd

CLASSES = ["car", "drone", "person"]
VAL_FRAC = 0.15


# ----------------------------------------------------------------- build

def build(args):
    grouped = json.load(open(os.path.join(args.splits, "split_grouped.json")))
    randomd = json.load(open(os.path.join(args.splits, "split_random.json")))

    side_g, side_r = {}, {}
    for tag, sp in (("train", grouped["train"]), ("test", grouped["test"])):
        for p in sp:
            side_g[p] = tag
    for tag, sp in (("train", randomd["train"]), ("test", randomd["test"])):
        for p in sp:
            side_r[p] = tag

    paths = sorted(side_g)
    print("loading %d samples..." % len(paths))
    X, y, unit, keep = [], [], [], []
    for i, p in enumerate(paths):
        try:
            m = rdrd.load_matrix(p)
        except ValueError:
            continue
        lab = None
        for part in p.split(os.sep):
            g = rdrd.class_of(part)
            if g:
                lab = g
        if lab is None:
            continue
        X.append(m)
        y.append(lab)
        unit.append(lab + "/" + rdrd.timestamp_of(os.path.dirname(p)))
        keep.append(p)
        if (i + 1) % 4000 == 0:
            print("  %d" % (i + 1))

    shapes = Counter(m.shape for m in X)
    dom = shapes.most_common(1)[0][0]
    sel = [i for i, m in enumerate(X) if m.shape == dom]
    print("shapes %s -> keeping %d" % (dict(shapes), len(sel)))

    X = np.stack([X[i] for i in sel]).astype(np.float32)
    y = np.array([y[i] for i in sel])
    unit = np.array([unit[i] for i in sel])
    keep = [keep[i] for i in sel]

    # grouped: carve a validation set out of TRAIN UNITS, never train frames
    rng = np.random.default_rng(12345)
    grp = np.array([side_g[p] for p in keep], dtype=object)
    for c in CLASSES:
        tr_units = sorted({unit[i] for i in range(len(keep))
                           if y[i] == c and grp[i] == "train"})
        rng.shuffle(tr_units)
        n_val = max(1, int(round(VAL_FRAC * len(tr_units))))
        val_units = set(tr_units[:n_val])
        for i in range(len(keep)):
            if grp[i] == "train" and unit[i] in val_units:
                grp[i] = "val"

    rnd = np.array([side_r[p] for p in keep], dtype=object)
    idx = np.where(rnd == "train")[0]
    rng.shuffle(idx)
    rnd[idx[:int(round(VAL_FRAC * len(idx)))]] = "val"

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "dataset.npz")
    np.savez_compressed(out, X=X, y=y, unit=unit,
                        grouped=grp.astype(str), random=rnd.astype(str),
                        path=np.array(keep))
    print("\nwritten %s   X=%s" % (out, X.shape))
    for name, arr in (("grouped", grp), ("random", rnd)):
        print("  %-8s %s" % (name, dict(Counter(arr.tolist()))))
    print("  classes %s" % dict(Counter(y.tolist())))
    # sanity: no unit may span grouped train/val/test
    span = 0
    for u in set(unit.tolist()):
        s = {grp[i] for i in range(len(unit)) if unit[i] == u}
        if len(s) > 1:
            span += 1
    print("  units spanning grouped subsets: %d (must be 0)" % span)


# ----------------------------------------------------------------- model

def make_model(torch, nn, widths=(64, 128, 208, 288), n_class=3):
    """
    Compact depthwise-separable CNN, ~101k parameters.

    Sized to match the published RangeDopplerNet Type-2 reference on this
    dataset (101,419 parameters) so that its compression behaviour is
    representative of a model someone would actually deploy. A reviewer
    will ask why this architecture stands in for deployed models; matching
    the reference size is the answer.
    """
    class Sep(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
            self.pw = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)
        def forward(self, x):
            return nn.functional.relu(self.bn(self.pw(self.dw(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = widths
            self.stem = nn.Sequential(
                nn.Conv2d(1, w[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
            self.b1 = Sep(w[0], w[1], (1, 2))
            self.b2 = Sep(w[1], w[2], (2, 2))
            self.b3 = Sep(w[2], w[3], (1, 2))
            self.head = nn.Linear(w[3], n_class)
            self.drop = nn.Dropout(0.2)
        def forward(self, x):
            x = self.b3(self.b2(self.b1(self.stem(x))))
            x = x.mean(dim=(2, 3))
            return self.head(self.drop(x))
    return Net()


def normalise(X):
    """Peak-referenced: peak becomes 0 dB, scaled by 60 dB of headroom."""
    peak = X.reshape(len(X), -1).max(axis=1)[:, None, None]
    return ((X - peak) / 60.0).astype(np.float32)


# ----------------------------------------------------------------- train

def run_training(data, split, seed, epochs=40, bs=128, lr=3e-3, quiet=False):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = normalise(data["X"])
    y = data["y"]
    sub = data[split]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    yi = np.array([cls_idx[c] for c in y])

    def tensors(tag):
        m = sub == tag
        return (torch.tensor(X[m]).unsqueeze(1), torch.tensor(yi[m]))

    Xtr, ytr = tensors("train")
    Xva, yva = tensors("val")
    Xte, yte = tensors("test")

    model = make_model(torch, nn).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    if not quiet:
        print("  device %s | params %d | train %d val %d test %d"
              % (dev, n_par, len(Xtr), len(Xva), len(Xte)))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(Xtr) // bs + 1))
    lossf = nn.CrossEntropyLoss()

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            xb, yb = Xtr[j].to(dev), ytr[j].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(Xva), 512):
                pv.append(model(Xva[i:i + 512].to(dev)).argmax(1).cpu())
            acc = (torch.cat(pv) == yva).float().mean().item() if len(Xva) else 0.0
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pt = []
        for i in range(0, len(Xte), 512):
            pt.append(model(Xte[i:i + 512].to(dev)).argmax(1).cpu())
        pred = torch.cat(pt).numpy()
    true = yte.numpy()

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        rec = tp / max(1, cm[i].sum())
        pre = tp / max(1, cm[:, i].sum())
        per[c] = {"recall": float(rec), "precision": float(pre),
                  "f1": float(2 * rec * pre / max(1e-9, rec + pre))}
    return {"accuracy": float((pred == true).mean()), "per_class": per,
            "confusion": cm.tolist(), "val_acc": float(best),
            "params": int(n_par)}


def print_result(name, r):
    print("\n  %s   accuracy %.4f  (val %.4f, %d params)"
          % (name, r["accuracy"], r["val_acc"], r["params"]))
    print("    %-8s %9s %10s %8s" % ("class", "recall", "precision", "F1"))
    for c in CLASSES:
        p = r["per_class"][c]
        print("    %-8s %9.4f %10.4f %8.4f"
              % (c, p["recall"], p["precision"], p["f1"]))
    cm = np.array(r["confusion"])
    print("    confusion (rows true, cols predicted): %s" % CLASSES)
    for i, c in enumerate(CLASSES):
        print("      %-8s %s" % (c, "  ".join("%6d" % v for v in cm[i])))


def check_structure(r):
    """Does this reproduce the published error structure?"""
    d = r["per_class"]["drone"]
    cm = np.array(r["confusion"])
    ci, di = CLASSES.index("car"), CLASSES.index("drone")
    car_to_drone, drone_to_car = int(cm[ci, di]), int(cm[di, ci])
    f1s = {c: r["per_class"][c]["f1"] for c in CLASSES}
    weakest = min(f1s, key=f1s.get)
    print("\n  Published-structure check")
    print("    drone precision below drone recall : %s  (%.4f vs %.4f)"
          % (d["precision"] < d["recall"], d["precision"], d["recall"]))
    print("    car->drone exceeds drone->car      : %s  (%d vs %d)"
          % (car_to_drone > drone_to_car, car_to_drone, drone_to_car))
    print("    weakest class by F1                : %s" % weakest)
    return {"drone_prec_below_rec": bool(d["precision"] < d["recall"]),
            "car_to_drone": car_to_drone, "drone_to_car": drone_to_car,
            "weakest_f1": weakest}


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--splits", required=True)
    b.add_argument("--out", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--split", default="grouped", choices=["grouped", "random"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=40)

    c = sub.add_parser("compare")
    c.add_argument("--data", required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    c.add_argument("--epochs", type=int, default=40)
    c.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "build":
        build(args)
        return

    data = np.load(args.data, allow_pickle=False)

    if args.cmd == "train":
        r = run_training(data, args.split, args.seed, args.epochs)
        print_result("%s split, seed %d" % (args.split, args.seed), r)
        check_structure(r)
        return

    # compare
    res = defaultdict(list)
    for split in ("random", "grouped"):
        for s in args.seeds:
            print("\n[%s split, seed %d]" % (split, s))
            r = run_training(data, split, s, args.epochs, quiet=(s != args.seeds[0]))
            res[split].append(r)
            print("    acc %.4f  drone F1 %.4f"
                  % (r["accuracy"], r["per_class"]["drone"]["f1"]))

    print("\n" + "=" * 66)
    print("SPLIT COMPARISON  (mean +/- sd over %d seeds)" % len(args.seeds))
    print("=" * 66)
    print("  %-9s %16s %16s %16s"
          % ("split", "accuracy", "drone F1", "drone precision"))
    summ = {}
    for split in ("random", "grouped"):
        a = np.array([r["accuracy"] for r in res[split]])
        f = np.array([r["per_class"]["drone"]["f1"] for r in res[split]])
        p = np.array([r["per_class"]["drone"]["precision"] for r in res[split]])
        summ[split] = {"acc": [float(a.mean()), float(a.std())],
                       "drone_f1": [float(f.mean()), float(f.std())],
                       "drone_prec": [float(p.mean()), float(p.std())]}
        print("  %-9s %8.4f +/-%.4f %8.4f +/-%.4f %8.4f +/-%.4f"
              % (split, a.mean(), a.std(), f.mean(), f.std(), p.mean(), p.std()))
    drop = summ["random"]["acc"][0] - summ["grouped"]["acc"][0]
    print("\n  accuracy lost moving to grouped splitting: %+.4f" % (-drop))
    if drop > 0.03:
        print("  A real leakage effect the Step 1 screen did not detect.")
    else:
        print("  Consistent with the Step 1 screen: little leakage on RDRD.")

    print_result("grouped, seed %d" % args.seeds[0], res["grouped"][0])
    st = check_structure(res["grouped"][0])

    if args.out:
        json.dump({"summary": summ, "structure": st,
                   "runs": {k: v for k, v in res.items()}},
                  open(args.out, "w"), indent=2)
        print("\n  written %s" % args.out)


if __name__ == "__main__":
    main()
