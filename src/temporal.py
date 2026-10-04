"""Does the failure on 13-48 survive the published input formulation?

The reviewer's central objection to this paper is the strongest one available:
we demonstrate the acquisition-level failure for our own classifier, not for
the published models that report 98 to 99 percent on this benchmark. Until
that is addressed, a reader can reasonably answer "you have found a weakness
of your model, not of the benchmark".

The published baseline does not use the input we use. DopplerNet takes a
tensor the source paper describes as "11x61x3 (distance-doppler-time)": three
time-consecutive range-Doppler frames stacked as channels, 400 ms apart. Our
model sees one frame. That is a real difference in available information and
it is a candidate explanation both for the accuracy gap and for the failure.

There is a problem, and it is the reason this script exists in the shape it
does. The published Kaggle release carries no per-sample timestamp and no
frame counter. Each file is one CFAR-cropped detection. The only ordering
the release preserves is the trailing integer in the file name, and the
loader in this repository has warned since the first commit that file-name
order does not reliably track acquisition order. So the exact 400 ms triplets
cannot be reconstructed by us or by anyone else working from the public data.

What can be done is to build the best approximation the release supports and
then measure whether the approximation carries any temporal information at
all, rather than assuming it. Three arms:

  single    one channel, the centre frame alone. The existing input, on
            exactly the triplet-centre sample set, so it is comparable to
            the other two arms sample for sample rather than approximately.

  index     three channels: the frame before, the centre, the frame after,
            in file-name index order, formed only where the integers are
            actually consecutive. This is the closest available stand-in for
            the published input.

  shuffle   three channels: the centre, flanked by two frames drawn at
            random from the same folder and never the true neighbours. Same
            channel count, same folder, same labels, same sample count. The
            only thing removed is adjacency.

The shuffle arm is the point. It makes the ordering assumption testable
instead of load-bearing:

  If index beats shuffle, file-name order does carry usable temporal
  information, we have partially reconstructed the published input, and the
  13-48 number under that input is the number the reviewer asked for.

  If index and shuffle are indistinguishable, the extra channels are helping
  as within-acquisition context and not as motion, which is direct evidence
  that the release does not preserve the time axis the published model used.
  That is a finding about the benchmark, not a failure of this experiment,
  and it belongs in the paper either way.

  If both three-channel arms leave recall on 13-48 near the single-frame
  figure, the failure is not an artefact of our single-frame input, which is
  the outcome that most strengthens the paper.

All three readings are reportable. None is assumed, and this comment was
written before the script was run.

Normalisation is applied to the single-frame array before stacking. The
offset scheme is a fixed affine map, so normalising before or after stacking
is identical; this is the same property the manuscript relies on when
quantising before or after normalisation.

    python src/temporal.py --data DIR/dataset.npz \\
        --out DIR/reports/temporal.json --seeds 0 1 2 3 4 5 6 7 8 9

Runtime is three arms by two protocols by the seed count. The random-split
protocol runs once per arm because the split is the fixed one stored in the
archive, as in capacity.py.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2 import CLASSES, make_model                      # noqa: E402
from step2b import normalise                               # noqa: E402
from step3 import TARGET, evaluate                         # noqa: E402

NUM_RE = re.compile(r"(\d+)")
ARMS = ("single", "index", "shuffle")


# --------------------------------------------------------------- triplets

def folder_index(paths):
    """{folder: [(integer index, row)]} sorted by the integer, not by string.

    Sorting matters: a lexicographic sort puts 10 before 2, so the npz row
    order cannot be used directly. The integer is the trailing numeric field
    of the base name, which is what the published archive supplies.
    """
    out = defaultdict(list)
    for row, p in enumerate(paths):
        base = os.path.basename(p)
        nums = NUM_RE.findall(base)
        if not nums:
            continue
        out[os.path.dirname(p)].append((int(nums[-1]), row))
    for k in out:
        out[k].sort()
    return out


def triplet_rows(paths):
    """Rows (before, centre, after) where the three indices are consecutive.

    A gap in the numbering means the frames are not adjacent, so no triplet
    is formed across it. This is the only adjacency evidence the release
    carries and the rule is deliberately strict about it.
    """
    trips = []
    for _folder, items in sorted(folder_index(paths).items()):
        for j in range(len(items) - 2):
            (i0, r0), (i1, r1), (i2, r2) = items[j], items[j + 1], items[j + 2]
            if i1 == i0 + 1 and i2 == i1 + 1:
                trips.append((r0, r1, r2))
    return trips


def companion_rows(paths, trips, seed=0):
    """For each triplet, two rows from the same folder that are NOT its
    neighbours. The shuffle arm's flanking frames.

    Folders too small to supply two such rows fall back to sampling from the
    whole folder, excluding the centre. The count of such cases is reported
    so it cannot hide.
    """
    rng = np.random.default_rng(9000 + seed)
    by_folder = {k: [r for _i, r in v]
                 for k, v in folder_index(paths).items()}
    out, fallback = [], 0
    for r0, r1, r2 in trips:
        pool = by_folder[os.path.dirname(paths[r1])]
        cand = [r for r in pool if r not in (r0, r1, r2)]
        if len(cand) < 2:
            cand = [r for r in pool if r != r1]
            fallback += 1
        pick = rng.choice(len(cand), size=2, replace=len(cand) < 2)
        out.append((cand[pick[0]], cand[pick[1]]))
    return out, fallback


def build_arm(Xn1, trips, comps, arm):
    """(M, C, H, W) for one arm. Xn1 is the normalised (N, 1, H, W) array."""
    centre = np.array([t[1] for t in trips])
    if arm == "single":
        return Xn1[centre]
    if arm == "index":
        a = np.array([t[0] for t in trips]); c = np.array([t[2] for t in trips])
    elif arm == "shuffle":
        a = np.array([p[0] for p in comps]); c = np.array([p[1] for p in comps])
    else:
        raise ValueError(arm)
    return np.concatenate([Xn1[a], Xn1[centre], Xn1[c]], axis=1)


# ------------------------------------------------------------------ train

def fit_eval(Xn, yi, tr, va, te, seed, epochs, lr=3e-3, bs=128):
    """Train and evaluate. Follows the stem-replacement pattern in step2b,
    which is how this codebase already feeds a multi-channel input to the
    shared architecture."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr = torch.tensor(Xn[tr]); ytr = torch.tensor(yi[tr])
    Xva = torch.tensor(Xn[va]); yva = torch.tensor(yi[va])

    model = make_model(torch, nn).to(dev)
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


def unseen_masks_for(y, unit, seed):
    """13-48 held out of training and validation, evaluated on it alone.

    Reimplemented on the triplet arrays rather than imported, because the
    triplet sample set is smaller than the frame set and the mask lengths
    must match it.
    """
    rng = np.random.default_rng(500 + seed)
    te = unit == TARGET
    pool = np.where(~te)[0]
    rng.shuffle(pool)
    n_val = int(0.15 * len(pool))
    va = np.zeros(len(y), bool); va[pool[:n_val]] = True
    tr = np.zeros(len(y), bool); tr[pool[n_val:]] = True
    return tr, va, te


def stat(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def audit_overlap(trips, comps, rnd):
    """How much of the test set appears inside training inputs.

    A sliding window over frames plus a frame-level random partition is a
    leakage mechanism, and it is not a subtle one. A triplet centred on a
    training frame carries two companion frames whose own split tags are
    whatever they happen to be, so a training input can contain the pixels of
    a test frame. Because neighbouring detections within one acquisition are
    nearly identical, a test frame whose neighbour was seen in training is
    close to a memorised sample.

    This counts the effect instead of reasoning about it. It needs no GPU and
    no training, and it is reported for both three-channel arms so the
    difference between local and acquisition-wide companions is visible.
    """
    out = {}
    for name, flank in (("index", [(t[0], t[2]) for t in trips]),
                        ("shuffle", list(comps))):
        n_tr_contaminated = 0
        n_train = 0
        reached = set()
        for (r0, r1, r2), (a, c) in zip(trips, flank):
            if rnd[r1] != "train":
                continue
            n_train += 1
            hit = [r for r in (a, c) if rnd[r] == "test"]
            if hit:
                n_tr_contaminated += 1
                reached.update(hit)
        n_test_frames = int((rnd == "test").sum())
        out[name] = {
            "train_triplets": n_train,
            "train_triplets_touching_a_test_frame": n_tr_contaminated,
            "share_of_train_triplets": (n_tr_contaminated / n_train
                                        if n_train else float("nan")),
            "distinct_test_frames_inside_training_inputs": len(reached),
            "share_of_test_frames": (len(reached) / n_test_frames
                                     if n_test_frames else float("nan"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--scheme", default="offset")
    ap.add_argument("--audit", action="store_true",
                    help="report how much of the test set appears inside "
                         "training inputs under the random protocol, then "
                         "exit; no training, no GPU, runs in seconds")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X, y, unit, rnd = d["X"], d["y"], d["unit"], d["random"]
    if "path" not in d:
        raise SystemExit("this archive has no 'path' array, so file-name "
                         "order is not recoverable; rebuild it with step2.py")
    paths = [str(p) for p in d["path"]]

    Xn1 = normalise(X, args.scheme)
    trips = triplet_rows(paths)
    comps, fallback = companion_rows(paths, trips)
    centre = np.array([t[1] for t in trips])
    yt, ut, rt = y[centre], unit[centre], rnd[centre]
    yi = np.array([CLASSES.index(c) for c in yt])

    print("=" * 74)
    print("TEMPORAL INPUT: does the 13-48 failure survive a 3-frame input?")
    print("=" * 74)
    print("  frames in archive          %d" % len(X))
    print("  consecutive-index triplets %d  (%.1f%% of frames)"
          % (len(trips), 100.0 * len(trips) / len(X)))
    print("  folders contributing       %d" % len(folder_index(paths)))
    print("  shuffle-arm fallbacks      %d" % fallback)
    print("  triplets on 13-48          %d" % int((ut == TARGET).sum()))
    print("  published input was 11x61x3 distance-doppler-time, 400 ms apart;")
    print("  the release carries no timestamp, so 'index' is an approximation")
    print("  and 'shuffle' measures whether that approximation carries")
    print("  anything beyond within-folder context.\n")

    out = {"n_frames": int(len(X)), "n_triplets": int(len(trips)),
           "n_triplets_target": int((ut == TARGET).sum()),
           "shuffle_fallbacks": int(fallback),
           "epochs": args.epochs, "seeds": list(args.seeds), "arms": {}}

    over = audit_overlap(trips, comps, rnd)
    out["overlap_audit"] = over
    print("  OVERLAP AUDIT, random protocol. A sliding window plus a")
    print("  frame-level random split puts test frames inside training")
    print("  inputs, so the three-channel accuracies below are inflated and")
    print("  are reported only to establish comparability with the published")
    print("  protocol, never as a measure of generalisation.")
    for name in ("index", "shuffle"):
        o = over[name]
        print("    %-8s %6.1f%% of training triplets contain a test frame;"
              " %6.1f%% of test frames appear in some training input"
              % (name, 100.0 * o["share_of_train_triplets"],
                 100.0 * o["share_of_test_frames"]))
    print("  The single-frame arm cannot leak this way: its input is one")
    print("  frame and that frame carries its own split tag.\n")
    if args.audit:
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                        exist_ok=True)
            json.dump(out, open(args.out, "w"), indent=2)
            print("written %s (audit only, no training)" % args.out)
        return

    tr_r = np.where(rt == "train")[0]
    va_r = np.where(rt == "val")[0]
    te_r = np.where(rt == "test")[0]
    print("  random protocol on triplet centres: %d train %d val %d test"
          % (len(tr_r), len(va_r), len(te_r)))
    print("  published single-partition figures on this data: 0.9948, 0.9808")
    print("  this manuscript, single frame, same split: 0.9537\n")

    for arm in args.arms:
        Xa = build_arm(Xn1, trips, comps, arm)
        rec = {"channels": int(Xa.shape[1])}
        print("  %-8s input %s" % (arm, tuple(Xa.shape[1:])))

        accs = []
        for s in args.seeds[:3]:
            r, npar = fit_eval(Xa, yi, tr_r, va_r, te_r, s, args.epochs)
            accs.append(r["accuracy"])
            rec["params"] = int(npar)
        am, asd = stat(accs)
        rec["random_accuracy"] = am
        rec["random_accuracy_sd"] = asd
        rec["random_runs"] = accs
        print("    random split   accuracy %.4f +/- %.4f  (params %d)"
              % (am, asd, rec["params"]))

        uns = []
        for s in args.seeds:
            tr, va, te = unseen_masks_for(yt, ut, s)
            r, _ = fit_eval(Xa, yi, tr, va, te, s, args.epochs)
            uns.append(r["drone"]["recall"])
        um, usd = stat(uns)
        rec["unseen_recall"] = um
        rec["unseen_recall_sd"] = usd
        rec["unseen_runs"] = uns
        print("    13-48 held out recall %.4f +/- %.4f   min %.4f max %.4f"
              % (um, usd, min(uns), max(uns)))
        out["arms"][arm] = rec

    have = [a for a in ("index", "shuffle", "single") if a in out["arms"]]
    if "index" in have and "shuffle" in have:
        a = np.array(out["arms"]["index"]["unseen_runs"])
        b = np.array(out["arms"]["shuffle"]["unseen_runs"])
        n = min(len(a), len(b))
        dif = a[:n] - b[:n]
        sd_d = dif.std(ddof=1) if n > 1 else 0.0
        t = dif.mean() / (sd_d / np.sqrt(n)) if sd_d > 0 else float("nan")
        out["index_minus_shuffle"] = {
            "mean": float(dif.mean()), "sd": float(sd_d),
            "paired_t": float(t), "n": int(n),
            "per_seed": [float(x) for x in dif]}
        print("\n  adjacency test, index minus shuffle on 13-48:")
        print("    mean %+.4f  sd %.4f  paired t %+.2f over %d seeds"
              % (dif.mean(), sd_d, t, n))
        print("    A null here means file-name order carries no usable time")
        print("    axis, which is a statement about the release.")
    if "index" in have and "single" in have:
        a = np.array(out["arms"]["index"]["unseen_runs"])
        b = np.array(out["arms"]["single"]["unseen_runs"])
        n = min(len(a), len(b))
        dif = a[:n] - b[:n]
        sd_d = dif.std(ddof=1) if n > 1 else 0.0
        t = dif.mean() / (sd_d / np.sqrt(n)) if sd_d > 0 else float("nan")
        out["index_minus_single"] = {
            "mean": float(dif.mean()), "sd": float(sd_d),
            "paired_t": float(t), "n": int(n),
            "per_seed": [float(x) for x in dif]}
        print("\n  three frames minus one frame on 13-48:")
        print("    mean %+.4f  sd %.4f  paired t %+.2f over %d seeds"
              % (dif.mean(), sd_d, t, n))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("\nwritten %s" % args.out)


if __name__ == "__main__":
    main()
