"""Synthetic RDRD-shaped tree for the repository self-test.

This does NOT reproduce the Real Doppler RAD-DAR measurements. It
reproduces the *structure* those measurements have, so that every stage of
the pipeline can be exercised without the real archive and without a GPU:

  * the real on-disk layout            data/<Class>/<HH-MM[suffix]>/<n>.csv
  * the duplicated archive tree        data/<Class>/... and <Class>/...
  * reprocessed-session twins          17-09 and 17-09p with identical bytes
  * a truncated twin                   12-41 with fewer files than its pair
  * an off-regime acquisition          drone/13-48 shifted in Doppler

The last item gives every stage a target to address: acquisition 13-48 is
placed at a Doppler centroid far from every other drone acquisition, so
step2c, step2d, step3 and step5 all have something to point at.

What this tree does NOT do is reproduce the paper's finding. On the real
data, acquisition 13-48 is misclassified; here it is not, because a
convolutional network is largely insensitive to where in the Doppler axis a
compact signature sits, and the paper reports that Doppler positional
insensitivity is one of four candidate mechanisms the measurements refute.
A generator cannot synthesise a failure whose mechanism is unknown, and
tuning one until the failure appeared would be manufacturing the result
rather than testing the code.

So the self-test is a self-test and nothing more. It checks that every
stage runs to completion on a tree with the real one's structure, that the
array shapes, split disjointness and parameter count are what the paper
states, and that no stage throws. Whether the paper's numbers reproduce is
checked by verify.sh against the real archive, where it can actually be
answered.

Usage:
    python tools/make_selftest_data.py --out /tmp/selftest
"""
import argparse
import os
import shutil

import numpy as np

H, W = 11, 61
FLOOR = -120.0


def acquisition(n, peak, spread_r, spread_d, doppler_centre, seed,
                doppler_jitter=1.0):
    """One acquisition: a target drifting slowly through the crop.

    Peak level is drawn per sample from a range shared by all three
    classes, so absolute level carries no class information and the model
    is forced onto the shape of the signature. That is deliberate. If peak
    level separated the classes the synthetic task would be solved by one
    scalar, every stage would report perfect accuracy and the self-test
    would check nothing beyond the absence of exceptions.
    """
    rng = np.random.default_rng(seed)
    out = []
    r0 = rng.uniform(3.0, 8.0)
    d0 = doppler_centre + rng.normal(0.0, doppler_jitter)
    rr, dd = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for _ in range(n):
        r0 += rng.normal(0.0, 0.04)
        d0 += rng.normal(0.0, 0.12)
        a = peak * rng.lognormal(0.0, 0.55)
        m = a * np.exp(-(((rr - r0) / spread_r) ** 2
                         + ((dd - d0) / spread_d) ** 2))
        m = 20.0 * np.log10(np.maximum(m + rng.lognormal(-1.2, 0.5, (H, W)),
                                       1e-6))
        out.append(np.maximum(m, FLOOR))
    return out


def write(d, mats):
    os.makedirs(d, exist_ok=True)
    for i, m in enumerate(mats, 1):
        np.savetxt(os.path.join(d, "%d.csv" % i), m, delimiter=",",
                   fmt="%.4f")


# Peak is the same for all three classes on purpose (see acquisition
# above); the classes differ in the shape of the signature, vehicles broad
# in range, pedestrians broad in Doppler, drones compact in both. Doppler
# centre 30 is the "normal" regime; 13-48 sits at 8.
PLAN = {
    "Cars": dict(peak=12.0, sr=2.6, sd=3.0, acqs=[
        ("13-13", 60, 30.0), ("13-23", 60, 29.0), ("13-44", 60, 31.0),
        ("15-37", 60, 30.5), ("16-07", 60, 29.5), ("17-09", 60, 30.0),
    ]),
    "Drones": dict(peak=12.0, sr=0.7, sd=1.2, acqs=[
        ("12-34", 60, 30.0), ("12-41", 60, 29.5), ("13-21", 60, 30.5),
        ("15-21", 60, 30.0), ("16-09", 60, 29.0),
        ("13-48", 60, 8.0),
    ]),
    "People": dict(peak=12.0, sr=1.6, sd=7.0, acqs=[
        ("11-00f", 60, 30.0), ("11-09f", 60, 29.0), ("11-23f", 60, 31.0),
        ("12-50f", 60, 30.0), ("12-57f", 60, 29.5), ("15-58", 60, 30.0),
    ]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/selftest")
    ap.add_argument("--per-acq", type=int, default=0,
                    help="override files per acquisition (0 = use plan)")
    args = ap.parse_args()

    root = args.out
    shutil.rmtree(root, ignore_errors=True)

    for cls, cfg in PLAN.items():
        for name, n, dop in cfg["acqs"]:
            if args.per_acq:
                n = args.per_acq
            jitter = 12.0 if name == "13-48" else 1.0
            mats = acquisition(n, cfg["peak"], cfg["sr"], cfg["sd"], dop,
                               seed=abs(hash(cls + name)) % 100000,
                               doppler_jitter=jitter)
            write(os.path.join(root, "data", cls, name), mats)

        # a reprocessed twin: identical bytes under a different folder name
        if cls == "Cars":
            shutil.copytree(os.path.join(root, "data", cls, "17-09"),
                            os.path.join(root, "data", cls, "17-09p"))
        if cls == "People":
            shutil.copytree(os.path.join(root, "data", cls, "15-58"),
                            os.path.join(root, "data", cls, "15-58i"))

    # a truncated twin: same name, fewer files, under the second root copy
    for cls in PLAN:
        shutil.copytree(os.path.join(root, "data", cls),
                        os.path.join(root, cls))
    trunc = os.path.join(root, "Drones", "12-41")
    for f in sorted(os.listdir(trunc))[45:]:
        os.remove(os.path.join(trunc, f))

    n = sum(len(f) for _, _, f in os.walk(root))
    print("self-test tree written to %s" % root)
    print("total files on disk %d (the deduplicated count is about half)" % n)


if __name__ == "__main__":
    main()
