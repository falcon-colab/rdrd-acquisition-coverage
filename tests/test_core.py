"""Unit tests for the claims the manuscript makes about the code itself.

These do not need the archive and do not need a GPU. Each one checks a
statement the paper makes in prose, so that a reader who doubts the
statement can run it rather than take it on trust.

    python -m pytest tests/ -v          (or: python tests/test_core.py)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "src"))

import rdrd                                  # noqa: E402
from step2 import make_model                 # noqa: E402
from step2b import normalise                 # noqa: E402
from step3 import quantise, reduce_doppler    # noqa: E402
import temporal                               # noqa: E402


# --------------------------------------------------------------------------
# Section III: "Aggregation sums linear power across adjacent Doppler bins."
# --------------------------------------------------------------------------

def test_aggregation_conserves_linear_power():
    rng = np.random.default_rng(0)
    X = rng.uniform(-110.0, -40.0, (16, 11, 61)).astype(np.float32)
    for factor in (2, 4, 8):
        keep = (61 // factor) * factor
        before = (10.0 ** (X[:, :, :keep] / 10.0)).sum(axis=2)
        after = (10.0 ** (reduce_doppler(X, factor) / 10.0)).sum(axis=2)
        assert np.allclose(before, after, rtol=1e-5), factor


def test_aggregation_shapes_match_reported_cell_counts():
    """Table V reports 671, 330, 165 and 77 cells at 1x, 2x, 4x and 8x."""
    X = np.zeros((4, 11, 61), dtype=np.float32)
    expected = {1: 671, 2: 330, 4: 165, 8: 77}
    for factor, cells in expected.items():
        out = reduce_doppler(X, factor)
        assert out.shape[1] * out.shape[2] == cells, (factor, out.shape)


def test_aggregation_is_identity_at_unit_factor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 11, 61)).astype(np.float32)
    assert reduce_doppler(X, 1) is X


# --------------------------------------------------------------------------
# Section III: quantisation, and the reason 'offset' normalisation was chosen
# --------------------------------------------------------------------------

def test_quantisation_error_is_bounded_by_half_a_step():
    rng = np.random.default_rng(2)
    a = rng.uniform(-100.0, -30.0, 4096).astype(np.float32)
    for bits in (16, 8, 4):
        step = (a.max() - a.min()) / (2 ** bits - 1)
        err = np.abs(quantise(a, bits) - a).max()
        assert err <= step / 2 + 1e-4, (bits, err, step / 2)


def test_quantisation_error_shrinks_as_bits_grow():
    rng = np.random.default_rng(3)
    a = rng.uniform(-100.0, -30.0, 4096).astype(np.float32)
    errs = [np.abs(quantise(a, b) - a).mean() for b in (4, 8, 16)]
    assert errs[0] > errs[1] > errs[2], errs


def test_quantisation_is_a_no_op_at_full_precision():
    rng = np.random.default_rng(4)
    a = rng.normal(size=256).astype(np.float32)
    assert quantise(a, 32) is a


def test_offset_normalisation_commutes_with_quantisation():
    """The manuscript's stated reason for preferring 'offset': it is a fixed
    affine map, so quantising before or after normalisation is the same
    operation up to a known scale factor. Quantising first and then
    normalising must agree with normalising first and then quantising on the
    correspondingly scaled grid."""
    rng = np.random.default_rng(5)
    X = rng.uniform(-110.0, -40.0, (32, 11, 61)).astype(np.float32)
    for bits in (16, 8):
        a = normalise(quantise(X, bits), "offset")
        lo, hi = X.min(), X.max()
        b = quantise(normalise(X, "offset"), bits,
                     (lo + 60.0) / 60.0, (hi + 60.0) / 60.0)
        assert np.abs(a - b).max() < 1e-3, (bits, np.abs(a - b).max())


def test_offset_normalisation_is_the_documented_affine_map():
    X = np.array([[[-60.0, 0.0, -120.0]]], dtype=np.float32)
    out = normalise(X, "offset")
    assert np.allclose(out, [[[0.0, 1.0, -1.0]]]), out


# --------------------------------------------------------------------------
# Section III: "101,139 parameters, matched to RangeDopplerNet Type-2"
# --------------------------------------------------------------------------

def test_model_parameter_count_is_the_reported_figure():
    import torch
    import torch.nn as nn
    model = make_model(torch, nn)
    n = sum(p.numel() for p in model.parameters())
    assert n == 101139, n


def test_model_accepts_every_aggregation_level():
    """The same architecture has to take all four input resolutions, which is
    what makes the grid in Table V a comparison of inputs rather than of
    models. Global pooling before the classifier is what allows it."""
    import torch
    import torch.nn as nn
    model = make_model(torch, nn).eval()
    for factor in (1, 2, 4, 8):
        X = reduce_doppler(np.zeros((2, 11, 61), dtype=np.float32), factor)
        with torch.no_grad():
            out = model(torch.tensor(X)[:, None])
        assert out.shape == (2, 3), (factor, out.shape)


# --------------------------------------------------------------------------
# Section III: acquisition grouping and deduplication
# --------------------------------------------------------------------------

def test_timestamp_grouping_merges_suffixed_folders():
    """15-55, 15-55a, 15-55m and 15-55p are one acquisition; 15-56 is not.
    group_by_timestamp returns folder -> unit label, so four folders must
    carry the same label and the fifth a different one."""
    sessions = {"car": {k: [] for k in
                        ("15-55", "15-55a", "15-55m", "15-55p", "15-56")}}
    unit_of = rdrd.group_by_timestamp(sessions)["car"]
    assert len(set(unit_of.values())) == 2, unit_of
    for k in ("15-55", "15-55a", "15-55m", "15-55p"):
        assert unit_of[k] == "15-55", (k, unit_of[k])
    assert unit_of["15-56"] == "15-56"


def test_timestamp_grouping_keeps_distinct_timestamps_apart():
    sessions = {"drone": {k: [] for k in ("13-48", "13-49", "16-09")}}
    unit_of = rdrd.group_by_timestamp(sessions)["drone"]
    assert len(set(unit_of.values())) == 3, unit_of


def test_class_of_maps_every_directory_spelling():
    assert rdrd.class_of("Cars") == "car"
    assert rdrd.class_of("Drones") == "drone"
    assert rdrd.class_of("People") == "person"


def test_dedup_drops_byte_identical_and_truncated_twins(tmp_path=None):
    """The archive carries the whole tree twice, and one interrupted
    extraction left a folder with the same name but fewer files. Both must
    go, and the larger copy must be the survivor."""
    import shutil
    import tempfile

    root = tempfile.mkdtemp()
    try:
        full = os.path.join(root, "data", "Drones", "12-41")
        os.makedirs(full)
        rng = np.random.default_rng(6)
        for i in range(1, 21):
            np.savetxt(os.path.join(full, "%d.csv" % i),
                       rng.normal(size=(11, 61)), delimiter=",", fmt="%.4f")
        # whole-tree duplicate
        shutil.copytree(os.path.join(root, "data", "Drones"),
                        os.path.join(root, "Drones"))
        # truncate the duplicate so it is a partial twin, not an exact one
        for f in sorted(os.listdir(os.path.join(root, "Drones", "12-41")))[12:]:
            os.remove(os.path.join(root, "Drones", "12-41", f))

        sessions = rdrd.find_sessions(root)
        assert len(sessions["drone"]) == 2, sorted(sessions["drone"])
        kept, dropped = rdrd.dedupe_tree(sessions)
        assert len(kept["drone"]) == 1, sorted(kept["drone"])
        survivors = list(kept["drone"].values())[0]
        assert len(survivors) == 20, len(survivors)
        assert os.path.join("data", "Drones") in survivors[0], survivors[0]
        assert len(dropped) >= 1, dropped
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------
# Section III: "no acquisition spans two subsets of the grouped split"
# --------------------------------------------------------------------------

def test_grouped_splits_are_unit_disjoint():
    """Table III compares a grouped split against a random one. The grouped
    split is only meaningful if no acquisition contributes samples to both
    sides, so check that directly on the returned path lists. Each folder
    below carries two files, which is what lets a split straddle one."""
    sessions, unit_of = {}, {}
    plan = {"car": ["13-13", "13-13p", "13-23", "15-37", "16-07", "17-09"],
            "drone": ["12-34", "12-41", "13-21", "13-48", "15-21", "16-09"],
            "person": ["11-00f", "11-00i", "11-23f", "12-50f", "12-57f",
                       "15-58"]}
    for cls, folders in plan.items():
        sessions[cls] = {f: ["%s/%s/%d.csv" % (cls, f, i) for i in (1, 2)]
                         for f in folders}
    unit_of = rdrd.group_by_timestamp(sessions)

    grouped, rand = rdrd.build_splits(sessions, unit_of,
                                      np.random.default_rng(0), test_frac=0.3)

    def unit(path):
        cls, folder, _ = path.split("/")
        return cls, unit_of[cls][folder]

    tr = {unit(p) for p in grouped["train"]}
    te = {unit(p) for p in grouped["test"]}
    assert tr and te
    assert not (tr & te), tr & te
    # every sample is placed exactly once, and the random split sees the same
    # pool, which is the premise of the comparison
    assert len(grouped["train"]) + len(grouped["test"]) == 36
    assert len(rand["train"]) + len(rand["test"]) == 36


def test_random_split_straddles_acquisitions_grouped_split_does_not():
    """The point of Table III stated as a test: on the same pool, the random
    split puts at least one acquisition on both sides and the grouped split
    puts none."""
    plan = {"car": ["13-13", "13-23", "15-37", "16-07"],
            "drone": ["12-34", "12-41", "13-48", "16-09"],
            "person": ["11-00f", "11-23f", "12-50f", "15-58"]}
    sessions = {cls: {f: ["%s/%s/%d.csv" % (cls, f, i) for i in range(12)]
                      for f in folders} for cls, folders in plan.items()}
    unit_of = rdrd.group_by_timestamp(sessions)
    grouped, rand = rdrd.build_splits(sessions, unit_of,
                                      np.random.default_rng(0), test_frac=0.3)

    def straddling(split):
        def unit(p):
            cls, folder, _ = p.split("/")
            return cls, unit_of[cls][folder]
        return ({unit(p) for p in split["train"]}
                & {unit(p) for p in split["test"]})

    assert not straddling(grouped), straddling(grouped)
    assert straddling(rand), "random split unexpectedly unit-disjoint"


# --------------------------------------------------------------------------
# The package tells a reader where to get the archive. If that is wrong, the
# reproduction fails at step one, which is the one failure a reproducibility
# package cannot afford. An earlier version of this repository carried the
# slug misspelled with a single "p" in "doppler", in the README, in both
# notebooks and in the Zenodo metadata, together with a note asserting that
# the misspelling was correct and should not be fixed. That URL is a 404.
# This test does not check that the dataset is reachable, which would need
# the network; it checks that the package names it with one consistent
# spelling and never the known-bad one.
# --------------------------------------------------------------------------

DATASET_SLUG = "iroldan/real-doppler-raddar-database"
KNOWN_BAD_SLUGS = ("iroldan/real-dopler-raddar-database",
                   "iroldan/real-doppler-rad-dar-database")

ROOT = os.path.join(HERE, os.pardir)


def _package_files():
    """Every file that could name the dataset, except this one, which names
    the bad spellings on purpose so it can forbid them."""
    skip = {".git", "__pycache__", ".ipynb_checkpoints", "runs", ".venv"}
    me = os.path.abspath(__file__)
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if not f.endswith((".md", ".py", ".json", ".ipynb", ".sh",
                               ".tex", ".cff", ".txt")):
                continue
            path = os.path.join(base, f)
            if os.path.abspath(path) == me:
                continue
            yield path


def test_no_file_carries_a_known_bad_dataset_slug():
    bad = []
    for path in _package_files():
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for slug in KNOWN_BAD_SLUGS:
            # the corrective note in the README names the bad spelling on
            # purpose, so allow a line that also says it returns 404
            for line in text.splitlines():
                if slug in line and "404" not in line:
                    bad.append((os.path.relpath(path, ROOT), slug))
    assert not bad, bad


def test_the_dataset_slug_is_stated_the_same_way_everywhere():
    """Every file that names the dataset must name it identically. A package
    that spells its own data source two ways has already lost the reader."""
    seen = set()
    pattern = re.compile(r"iroldan/[A-Za-z0-9._-]+")
    for path in _package_files():
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for m in pattern.findall(text):
            seen.add(m)
    assert seen, "no dataset slug found anywhere in the package"
    assert seen == {DATASET_SLUG}, sorted(seen)


# --------------------------------------------------------------------------
# Section VIII: the three-frame input. The manuscript claims the triplets are
# formed in file-name INDEX order and only across genuinely consecutive
# integers, and that the shuffle arm removes adjacency while keeping the
# folder. All three are properties of the construction, so they are testable
# without the archive.
# --------------------------------------------------------------------------

def test_triplets_use_numeric_not_lexicographic_order():
    """9, 10, 11 are consecutive; a string sort would scatter them."""
    paths = ["Drones/13-48/%d.csv" % i for i in (1, 2, 3, 9, 10, 11)]
    trips = temporal.triplet_rows(paths)
    assert (0, 1, 2) in trips, trips
    assert (3, 4, 5) in trips, "numeric order lost: %s" % (trips,)
    assert len(trips) == 2, trips


def test_triplets_never_span_a_gap_in_numbering():
    """A missing file means the frames are not adjacent, so no triplet."""
    paths = ["Cars/15-37/%d.csv" % i for i in (1, 2, 4, 5)]
    assert temporal.triplet_rows(paths) == []
    paths = ["Cars/15-37/%d.csv" % i for i in (1, 2, 3, 5, 6, 7)]
    trips = temporal.triplet_rows(paths)
    assert trips == [(0, 1, 2), (3, 4, 5)], trips


def test_triplets_never_cross_a_folder_boundary():
    """Two folders whose numbering runs on must not be joined."""
    paths = (["Drones/13-48/%d.csv" % i for i in (1, 2)]
             + ["Drones/12-34/%d.csv" % i for i in (3, 4)])
    assert temporal.triplet_rows(paths) == []


def test_shuffle_arm_keeps_the_folder_and_drops_adjacency():
    paths = ["Drones/13-48/%d.csv" % i for i in range(1, 11)]
    trips = temporal.triplet_rows(paths)
    comps, fallback = temporal.companion_rows(paths, trips, seed=0)
    assert fallback == 0
    assert len(comps) == len(trips)
    for (r0, r1, r2), (c0, c1) in zip(trips, comps):
        assert c0 not in (r0, r1, r2), "companion is a true neighbour"
        assert c1 not in (r0, r1, r2), "companion is a true neighbour"


def test_arms_agree_on_sample_count_and_centre():
    """single, index and shuffle must describe the SAME samples, or the
    comparison between them is not paired."""
    paths = ["Drones/13-48/%d.csv" % i for i in range(1, 13)]
    trips = temporal.triplet_rows(paths)
    comps, _ = temporal.companion_rows(paths, trips, seed=0)
    Xn1 = np.arange(12 * 1 * 11 * 61, dtype=np.float32).reshape(12, 1, 11, 61)
    a = temporal.build_arm(Xn1, trips, comps, "single")
    b = temporal.build_arm(Xn1, trips, comps, "index")
    c = temporal.build_arm(Xn1, trips, comps, "shuffle")
    assert a.shape[0] == b.shape[0] == c.shape[0] == len(trips)
    assert a.shape[1] == 1 and b.shape[1] == 3 and c.shape[1] == 3
    # the centre channel is the same frame in every arm
    assert np.array_equal(a[:, 0], b[:, 1])
    assert np.array_equal(a[:, 0], c[:, 1])


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print("  pass  %s" % name)
        except AssertionError as e:
            bad += 1
            print("  FAIL  %s  %s" % (name, e))
        except Exception as e:                       # noqa: BLE001
            bad += 1
            print("  ERROR %s  %s: %s" % (name, type(e).__name__, e))
    print("\n%d of %d passed" % (len(fns) - bad, len(fns)))
    return bad


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
