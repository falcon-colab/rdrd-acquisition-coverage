"""
rdrd.py -- single module for the radar compression project.

Everything the pipeline needs lives here: loading, session discovery,
deduplication, duplicate-session merging, leakage measurement, signal
statistics, and split construction.

Dataset layout (as published):
    <root>/[data/]<Class>/<session>/<n>.csv

Known pathologies this module handles:
  * the archive contains the whole tree twice (data/Cars/... and Cars/...)
  * some session folders are duplicates or reprocessings of each other
  * filename order does not reliably track acquisition order
  * a dB-based similarity mask biases comparisons by class signal level
"""

import hashlib
import json
import os
import re
from collections import defaultdict

import numpy as np

DATA_EXT = {".csv", ".txt", ".dat"}
NUM_RE = re.compile(r"(\d+)")

CLASS_KEYS = [("drone", "drone"), ("uav", "drone"), ("dron", "drone"),
              ("car", "car"), ("vehic", "car"), ("coche", "car"),
              ("people", "person"), ("pedestr", "person"),
              ("person", "person"), ("human", "person")]


# ----------------------------------------------------------------- loading

def class_of(name):
    low = name.lower()
    for key, label in CLASS_KEYS:
        if key in low:
            return label
    return None


def load_matrix(path):
    for kw in ({"delimiter": ","}, {}, {"delimiter": ";"}):
        try:
            m = np.loadtxt(path, **kw)
            if m.ndim == 2 and m.size > 1:
                return m
        except Exception:
            continue
    raise ValueError("unparseable: %s" % path)


def find_sessions(root):
    """{class: {session_key: [file paths]}}; session_key is the full rel path."""
    out = defaultdict(dict)
    for dirpath, _d, files in os.walk(root):
        data = sorted(f for f in files
                      if os.path.splitext(f)[1].lower() in DATA_EXT)
        if not data:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)
        label = None
        for p in parts:
            g = class_of(p)
            if g:
                label = g
        if label is None:
            continue
        out[label][rel] = [os.path.join(dirpath, f) for f in data]
    return dict(out)


def short_name(session_key):
    return session_key.split(os.sep)[-1]


def numbering(paths):
    nums = [int(NUM_RE.findall(os.path.basename(p))[-1])
            for p in paths if NUM_RE.findall(os.path.basename(p))]
    if not nums:
        return {"n": len(paths)}
    a = np.array(sorted(nums))
    span = int(a.max() - a.min() + 1)
    return {"n": len(paths), "min": int(a.min()), "max": int(a.max()),
            "missing": int(span - a.size), "contiguous": bool(span == a.size)}


# ------------------------------------------------------------- dedup (tree)

def file_digest(path, nbytes=65536):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def session_digest(paths, n_probe=12):
    if not paths:
        return None
    idx = np.linspace(0, len(paths) - 1, min(n_probe, len(paths))).astype(int)
    joined = "|".join(file_digest(paths[i]) for i in idx) + "|%d" % len(paths)
    return hashlib.sha1(joined.encode()).hexdigest()


def dedupe_tree(sessions):
    """
    Remove duplicate session folders in two passes.

    Pass 1, byte-identical content: the archive ships the whole tree twice
    (data/Cars/... and Cars/...), so identical folders are collapsed to the
    shallowest path.

    Pass 2, same name under different parents: an interrupted extraction can
    leave a TRUNCATED twin, which pass 1 misses because a different file
    count changes the digest. Two folders of the same class with the same
    folder name are the same recording, so the one with more files wins.
    """
    kept, dropped = {}, []
    for label, sess in sessions.items():
        by_dig = defaultdict(list)
        for key, paths in sess.items():
            by_dig[session_digest(paths)].append(key)
        stage1 = {}
        for _dig, keys in by_dig.items():
            keys.sort(key=lambda k: (k.count(os.sep), len(k), k))
            stage1[keys[0]] = sess[keys[0]]
            dropped += [(label, k, keys[0], "identical") for k in keys[1:]]

        by_name = defaultdict(list)
        for key in stage1:
            by_name[short_name(key)].append(key)
        keep = {}
        for _name, keys in by_name.items():
            keys.sort(key=lambda k: (-len(stage1[k]), k.count(os.sep), k))
            keep[keys[0]] = stage1[keys[0]]
            for k in keys[1:]:
                dropped.append((label, k, keys[0],
                                "partial (%d of %d files)"
                                % (len(stage1[k]), len(stage1[keys[0]]))))
        kept[label] = keep
    return kept, dropped


# ------------------------------------------------------------- features

def session_features(paths, limit=120, top_k=12):
    """
    Sparse peak-rank features. A FIXED CELL COUNT is used rather than a dB
    mask: a dB mask keeps fewer cells for a high-SNR class, which changes
    correlation magnitudes on its own and makes classes incomparable.
    """
    take = paths[:limit] if limit else paths
    mats = []
    for p in take:
        try:
            mats.append(load_matrix(p))
        except ValueError:
            pass
    if not mats:
        return None, None
    shapes = defaultdict(int)
    for m in mats:
        shapes[m.shape] += 1
    dom = max(shapes, key=shapes.get)
    stack = np.stack([m for m in mats if m.shape == dom])

    flat = stack.reshape(len(stack), -1).astype(np.float32)
    order = np.argsort(flat, axis=1)[:, ::-1][:, :top_k]
    feat = np.zeros_like(flat)
    rows = np.arange(len(flat))[:, None]
    vals = np.take_along_axis(flat, order, axis=1)
    feat[rows, order] = vals - vals[:, :1]
    feat -= feat.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(feat, axis=1)
    nrm[nrm == 0] = 1.0
    return stack, feat / nrm[:, None]


def signal_stats(stack):
    x = stack.reshape(len(stack), -1).astype(np.float64)
    lin = 10.0 ** ((x - x.max(axis=1, keepdims=True)) / 10.0)
    floor = np.median(lin, axis=1)
    ptf = -10 * np.log10(np.maximum(floor, 1e-30))
    within10 = (lin > 10 ** -1.0).sum(axis=1)
    return {"ptf_median": float(np.median(ptf)),
            "ptf_p10": float(np.percentile(ptf, 10)),
            "cells_within_10dB": float(np.median(within10)),
            "n": int(len(stack))}


# --------------------------------------------------- grouping units

TS_RE = re.compile(r"^(\d{1,2}-\d{2})")


def timestamp_of(session_key):
    """
    Acquisition timestamp prefix of a session folder.

    Folder names look like 13-44, 13-44p, 12-50f, 15-55a, 15-58i. The leading
    HH-MM is the recording time; the suffix distinguishes variants captured
    within that same recording. Variants of one timestamp are therefore NOT
    statistically independent and must share a grouping unit.
    """
    m = TS_RE.match(short_name(session_key))
    return m.group(1) if m else short_name(session_key)


def group_by_timestamp(sessions):
    """
    {class: {session_key: unit_id}} using the acquisition timestamp.

    This replaces content-similarity merging, which failed badly: every crop
    is centred on its own detection, so session mean-vectors all look like a
    central blob and correlate above 0.9 regardless of provenance. A
    timestamp is a fact about acquisition and cannot drift with a parameter.
    """
    return {label: {key: timestamp_of(key) for key in sess}
            for label, sess in sessions.items()}


def suspicious_pairs(stats, ptf_tol=0.05):
    """
    Report (do NOT merge) session pairs that share a file count and have
    near-identical signal statistics. These may be reprocessings of one
    recording. Reporting keeps the judgement with the analyst rather than
    burying it in an automatic merge.
    """
    keys = sorted(stats)
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = stats[keys[i]], stats[keys[j]]
            if a["n"] != b["n"]:
                continue
            if abs(a["ptf_median"] - b["ptf_median"]) > ptf_tol:
                continue
            if abs(a["ptf_p10"] - b["ptf_p10"]) > ptf_tol:
                continue
            out.append((keys[i], keys[j], a["n"], a["ptf_median"]))
    return out


# ----------------------------------------------------------------- leakage

def leakage_test(feats, unit_of, rng, n_pairs=6000):
    by_unit = defaultdict(list)
    for name, f in feats.items():
        if f is not None and len(f) > 1:
            by_unit[unit_of.get(name, name)].append(f)
    units = {u: np.concatenate(v) for u, v in by_unit.items()}
    units = {u: v for u, v in units.items() if len(v) > 1}
    if len(units) < 2:
        return None
    keys = list(units)

    per = max(50, n_pairs // len(keys))
    within = []
    for u in keys:
        f = units[u]
        a = rng.integers(0, len(f), per)
        b = rng.integers(0, len(f), per)
        m = a != b
        if m.any():
            within.append(np.einsum("ij,ij->i", f[a[m]], f[b[m]]))
    within = np.concatenate(within)

    cross = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(keys), 2, replace=False)
        fi, fj = units[keys[i]], units[keys[j]]
        cross.append(float(fi[rng.integers(len(fi))] @ fj[rng.integers(len(fj))]))
    cross = np.array(cross)

    return {"n_units": len(keys),
            "within_mean": float(within.mean()),
            "cross_mean": float(cross.mean()),
            "gap": float(within.mean() - cross.mean()),
            "frac_within_above_cross_p95": float(
                (within > np.percentile(cross, 95)).mean())}


# ------------------------------------------------------------------ splits

def build_splits(sessions, unit_of, rng, test_frac=0.3):
    grouped = {"train": [], "test": []}
    for label in sorted(sessions):
        units = defaultdict(list)
        for key, paths in sessions[label].items():
            units[unit_of[label].get(key, key)] += paths
        keys = sorted(units)
        rng.shuffle(keys)
        n_test = max(1, int(round(test_frac * len(keys))))
        for k in keys[:n_test]:
            grouped["test"] += units[k]
        for k in keys[n_test:]:
            grouped["train"] += units[k]

    allf = grouped["train"] + grouped["test"]
    idx = rng.permutation(len(allf))
    cut = int(round(test_frac * len(allf)))
    rand = {"train": [allf[i] for i in idx[cut:]],
            "test": [allf[i] for i in idx[:cut]]}
    return grouped, rand


def verify_split_by_content(split, rng, n_test=600, n_train=4000):
    """Path checks are not enough when duplicate folders exist."""
    te = split["test"]
    tr = split["train"]
    ti = rng.choice(len(te), min(n_test, len(te)), replace=False)
    hashes = {file_digest(te[i]) for i in ti}
    ri = rng.choice(len(tr), min(n_train, len(tr)), replace=False)
    hits = sum(1 for i in ri if file_digest(tr[i]) in hashes)
    return {"probed_train": int(len(ri)), "probed_test": int(len(ti)),
            "collisions": int(hits), "clean": bool(hits == 0)}
