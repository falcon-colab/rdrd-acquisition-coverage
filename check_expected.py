"""Compare a run's reports against every number the manuscript states.

    python check_expected.py --reports reports/ [--expected expected.json]

Reads expected.json, reads the report files a run produced, and prints one
line per claim: the paper's value, the value this run produced, the
difference, and whether the difference is inside the tolerance that
expected.json records for that claim along with its reason.

Exit status is 0 when every check inside a present report agrees, and 1
otherwise. Missing reports are reported separately and do not count as
failures, because a partial run is a partial run rather than a
contradiction. The summary says plainly how many claims were actually
tested, so a run that checked six numbers cannot be mistaken for one that
checked all of them.
"""
import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------- helpers

def dig(obj, path):
    """Walk a dotted path, taking integer segments as list indices.

    Keys in these reports contain spaces ("drone recall") but never dots,
    so splitting on dots is unambiguous.
    """
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
    return cur


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch_t(a, b):
    """Welch's t for two independent samples, a minus b."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = sd(a) ** 2 / len(a), sd(b) ** 2 / len(b)
    if va + vb == 0.0:
        return float("nan")
    return (mean(a) - mean(b)) / math.sqrt(va + vb)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return float("nan")
    return num / (dx * dy)


# ------------------------------------------------------------ derivations
# Each takes the loaded report and the check's "args" and returns a number.
# These exist for the paper's values that are computed from a report rather
# than stored in one.

def d_session_recall(rep, a):
    for s in rep[a["cls"]]["sessions"]:
        if s["name"] == a["name"]:
            return s["recall"]
    raise KeyError("acquisition %s not in the %s sessions"
                   % (a["name"], a["cls"]))


def d_min_session_recall(rep, a):
    return min(s["recall"] for s in rep[a["cls"]]["sessions"])


def d_recovery_extreme(rep, a):
    """The largest or smallest recovery over the 52-unit sweep.

    The paper's strongest claim is now a bound rather than a comparison:
    no acquisition other than 13-48 recovers as much as 0.05. A bound is
    exactly the kind of statement that rots silently when a rerun shifts
    one unit, so it gets a check of its own.
    """
    vals = [u["recovery"] for k, u in rep["units"].items()
            if not k.endswith(a.get("exclude", "\x00"))]
    return max(vals) if a.get("which", "max") == "max" else min(vals)


def d_recovery_of(rep, a):
    return rep["units"][a["unit"]]["recovery"]


def d_predicted_fraction(rep, a):
    return rep["units"][a["unit"]]["predicted_fraction"][a["cond"]][a["cls"]]


def d_units_above(rep, a):
    """How many of the 52 units clear a threshold.

    The manuscript and a figure caption both count units above a cut, and
    counting by eye off a strip plot is exactly how a caption acquires a
    wrong number. Count it from the report instead. 'cls' restricts to one
    class, 'exclude' drops a named unit.
    """
    classes = [a["cls"]] if a.get("cls") else sorted(rep)
    n = 0
    for c in classes:
        for s in rep[c]["sessions"]:
            if s["name"] == a.get("exclude"):
                continue
            if s["recall"] > a["above"]:
                n += 1
    return n


def d_session_recall_extreme(rep, a):
    """The smallest or largest per-acquisition recall in a class.

    Used for the stated band of the drone class once 13-48 is set aside.
    A band quoted as 'between x and y' is two claims, so it gets two checks.
    """
    vals = [s["recall"] for s in rep[a["cls"]]["sessions"]
            if s["name"] != a.get("exclude")]
    return max(vals) if a.get("which") == "max" else min(vals)


def d_recall_sd_excluding(rep, a):
    rows = [s for s in rep[a["cls"]]["sessions"] if s["name"] != a["exclude"]]
    return sd([s["recall"] for s in rows])


def d_pearson_excluding(rep, a):
    rows = [s for s in rep[a["cls"]]["sessions"] if s["name"] != a["exclude"]]
    return pearson([s["ptf_median"] for s in rows], [s["recall"] for s in rows])


def d_welch_t(rep, a):
    return welch_t(dig(rep, a["a"]), dig(rep, a["b"]))


def d_welch_t_abs(rep, a):
    """Magnitude of Welch's t, for a two-sided claim.

    Used only where the manuscript's claim is a failure to detect a
    difference. There the test is two-sided, the sign carries no part of the
    claim, and the manuscript quotes the magnitude while the code computes
    the signed statistic in the direction aggregated minus full resolution.
    Comparing magnitudes is what the claim actually asserts. Anywhere the
    manuscript claims a direction, the signed d_welch_t is used instead, so a
    sign flip there still fails.
    """
    return abs(welch_t(dig(rep, a["a"]), dig(rep, a["b"])))


AUGMENT_KEYS = {"baseline": "1x, no augmentation",
                "aggregation": "4x aggregation",
                "shift4": "1x + shift +/-4 bins",
                "shift8": "1x + shift +/-8 bins",
                "shift16": "1x + shift +/-16 bins"}


def d_augment_value(rep, a):
    return rep[AUGMENT_KEYS[a["key"]]]["mean"]


def d_ablate_drop(rep, a):
    return rep["neither"]["accuracy"] - rep[a["condition"]]["accuracy"]


def d_interaction_contrast(rep, a):
    """I = Y_both - Y_input - Y_weights + Y_neither, on accuracy."""
    return (rep["both"]["accuracy"] - rep["input only"]["accuracy"]
            - rep["weights only"]["accuracy"] + rep["neither"]["accuracy"])


# ------------------------------------------------- paired derivations
# Every condition in step4.py and step5.py is run at the same seeds, and the
# seed fixes both the partition and the initialisation. Conditions therefore
# differ only in the quantity under test, and the per-seed difference is the
# unit of analysis. Welch's test, which the first version of this manuscript
# used, assumes the two samples are independent and throws that pairing away.
# These derivations replace it.

def _paired(a, b):
    """Per-seed differences a minus b, truncated to the shorter array."""
    n = min(len(a), len(b))
    if n < 2:
        return []
    return [a[i] - b[i] for i in range(n)]


def paired_t(a, b):
    d = _paired(a, b)
    if len(d) < 2:
        return float("nan")
    s = sd(d)
    if s == 0.0:
        return float("nan")
    return mean(d) / (s / math.sqrt(len(d)))


def d_paired_t(rep, a):
    return paired_t(dig(rep, a["a"]), dig(rep, a["b"]))


def d_paired_t_abs(rep, a):
    """Magnitude of the paired t, for a claim that is a failure to detect.

    Same reasoning as d_welch_t_abs: where the manuscript's claim is
    two-sided the sign carries no part of it, and the manuscript quotes the
    magnitude. Anywhere a direction is claimed, d_paired_t is used instead so
    that a sign flip still fails the check.
    """
    return abs(paired_t(dig(rep, a["a"]), dig(rep, a["b"])))


def d_paired_mean_diff(rep, a):
    d = _paired(dig(rep, a["a"]), dig(rep, a["b"]))
    return mean(d) if d else float("nan")


def d_paired_n_positive(rep, a):
    """How many seeds move in the claimed direction.

    This is the assumption-free version of the claim and the manuscript
    quotes it alongside the t. It needs no distributional assumption at all.
    """
    return sum(1 for x in _paired(dig(rep, a["a"]), dig(rep, a["b"])) if x > 0)


def d_interaction_contrast_t(rep, a):
    """One-sample t on the per-seed interaction contrast.

    I_s is formed within each seed, so the four cells' shared run-to-run
    component cancels. The mean of I_s equals the contrast of the cell means,
    because the contrast is linear, so the point estimate is unchanged and
    only its uncertainty differs from the unpaired version.
    """
    cells = ("both", "input only", "weights only", "neither")
    runs = {k: rep[k]["runs"] for k in cells}
    n = min(len(v) for v in runs.values())
    I = [runs["both"][i] - runs["input only"][i]
         - runs["weights only"][i] + runs["neither"][i] for i in range(n)]
    if len(I) < 2:
        return float("nan")
    s = sd(I)
    if s == 0.0:
        return float("nan")
    return mean(I) / (s / math.sqrt(len(I)))


def d_separation_gap(rep, a):
    """Smallest value of a minus largest value of b.

    Positive means the two conditions do not overlap on any seed. For the
    coverage experiment this is a stronger statement than any p-value, and it
    is what the manuscript leads with.
    """
    return min(dig(rep, a["a"])) - max(dig(rep, a["b"]))


def d_sign_test_p(rep, a):
    """Exact binomial sign-test p on per-acquisition deltas.

    args: hi, lo  -- the two by_factor keys
          sided    -- "one" or "two"; the manuscript must say which, because
                      with three of five positive the one-sided value is 0.50
                      and the two-sided value is 1.00.
    """
    rows = rep["per_session"]
    deltas = []
    for _name, r in rows.items():
        hi = r["by_factor"][a["hi"]]
        lo = r["by_factor"][a["lo"]]
        deltas.append(mean(hi) - mean(lo))
    n = len(deltas)
    pos = sum(1 for x in deltas if x > 0)
    upper = sum(_comb(n, k) for k in range(pos, n + 1)) / float(2 ** n)
    if a.get("sided", "two") == "one":
        return upper
    lower = sum(_comb(n, k) for k in range(0, pos + 1)) / float(2 ** n)
    return min(1.0, 2.0 * min(upper, lower))


def _comb(n, k):
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def check_scope(c):
    """manuscript or artefact.

    Removing the compression study from the manuscript did not remove it from
    the released analysis, and the numbers it produced are still worth
    guarding. A check whose number the manuscript no longer prints is marked
    artefact-only: it still runs and still has to agree, but it is counted
    separately so that "every number in the paper is checked" stays a true
    statement about the paper rather than about the repository.
    """
    return c.get("scope", "manuscript")


def report_label(name):
    """How a report is named in the 'absent' list: all its candidates."""
    return name if isinstance(name, str) else " or ".join(name)


DERIVATIONS = {
    "session_recall": d_session_recall,
    "min_session_recall": d_min_session_recall,
    "units_above": d_units_above,
    "recovery_extreme": d_recovery_extreme,
    "recovery_of": d_recovery_of,
    "predicted_fraction": d_predicted_fraction,
    "session_recall_extreme": d_session_recall_extreme,
    "recall_sd_excluding": d_recall_sd_excluding,
    "pearson_excluding": d_pearson_excluding,
    "welch_t": d_welch_t,
    "welch_t_abs": d_welch_t_abs,
    "augment_value": d_augment_value,
    "ablate_drop": d_ablate_drop,
    "interaction_contrast": d_interaction_contrast,
    "paired_t": d_paired_t,
    "paired_t_abs": d_paired_t_abs,
    "paired_mean_diff": d_paired_mean_diff,
    "paired_n_positive": d_paired_n_positive,
    "interaction_contrast_t": d_interaction_contrast_t,
    "separation_gap": d_separation_gap,
    "sign_test_p": d_sign_test_p,
}


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True,
                    help="directory holding the report JSON files")
    ap.add_argument("--expected", default=None,
                    help="expected.json (default: next to this script)")
    ap.add_argument("--only", default=None,
                    help="check only ids containing this substring")
    ap.add_argument("--brief", action="store_true",
                    help="print the table and the summary, without the "
                         "per-disagreement detail")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    spec = json.load(open(args.expected
                          or os.path.join(here, "expected.json")))
    checks = spec["checks"]
    if args.only:
        checks = [c for c in checks if args.only in c["id"]]

    cache = {}

    def load(name):
        """Load a report, accepting any of the names it may carry.

        A check's "report" may be a single filename or a list of candidates,
        tried in order. This exists because the notebook and verify.sh do not
        agree on two names: the notebook writes the coverage experiment to
        step2d_13-48.json and the ten-seed confirmation to step4_seeds10.json,
        while verify.sh writes step2d.json and step4_seeds.json. Both are the
        same experiment. Renaming files by hand to satisfy a checker is how
        the wrong file gets checked, so the checker accepts both instead.
        """
        names = [name] if isinstance(name, str) else list(name)
        key = tuple(names)
        if key not in cache:
            cache[key] = (None, None)
            for n in names:
                p = os.path.join(args.reports, n)
                if os.path.exists(p):
                    cache[key] = (json.load(open(p)), n)
                    break
        return cache[key]

    rows, missing, failures, used = [], {}, [], {}
    for c in checks:
        rep, from_file = load(c["report"])
        if rep is None:
            missing.setdefault(report_label(c["report"]), []).append(c["id"])
            continue
        used.setdefault(from_file, 0)
        used[from_file] += 1
        try:
            if "derive" in c:
                got = DERIVATIONS[c["derive"]](rep, c.get("args") or {})
            else:
                got = dig(rep, c["path"])
        except (KeyError, IndexError, TypeError) as e:
            rows.append((c, None, "no such value: %s" % e))
            failures.append(c)
            continue
        got = float(got)
        if math.isnan(got):
            rows.append((c, got, "value is not a number"))
            failures.append(c)
            continue
        diff = got - float(c["expect"])
        ok = abs(diff) <= float(c["tol"]) + 1e-9
        rows.append((c, got, None if ok else "outside tolerance"))
        if not ok:
            failures.append(c)

    # ------------------------------------------------------------- output
    def fmt(v, as_count=False, signed=False):
        """Counts read better as integers; measurements to four places."""
        if v is None:
            return "-"
        if as_count and abs(v - round(v)) < 1e-9:
            return "%+d" % round(v) if signed else "%d" % round(v)
        return "%+.4f" % v if signed else "%.4f" % v

    w = max([len(c["id"]) for c in checks] + [8])
    print("=" * (w + 56))
    print("%-*s %11s %11s %11s  %s" % (w, "check", "paper", "this run",
                                       "diff", "verdict"))
    print("=" * (w + 56))
    for c, got, why in rows:
        exp = float(c["expect"])
        # a claim whose expected value and tolerance are both whole numbers
        # is a count, and prints as one
        cnt = (abs(exp - round(exp)) < 1e-9
               and abs(float(c["tol"]) - round(float(c["tol"]))) < 1e-9)
        if got is None:
            print("%-*s %11s %11s %11s  FAIL  %s"
                  % (w, c["id"], fmt(exp, cnt), "-", "-", why))
            continue
        mark = "ok" if why is None else "FAIL"
        print("%-*s %11s %11s %11s  %-5s %s"
              % (w, c["id"], fmt(exp, cnt), fmt(got, cnt),
                 fmt(got - exp, cnt, signed=True), mark,
                 "" if why is None else "tol %s" % fmt(float(c["tol"]), cnt)))

    tested = len(rows)
    print("=" * (w + 56))
    print("%d of %d claims tested, %d agreed, %d disagreed"
          % (tested, len(checks), tested - len(failures), len(failures)))
    man = [r for r in rows if check_scope(r[0]) == "manuscript"]
    art = [r for r in rows if check_scope(r[0]) == "artefact"]
    if art:
        print("  of these, %d are numbers the manuscript prints and %d guard"
              % (len(man), len(art)))
        print("  analysis released in the artefact but not printed in the paper")

    if used:
        print("\nread from:")
        for name in sorted(used):
            print("  %-26s %d claims" % (name, used[name]))

    if missing:
        n = sum(len(v) for v in missing.values())
        print("\n%d claims could not be tested because their report is "
              "absent:" % n)
        for name in sorted(missing):
            print("  %-26s covers %d claims" % (name, len(missing[name])))
        print("These are not failures. They are claims this run did not "
              "reach.")

    if failures and args.brief:
        print("\n%d disagreements. Rerun without --brief for the paper "
              "reference and the tolerance\nreason recorded for each."
              % len(failures))
        return 1

    if failures:
        print("\nDisagreements, with the tolerance reason recorded for each:")
        for c in failures:
            print("\n  %s" % c["id"])
            print("    paper      %s" % c["paper"])
            print("    tolerance  %s because %s" % (c["tol"], c["why_tol"]))
        print("\nA disagreement on an exact count means the archive differs "
              "from the one used.\nA disagreement on a trained quantity, "
              "outside the spread the paper itself\nreports for it, means "
              "the conclusion drawn from that number does not hold\nhere. "
              "Either is worth reporting.")
        return 1

    if tested == 0:
        print("\nNothing was tested. Point --reports at a directory holding "
              "the report\nfiles a run produced.")
        return 1

    print("\nEvery claim this run reached agreed with the manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
