"""Figure 2: the failing acquisition against a typical one and the class it
is mistaken for, as mean range-Doppler maps.

WHY THIS SCRIPT EXISTS. The published version of this figure was three
bare `imshow` panels with titles and nothing else: no axes, no tick
labels, no colour scale and no colourbar. That is a problem specific to
this figure rather than a matter of taste. The paper's central
descriptive claim is a position on the Doppler axis, bin 8.50 against
29.70 to 31.19 for every other drone acquisition, and a panel with no
Doppler axis gives the reader no way to read that position off the
picture. A reviewer asked for the figure to be improved and was right to.

WHAT CHANGED, and why each change is needed rather than decorative:

  Axes. Doppler bin on x, range bin on y, with ticks. Zero Doppler sits
  at bin 30, marked with a faint line, so "left of centre" has a
  meaning on the page.

  One colour scale across all three panels, with a single colourbar.
  Three panels scaled independently cannot be compared, and the old
  figure gave no way to tell whether they had been. Levels are
  peak-referenced per sample before averaging, so a panel's brightness
  reports shape rather than how strong that day's returns happened to
  be, and the colourbar says so.

  A Doppler profile drawn inside each panel: the panel's energy summed
  over range, scaled to the panel height. This costs no vertical space
  and turns an impression about where the energy sits into a line a
  reader can follow, with the centroid marked and its value printed.

  Viridis, which is monotonic in lightness, so the figure survives
  greyscale printing and colour-vision deficiency. The Okabe-Ito palette
  used elsewhere in the paper is categorical and is wrong for a
  continuous field; one perceptually uniform ramp is the right choice
  here, and the two conventions do not clash because they encode
  different kinds of thing.

RUNNING IT. Needs the prepared dataset, so it runs where the data is
rather than from the paper directory:

    python tools/make_fig_session.py --data DIR/dataset.npz \\
        --out figures --also paper/figures

The defaults reproduce the published panels: the failing acquisition
drone/13-48, drone/12-34 as the typical drone comparison, and the whole
pedestrian class, which is where 79 per cent of 13-48's held-out samples
are sent. Pass --typical to choose a different middle panel; the script
prints that unit's out-of-fold recall if the report is available, so the
choice can be stated in the caption rather than left to the reader to
wonder about.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

INK, MUTED, LINE = "#1a1a1a", "#555555", "#ffffff"

# Two targets. "column" is drawn at the width it is printed at, 3.45 in,
# so its 6.5 pt type reaches the page as 6.5 pt. The previous version was
# drawn at 7.2 in and scaled down to fit one column, which took its
# labels to roughly 3 pt, and is the reason the figure was unreadable.
# "page" is the same figure for a two-column float or a slide.
SIZES = {"column": {"fig": (3.45, 1.26), "base": 6.5, "tick": 6.0,
                    "title": 6.5, "note": 6.0, "dop": 30,
                    "roles": False, "wspace": 0.10},
         "page":   {"fig": (7.20, 1.85), "base": 8.0, "tick": 7.0,
                    "title": 7.5, "note": 7.0, "dop": 15,
                    "roles": True, "wspace": 0.07}}


def style(z):
    plt.rcParams.update({
        "font.size": z["base"], "axes.labelsize": z["base"],
        "xtick.labelsize": z["tick"], "ytick.labelsize": z["tick"],
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "figure.dpi": 600})


def peak_referenced(X):
    """Each sample expressed in dB below its own peak, then averaged.

    Averaging raw dB would let one loud acquisition dominate a panel and
    would make the three panels incomparable, which is exactly the fault
    the shared colour scale is meant to remove. Referencing to the
    per-sample peak first is the same normalisation the feature code in
    src/step2d.py uses, so the figure and the numbers in the text are
    describing the same quantity.
    """
    peak = X.reshape(len(X), -1).max(axis=1)
    return (X - peak[:, None, None]).mean(axis=0)


def doppler_profile(M):
    """Energy summed over range, in linear units, normalised to [0, 1].

    Linear rather than dB: summing dB across range has no physical
    meaning, and the profile is here to show where the energy is, which
    is a statement about power.
    """
    lin = 10.0 ** (M / 10.0)
    p = lin.sum(axis=0)
    p = p - p.min()
    return p / max(p.max(), 1e-30)


def _features(X):
    """src/step2d.py's own feature code, imported rather than copied.

    THIS IS THE BUG THIS FIGURE SHIPPED WITH ONCE. The first version
    computed the Doppler centroid on the mean map, taking that map's
    median as the floor. The paper computes it per sample, each against
    its own floor, and reports the mean of those. For an acquisition
    whose samples all sit at the same Doppler bin the two agree, which
    is why the typical drone and the pedestrian panels looked right. For
    \texttt{13-48}, whose per-sample Doppler scatter is 10.64 bins, the
    mean map smears and its centroid is pulled towards the middle: the
    figure printed 12.5 against the 8.50 the text and Fig. 3 both quote,
    two different numbers for one named quantity on one page.

    Importing the real function removes the possibility of the two
    definitions drifting apart again.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from step2d import sample_features           # noqa: E402
    return sample_features(X)


def centroid_bin(X):
    """Mean of the per-sample Doppler centroids, as the text defines it."""
    return float(_features(X)["doppler_centroid"].mean())


def doppler_profile_mean(X):
    """Mean of the per-sample Doppler profiles, normalised to [0, 1].

    The profile of the mean map would disagree with the centroid marker
    for the same reason the old centroid did, so the curve is averaged
    the same way the number is: per sample first, then across samples.
    """
    peak = X.reshape(len(X), -1).max(axis=1)
    lin = 10.0 ** ((X - peak[:, None, None]) / 10.0)
    p = lin.sum(axis=1)                          # over range -> (n, W)
    p = p / np.maximum(p.max(axis=1, keepdims=True), 1e-30)
    p = p.mean(axis=0)
    p = p - p.min()
    return p / max(p.max(), 1e-30)


def panels(X, y, unit, target, typical, confused):
    """The three mean maps, in the order the argument is made."""
    out = []
    for name, role, sel in (
            (target.split("/")[-1], "the failing acquisition", unit == target),
            (typical.split("/")[-1], "a typical drone acquisition",
             unit == typical),
            ("pedestrian", "the class it is sent to", y == confused)):
        Xs = X[sel]
        out.append((name, role, peak_referenced(Xs), int(sel.sum()),
                    centroid_bin(Xs), doppler_profile_mean(Xs)))
    return out


def draw(items, out_dir, z, name="session_13-48_compact.png"):
    style(z)
    H, W = items[0][2].shape
    lo = min(np.percentile(m, 2.0) for _, _, m, _, _, _ in items)
    hi = max(m.max() for _, _, m, _, _, _ in items)

    fig, axes = plt.subplots(1, 3, figsize=z["fig"], sharey=True,
                             gridspec_kw={"wspace": z["wspace"]})

    for ax, (title, role, M, n, c, prof) in zip(axes, items):
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       vmin=lo, vmax=hi, interpolation="nearest",
                       extent=(-0.5, W - 0.5, -0.5, H - 0.5))
        # zero Doppler, so that "left of centre" is readable as a fact
        ax.axvline((W - 1) / 2.0, color=LINE, lw=0.7, alpha=0.5,
                   linestyle=(0, (2.5, 2)))

        ax.plot(np.arange(W), prof * (H - 1) * 0.80 - 0.5, color=LINE,
                lw=1.0, alpha=0.9, solid_joinstyle="round")

        ax.plot([c], [-0.5], marker="^", ms=5.5, color=LINE, clip_on=False,
                markeredgecolor=INK, markeredgewidth=0.4, zorder=5)
        # inside the panel, so it cannot crowd the title
        ax.text(0.96, 0.05, "%.1f" % c, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=z["note"], color=LINE)

        head = ("%s   $n=%d$\n%s" % (title, n, role) if z["roles"]
                else "%s   $n=%d$" % (title, n))
        ax.set_title(head, fontsize=z["title"], color=INK,
                     linespacing=1.25, pad=3)
        if z["roles"]:
            ax.set_xlabel("Doppler bin (of %d)" % W, labelpad=2)
        ax.xaxis.set_major_locator(MultipleLocator(z["dop"]))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=2)

    axes[0].set_ylabel("range bin", labelpad=2)
    if not z["roles"]:
        # one label under the middle panel rather than three, which
        # centres it and reserves its space without hand-placing it; and
        # the rightmost tick label dropped on every panel but the last,
        # where adjacent panels would otherwise print 60 and 0 on top of
        # each other
        axes[1].set_xlabel("Doppler bin (of %d)" % W, labelpad=2)
        ticks = list(range(0, W, z["dop"]))
        for i, ax in enumerate(axes):
            ax.set_xticks(ticks)
            lab = [str(t) for t in ticks]
            if i < len(axes) - 1:
                lab[-1] = ""
            ax.set_xticklabels(lab)

    cb = fig.colorbar(im, ax=list(axes), location="right", fraction=0.018,
                      pad=0.012)
    # a horizontal caption above the bar: a rotated label on a figure this
    # short gets clipped by the tight bounding box
    cb.set_label("dB below per-sample peak", fontsize=z["note"],
                 labelpad=3)
    cb.ax.tick_params(labelsize=z["note"], length=2)
    cb.outline.set_visible(False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  %s  %d panels, shared scale %.1f to %.1f dB"
          % (path, len(items), lo, hi))
    for title, role, M, n, c, _ in items:
        print("    %-26s centroid %6.2f   n=%d" % (title, c, n))
    return path


def recall_of(reports, unit):
    """The unit's out-of-fold recall, so the caption can state it."""
    p = os.path.join(reports, "step2c_sessions.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for cls in d:
        for r in d[cls].get("sessions", []):
            if r["name"] == unit or r["name"].endswith("/" + unit):
                return float(r["recall"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="drone/13-48")
    ap.add_argument("--typical", default="drone/12-34")
    ap.add_argument("--confused", default="person")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--width", default="column",
                    choices=sorted(SIZES),
                    help="column: drawn at the 3.45 in it is printed at. "
                         "page: a two-column float or a slide.")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--also", default="paper/figures",
                    help="second directory to copy into, so the copy the "
                         "paper compiles against cannot go stale behind "
                         "the one this script writes")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=False)
    X, y, unit = d["X"], d["y"], d["unit"]
    y = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in y])
    unit = np.array([s.decode() if isinstance(s, bytes) else str(s)
                     for s in unit])

    for u in (a.target, a.typical):
        if not (unit == u).any():
            raise SystemExit("no such unit in the data: %s" % u)
    if not (y == a.confused).any():
        raise SystemExit("no such class in the data: %s" % a.confused)

    path = draw(panels(X, y, unit, a.target, a.typical, a.confused),
                a.out, SIZES[a.width])

    r = recall_of(a.reports, a.typical)
    if r is not None:
        print("    %s out-of-fold recall %.4f (state this in the caption)"
              % (a.typical, r))

    if a.also:
        os.makedirs(a.also, exist_ok=True)
        shutil.copyfile(path, os.path.join(a.also, os.path.basename(path)))
        print("  copied into %s" % a.also)


if __name__ == "__main__":
    main()
