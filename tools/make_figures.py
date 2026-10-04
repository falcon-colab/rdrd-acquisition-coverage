"""Two print figures built from the released reports, no GPU and no retraining.

Both answer requests a reviewer made for visual evidence that the manuscript
previously carried only as prose: the spread of per-acquisition recall across
all 52 units, and where 13-48 sits in Doppler among the 21 drone units.

Colour is Okabe-Ito, which passes a colour-vision-deficiency check, but no
figure relies on it: class identity is carried by row position and marker
shape as well, so both survive greyscale printing.

    python tools/make_figures.py --reports reports --out figures
"""
import argparse, json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
STYLE = {"drone": (BLUE, "o"), "car": (VERM, "s"), "person": (GREEN, "^")}
LABEL = {"drone": "drone", "car": "vehicle", "person": "pedestrian"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 400})


def fig_recall(sess, out):
    fig, ax = plt.subplots(figsize=(5.4, 2.0))
    rng = np.random.default_rng(0)
    for row, cls in enumerate(("drone", "car", "person")):
        colour, marker = STYLE[cls]
        rows = sess[cls]["sessions"]
        x = np.array([r["recall"] for r in rows], float)
        y = row + (rng.random(len(x)) - 0.5) * 0.26      # jitter, not data
        ax.scatter(x, y, s=22, facecolor="none", edgecolor=colour,
                   marker=marker, linewidths=0.9, zorder=3)
        tgt = [r for r in rows if r["name"].endswith("13-48")]
        if tgt:
            ax.scatter([tgt[0]["recall"]], [row], s=46, color=colour,
                       marker=marker, zorder=4)
            ax.annotate("13-48", (tgt[0]["recall"], row),
                        textcoords="offset points", xytext=(7, 7),
                        fontsize=7.5, color=INK)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([LABEL[c] for c in ("drone", "car", "person")])
    ax.set_xlabel("out-of-fold recall, one point per acquisition")
    ax.set_xlim(0.0, 1.02); ax.set_ylim(-0.6, 2.6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "recall_by_acquisition.png"),
                bbox_inches="tight")
    plt.close(fig)
    n = sum(len(sess[c]["sessions"]) for c in ("drone", "car", "person"))
    print("  recall_by_acquisition.png   %d acquisitions" % n)


def fig_doppler(cen, out):
    rows = cen["by_acquisition"]
    tgt = [r for r in rows if r["acquisition"].endswith("13-48")][0]
    oth = [r for r in rows if r is not tgt]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    ax.scatter([r["centroid_mean"] for r in oth], [r["centroid_sd"] for r in oth],
               s=24, facecolor="none", edgecolor=BLUE, marker="o",
               linewidths=0.9, zorder=3)
    ax.scatter([tgt["centroid_mean"]], [tgt["centroid_sd"]], s=60, color=VERM,
               marker="D", zorder=4)
    ax.annotate("13-48", (tgt["centroid_mean"], tgt["centroid_sd"]),
                textcoords="offset points", xytext=(9, -2), fontsize=8,
                color=INK)
    ax.annotate("the other 20 drone\nacquisitions",
                (float(np.mean([r["centroid_mean"] for r in oth])),
                 float(np.mean([r["centroid_sd"] for r in oth]))),
                textcoords="offset points", xytext=(-16, 20), fontsize=7.5,
                color=MUTED, ha="center")
    ax.set_xlabel("mean Doppler centroid (bin, of 61)")
    ax.set_ylabel("within-acquisition sd (bins)")
    ax.set_xlim(0, 36); ax.set_ylim(-0.6, 12)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "doppler_by_acquisition.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  doppler_by_acquisition.png  %d drone acquisitions" % len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--also", default="paper/figures",
                    help="second directory to copy into, so the copy the "
                         "paper compiles against cannot go stale behind the "
                         "one this script writes. Pass an empty string to "
                         "skip.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_recall(json.load(open(os.path.join(a.reports, "step2c_sessions.json"))), a.out)
    fig_doppler(json.load(open(os.path.join(a.reports, "step2d_centroids.json"))), a.out)

    if a.also and os.path.isdir(os.path.dirname(a.also.rstrip("/")) or "."):
        os.makedirs(a.also, exist_ok=True)
        for name in ("recall_by_acquisition.png", "doppler_by_acquisition.png"):
            src = os.path.join(a.out, name)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(a.also, name))
        print("  copied both into %s" % a.also)


if __name__ == "__main__":
    main()
