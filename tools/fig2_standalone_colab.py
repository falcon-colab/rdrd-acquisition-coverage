# =====================================================================
# FIGURE 2, STANDALONE. Paste this whole thing into one Colab cell.
#
# No git, no clone, no pull. The only thing it needs is the dataset.npz
# this notebook already built; it looks for that itself.
#
# It stops with an error rather than drawing a wrong figure. The check
# that matters: acquisition 13-48 must come out at a Doppler centroid of
# 8.50, the number Section IV-C and Fig. 3 both quote. The earlier
# version of this figure printed 12.5 because it measured the centroid
# on the averaged map instead of per sample, and 12.5 contradicted the
# text three lines below it. The assertion at the bottom is what makes
# that failure impossible to ship a second time.
# =====================================================================
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ------------------------------------------------- find the dataset
CANDIDATES = ["/content/work/dataset.npz", "/content/dataset.npz"]
CANDIDATES += sorted(glob.glob("/content/**/dataset.npz", recursive=True))
CANDIDATES += sorted(glob.glob("/content/drive/**/dataset.npz", recursive=True))
DATA = next((p for p in CANDIDATES if os.path.exists(p)), None)
if DATA is None:
    sys.exit("dataset.npz not found under /content. Set DATA by hand:\n"
             "    DATA = '/path/to/dataset.npz'\n"
             "and delete the three lines above this message.")
print("dataset:", DATA)

d = np.load(DATA, allow_pickle=False)
X, y, unit = d["X"], d["y"], d["unit"]
dec = lambda a: np.array([s.decode() if isinstance(s, bytes) else str(s)
                          for s in a])
y, unit = dec(y), dec(unit)
print("X", X.shape, " classes", sorted(set(y.tolist())))

TARGET, TYPICAL, CONFUSED = "drone/13-48", "drone/12-34", "person"

# ---------------------------------------- the paper's own definition
# Copied verbatim from the Doppler part of sample_features() in
# src/step2d.py. Per sample: reference to that sample's own peak, take
# the excess above that sample's own median floor, collapse over range,
# then take the mass-weighted mean bin. The figure averages these
# per-sample centroids; it does NOT measure the averaged map.
def doppler_centroid_per_sample(X):
    n, H, W = X.shape
    flat = X.reshape(n, -1)
    peak = flat.max(axis=1)
    lin = 10.0 ** ((X - peak[:, None, None]) / 10.0)
    linf = lin.reshape(n, -1)
    floor = np.median(linf, axis=1)
    excess = np.maximum(linf - floor[:, None], 0.0).reshape(n, H, W)
    cmass = excess.sum(axis=1)
    cols = np.arange(W)[None, :]
    return (cmass * cols).sum(1) / np.maximum(cmass.sum(1), 1e-12)

def peak_referenced(X):
    peak = X.reshape(len(X), -1).max(axis=1)
    return (X - peak[:, None, None]).mean(axis=0)

def doppler_profile_mean(X):
    # averaged the same way the centroid is, per sample then across
    # samples, so the curve and the marker cannot disagree
    peak = X.reshape(len(X), -1).max(axis=1)
    lin = 10.0 ** ((X - peak[:, None, None]) / 10.0)
    p = lin.sum(axis=1)
    p = p / np.maximum(p.max(axis=1, keepdims=True), 1e-30)
    p = p.mean(axis=0)
    p = p - p.min()
    return p / max(p.max(), 1e-30)

items = []
for name, sel in ((TARGET.split("/")[-1], unit == TARGET),
                  (TYPICAL.split("/")[-1], unit == TYPICAL),
                  ("pedestrian", y == CONFUSED)):
    if not sel.any():
        sys.exit("nothing selected for %s. Check the unit names in the "
                 "npz: %s" % (name, sorted(set(unit.tolist()))[:8]))
    Xs = X[sel]
    items.append((name, peak_referenced(Xs), int(sel.sum()),
                  float(doppler_centroid_per_sample(Xs).mean()),
                  doppler_profile_mean(Xs)))

# ----------------------------------------------------------- drawing
INK, MUTED, LINE = "#1a1a1a", "#555555", "#ffffff"
# drawn at 3.45 in, the width it is printed at, so 6.5 pt type reaches
# the page as 6.5 pt rather than being scaled down to about 3 pt
FIG, BASE, TICK, TITLE, NOTE, DOP = (3.45, 1.26), 6.5, 6.0, 6.5, 6.0, 30
plt.rcParams.update({
    "font.size": BASE, "axes.labelsize": BASE, "xtick.labelsize": TICK,
    "ytick.labelsize": TICK, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 600})

H, W = items[0][1].shape
lo = min(np.percentile(m, 2.0) for _, m, _, _, _ in items)
hi = max(m.max() for _, m, _, _, _ in items)

fig, axes = plt.subplots(1, 3, figsize=FIG, sharey=True,
                         gridspec_kw={"wspace": 0.10})
for ax, (title, M, n, c, prof) in zip(axes, items):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                   vmin=lo, vmax=hi, interpolation="nearest",
                   extent=(-0.5, W - 0.5, -0.5, H - 0.5))
    ax.axvline((W - 1) / 2.0, color=LINE, lw=0.7, alpha=0.5,
               linestyle=(0, (2.5, 2)))                      # zero Doppler
    ax.plot(np.arange(W), prof * (H - 1) * 0.80 - 0.5, color=LINE,
            lw=1.0, alpha=0.9, solid_joinstyle="round")
    ax.plot([c], [-0.5], marker="^", ms=5.5, color=LINE, clip_on=False,
            markeredgecolor=INK, markeredgewidth=0.4, zorder=5)
    ax.text(0.96, 0.05, "%.1f" % c, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=NOTE, color=LINE)
    ax.set_title("%s   $n=%d$" % (title, n), fontsize=TITLE, color=INK,
                 linespacing=1.25, pad=3)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=2)

axes[0].set_ylabel("range bin", labelpad=2)
axes[1].set_xlabel("Doppler bin (of %d)" % W, labelpad=2)
ticks = list(range(0, W, DOP))
for i, ax in enumerate(axes):                 # drop the colliding 60 / 0
    ax.set_xticks(ticks)
    lab = [str(t) for t in ticks]
    if i < len(axes) - 1:
        lab[-1] = ""
    ax.set_xticklabels(lab)

cb = fig.colorbar(im, ax=list(axes), location="right", fraction=0.018,
                  pad=0.012)
cb.set_label("dB below per-sample peak", fontsize=NOTE, labelpad=3)
cb.ax.tick_params(labelsize=NOTE, length=2)
cb.outline.set_visible(False)

OUT = "/content/session_13-48_compact.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("\nshared colour scale %.1f to %.1f dB" % (lo, hi))
for title, M, n, c, _ in items:
    print("  %-12s n=%5d   Doppler centroid %6.2f" % (title, n, c))

# ------------------------------------------------- the one real check
got = items[0][3]
assert abs(got - 8.50) < 0.05, (
    "13-48 came out at %.2f, not 8.50. The figure would contradict "
    "Section IV-C and Fig. 3. Do not use it." % got)
print("\nOK: 13-48 at %.2f matches the 8.50 in the text." % got)
print("written", OUT)

from IPython.display import Image, display
display(Image(OUT))
try:
    from google.colab import files
    files.download(OUT)
except Exception:
    print("(download it from the file browser on the left)")
