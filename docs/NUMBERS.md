# Every number in the paper, and the command that produces it

Paths below are relative to the repository root. `$DS` is
`runs/<date>/cache/dataset.npz`, built by stage 2. `$R` is
`runs/<date>/reports/`. `verify.sh` runs all of these in order; this file is
for checking one number without rerunning everything.

Add `--out $R/<name>.json` to any command to write its report, which is what
`check_expected.py` then reads.

---

## Section III, the data

**17,485 samples after deduplication; 5720 vehicle, 5065 drone, 6700
pedestrian. 80 folders to 52 acquisitions; 27 vehicle folders to 15, 21 drone
to 21, 32 pedestrian to 16.**

```bash
python src/run_step1.py --root rdrd/ --out runs/x/splits --per-session 120 --top-k 12
```

Printed by stages 1, 2 and 4 of that script's output. The total also lands in
`step1_report.json` at `dedup.total`, which is the `samples-total` check.

**Published acquisition parameters** (8.75 GHz, 500 MHz bandwidth, 0.878 m
range resolution, 3.596 km maximum unambiguous range, 0.34 km/h Doppler
resolution, 512 integrated ramps) are quoted from the source publication and
are not recomputed here. The paper says so where it quotes them.

**101,139 parameters.**

```bash
python -c "import sys; sys.path.insert(0,'src'); import torch, torch.nn as nn; from step2 import make_model; print(sum(p.numel() for p in make_model(torch,nn).parameters()))"
```

Also asserted by `tests/test_core.py::test_model_parameter_count_is_the_reported_figure`,
and present in `step2_compare.json` at `runs.grouped.0.params`.

**Table II, the normalisation ablation** (offset 0.925 / 0.046, global 0.922 /
0.045, peak+level 0.922 / 0.037, peak 0.908 / 0.044).

```bash
python src/step2b.py norm --data $DS --folds 5 --epochs 40
```

The claim that quantising before or after `offset` normalisation is the same
operation up to scale is asserted by
`tests/test_core.py::test_offset_normalisation_commutes_with_quantisation`,
which does not need the archive.

**Table V's input sizes** (671, 330, 165, 77 cells; 2684, 1320, 660, 308
bytes at 32 bits) come from `reduce_doppler` and are asserted by
`tests/test_core.py::test_aggregation_shapes_match_reported_cell_counts`.
That aggregation conserves total linear power exactly is asserted by
`test_aggregation_conserves_linear_power`.

---

## Section IV, the split comparison

**Table III** (accuracy 0.9335 / 0.0055 / 0.0030 random against 0.9099 /
0.0427 / 0.0062 grouped; drone recall 0.9358 / 0.0078 / 0.0144 against
0.8376 / 0.1389 / 0.0383; vehicle F1 0.9055 / 0.0073 / 0.0043 against
0.8979 / 0.0117 / 0.0095).

```bash
python src/step2b.py kfold --data $DS --folds 5 --seeds 0 1 2 --scheme offset --epochs 40
```

The three columns are `mean`, `between_fold_sd` and `within_fold_sd` under
`summary.<metric>.<split>`. Metric keys carry spaces: `accuracy`,
`drone recall`, `drone f1`, `drone precision`, `car f1`.

**t = -1.19 for accuracy and -1.12 for drone F1, neither significant at five
folds.** Same command; printed at the end as the paired comparison, and
stored under `paired`.

**Per-fold drone F1 differences +0.003, -0.023, +0.007, -0.226, -0.010.**
Same command; the `fold_means` arrays under each metric.

---

## Section IV, the acquisition-level failure

**Table IV's three recall figures come from three different experiments and
must not be conflated.** The paper has a table naming each one, and it is
worth repeating here:

| Recall | Experiment | Command |
|---|---|---|
| 0.249 | cross-validation, whole acquisition as one out-of-fold test fold | `step2c.py` |
| 0.157 | the coverage test, a held-back half of the acquisition | `step2d.py` |
| 0.222 | the compression baseline, whole acquisition, ten seeds, full resolution | `step4.py seeds` |

**Per-acquisition recall; twenty drone acquisitions between 0.90 and 0.99 and
13-48 at 0.249; weakest pedestrian 0.7776 and weakest vehicle 0.8189; recall
sd 0.0262 drone (excluding 13-48), 0.0505 vehicle, 0.0549 pedestrian.**

```bash
python src/step2c.py --data $DS --folds 5 --seeds 0 1 --scheme offset --epochs 40
```

The drone sd of 0.0262 is computed over the drone acquisitions **with 13-48
excluded**, which is what makes it lower than the other two classes.
`check_expected.py` recomputes it that way from the `sessions` list rather
than reading the report's own `recall_sd`, which includes 13-48. An earlier
draft claimed a 17.8-fold variance asymmetry from this stage; that claim was
one recording rather than a class property and was withdrawn.

**Pearson -0.62 between drone recall and headroom, falling to -0.16 once
13-48 is excluded, with the rank correlation never significant.** Same
command. The fall is the refutation of the headroom mechanism.

**Table IV's acquisition profile** (Doppler centroid 8.50 against 30.04,
z = -15.67; peak Doppler bin 7.58 against 30.01, z = -8.16; headroom 43.74
against 24.70, z = +2.57; dynamic range 72.95 against 53.99, z = +2.07; peak
level -72.33 against -88.95, z = +2.26), **the labelling check at 100
percent nearest the drone centroid**, and **the coverage test** (held out:
6.0 vehicle, 31.0 drone, 160.0 pedestrian predictions, recall 0.157; half in
training: 4.5, 190.5, 2.0, recall 0.967).

```bash
python src/step2d.py --data $DS --target drone/13-48 --seeds 0 1 --epochs 40
```

The confusion row sums to 197, not 191: all three predicted classes are shown
in the paper's table because an earlier draft dropped the vehicle column and
the recall could not then be reconciled.

**Doppler centroid 7.6 with sd 12.1 on 13-48, against 0.4 to 6.0 elsewhere.**
Same command, the `[A]` profile section.

---

## Section V, compression

**Table V, the full grid.**

```bash
python src/step3.py grid --data $DS --seeds 0 1 --factors 1 2 4 8 --bits 32 16 8 4 --epochs 40
```

**The ten-seed confirmation the paper's Table V values come from** (32-bit:
standard 0.9210 / 0.0414, 0.9322 / 0.0207, 0.9056 / 0.0457, 0.9011 / 0.0596
and unseen 0.2221 / 0.0408, 0.3295 / 0.1434, 0.3975 / 0.1264, 0.1916 /
0.0267 at 1x, 2x, 4x, 8x; 8-bit: standard 0.9235, 0.9156, 0.9212, 0.9128 and
unseen 0.2341, 0.3074, 0.3944, 0.1911).

```bash
python src/step4.py seeds --data $DS --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 40
```

**Welch t = +4.17 at 32 bits and +3.08 at 8 bits for 4x against 1x on the
unseen regime; t = 0.79 and 0.16 on the standard split.** Printed by the same
command. `check_expected.py` recomputes all four from the `standard_runs` and
`unseen_runs` arrays, so the statistic is checked rather than trusted.

**Bootstrap intervals [+0.049, +0.307] and [+0.058, +0.392].**

```bash
python src/analyse4.py $R/step4_seeds.json
```

No retraining. Also prints the gap statistic, which the paper notes is
scale-free and therefore unreliable where the spread is small.

**Which quantisation axis does the damage** (input alone 0.0159, weights
alone 0.1028, joint 0.1412 against an additive prediction of 0.1187; accuracy
sd 0.176 weights-only and 0.134 joint, with individual seeds collapsing to
0.534 and 0.585).

```bash
python src/step4.py ablate --data $DS --seeds 0 1 2 3 4 --epochs 40
python src/step3.py ablate --data $DS --seeds 0 1 --epochs 40
```

**The interaction contrast I = -0.0225, 95 percent CI [-0.185, +0.161],
t = -0.23.**

```bash
python src/interaction.py $R/step4_ablate.json
```

No retraining. The paper states this as a failure to distinguish the
interaction from zero, not as evidence of additivity, and the script prints
that distinction itself.

---

## Section V and VI, what did not hold

**Table VI, leave one acquisition out** (13-48 +0.136, 15-37 +0.048, 15-21
+0.020, 12-34 -0.017, 11-23 -0.022; mean +0.033, paired t +1.16, three of
five improved, exact sign test p = 0.50).

```bash
python src/step5.py --data $DS \
  --sessions drone/13-48 car/15-37 drone/15-21 drone/12-34 person/11-23 \
  --seeds 0 1 2 --factors 1 4 --epochs 40
```

This is the test that stops the aggregation effect from being stated as a
general result.

**The shift-augmentation refutation** (0.1608, 0.1613, 0.1603 at plus or
minus 4, 8 and 16 bins against a baseline of 0.2214).

```bash
python src/step4.py augment --data $DS --seeds 0 1 2 3 4 --shifts 4 8 16 --epochs 40
```

Augmentation fills vacated cells with each sample's own noise floor rather
than rolling the array around, because a circular roll would wrap a strong
return to the opposite Doppler extreme and create a signature no radar would
produce.

**The leakage screen** (within-unit against cross-unit similarity gaps of
+0.024 vehicle, +0.050 drone, +0.034 pedestrian, small enough that random
partitioning on this database is closer to honest than expected).

```bash
python src/run_step1.py --root rdrd/ --out runs/x/splits --per-session 120 --top-k 12
```

Stage 6 of that output. This null is reported in the paper rather than
dropped, and it is the reason the paper's claim is about coverage rather than
about leakage.

**The rotor-sideband mechanism** is refuted arithmetically, not
experimentally: 61 Doppler bins at 0.34 km/h spans about 21 km/h, far below
blade-tip Doppler. No command.

**The cell-count redundancy mechanism** is refuted by the peak-relative
concentration measure in `step2d.py`'s `[A]` section. An earlier version of
that measure summed dB values, which is meaningless, and made all three
classes look identical at roughly 510 of 671 cells. The current measure
counts cells within 10 dB of each sample's own peak.
