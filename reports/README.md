# reports

This directory is where the reports from the manuscript's own run belong,
one JSON file per stage:

```
step1_report.json      inventory, deduplication, acquisition grouping, splits
step2_compare.json     the grouped against random split comparison
step2b_kfold.json      the cross-validated protocol comparison (Table III)
step2b_norm.json       the normalisation ablation (Table II)
step2c_sessions.json   per-acquisition recall against headroom (Table IV)
step2d_13-48.json      the coverage experiment on drone/13-48
step3_grid.json        the compression grid
step3_ablate.json      which quantisation axis does the damage
step4_seeds10.json     the ten-seed confirmation (Table V)
step4_augment.json     the shift-augmentation mechanism test
step4_ablate.json      the factorial quantisation experiment
step5_loo.json         leave one acquisition out (Table VI)
step2d_13-48_10seeds.json  the ten-seed coverage test (Table V)
step2d_control.json    the coverage test on the pedestrian control 11-23
step2d_control2.json   the coverage test on the vehicle control 15-37
step2d_centroids.json  per-acquisition Doppler centroid, mean and spread
capacity.json          the capacity and schedule sweep behind the gap analysis
temporal.json          the three-frame input experiment (Table VII)
```

With these present, the manuscript's numbers can be checked without
retraining anything:

```bash
python check_expected.py --reports reports/
```

That is a check of internal consistency: it confirms that the numbers printed
in the paper are the numbers these files contain. It is not a reproduction,
because it does not rerun the training. For that, use `verify.sh`, which
produces a fresh set of these files from the archive and then runs the same
comparison against them.

If this directory is empty apart from this file, the reports have not been
committed yet and `check_expected.py` will report every claim as untested
rather than as failing.

Two files go by more than one name, and the checker accepts either, so there
is never a reason to rename one by hand. The notebook writes
`step2d_13-48.json` and `step4_seeds10.json`; `verify.sh` writes the same two
experiments under those names too, and older runs may have called them
`step2d.json` and `step4_seeds.json`. Where both a five-seed
`step4_seeds.json` and a ten-seed `step4_seeds10.json` are present, the
checker reads the ten-seed file, because Table V reports ten seeds. The
summary prints which file each claim was read from, so you can confirm it
took the one you meant.
