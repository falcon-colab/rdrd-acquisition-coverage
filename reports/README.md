# reports

This directory is where the reports from the manuscript's own run belong,
one JSON file per stage:

```
step1_report.json      inventory, deduplication, acquisition grouping, splits
step2_compare.json     the grouped against random split comparison
step2b_kfold.json      the cross-validated protocol comparison (Table III)
step2b_norm.json       the normalisation ablation (Table II)
step2c_sessions.json   per-acquisition recall against headroom (Table IV)
step2d.json            the coverage experiment on drone/13-48
step3_grid.json        the compression grid
step3_ablate.json      which quantisation axis does the damage
step4_seeds.json       the ten-seed confirmation (Table V)
step4_augment.json     the shift-augmentation mechanism test
step4_ablate.json      the factorial quantisation experiment
step5_loo.json         leave one acquisition out (Table VI)
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
