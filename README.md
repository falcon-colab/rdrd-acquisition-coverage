# Aggregate accuracy conceals an acquisition-level failure in a public radar benchmark

Code and analysis for the paper of that name, on the Real Doppler RAD-DAR
database (RDRD).

A three-class range-Doppler classifier reaches about 0.93 accuracy on this
benchmark under either a random or an acquisition-grouped partition, and the
difference between the two protocols does not reach significance in
aggregate. Looking past the aggregate, twenty of the twenty-one drone
acquisitions reach out-of-fold recall between 0.90 and 0.99 and one reaches
0.249. Held out of training entirely, recall on a held-back half of that
acquisition is 0.178 over ten seeds, with 79 percent of its samples called
pedestrian. Adding the other half to training raises it to 0.948. The same
experiment on the weakest acquisition of another class recovers 0.056, so
the effect is not a generic property of a difficult recording. A labelling check
clears the acquisition: all of its samples sit nearest the drone class
centroid. What distinguishes it is essentially one property, a Doppler
centroid near 7.6 bins where every other drone acquisition sits near 30.

The paper reports four candidate physical mechanisms for the failure and
refutes all four against the data, reports a secondary aggregation effect
that does not survive a leave-one-acquisition-out test, and reports a null
interaction between the two quantisation axes. Those negative results are in
the paper and they are in this repository. Nothing here is tuned to make
them look better than they are.

## What is verified, and by what

Three different things get called reproduction, and this repository keeps
them apart.

| | What it checks | Needs the archive | Needs a GPU | Runtime |
|---|---|---|---|---|
| `tests/test_core.py` | the claims the paper makes about the code: power conservation under aggregation, the 671/330/165/77 cell counts, the quantiser's error bound, the 101,139-parameter model, why `offset` normalisation was chosen, acquisition grouping, deduplication, split disjointness | no | no | seconds |
| `./selftest.sh` | that all eleven stages run end to end on a synthetic tree with the real one's structure | no | no | minutes |
| `./verify.sh` | every number the paper states, against a fresh run on the real archive | yes | yes | hours |

Run the first two now, on any machine:

```bash
pip install -r requirements.txt
python tests/test_core.py        # 18 tests
./selftest.sh
```

The self-test deliberately does not reproduce the paper's finding, and prints
a table of disagreements at the end to show that it did not. The reason is
in the header of `tools/make_selftest_data.py`: the paper reports the
mechanism behind the failure as unknown, having refuted four candidates, and
a generator cannot synthesise a failure whose mechanism is unknown. Tuning
one until the failure appeared would be manufacturing the result rather than
testing the code. What the self-test earns is narrower and real: every stage
runs, the shapes and splits are what the paper describes, and the diagnostic
in stage 2d identifies the planted off-regime acquisition as the odd one out.

Only `verify.sh` answers whether the paper's numbers hold, and only against
the archive.

## Getting the data

The RDRD archive is not redistributed here. It comes from Kaggle, under the
slug `iroldan/real-doppler-raddar-database`, spelled with two `p`s in
`doppler`. An earlier version of this README claimed the slug carried a
single `p`. It does not, and that URL returns 404, so anyone who followed
the old instructions could not download the archive at all.

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d iroldan/real-doppler-raddar-database -p archive/
unzip -q archive/real-doppler-raddar-database.zip -d rdrd/
```

Two things about the archive are worth knowing before you count anything.

**It contains the whole tree twice.** Once as `data/<Class>/<session>/` and
again as `<Class>/<session>/`. A naive file count therefore comes to 34,970,
exactly double the 17,485 the source publication reports. Stage 1
deduplicates by content hash and recovers 17,485 with the published per-class
counts of 5720 vehicle, 5065 drone and 6700 pedestrian. If your count comes
out at double, nothing is wrong; run stage 1.

**An interrupted extraction can leave a truncated twin.** We saw a drone
folder `12-41` present twice with 414 and 500 files. Byte-identical hashing
does not catch that, because the folders are not identical. Stage 1 runs a
second pass that keeps the larger of two same-named folders under different
parents. If your total is between 17,485 and 34,970 rather than one of the
two, this is the likely cause, and re-extracting cleanly is worth doing
before anything else.

Acquisition structure is recovered from the directory names, not from
metadata. Folders are grouped by their leading `HH-MM` timestamp, so
`15-55`, `15-55a`, `15-55m` and `15-55p` are one acquisition. That takes 80
folders to 52 acquisitions: 27 vehicle folders to 15, 21 drone folders to 21,
32 pedestrian folders to 16.

One earlier approach is recorded here because it failed instructively.
Grouping acquisitions by centroid similarity collapsed 27 vehicle folders to
3, because every sample is a CFAR detection cropped around its own peak, so
every acquisition's mean vector is a blob in the middle of the window and all
of them correlate above 0.9. It corrupted the splits before the error was
caught. `rdrd.suspicious_pairs` reports candidate reprocessed pairs and
deliberately does not merge them.

## Reproducing the paper

```bash
./verify.sh --data rdrd/ --out runs/$(date +%F)
```

This runs all eight stages with the settings the paper used, writes one
report per stage under `runs/<date>/reports/`, and finishes by running
`check_expected.py`, which prints every claim in the paper against what the
run produced. Expect four to six hours on one A100, nearly all of it in
stage 4, which trains ten seeds across sixteen grid cells in two evaluation
regimes.

To confirm the pipeline works on your machine first:

```bash
./verify.sh --data rdrd/ --out runs/quick --quick
```

`--quick` runs every stage with fewer seeds and folds and finishes in well
under an hour. It will not reproduce the paper's numbers, because several of
them are means over the seeds it drops, and it says so when it finishes.

A Colab notebook that does the same thing cell by cell, including the Kaggle
download and a resume block for when the runtime disconnects, is in
`notebooks/RDRD_Master.ipynb`.

## How the numbers are checked

`expected.json` holds all 61 numbers the paper states, each with the report
file and the exact location inside it that produces the number, a tolerance,
and a one-line reason for that tolerance. `check_expected.py` reads it and
prints a table, naming the file each number was read from.

Two of those reports go by more than one name, because the notebook and
`verify.sh` once disagreed, and the checker accepts either rather than
asking anyone to rename a file. That matters more than it sounds: the
ten-seed confirmation and an earlier five-seed run of the same command
differ only in filename, and renaming one over the other silently checks
Table V against half the seeds it reports. Where both are present the
checker takes the ten-seed file and says so.

The tolerances are the part worth reading. Counts and the parameter count are
exact, because a mismatch there means a different dataset or a different
model rather than noise. Trained quantities are not exact and are not claimed
to be: training is stochastic, cuDNN kernel selection is not deterministic
across GPU models, and the paper itself reports between-seed standard
deviations as high as 0.16 on some cells of the grid. Each trained
quantity's tolerance is set from the spread the paper reports for that
quantity. An entry whose tolerance looks generous is one the paper already
describes as noisy, and the paper's claim about it is correspondingly weak.
The `why_tol` field states the reasoning for every entry, so a reader who
thinks a tolerance is too loose can see what it was set from and argue with
it.

A disagreement on an exact count means the archive differs. A disagreement on
a trained quantity, outside the spread the paper reports for it, means the
conclusion drawn from that number does not hold on your machine. Both are
worth reporting, and `check_expected.py` keeps untested claims separate from
failing ones so a partial run cannot be mistaken for a clean one.

`docs/NUMBERS.md` maps every table and figure in the paper to the command
that produces it.

## Layout

```
src/                 the pipeline, one module per stage
  rdrd.py            inventory, deduplication, grouping, splits, leakage test
  run_step1.py       stage 1 driver
  step2.py           dataset build, training, the split comparison
  step2b.py          cross-validated protocol comparison, normalisation
  step2c.py          per-acquisition recall against headroom
  step2d.py          the coverage experiment
  step3.py           the compression grid and the axis ablation
  step4.py           ten-seed confirmation, shift augmentation, factorial axes
  step5.py           leave one acquisition out
  centroids.py       per-acquisition Doppler centroid, mean and spread
  analyse4.py        re-analysis of stage 4 without retraining
  interaction.py     the factorial interaction contrast
tests/test_core.py   18 unit tests of the paper's claims about the code
tools/               synthetic data generators and the exploratory scripts
notebooks/           the Colab master notebook, plus earlier probes
paper/               the LaTeX source
figures/             the figure the paper uses, and the four it does not
reports/             where the paper's own reports belong
expected.json        every number in the paper, with its tolerance and why
check_expected.py    compares a run's reports against expected.json
verify.sh            full reproduction on the real archive
selftest.sh          whole pipeline on synthetic data
```

`tools/` holds the exploratory scripts from the investigation, including the
ones whose conclusions were overturned. They are kept because the paper
discusses the refuted mechanisms and a reader may want to see what was
actually run. `src/` is what the paper's numbers come from.

## A note on the figures

The paper uses one figure, `session_13-48_vs_others.png`, placed as a
two-column `figure*` in Section IV. The notebook produces it, under the
heading "Look at the samples", into `MyDrive/radar_compression/figures/`,
along with four others the paper does not use.

To build the paper, the figures must sit in a `figures/` directory beside
`paper_main.tex`, which is what `\graphicspath{{figures/}}` in the preamble
expects. On Overleaf that means a folder named `figures`, not the project
root. `figures/README.md` here lists what each one is and which stage
produced it.

## Citing

Archived at [doi:10.5281/zenodo.23118075](https://doi.org/10.5281/zenodo.23118075),
which is the concept DOI and always resolves to the newest version. See
`CITATION.cff`. The dataset itself should be cited separately, to
I. Roldan et al., and the paper's reference list has the citation.

`docs/RELEASING.md` has the steps for cutting a release and minting the
Zenodo DOI, including which of the two DOIs Zenodo mints belongs in the
paper.

## License

MIT, see `LICENSE`. The RDRD database is not covered by it and carries its own
terms on Kaggle.
