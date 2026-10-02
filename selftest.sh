#!/usr/bin/env bash
#
# Run the whole pipeline on synthetic data. No archive, no GPU, no Kaggle
# account. Finishes in a few minutes on a laptop.
#
#   ./selftest.sh [--out DIR]
#
# What this proves: every module imports, every stage runs to completion, the
# array shapes and split disjointness are what the manuscript describes, the
# model has the parameter count the manuscript reports, and the diagnostic in
# stage 2d identifies the acquisition that was planted as the odd one out.
#
# What this does NOT prove: that the paper's numbers reproduce. They cannot
# reproduce here, because the synthetic tree is not the measurements. It also
# does not reproduce the paper's finding, and deliberately so. See the
# comment at the top of tools/make_selftest_data.py for why a generator
# cannot synthesise a failure whose mechanism the paper reports as unknown.
#
# For the real thing, use verify.sh.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${PYTHON:-python3}"

OUT="${TMPDIR:-/tmp}/rdrd-selftest"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TREE="$OUT/tree"
SPLITS="$OUT/splits"
CACHE="$OUT/cache"
REPORTS="$OUT/reports"
rm -rf "$OUT"
mkdir -p "$SPLITS" "$CACHE" "$REPORTS"

step () { echo; echo "--- $1"; }

step "unit tests"
"$PY" "$HERE/tests/test_core.py"

step "synthetic tree"
"$PY" "$HERE/tools/make_selftest_data.py" --out "$TREE"

step "stage 1"
"$PY" "$SRC/run_step1.py" --root "$TREE" --out "$SPLITS" \
    --per-session 40 --top-k 12 --quiet-sessions | tail -20
cp "$SPLITS/step1_report.json" "$REPORTS/"

step "stage 2"
"$PY" "$SRC/step2.py" build --splits "$SPLITS" --out "$CACHE" | tail -6
DS="$CACHE/dataset.npz"
"$PY" "$SRC/step2.py" compare --data "$DS" --seeds 0 1 --epochs 12 \
    --out "$REPORTS/step2_compare.json" | tail -8

step "stage 2b"
"$PY" "$SRC/step2b.py" kfold --data "$DS" --folds 3 --seeds 0 --epochs 10 \
    --scheme offset --out "$REPORTS/step2b_kfold.json" | tail -4
"$PY" "$SRC/step2b.py" norm --data "$DS" --folds 3 --epochs 10 \
    --out "$REPORTS/step2b_norm.json" | tail -7

step "stage 2c"
"$PY" "$SRC/step2c.py" --data "$DS" --folds 3 --seeds 0 --epochs 10 \
    --scheme offset --out "$REPORTS/step2c_sessions.json" | tail -8

step "stage 2d"
"$PY" "$SRC/step2d.py" --data "$DS" --target drone/13-48 --seeds 0 \
    --epochs 10 --out "$REPORTS/step2d_13-48.json" | tail -12

"$PY" "$SRC/centroids.py" --data "$DS" --target drone/13-48 \
    --out "$REPORTS/step2d_centroids.json" | tail -6

step "stage 3"
"$PY" "$SRC/step3.py" grid --data "$DS" --seeds 0 --epochs 8 \
    --factors 1 2 4 8 --bits 32 8 --out "$REPORTS/step3_grid.json" | tail -14
"$PY" "$SRC/step3.py" ablate --data "$DS" --seeds 0 --epochs 8 \
    --out "$REPORTS/step3_ablate.json" | tail -8

step "stage 4"
"$PY" "$SRC/step4.py" seeds --data "$DS" --seeds 0 1 2 --epochs 8 \
    --out "$REPORTS/step4_seeds10.json" | tail -8
"$PY" "$SRC/step4.py" augment --data "$DS" --seeds 0 1 --epochs 8 \
    --shifts 4 8 --out "$REPORTS/step4_augment.json" | tail -6
"$PY" "$SRC/step4.py" ablate --data "$DS" --seeds 0 1 --epochs 8 \
    --out "$REPORTS/step4_ablate.json" | tail -8

step "stage 5"
"$PY" "$SRC/step5.py" --data "$DS" \
    --sessions drone/13-48 car/15-37 drone/15-21 \
    --seeds 0 1 --factors 1 4 --epochs 8 \
    --out "$REPORTS/step5_loo.json" | tail -8

step "re-analysis without retraining"
"$PY" "$SRC/analyse4.py" "$REPORTS/step4_seeds10.json" | tail -5
"$PY" "$SRC/interaction.py" "$REPORTS/step4_ablate.json" | tail -5

step "the manuscript's numbers against this synthetic run"
set +e
"$PY" "$HERE/check_expected.py" --reports "$REPORTS" --brief
set -e

cat <<'EOF'

--- self-test complete

Every stage ran. The table above compares the manuscript's numbers against
this synthetic run, and most of them disagree, which is the correct result:
the synthetic tree is not the measurements. The table is printed here only
to show that the checker reads every report and is capable of reporting
disagreement, so that a green table on the real data means something.

To reproduce the paper, run verify.sh against the real archive.
EOF
