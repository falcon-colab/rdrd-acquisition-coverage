#!/usr/bin/env bash
#
# Reproduce every number in the manuscript and check each one.
#
#   ./verify.sh --data /path/to/extracted/archive --out runs/$(date +%F)
#
# The archive is the Real Doppler RAD-DAR database, extracted so that the
# class folders sit under it (see README, "Getting the data"). The script
# runs all eight stages with the settings the manuscript used, writes one
# report per stage, and then runs check_expected.py, which prints a table of
# every claim against what this run produced.
#
# This needs a GPU. On one A100 or equivalent the whole sequence takes
# roughly four to six hours, almost all of it in stage 4, which trains ten
# seeds across sixteen grid cells twice over. --quick cuts the seed counts
# and fold counts to make the sequence finish in well under an hour; it
# exercises every stage but it will NOT reproduce the paper's numbers,
# because several of them are means over the seeds it drops. Use --quick to
# confirm the pipeline works on your machine, then run without it.
#
# The script stops at the first failing stage rather than carrying on with a
# missing input, and check_expected.py reports untested claims separately
# from disagreements, so a partial run cannot be mistaken for a clean one.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

DATA=""
OUT=""
QUICK=0
EPOCHS=40

while [ $# -gt 0 ]; do
  case "$1" in
    --data)   DATA="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --quick)  QUICK=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$DATA" ] || [ -z "$OUT" ]; then
  echo "usage: $0 --data DIR --out DIR [--quick] [--epochs N]" >&2
  exit 2
fi
if [ ! -d "$DATA" ]; then
  echo "no such directory: $DATA" >&2
  exit 2
fi

if [ "$QUICK" = "1" ]; then
  PER_SESSION=40;  FOLDS=3
  S_CMP="0 1";     S_KFOLD="0";    S_2C="0";   S_2D="0"
  S_GRID="0";      S_SEEDS="0 1 2"; S_AUG="0 1"; S_ABL="0 1"; S_LOO="0 1"
  FACTORS="1 2 4 8"; BITS="32 8"
  LOO_SESSIONS="drone/13-48 car/15-37 drone/15-21"
  echo "### QUICK MODE: every stage runs, the paper's numbers will NOT reproduce"
else
  PER_SESSION=120; FOLDS=5
  S_CMP="0 1 2";   S_KFOLD="0 1 2"; S_2C="0 1"; S_2D="0 1"
  S_GRID="0 1";    S_SEEDS="0 1 2 3 4 5 6 7 8 9"
  S_AUG="0 1 2 3 4"; S_ABL="0 1 2 3 4"; S_LOO="0 1 2"
  FACTORS="1 2 4 8"; BITS="32 16 8 4"
  LOO_SESSIONS="drone/13-48 car/15-37 drone/15-21 drone/12-34 person/11-23"
fi

SPLITS="$OUT/splits"
CACHE="$OUT/cache"
REPORTS="$OUT/reports"
LOGS="$OUT/logs"
mkdir -p "$SPLITS" "$CACHE" "$REPORTS" "$LOGS"

PY="${PYTHON:-python3}"

banner () {
  echo
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
}

run () {                      # run <logname> <command...>
  local name="$1"; shift
  echo "    \$ $*" | tee "$LOGS/$name.cmd"
  "$@" 2>&1 | tee "$LOGS/$name.log"
}

START=$(date +%s)

"$PY" - <<'PY'
import sys
print("python  %s" % sys.version.split()[0])
try:
    import torch
    print("torch   %s   cuda %s   device %s"
          % (torch.__version__, torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available()
             else "cpu"))
    if not torch.cuda.is_available():
        print("\nNo GPU visible. The sequence will still run but will take a"
              "\nvery long time. Consider --quick to check the pipeline"
              "\nfirst.\n")
except ImportError:
    sys.exit("torch is not installed: pip install -r requirements.txt")
PY

banner "unit tests (no data, no GPU)"
run tests "$PY" "$HERE/tests/test_core.py"

banner "stage 1  inventory, deduplication, acquisition grouping, splits"
run step1 "$PY" "$SRC/run_step1.py" --root "$DATA" --out "$SPLITS" \
    --per-session "$PER_SESSION" --top-k 12
cp "$SPLITS/step1_report.json" "$REPORTS/step1_report.json"

banner "stage 2  dataset build and the split comparison"
run step2_build "$PY" "$SRC/step2.py" build --splits "$SPLITS" --out "$CACHE"
DS="$CACHE/dataset.npz"
run step2_compare "$PY" "$SRC/step2.py" compare --data "$DS" \
    --seeds $S_CMP --epochs "$EPOCHS" --out "$REPORTS/step2_compare.json"

banner "stage 2b  cross-validated protocol comparison and normalisation"
run step2b_kfold "$PY" "$SRC/step2b.py" kfold --data "$DS" --folds "$FOLDS" \
    --seeds $S_KFOLD --scheme offset --epochs "$EPOCHS" \
    --out "$REPORTS/step2b_kfold.json"
run step2b_norm "$PY" "$SRC/step2b.py" norm --data "$DS" --folds "$FOLDS" \
    --epochs "$EPOCHS" --out "$REPORTS/step2b_norm.json"

banner "stage 2c  per-acquisition recall against headroom"
run step2c "$PY" "$SRC/step2c.py" --data "$DS" --folds "$FOLDS" \
    --seeds $S_2C --epochs "$EPOCHS" --scheme offset \
    --out "$REPORTS/step2c_sessions.json"

banner "stage 2d  the coverage experiment on drone/13-48"
run step2d "$PY" "$SRC/step2d.py" --data "$DS" --target drone/13-48 \
    --seeds $S_2D --epochs "$EPOCHS" --out "$REPORTS/step2d_13-48.json"

run centroids "$PY" "$SRC/centroids.py" --data "$DS" \
    --target drone/13-48 --out "$REPORTS/step2d_centroids.json"

banner "stage 3  the compression grid and the axis ablation"
run step3_grid "$PY" "$SRC/step3.py" grid --data "$DS" --seeds $S_GRID \
    --epochs "$EPOCHS" --factors $FACTORS --bits $BITS \
    --out "$REPORTS/step3_grid.json"
run step3_ablate "$PY" "$SRC/step3.py" ablate --data "$DS" --seeds $S_GRID \
    --epochs "$EPOCHS" --out "$REPORTS/step3_ablate.json"

banner "stage 4  ten-seed confirmation, shift augmentation, factorial axes"
run step4_seeds "$PY" "$SRC/step4.py" seeds --data "$DS" --seeds $S_SEEDS \
    --epochs "$EPOCHS" --out "$REPORTS/step4_seeds10.json"
run step4_augment "$PY" "$SRC/step4.py" augment --data "$DS" --seeds $S_AUG \
    --epochs "$EPOCHS" --shifts 4 8 16 --out "$REPORTS/step4_augment.json"
run step4_ablate "$PY" "$SRC/step4.py" ablate --data "$DS" --seeds $S_ABL \
    --epochs "$EPOCHS" --out "$REPORTS/step4_ablate.json"

banner "stage 5  leave one acquisition out"
run step5 "$PY" "$SRC/step5.py" --data "$DS" --sessions $LOO_SESSIONS \
    --seeds $S_LOO --factors 1 4 --epochs "$EPOCHS" \
    --out "$REPORTS/step5_loo.json"

banner "re-analysis without retraining"
run analyse4 "$PY" "$SRC/analyse4.py" "$REPORTS/step4_seeds10.json"
run interaction "$PY" "$SRC/interaction.py" "$REPORTS/step4_ablate.json"

MINS=$((($(date +%s) - START) / 60))
banner "every claim in the manuscript against this run  (${MINS} min elapsed)"
set +e
"$PY" "$HERE/check_expected.py" --reports "$REPORTS" \
    | tee "$LOGS/check_expected.log"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "reports  $REPORTS"
echo "logs     $LOGS"
if [ "$QUICK" = "1" ]; then
  echo
  echo "This was --quick. Disagreements above are expected: several of the"
  echo "paper's numbers are means over seeds this mode does not run. Rerun"
  echo "without --quick before drawing any conclusion from the table."
fi
exit "$STATUS"
