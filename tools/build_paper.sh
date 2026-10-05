#!/bin/sh
# Build both versions of the single-column paper.
#
# ICCI-2026 reviews double-blind, so the PDF that gets uploaded must carry no
# author names, no affiliation and no identifying link, while the camera-ready
# must carry all three. Keeping two .tex files is how an author name ends up in
# a blind submission, so there is one source with a \blindtrue switch and this
# script flips it. The two PDFs get different names so the file you upload is
# never the one you last previewed.
#
#   sh tools/build_paper.sh          both versions into paper/
#
set -e
cd "$(dirname "$0")/.."
python3 tools/make_tables.py >/dev/null
python3 tools/make_figures.py >/dev/null
cd paper

build () {                      # $1 = true|false, $2 = output name
    sed "s/^\\\\blind\\(true\\|false\\)/\\\\blind$1/" paper_long.tex > _build.tex
    for i in 1 2 3; do pdflatex -interaction=nonstopmode _build.tex >/dev/null 2>&1 || true; done
    [ -f _build.pdf ] || { echo "FAILED to build $2"; exit 1; }
    mv _build.pdf "$2"
    pages=$(pdfinfo "$2" | sed -n 's/^Pages: *//p')
    bad=$(grep -c '^!' _build.log || true)
    und=$(grep -c 'undefined' _build.log || true)
    ovf=$(grep -c 'Overfull' _build.log || true)
    printf '%-34s %2s pages  errors %s  undefined %s  overfull %s\n' \
           "$2" "$pages" "$bad" "$und" "$ovf"
    case "$2" in
      *blind*) [ "$pages" -le 12 ] || echo "  *** OVER THE 12-PAGE SUBMISSION LIMIT ***" ;;
      *)       [ "$pages" -le 12 ] || echo "  (over 12; the camera-ready follows the"
               [ "$pages" -le 12 ] || echo "   publishing journal's limit, not this one)" ;;
    esac
}

# Only the blind PDF is bound by the 12-page call for papers. The camera-ready
# carries the author block and follows whatever the publishing journal sets.
build true  icci2026_submission_blind.pdf
build false icci2026_cameraready.pdf

# The blind PDF is the one that gets uploaded, so check it rather than trust it.
echo
echo "anonymity check on icci2026_submission_blind.pdf:"
hits=$(pdftotext icci2026_submission_blind.pdf - \
       | grep -ciE 'mustafa|murtaza|nawaz|zohaib|nust|aeronautical|islamabad|falcon-colab|github|zenodo' || true)
if [ "$hits" = "0" ]; then echo "  clean: no identifying string found"
else echo "  *** $hits identifying strings present, DO NOT SUBMIT ***"; fi
# pdffonts columns end: ... emb sub uni object ID, so the emb flag is $(NF-4).
pdffonts icci2026_submission_blind.pdf | awk 'NR>2 && NF>5 {t++; if ($(NF-4)!="yes") n++} END {
    if (n) print "  *** " n " of " t " fonts NOT embedded ***";
    else print "  all " t " fonts embedded"}'
rm -f _build.*
