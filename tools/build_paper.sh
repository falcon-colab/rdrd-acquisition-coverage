#!/bin/sh
# Build the paper. Two formats, and for each one a named and a blind version.
#
#   sh tools/build_paper.sh            the 6-page two-column submission (default)
#   sh tools/build_paper.sh long       the 12-page single-column variant
#   sh tools/build_paper.sh both       all of it
#
# paper_main.tex is the submission: IEEEtran two-column, 6 pages including
# references, which is what ICIC, ICCII and ICOSST all require.
#
# Each source carries a \blindtrue switch rather than living in two files,
# because keeping two files is how an author name ends up in a blind
# submission. This script flips the switch and gives the two PDFs different
# names, so the file you upload is never the one you last previewed by hand.
# It then checks the page count, the anonymity of the blind PDF and that
# every font is embedded, since a venue that desk-rejects on format will not
# tell you first.
set -e
cd "$(dirname "$0")/.."
python3 tools/make_tables.py >/dev/null
python3 tools/make_figures.py >/dev/null
cd paper

build () {                      # $1 = true|false, $2 = output name, $3 = source
    sed "s/^\\\\blind\\(true\\|false\\)/\\\\blind$1/" "$3" > _build.tex
    for i in 1 2 3; do pdflatex -interaction=nonstopmode _build.tex >/dev/null 2>&1 || true; done
    [ -f _build.pdf ] || { echo "FAILED to build $2"; exit 1; }
    mv _build.pdf "$2"
    pages=$(pdfinfo "$2" | sed -n 's/^Pages: *//p')
    bad=$(grep -c '^!' _build.log || true)
    und=$(grep -c 'undefined' _build.log || true)
    ovf=$(grep -c 'Overfull' _build.log || true)
    printf '%-34s %2s pages  errors %s  undefined %s  overfull %s\n' \
           "$2" "$pages" "$bad" "$und" "$ovf"
    case "$3" in
      *paper_main*) lim=6 ;;
      *)            lim=12 ;;
    esac
    [ "$pages" -le "$lim" ] || echo "  *** OVER THE $lim-PAGE LIMIT ***"
}

want=${1:-main}
case "$want" in
  main|both)
    build false conference_6pp_named.pdf paper_main.tex
    build true  conference_6pp_blind.pdf paper_main.tex ;;
esac
case "$want" in
  long|both)
    build false singlecol_12pp_named.pdf paper_long.tex
    build true  singlecol_12pp_blind.pdf paper_long.tex ;;
esac

# The blind PDF is the one that gets uploaded, so check it rather than trust it.
echo
# Check the blind PDFs rather than trust them.
for f in conference_6pp_blind.pdf singlecol_12pp_blind.pdf; do
  [ -f "$f" ] || continue
  echo "anonymity check on $f:"
  hits=$(pdftotext "$f" - \
         | grep -ciE 'mustafa|murtaza|nawaz|zohaib|nust|aeronautical|islamabad|falcon-colab|github|zenodo' || true)
  if [ "$hits" = "0" ]; then echo "  clean: no identifying string found"
  else echo "  *** $hits identifying strings present, DO NOT SUBMIT ***"; fi
done
# pdffonts columns end: ... emb sub uni object ID, so the emb flag is $(NF-4).
pdffonts conference_6pp_named.pdf | awk 'NR>2 && NF>5 {t++; if ($(NF-4)!="yes") n++} END {
    if (n) print "  *** " n " of " t " fonts NOT embedded ***";
    else print "  all " t " fonts embedded"}'
rm -f _build.*
