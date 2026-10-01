# Cutting a release and minting the DOI

Zenodo mints a DOI from a GitHub **release**, not from a repository. Nothing
happens until a release is published, and the switch has to be on **before**
the release is cut. A release published first and the switch flipped after
will not be archived; you would have to cut a second release.

## Once, to set it up

1. Sign in at [zenodo.org](https://zenodo.org) with the GitHub account that
   owns the repository. Signing in with GitHub is what authorises the link;
   an email-registered Zenodo account will not see your repositories.
2. Go to the GitHub tab in your Zenodo account settings.
3. Find `falcon-colab/rdrd-acquisition-coverage` and switch it **on**. If it
   is not listed, use the sync button; a repository created minutes ago can
   take a moment to appear.

`.zenodo.json` in the repository root supplies the title, description,
authors, keywords, licence and the link to the source dataset, so Zenodo does
not fall back on the repository blurb. Check the author name in it reads the
way you want it cited before releasing.

## For each release

```bash
git tag -a v1.0.0 -m "Release accompanying the submitted manuscript"
git push origin v1.0.0
```

Then publish it as a release on GitHub, from the Releases page, choosing the
tag you just pushed. Zenodo picks it up within a few minutes and the DOI
appears on your Zenodo uploads page.

## Which DOI goes in the paper

Zenodo mints two.

- The **concept DOI** always resolves to the newest version. It is the one
  that belongs in the paper, because a reader who follows it a year from now
  reaches the current code rather than whatever happened to be tagged on
  submission day.
- The **version DOI** pins one release. Useful if a reviewer asks for the
  exact state the results came from.

Zenodo labels the concept one "Cite all versions" on the record page.

## Putting it in the paper

One line in `paper/paper_main.tex`, in the Reproducibility section, currently
reads:

```latex
TODO: Zenodo DOI
```

Replace it with the concept DOI:

```latex
\url{https://doi.org/10.5281/zenodo.XXXXXXX}
```

Then add the same DOI to `CITATION.cff` as a top-level field, so that the
"Cite this repository" button on GitHub offers it:

```yaml
doi: 10.5281/zenodo.XXXXXXX
```

## Before the release, worth doing

Drop the reports from the manuscript's own run into `reports/` and commit
them, then run:

```bash
python check_expected.py --reports reports/
```

Every claim should agree, because this is checking the paper's numbers
against the files the paper's numbers came from. If anything disagrees here,
a number in the paper does not match its own report, and that is worth
finding before the archive is permanent. A DOI cannot be withdrawn, only
superseded by a new version.
