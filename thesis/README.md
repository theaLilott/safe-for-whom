# Thesis LaTeX sources

Source of the bachelor thesis "Safe for Whom? Evaluating AI Safety Evaluations
for Demographic Blindspots" (Manon Kempermann, Saarland University).

Not included here (too large for the repo, needed to compile):
- `anthology-1.bib` / `anthology-2.bib`: full ACL Anthology BibTeX dumps,
  available from https://aclanthology.org/anthology.bib (split locally).
  Place them next to `Thesis.tex`.

Build: `latexmk -pdf Thesis.tex` (uses biber).

A compiled version is included as `Thesis.pdf`.
