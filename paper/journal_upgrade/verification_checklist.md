# Verification checklist

The manuscript reports three training-label noise conditions: **0 %, 6 % and 6.8 %**.
Checks below are run by `check_final.py` against the compiled PDF, plus inspection of
every rendered page and an independent multi-lens audit of the final build.

## Hard rules

- **No invented numbers.** Every value in the results tables and figures is read from
  the frozen CSVs in `results/` by `build_tables.py` and `make_result_figures.py` and
  printed unchanged. An automated check confirms that all 204 `mean ± SD` values for the
  three reported conditions appear verbatim in the PDF. `NUMBER_PROVENANCE.md` lists
  every number and its source; `number_provenance.csv` lists all 204 table cells.
- **No leakage from dropped conditions.** 544 values belonging to conditions the paper
  no longer reports (2, 4, 8, 10, 20 %, the 6.2–7.8 % sweep, and the 30 / 40 / boundary
  stress conditions) were checked against the PDF text; none appears.
- **No invented references.** 40 entries. 37 DOIs were resolved against the Crossref
  REST API and 3 against the arXiv API on 2026-11-19; the returned title, container and
  year matched each entry. One correction followed: Deng et al. was dated 2021 but the
  IEEE TPEL issue is 2022, so the year was fixed and the entry moved in the ordering.
  The PDF carries 40 distinct DOI hyperlinks.
- **No model hidden.** All 17 models appear in every results table and figure, including
  the five that beat the proposed model on clean labels. The two PF candidate variants
  (L8, J) are marked with a dagger rather than omitted.
- **Page count: 14** (limit 18).

## Honesty

- The abstract, Section 4.1 and the conclusion all state that the proposed model ranks
  **twelfth of seventeen on clean labels** and first only at the two noisy conditions.
- The 6.8 % condition is identified as post-hoc in the abstract, Section 3.4, Section
  4.2, the limitations and the conclusion: it was run only after the 6 % result had been
  seen, so two of the three reported conditions are not independent of each other.
- Section 4.2 states that no significance test is available: four paired observations
  per condition floor the smallest attainable two-sided *p*-value at 0.125, before any
  correction for comparing seventeen models. No test statistic is quoted that was not
  computed on the reported conditions.
- Table 11 compares the three attached papers on task, data, split and metric family
  only. Their RMSE/MAE/R² values stay in their own column in their own units, and no
  claim is made that this model beats them.
- Dropped experiments are not deleted from the project: the 2–20 % sweep, the 6.2–7.8 %
  exploratory grid and the stress conditions remain in `results/` and in the
  pre-registration documents. They are outside this manuscript's reported scope, not
  discarded.

## Layout

- Two-column `elsarticle` 5p, Times-family text, consistent sizes for title, headings,
  body and captions; references set one step smaller, as is usual for the class.
- No author block in the title area and no author name in any page header; the running
  head is "Battery degradation classification under label noise". PDF author metadata
  is empty.
- Whitespace: every page except the last carries 15–29 % ink with no internal blank band
  wider than the page margins. The final page holds the tail of the reference list.
- No overlap: every page was rendered and inspected. Diagram boxes auto-shrink their
  text to fit (`fit()` in `make_diagrams.py`).
- Tables: no overflow. Results are given as one ACC/BACC table in the body and one
  F1/AUC table in the appendix, each 17 rows × 6 value columns.
- Wide figures span both columns; Figs. 8, 13 and 14 sit in one column.
- The proposed row is shaded and labelled "(proposed)" in both results tables.

## Figures

- Fig. 1 (workflow) and Fig. 5 (EF-GBTSVM-PF architecture) were redrawn as numbered,
  coloured, rounded-box diagrams with short key points only. They show only steps the
  method performs: granular balls with purity threshold τ, the RVFL enhanced map, twin
  planes, per-fold inner-CV variant selection, and LOBO evaluation.
- Fig. 2 (18650 cell) was redrawn as a labelled cross-section with the casing, Cu and Al
  current collectors, graphite anode, separator and electrolyte, LiCoO₂ cathode, Li⁺
  transport and electron direction through the external load. It is drawn from scratch
  and is labelled in the caption as conceptual.
- Every results figure has a 5–10 line explanation in the body text giving what it
  shows, the main observation with numbers, and what it means.
- Per-cell values at 6 % and 6.8 % are reassembled from the per-fold run records.
  `make_result_figures.py` asserts that the reassembled per-cell means reproduce the
  published summary means exactly before drawing Fig. 10.

## Independent audit of the final build

Thirty reviewer agents audited the compiled PDF across five lenses (stale scope,
numerical integrity, overclaiming, layout, references), each finding then checked by an
independent agent instructed to refute it. 25 findings were raised and 12 were dismissed
on refutation. All 13 that survived were fixed:

| Fixed | What was wrong |
|---|---|
| Contribution (iv), abstract, Section 4.1 | Said five models beat the proposed one on clean labels. Eleven do (rank 12 of 17). All eleven are now named in Section 4.1, and the count is corrected in the abstract and the contribution list. |
| Conclusion | Claimed the single-feature threshold stays "within two and a half points of every learned model". The accuracy gaps are 1.83, 2.41 and 2.53 points, and the balanced-accuracy gap reaches 4.02. The sentence now gives the metric and the true figures. |
| Section 4.5 | Hard-coded cross-reference to "Section 4.6" pointed at the literature comparison instead of the ablation. Replaced with `\ref{sec:ablation}`. |
| `number_provenance.csv` | 102 F1/AUC rows cited `rank_F1_all17` / `rank_AUC_all17`, which do not exist in the source CSVs. Those ranks are recomputed from the printed means, and the file now says so. |
| References | `dini2024`, `zhou2024gnn` and `xia2024fusion` were in the bibliography but never cited; body numbering jumped [30] to [34]. All three are now cited in Section 2.1, and a check confirms all 40 entries are cited. |
| Figures 3, 4, 6, 11, 12 | Panel lettering printed at 2–4 pt because wide source figures were scaled to 0.31–0.46 of the text width. Placements were enlarged (Figure 3 is now two panels per row instead of three) so the lettering prints substantially larger. |
| Page 5 | The equation block overflowed past the footer under two tall floats. Float heights were reduced until the column fits. |

Known limitation the audit raised that is **not** fully fixed: some lettering inside the
descriptive figures (Figures 3, 4, 6 and 12) is still below the 6 pt artwork minimum
some publishers apply. Those plots are pre-rendered artefacts carried over from an
earlier build (`paper_figures/new21/`), and no generator source survives for most of
them, so redrawing them at print size would mean reconstructing the plots rather than
regenerating them. Enlarging the placements was the correction available without that
risk.

## Not verified here

- Resolution of the 40 DOI links was checked against the Crossref and arXiv metadata
  APIs, not by loading each publisher landing page.
- `quadir2025ef` is cited as the arXiv version (2410.05786v2, February 2025), which is
  what was verified. The arXiv record's author-supplied `journal_ref` field names ICPR
  2024 as the venue; that published version was not independently confirmed, so the
  preprint is cited instead.
