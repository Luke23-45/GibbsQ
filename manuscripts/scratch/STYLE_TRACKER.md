# Manuscript Style Tracker

This file is the phase-1 and phase-2 tracker for the LaTeX design upgrade of the GibbsQ manuscript.

## Scope

The current manuscript uses mostly default `article` styling plus local one-off spacing tweaks. The upgrade goal is to make the manuscript look cleaner, more deliberate, and more publication-ready without destabilizing the math or references.

## Phase 1 Audit

The audit below was performed by reading:

- `main.tex`
- `preamble.tex`
- all files in `sections/`
- all files in `appendix/`

It was also cross-checked against current package documentation and LaTeX references for:

- section titles and spacing (`titlesec`)
- font stack modernization (`newtxtext`, `newtxmath`)
- captions (`caption`)
- theorem styling (`amsthm`, `thmtools`)
- table rules and spacing (`booktabs`)
- microtypography (`microtype`)

## Enhancement Targets

| Area | Current issue | Candidate solutions | Selected direction | Status |
|---|---|---|---|---|
| Document font stack | Uses `mathptmx`, which is older and visually dated | 1. Keep `mathptmx` and tune spacing. 2. Move to `newtxtext` + `newtxmath`. 3. Move to TeX Gyre Termes stack. | Use `newtxtext` + `newtxmath` for a cleaner Times-like research-paper look while staying compatible with `pdflatex`. | Completed |
| Section headings | Default `article` headings look plain; numbering presentation is weak | 1. Default LaTeX only. 2. `sectsty`. 3. `titlesec` with explicit spacing and labels. | Use `titlesec` for controlled heading weight, spacing, and label formatting. | Completed |
| Section numbering hierarchy | User wants explicit hierarchy like `1.1`, `1.2.3`; display should feel more structured | 1. Leave defaults. 2. Raise `secnumdepth`. 3. Raise `secnumdepth` and style labels with `titlesec`. | Set numbering depth explicitly and format visible labels for section, subsection, and subsubsection. | Completed |
| Paragraph heads | Many files use `\paragraph{...}`; default run-in style looks cramped | 1. Keep run-in paragraphs. 2. Convert some to subsections manually. 3. Style `\paragraph` consistently with visible spacing. | Keep structure intact and style `\paragraph` as a deliberate run-in/display hybrid with better spacing. | Completed |
| Theorem-like environments | Definitions, theorems, lemmas, remarks all use plain defaults | 1. Default `amsthm`. 2. `amsthm` custom theorem styles. 3. `thmtools` declarations over `amsthm`. | Use `thmtools` over `amsthm` to define cleaner theorem, definition, and remark styles while preserving numbering. | Completed |
| Proof environment | Proof blocks are readable but visually plain; step headers are manual and inconsistent | 1. Keep defaults. 2. Tune `proof` spacing and head font. 3. Box proofs. | Keep proofs unboxed, but strengthen proof heading and theorem spacing for a cleaner mathematical presentation. | Completed |
| Display math spacing | Dense proofs use many equations/align blocks; no explicit global tuning | 1. Leave defaults. 2. Add `mathtools` and mild display spacing control. 3. Aggressively compress displays. | Add `mathtools` and keep spacing conservative to avoid harming readability. | Completed |
| Tables | Many local `\arraystretch` overrides and manual horizontal padding; no global caption policy | 1. Leave local tables untouched. 2. Add caption and table defaults only. 3. Full rewrite with `siunitx`. | Introduce global table/caption styling now; leave a later pass for optional numeric alignment cleanup. | Completed |
| Figures | Captions are serviceable but uncoordinated with table captions | 1. Default captions. 2. `caption` package styling. 3. Full float redesign. | Use `caption` for consistent font, width, and spacing. | Completed |
| Lists | `itemize` and `enumerate` rely on defaults; spacing varies by context | 1. Keep defaults. 2. `enumitem` tuning. 3. Manual local spacing. | Use `enumitem` for consistent list spacing and indentation. | Completed |
| Float spacing | Some float spacing is already tuned, but style is incomplete | 1. Keep current float counters only. 2. Add global caption/float spacing and remove local hacks over time. | Keep existing float-count tuning and complement it with consistent caption/table spacing. | Completed |
| Title block | Title/author/date are functional but plain | 1. Leave as-is. 2. Tune `\maketitle`. 3. Replace with a custom title page. | Mild `\maketitle` refinement only; avoid custom title-page complexity during the first style pass. | Completed |
| Hyperlinks and cross-references | `hyperref` is present; `\cref` is a manual alias to `\autoref` | 1. Keep alias. 2. Move to `cleveref`. 3. Remove auto text refs. | Keep current behavior for now to avoid regression; revisit only if ref wording becomes inconsistent. | Deferred |
| Appendix headings | Appendices inherit the same plain heading style | 1. Leave default appendix behavior. 2. Let heading package style them automatically. | Use the same section system so appendices inherit the upgraded look. | In progress |
| Manual body hacks | Repeated `\vspace`, repeated `\renewcommand{\arraystretch}{...}`, `\clearpage` separators, mojibake comments | 1. Ignore them. 2. Remove only the style-critical ones. 3. Full manual cleanup. | Remove only style-critical hacks in this pass and log the rest for follow-up. | In progress |

## Files and Commands Affected

Primary style files:

- `main.tex`
- `preamble.tex`

Files with repeated local table spacing overrides:

- `sections/03_model.tex`
- `sections/08_experiments.tex`
- `appendix/B_gradient_check.tex`
- `appendix/C_training_details.tex`
- `appendix/D_stress_sweep.tex`
- `appendix/E_engine_consistency.tex`

Files heavily dependent on theorem/proof/math presentation:

- `sections/04_softmax_theorem.tex`
- `sections/05_uas_theorem.tex`
- `sections/03_model.tex`
- `sections/06_calibrated_uas.tex`

Files with many paragraph heads that depend on heading polish:

- `sections/01_introduction.tex`
- `sections/02_related_work.tex`
- `sections/08_experiments.tex`
- `appendix/A_drift_verification.tex`
- `appendix/B_gradient_check.tex`
- `appendix/D_stress_sweep.tex`

## Phase 2 Implementation Plan

### Pass A: Global style foundation

- Replace the old Times stack with a modern Times-like package pair.
- Add explicit section, subsection, subsubsection, and paragraph styling.
- Set section-number depth explicitly.
- Add global theorem styles.
- Add caption and list styling.
- Add small math-presentation improvements that are safe under `pdflatex`.

### Pass B: Body cleanup for consistency

- Remove local table spacing hacks where the new global rules make them unnecessary.
- Remove local caption spacing hacks where possible.
- Normalize obvious visual separators that do not belong in source files.

### Pass C: Verification

- Compile with `latexmk` after the preamble change.
- Inspect log warnings and fix style-related regressions.
- Recompile after body cleanup.
- Check numbering, theorem labels, captions, list spacing, and appendix appearance.

## Verification Checklist

- [x] Main sections use an improved display style.
- [x] Subsections display consistent hierarchical numbering.
- [x] Section spacing is consistent across all main files and appendices.
- [x] Paragraph heads no longer look cramped.
- [x] Theorem, definition, lemma, remark, and proof blocks share a coherent style.
- [x] Display equations still compile cleanly and remain readable.
- [x] Table captions and figure captions use a unified style.
- [x] Lists have consistent indentation and vertical spacing.
- [x] Appendix sections inherit the same heading system.
- [x] Local manual spacing hacks are reduced, not increased.
- [x] The manuscript compiles without new fatal errors.
- [x] Cross-references and theorem numbering remain intact.

## Deferred Items

These are real improvement opportunities, but they are intentionally deferred until the basic design system is stable:

- Full numeric alignment refactor with `siunitx`
- More aggressive title-page redesign
- Running headers/footers
- Bibliography style modernization
- Custom theorem boxes or colored statement blocks
- A full figure-size and placement pass
