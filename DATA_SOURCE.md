# Data sources

## Simulation results
Everything the simulation analyses produce is in this repository: `results/` (cached outputs of every experiment),
`CICADAS_II_Paper_Draft/results_macros.tex` (every reported number), `CICADAS_II_Paper_Draft/rr_tables/` and
`CICADAS_II_Paper_Draft/figures_rr/`. `python -m cicadas2.rr.run` regenerates them from code (see `REPRODUCIBILITY.md`).

## Real cohort (Section 6 of the manuscript)
De-identified patient-level data for the 9,108-patient real cohort are available to credentialed users under a data
use agreement through the Brain Data Science Platform:

- bdsp.io project: **CICADAS 2** (https://bdsp.io; DOI to be added on publication)
- S3: `s3://bdsp-opendata-credentialed/cicadas2/`

The data are not in this repository. [`real_cohort/MANIFEST.csv`](real_cohort/MANIFEST.csv) lists every file with its
size, SHA-256 hash and S3 location, and [`real_cohort/reproduce_section6.py`](real_cohort/reproduce_section6.py)
recomputes every number in Section 6 from the downloaded package (run it from the package root:
`python code/reproduce_section6.py`).

**Contents:** minute-level automated EEG labels (seizure, LPD, GPD, LRDA, GRDA; first 5 days of EEG), anti-seizure and
sedative medication administrations, and clinical covariates (age capped at 90, sex, SOFA, diagnosis category), on a
per-patient relative time axis with BDSP surrogate identifiers. No outcomes, calendar dates or direct identifiers.

**Provenance:** EEG and medication administration records from Massachusetts General Brigham and Beth Israel Deaconess
Medical Center, held in BDSP; EEG labelled by the expert-level classifier of Jing et al., Neurology 2023;100:e1750.
Collected under IRB protocols MGH #2023P000487 and #2024P002630, BIDMC #2022P000481 and #2022P000417, and Stanford
#83833, with a waiver of informed consent.
