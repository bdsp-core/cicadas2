"""R&R (JNE major revision, Oct 2026) toolkit for CICADAS-II.

Everything added for the revision lives in this subpackage so the
original port stays untouched and comparable. Modules:

* cohort    - reproducible cohorts, one random stream per patient
* engine    - vectorised simulator (all patients at once), bit-identical
              to strategies.run_strategy given the same uniform tape
* survival  - exact survival probabilities and alternative mortality shapes
* stats     - MC standard errors, paired bootstrap, winner stability, E-value
* gformula  - treatment assignment, fitted mortality model, g-formula
* theta     - registry of every mortality parameter vector in the paper
* macros    - LaTeX macro writer (one source for every reported number)
* ledger    - append-only log tying results to the runs that produced them
"""
