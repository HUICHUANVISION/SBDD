# SBDD — Search Breadth in Software Defect Prediction

This repository holds the experimental artifacts accompanying:

## Repository layout

```
SBDD/
├── README.md            — this file
├── LICENSE              — MIT license
├── CITATION.cff         — citation metadata
├── requirements.txt     — Python dependencies for the analysis
├── src/                 — analysis code (Python 3)
│   ├── data_loader.py   — load the 35 SDP datasets from data/
│   ├── classifiers.py   — LR / RF / XGBoost / MLP learner families
│   ├── da_methods.py    — none / SMOTE / ADASYN / Borderline-SMOTE
│   ├── intrinsic_features.py — dataset descriptors used in RQ4 routing
│   ├── grid_a2_nested_cv.py  — main nested-CV runner (catalog × K)
│   ├── run_v9.py        — orchestration of v9 grid (compatible with this STVR run)
│   ├── lodo_cv.py       — leave-one-family-out CV harness (RQ4)
│   ├── lodo_signal_v9.py — LODO with the four descriptor families
│   ├── recommender_v9.py — LOFO recommendation
│   ├── compute_rq5_ols.py — RQ5 OLS on inner/outer/optimism slopes
│   ├── aggregate.py     — bootstrap aggregation utilities
│   └── signals.py       — descriptor signal extraction
├── data/                — 35 benchmark datasets in Weka ARFF format
│   ├── AEEEM/           — 5 datasets (EQ, JDT, Lucene, Mylyn, PDE)
│   ├── JIRA/            — 7 datasets (activemq, derby, groovy, hbase, hive, jruby, wicket)
│   ├── NASA/            — 5 datasets (CM1, MW1, PC1, PC3, PC4)
│   ├── PROMISE/         — 10 datasets (ant, camel, ivy, jedit, log4j, poi, tomcat, velocity, xalan, xerces)
│   ├── ReLink/          — 3 datasets (Apache, Safe, Zxing)
│   └── SOFTLAB/         — 5 datasets (ar1, ar3, ar4, ar5, ar6)
├── results/             — paper-level summary tables (CSV)
├── figures/             — six PDF figures used by the paper
└── paper/               — STVR manuscript (PDF + .tex + .bib)
```

## Datasets

All 35 datasets are in Weka ARFF format and grouped into six benchmark families drawn from the prior public SDP literature:

| Family | Datasets | Source |
|---|---|---|
| AEEEM | 5 | D'Ambros et al., "An empirical comparison of software defect prediction techniques" |
| JIRA | 7 | Various defect-tracking projects |
| NASA | 5 | NASA MDP / Shepperd et al. |
| PROMISE | 10 | PROMISE Software Engineering Repository |
| ReLink | 3 | Wu et al., "Relinking feature models" |
| SOFTLAB | 5 | Zimmermann et al. cross-project defect data |

The five SOFTLAB datasets are referred to in the paper as ar1, ar3, ar4, ar5, ar6. NASA MDP datasets retain their canonical CM1/MW1/PC1/PC3/PC4 names.

## Reproducing the analysis

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main nested-CV grid (catalog of 20 policies × K∈{1,2,4,8,16,20}):
   ```bash
   python src/grid_a2_nested_cv.py \
       --datasets data/ \
       --catalog src/catalogs/catalog20.yaml \
       --ks 1,2,4,8,16,20 \
       --outer-seeds 42,2027,1729 \
       --output results/
   ```

3. Aggregate per-fold outputs into the summary CSVs in `results/`:
   ```bash
   python src/aggregate.py --run-dir runs/ --out results/
   python src/compute_rq5_ols.py --in results/ --out results/
   python src/signals.py --in results/ --out results/
   ```

4. Generate the six figures (matching `figures/`):
   ```bash
   python src/gen_figures.py --results results/ --out figures/
   ```

The provided `results/` directory contains the summary CSVs that feed the paper's Tables 1–4 and the underlying per-family, per-dataset aggregates. The `paper/` directory contains the STVR manuscript.

## Catalog of 20 policies

The frozen policy catalog combines four learner families (LR, RF, XGBoost, MLP) with four augmentation components (none, SMOTE, ADASYN, Borderline-SMOTE). The exact 20 tuples (with hyperparameters) are enumerated in Table 2 of the paper and mirrored in `src/catalogs/catalog20.yaml`.

## Citation

If you use this artifact, please cite the STVR paper. The CITATION.cff file contains machine-readable metadata for GitHub's citation panel.

## License

Code and results: MIT License (see `LICENSE`).
Datasets are redistributed from their original public sources; please respect the licenses of those upstream collections when reusing them.

## Provenance

This artifact accompanies the manuscript submitted to *Software Testing, Verification and Reliability* (Wiley) in September 2026. The repository will be tagged with the accepted-manuscript commit upon publication.
