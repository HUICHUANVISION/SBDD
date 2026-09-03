# Benchmark datasets

This directory contains the 35 static software-defect-prediction datasets used in the paper, in Weka ARFF format. The datasets come from six public benchmark families. All datasets are placed here as ARFF for direct loading via `scipy`/`liac-arff` or Weka.

## Family sizes and dataset names

| Family | n | Datasets |
|---|---|---|
| AEEEM | 5 | EQ, JDT, Lucene, Mylyn, PDE |
| JIRA | 7 | activemq-5.0.0, derby-10.5.1.1, groovy-1_6_BETA_1, hbase-0.94.0, hive-0.9.0, jruby-1.1, wicket-1.3.0-beta2 |
| NASA | 5 | CM1, MW1, PC1, PC3, PC4 |
| PROMISE | 10 | ant-1.7, camel-1.4, ivy-2.0, jedit-4.0, log4j-1.0, poi-2.0, tomcat, velocity-1.6, xalan-2.4, xerces-1.3 |
| ReLink | 3 | Apache, Safe, Zxing |
| SOFTLAB | 5 | ar1, ar3, ar4, ar5, ar6 |

Total: 35 datasets.

## Loading

`src/data_loader.py` reads any ARFF file under a family directory:

```python
from data_loader import load_dataset
X, y, feature_names = load_dataset("AEEEM/EQ.arff")
```

The `bug` attribute is treated as the binary label. Features are returned as a NumPy array with column order matching the ARFF `@attribute` declarations.

## Provenance and licensing

These datasets are redistributed from their original public sources. When reusing them, please respect the upstream licenses:

- **AEEEM** — D'Ambros et al., "An empirical comparison of software defect prediction techniques" (reproduced from the supplementary materials of that paper).
- **JIRA** — Defect-tracking datasets distributed by Shepperd et al. and others; see the JIRA subset of the PROMISE/NASA ecosystems.
- **NASA** — NASA Metrics Data Program (MDP) datasets, as curated by Shepperd et al.
- **PROMISE** — PROMISE Software Engineering Repository, http://promise.site.uottawa.ca/SERepository/datasets-page.html.
- **ReLink** — Apache, Safe, and ZXing defect subsets from Wu et al., "Relinking feature models in software product lines" (originally hosted at the ReLink project site).
- **SOFTLAB** — Five SOFTLAB cross-project defect subsets distributed by Zimmermann et al.

If a particular dataset's upstream license is more restrictive than MIT, that dataset's redistribution is non-commercial / academic-only; see the upstream source for the exact terms.