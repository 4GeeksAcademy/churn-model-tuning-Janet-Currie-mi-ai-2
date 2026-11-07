# Technology notes

Versions used for the finished run:

- Python 3.13.14
- pandas 3.1.1
- scikit-learn 1.9.1
- scipy 1.18.1
- jupyter 1.1.1
- certifi 2026.10.23

scikit-learn 1.9 still accepts `class_weight="balanced"` and `C` on `LogisticRegression`. The `penalty` argument is deprecated and was not set.

`GridSearchCV` accepts a function for `refit`. That function returns the chosen row, and the search trains that row once. `best_score_` is not filled in when `refit` is a function, so the notebook reads the mean from `cv_results_`.

This machine's Python cannot verify the dataset website certificate unless `SSL_CERT_FILE` points at the `certifi` bundle before `pandas.read_csv` opens the URL.
