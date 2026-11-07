# StreamLoop Churn Model Tuning

## Purpose

Compare a factory-settings logistic regression churn model with a tuned one, using a score that catches customers who leave. The work is a notebook experiment and a short decision report.

## Final Behavior

The notebook loads the public telecom churn table, checks the contract, splits once, and keeps cleaning inside one pipeline. It scores a factory model on the locked test customers, searches settings on the practice customers only, and scores the chosen model once on the same test customers.

On the locked test set, recall for quitters rose from 0.5588 to 0.7834. Precision fell from 0.6572 to 0.5034. The final settings are `C=1.0` and `class_weight="balanced"`.

## Architecture and Data Flow

`preprocess.py` holds `TotalChargesCleaner`, a stateless step that turns blank charges into missing numbers. The notebook builds a scikit-learn pipeline: cleaner, median fill and scaling for numbers, most-common fill and one-hot encoding for categories, then logistic regression.

Practice rows: 5,634. Test rows: 1,409. Seed 42. Five stratified folds. The broad search is `RandomizedSearchCV` with `refit=True`. The tight search is `GridSearchCV` with a selection function so a tied, simpler setting can be the one model the search retrains.

## Important Files

- `streamloop_churn_tuning.ipynb` — the experiment, with saved outputs from a clean run.
- `tuning_report.md` — the decision note. Its figures match that run.
- `preprocess.py` and `test_preprocess.py` — the blank-charge cleaner and its checks.
- `requirements.txt` — pinned pandas, scikit-learn, scipy, jupyter, and certifi.
- `README.md` — install and run steps.
- `main.py` and `server.py` — untouched course starter. They are not part of the experiment.

## Interfaces and Contracts

- Data URL: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- Positive class: `Churn == "Yes"`, coded as 1.
- Main score: scikit-learn `"recall"`.
- `customerID` is excluded.
- Eleven blank `TotalCharges` rows also have `tenure == 0`.
- `No internet service` and `No phone service` stay as categories.
- This Python install needs `certifi` so HTTPS certificate checks succeed. The notebook still reads the URL. It does not require a saved copy of the table.

## Decisions and Constraints

Logistic regression was chosen because the model family is not graded and the search stays small. The knobs are `C` and class weight. `max_iter=1000` and `random_state=42` are repeatability controls on both models.

The broad search showed class weight off stuck near 0.54 practice recall, while `balanced` sat near 0.80. `C` from 0.3 to 15 was flat. The tight grid is `C` in `[0.3, 0.5, 1, 2, 3.5, 5, 10, 15]` with `class_weight="balanced"` only.

The winner rule treats a gap smaller than the leader's fold spread as a tie, then prefers the simpler tied setting. `C=3.5` led by 0.0007, inside a spread of 0.0376, so the final model keeps factory `C=1` and balanced class weight. The tight search uses that rule as its `refit` function. The broad search still uses `refit=True`. Neither chosen estimator is fitted again by hand.

## Validation

- `python -m unittest test_preprocess.py` — 3 tests passed.
- `jupyter nbconvert --execute` ran the notebook from top to bottom twice. The second run matched `tuning_report.md`.
- Test scoring stages were `baseline`, then `final`.
- An unseen contract category transformed successfully.

## Maintenance Notes

Change the tight grid only after reading a new broad-search table. The notebook stops if the balanced `C` band from 0.3 to 15 is no longer flat, or if class weight off is no longer clearly worse.

The cleaner must stay importable from `preprocess.py`. Defining it only inside the notebook can break `n_jobs=-1`.

Do not turn this project into a website. `server.py` is the course starter, not the assignment.
