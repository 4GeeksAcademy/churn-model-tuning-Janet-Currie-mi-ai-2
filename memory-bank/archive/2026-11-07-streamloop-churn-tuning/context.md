# StreamLoop Churn Model Tuning

## Goal

Tune one churn classifier and compare it with a default-settings model. Deliver a notebook and `tuning_report.md`.

## Scope

Load the public telecom churn CSV from the given URL. Keep preprocessing inside one scikit-learn pipeline. Train a logistic-regression baseline, search settings on the practice split only, then score the locked test split twice.

## Out of scope

No website, API, database, deployment, or multi-model contest. `main.py` and `server.py` stay as the course starter.

## Constraints

Positive class is `Churn == "Yes"`. Main score is recall. `customerID` is not a feature. Blank `TotalCharges` values become missing numbers inside the pipeline. Seed 42. Split is 80/20 and stratified. Searches use 5 stratified folds and `n_jobs=-1`.

## Result

Completed. Durable notes are in `memory-bank/implementation-memory/streamloop-churn-tuning.md`.
