# Decisions

## Logistic regression

The model family is not graded. Logistic regression keeps the search small: `C` and class weight. `max_iter=1000` and `random_state=42` are repeatability controls on both the factory model and the tuned model.

## Recall

Missing a quitter is the costly mistake, so recall for `Yes` is the main score.

## Split and folds

Seed 42. 80 percent practice, 20 percent test, stratified. Five shuffled stratified folds.

## Cleaning

Median fill and scaling for `tenure`, `MonthlyCharges`, and `TotalCharges`. Most-common fill and one-hot encoding for the other features, including `SeniorCitizen`. Unknown categories are ignored. Blank charges are coerced in `preprocess.py` so the all-cores search can import the step.

## Grid and winner

The broad search showed balanced class weight far above class weight off, and a flat `C` ridge from about 0.3 to 15. The tight grid checks that ridge only.

A gap smaller than the leader's fold spread is a tie. The simpler tied setting wins. That kept `C=1` and `class_weight="balanced"`. The tight search uses this rule as `refit` so the chosen model is trained once. The broad search uses `refit=True`.

## Starter files

`main.py` and `server.py` stay untouched.
