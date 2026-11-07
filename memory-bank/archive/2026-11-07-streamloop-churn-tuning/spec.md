# Spec

The notebook loads the official CSV URL, checks the schema, splits once before fitting, and uses one pipeline. A factory logistic regression is scored once on the test split. `RandomizedSearchCV` explores `C` and class weight on the practice split. `GridSearchCV` uses a smaller grid taken from those results. The winner is chosen from practice-fold scores and fold spread. That winner is scored once on the same test split. `tuning_report.md` matches the notebook.

Primary metric: recall for quitters. Also recorded: precision, F1, confusion-matrix counts, accuracy, and PR-AUC.

Acceptance is the checklist in `churn-model-tuning-context.md` section 9. The completed notebook and report meet that checklist.
