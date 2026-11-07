# StreamLoop Churn Model Tuning — Project Context

## 1. Document Purpose

This file is the implementation handoff for the **StreamLoop churn model tuning** project. It translates the project brief into a precise scope, execution contract, and acceptance checklist for the developer or coding agent that will build the project.

This document is context only. No application, notebook, model, or tuning report has been implemented yet.

## 2. Source of Truth

Primary project brief:

- <https://github.com/4GeeksAcademy/ai-engineering-syllabus/blob/main/content/projects/streamloop-churn-model-tuning/README.md>

Dataset:

- <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv>

If this document and the project brief ever conflict, follow the project brief. Treat the dataset URL as a remote dependency and preserve a reproducible record of the library versions and random seeds used during implementation.

## 3. Project Summary

StreamLoop is a fictional mid-size subscription streaming platform. A first churn classifier already exists conceptually, but it was trained with default hyperparameters and judged primarily by its initial performance. StreamLoop's technical lead needs evidence that the model was tuned systematically, evaluated without data leakage, and selected using a metric that reflects the business cost of missing customers who will churn.

The project is an educational machine-learning exercise focused on **sound model-selection methodology**, not on finding the most sophisticated classifier or deploying a production service. The final work must demonstrate a disciplined progression:

1. Establish a default-hyperparameter baseline.
2. Explore a broad hyperparameter space with `RandomizedSearchCV`.
3. Refine the promising region with `GridSearchCV`.
4. Inspect both average cross-validation performance and variability across folds.
5. Select and justify a final estimator.
6. Compare the final tuned model with the baseline on a held-out test set.

## 4. Business Problem

The target is whether a customer churns:

- Positive class: `Churn == "Yes"`
- Negative class: `Churn == "No"`

The business states that failing to identify a customer who will churn is more expensive than sending a retention offer to a customer who would have stayed. In classification terms, **false negatives are more costly than false positives**.

Consequences for model evaluation:

- Accuracy must not be the primary tuning metric.
- The primary search metric must reward identifying actual churners.
- **Recall for the positive (`Yes`) class is the most direct default choice** unless the implementation explicitly chooses another metric and clearly demonstrates why it better captures the stated cost asymmetry.
- Precision, F1, confusion matrix counts, and optionally PR-AUC should be reported as supporting context so that a high-recall model is not selected without acknowledging the false-positive burden.
- The baseline and final model must be compared with the same metric definitions and the same untouched test split.

The exercise does not provide monetary costs or an operating threshold. Do not invent an ROI claim. Threshold optimization, cost-sensitive decision analysis, and retention-campaign economics may be discussed as future work, but they are outside the required scope unless intentionally added without compromising the core requirements.

## 5. Dataset Contract

### 5.1 Observed shape and target distribution

At the time this context was prepared, the source CSV contained:

- 7,043 customer rows
- 21 columns, including the target
- 1,869 `Churn == "Yes"` rows (approximately 26.5%)
- 5,174 `Churn == "No"` rows (approximately 73.5%)

This is an imbalanced binary-classification problem, although the imbalance is not extreme. A stratified train/test split and stratified cross-validation are appropriate.

### 5.2 Columns

| Column | Intended role / type | Implementation note |
| --- | --- | --- |
| `customerID` | Identifier | Exclude from model features; it has no intended predictive meaning. |
| `gender` | Categorical | Encode within the pipeline. |
| `SeniorCitizen` | Binary indicator | Usually arrives as `0`/`1`; may be treated as numeric or categorical consistently. |
| `Partner` | Categorical/binary | Encode within the pipeline. |
| `Dependents` | Categorical/binary | Encode within the pipeline. |
| `tenure` | Numeric | Impute if necessary within the pipeline. |
| `PhoneService` | Categorical/binary | Encode within the pipeline. |
| `MultipleLines` | Categorical | Values such as `No phone service` are legitimate categories, not missing data. |
| `InternetService` | Categorical | Encode within the pipeline. |
| `OnlineSecurity` | Categorical | Values such as `No internet service` are legitimate categories. |
| `OnlineBackup` | Categorical | Same handling principle as above. |
| `DeviceProtection` | Categorical | Same handling principle as above. |
| `TechSupport` | Categorical | Same handling principle as above. |
| `StreamingTV` | Categorical | Same handling principle as above. |
| `StreamingMovies` | Categorical | Same handling principle as above. |
| `Contract` | Categorical | Encode within the pipeline. |
| `PaperlessBilling` | Categorical/binary | Encode within the pipeline. |
| `PaymentMethod` | Categorical | Encode within the pipeline. |
| `MonthlyCharges` | Numeric | Impute if necessary within the pipeline. |
| `TotalCharges` | Numeric represented as text | Coerce whitespace/non-numeric values to missing, then impute inside the pipeline. |
| `Churn` | Target | Map `Yes` to the positive class and `No` to the negative class. |

### 5.3 Known cleaning edge case

The source currently contains 11 `TotalCharges` values consisting of whitespace rather than a number; these rows also have `tenure == 0`. The implementation must handle them deliberately. A robust approach is to perform numeric coercion as a pipeline transformation, convert invalid/blank values to missing, and impute them using a value learned from the training data. Do not silently fit an imputer on the complete dataset.

### 5.4 Data assumptions that must be verified in the notebook

- The target contains only `Yes` and `No`.
- `customerID` is unique or, at minimum, is not used as a feature.
- The observed columns match the expected schema.
- Duplicate rows, unexpected nulls, and unexpected categories are reported.
- Class counts are recorded before modeling.

Exploratory analysis should be sufficient to verify the data contract, but a large general-purpose EDA is not the purpose of this project.

## 6. Required Methodology

### 6.1 Reproducibility

- Set and record a fixed random seed for the train/test split, classifier where supported, and randomized search.
- Record relevant package versions or provide an environment file such as `requirements.txt`.
- Make the notebook runnable from a clean environment from top to bottom.
- Load the dataset from the supplied URL as required by the brief; optional local caching must not become a hidden prerequisite.

### 6.2 Data separation and test-set discipline

The held-out test set is a final evaluation resource, not a tuning tool.

Required sequence:

1. Load the raw data and verify its schema.
2. Separate features from the target and remove/ignore the identifier as part of the modeled workflow.
3. Create one train/test split before fitting preprocessing or models. Use stratification on `Churn`.
4. Fit the default baseline pipeline on training data only.
5. Perform one baseline evaluation event on the test set and record the agreed metrics.
6. Run both hyperparameter searches using only the training partition.
7. Select the final estimator using cross-validation evidence from the training partition only.
8. Perform one final tuned-model evaluation event on the same test set and report the same metrics.

The brief describes the test set as being touched exactly twice: once for the baseline and once for the final tuned model. Computing several predeclared metrics from each of those two evaluation predictions is acceptable; repeatedly checking test performance while making tuning decisions is not.

### 6.3 Pipeline requirement

Preprocessing and the classifier must form a single scikit-learn `Pipeline`. Data-dependent preprocessing must never be fit before the split or outside cross-validation.

The preprocessing stage should, at minimum:

- Exclude `customerID`.
- Coerce `TotalCharges` to numeric and treat blanks as missing.
- Impute missing numeric values using a training-derived strategy.
- Encode categorical variables, preferably with unknown-category tolerance such as `handle_unknown="ignore"`.
- Scale numeric features if the chosen classifier benefits from scaling.

A `ColumnTransformer` within the pipeline is the expected general pattern. Any custom or function transformer must work correctly when the pipeline is cloned by scikit-learn search objects.

### 6.4 Classifier choice

The classifier architecture is intentionally not graded. Select one classifier that:

- Supports a meaningful tunable hyperparameter space.
- Can be evaluated with the chosen churn-oriented metric.
- Runs within reasonable notebook time on approximately 7,000 rows.
- Has its preprocessing needs correctly represented in the pipeline.

Reasonable options include logistic regression, random forest, gradient boosting, or another scikit-learn-compatible binary classifier. Logistic regression offers interpretability and a compact search space; tree ensembles allow nonlinear interactions but may require a more carefully bounded search. Do not expand the project into a broad tournament of unrelated model families—the required comparison is default versus tuned behavior for the chosen model.

If supported by the classifier, class weighting may be included in the hyperparameter search. It must not replace deliberate metric selection.

### 6.5 Baseline

The baseline must use the complete preprocessing-and-classifier pipeline with the classifier's default hyperparameters, except for non-performance controls needed for reproducibility or convergence (for example, `random_state` or a sufficiently high iteration limit). Any such deviation from literal defaults should be disclosed.

Record at least:

- Primary business-aligned metric
- Precision for churn
- Recall for churn
- F1 for churn
- Confusion matrix
- Accuracy as secondary context, not the selection criterion

If probability estimates are available, PR-AUC may also be reported.

### 6.6 Broad random search

Run `RandomizedSearchCV` first.

Mandatory characteristics:

- Fit on training data only.
- Search parameters using pipeline-prefixed names (for example, `classifier__...`).
- Use cross-validation; stratified folds are expected for this target.
- Set `scoring` explicitly to the chosen business-aligned metric or scorer.
- Set `n_jobs=-1`.
- Use a fixed `random_state`.
- Leave `refit=True` so the best estimator is retrained automatically on all training data.
- Use a broad but valid search space based on the selected classifier's actual API.
- Ensure sampled combinations are legal; if parameters have conditional compatibility, use separate parameter dictionaries or otherwise prevent invalid combinations.

The random search exists to locate promising regions cheaply. It is not the final precision search.

### 6.7 Focused grid search

Use the random-search evidence to construct a smaller, explicit `GridSearchCV` space around promising values or categories.

Mandatory characteristics:

- Fit on training data only.
- Use the same primary scoring definition and a compatible cross-validation strategy.
- Set `n_jobs=-1`.
- Leave `refit=True`.
- Make the relationship between the random-search results and the narrowed grid visible in notebook commentary.

Do not manually call `.fit()` on `best_estimator_` after either search. With `refit=True`, the search object has already fitted the selected estimator on the full training partition.

### 6.8 Cross-validation stability and model selection

Do not select the final model solely because it has the numerically highest `mean_test_score`.

Inspect `cv_results_` for a small set of leading candidates, including:

- Hyperparameter values
- Rank
- Mean validation score
- Standard deviation across folds
- Individual split scores where useful
- Fit time as secondary operational context
- Training score/generalization gap only if `return_train_score=True`

The final choice may be:

- The candidate with the strongest mean score when its variance is acceptable; or
- A slightly lower-scoring candidate that is materially more stable or simpler.

The notebook and report must state the rule used and explain the trade-off. Do not claim a difference is meaningful merely because one mean exceeds another by a tiny amount. Where candidates are effectively tied relative to fold variation, prefer the stable or simpler choice and say so.

## 7. Required Deliverables

The submitted repository must include:

1. **A Jupyter notebook** containing the complete, reproducible workflow:
   - Data loading and focused validation
   - Train/test split
   - Pipeline construction
   - Default baseline and its single test evaluation
   - Broad randomized search
   - Narrowed grid search
   - `cv_results_` stability analysis
   - Final model selection
   - Final single test evaluation
   - Clear markdown reasoning throughout
2. **`tuning_report.md`** containing a concise stakeholder-readable comparison and decision record.
3. **Dependency/environment declaration**, strongly recommended for reproducibility (for example, `requirements.txt`).
4. **Repository instructions**, if the starter README does not already explain how to install dependencies and run the notebook.

The project brief explicitly requires the notebook and `tuning_report.md`. A deployed API, web UI, dashboard, database, and production monitoring stack are not required.

## 8. `tuning_report.md` Content Contract

The report should be understandable without reading every notebook cell and should contain:

### 8.1 Executive summary

- What was tuned
- Which primary metric was used
- Whether tuning improved the held-out result over the default baseline
- Which final configuration was selected

### 8.2 Business-aligned metric rationale

- Explain why missed churners are especially costly.
- Explain why the selected primary metric represents that priority.
- Acknowledge the precision/false-positive trade-off.

### 8.3 Experimental design

- Train/test split and random seed
- Cross-validation strategy and fold count
- Confirmation that searches used training data only
- Baseline → randomized search → grid search progression

### 8.4 Baseline versus tuned comparison

Use a compact table with identical metric definitions for both models. Include the primary metric and supporting precision, recall, F1, confusion-matrix counts, and accuracy. Include PR-AUC if adopted.

### 8.5 Search and selection evidence

- Broad random-search scope
- How the random results informed the grid
- Final hyperparameters
- Top-candidate mean and variation across folds
- Explanation of whether the final model maximized the mean or traded a small amount of mean performance for stability/simplicity

### 8.6 Limitations and next steps

Possible items include:

- The data is a public telecom proxy rather than proprietary StreamLoop behavior.
- The outcome is not time-indexed, so the split does not test temporal drift.
- No monetary cost matrix or retention-offer capacity was supplied.
- No probability calibration, threshold optimization, fairness audit, or production monitoring is required in this iteration.

Avoid presenting speculative future work as completed functionality.

## 9. Acceptance Criteria

The implementation is complete only when all of the following are true:

- [ ] The supplied dataset URL is used and the expected schema is validated.
- [ ] `Churn == "Yes"` is consistently treated as the positive class.
- [ ] `customerID` is excluded from predictive features.
- [ ] Blank/non-numeric `TotalCharges` values are handled safely.
- [ ] A stratified train/test split is created before fitting any preprocessing or model.
- [ ] All learned preprocessing is inside a scikit-learn pipeline.
- [ ] A default-hyperparameter baseline is trained and recorded before tuning.
- [ ] Baseline test evaluation occurs once using predeclared metrics.
- [ ] `RandomizedSearchCV` performs the broad search on training data only.
- [ ] The random search uses cross-validation, explicit business-aligned `scoring`, `n_jobs=-1`, a seed, and `refit=True`.
- [ ] The random-search findings visibly determine the narrower grid.
- [ ] `GridSearchCV` performs the focused search on training data only with the same primary scoring logic.
- [ ] `best_estimator_` is not manually refitted after a search with `refit=True`.
- [ ] Multiple leading candidates from `cv_results_` are examined for fold-to-fold variation.
- [ ] The final model selection is justified using both performance and stability/simplicity evidence.
- [ ] Final tuned test evaluation occurs once, on the original held-out test partition.
- [ ] Baseline and tuned test results use identical metric definitions and are directly compared.
- [ ] The notebook runs from top to bottom in a clean environment.
- [ ] `tuning_report.md` states the metric rationale, final hyperparameters, baseline/tuned results, stability trade-off, and limitations.
- [ ] No production deployment or unrelated application work is presented as part of the required scope.

## 10. Validation Expectations for the Builder

Because this is primarily a notebook project, validation should focus on reproducibility and leakage prevention rather than a large application test suite.

At minimum, the builder should:

- Restart the notebook kernel and run all cells in order without relying on hidden state.
- Confirm that no preprocessing object is fitted on the complete dataset.
- Confirm that search `.fit()` calls receive only training features and labels.
- Confirm that the test set is evaluated only in the baseline and final evaluation sections.
- Confirm the pipeline can transform test rows with categories not observed during fitting.
- Confirm all searched parameter combinations are valid or that failures are deliberately handled and explained.
- Confirm the report values match the notebook outputs exactly.
- Confirm all random seeds, CV settings, metric definitions, and final hyperparameters are recorded.

Focused automated checks may be added for custom transformers or data-contract assumptions if implementation choices introduce code outside the notebook. Do not add a testing framework solely for ceremony.

## 11. Explicit Non-Goals

Unless the project owner later expands the scope, do not build:

- A REST API or inference microservice
- A frontend, dashboard, or Streamlit application
- Authentication, a database, or customer-management workflow
- A production training pipeline or scheduler
- Cloud deployment or CI/CD infrastructure
- Real-time predictions or monitoring
- A deep-learning model
- A multi-model AutoML benchmark
- A retention campaign optimizer
- A new or synthetic dataset
- Model serialization as a substitute for the required analysis and report

Saving the final fitted model is optional and does not replace any graded deliverable.

## 12. Recommended Implementation Order

1. Initialize the Python environment and declare dependencies.
2. Create the notebook structure with markdown sections matching the methodology.
3. Load and validate the remote dataset.
4. Define the target, feature groups, and one stratified held-out split.
5. Build a clone-safe preprocessing/classifier pipeline.
6. Fit and evaluate the default baseline once on the test set.
7. Define and run the broad randomized search on training data.
8. Analyze its `cv_results_` and define the focused grid.
9. Run the focused grid search on training data.
10. Compare leading CV candidates for mean score and stability.
11. Select the final already-refitted estimator.
12. Evaluate it once on the held-out test set.
13. Write `tuning_report.md` from the recorded evidence.
14. Restart and run the notebook end to end; reconcile all report figures.

## 13. Decision Boundaries Left to the Builder

The brief deliberately leaves these implementation choices open:

- Classifier family
- Exact train/test ratio and random seed
- Cross-validation fold count
- Exact search distributions and grid values
- Exact primary metric, provided it is deliberately tied to the cost of missed churners
- Numeric imputation strategy
- Whether to use scaling, class weights, PR-AUC, confidence intervals, or visualizations

Each choice should be proportionate to the small tabular dataset, documented, and consistent across baseline and tuned comparisons. Choices must not weaken the core constraints around pipelines, test isolation, broad-to-focused search, or stability analysis.

## 14. Common Failure Modes to Prevent

- Encoding or imputing the full dataset before splitting
- Leaving `customerID` in the features
- Treating service-status strings such as `No internet service` as missing values
- Allowing blank `TotalCharges` strings to break numeric processing
- Using accuracy or the classifier's default score without business justification
- Tuning against the test set or checking it after every experiment
- Running grid search first without a broad randomized search
- Defining a grid unrelated to the random-search evidence
- Using invalid combinations of conditional hyperparameters
- Manually refitting `best_estimator_` despite `refit=True`
- Reporting only `best_score_` without examining fold variation
- Selecting a marginally higher mean score while ignoring instability
- Comparing baseline and tuned models with different splits or metrics
- Presenting notebook outputs that cannot be reproduced from a clean run
- Expanding into deployment work before satisfying the analysis deliverables

## 15. Definition of Done

This project is done when the repository provides a reproducible notebook and a concise `tuning_report.md` that together prove:

1. The default model was measured fairly.
2. Hyperparameters were explored broadly and then refined deliberately.
3. The held-out test set did not influence tuning or model selection.
4. The selection metric reflects StreamLoop's priority of catching churners.
5. Cross-validation stability was considered rather than trusting one mean score.
6. The final tuned model was compared fairly with the baseline and the decision was explained.

