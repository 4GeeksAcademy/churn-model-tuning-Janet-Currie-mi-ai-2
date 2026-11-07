# StreamLoop churn model: factory settings versus tuned settings

## Executive summary

The model is logistic regression. The main score is recall for customers who leave (`Churn == "Yes"`).

Tuning did improve that score on the locked test customers. Recall rose from **0.5588** to **0.7834**.

The final settings are `C=1.0` and `class_weight="balanced"`, with `max_iter=1000` and `random_state=42`. The factory model already used `C=1.0`. The change that mattered was turning class weight on, so the model pays more attention to the smaller group of quitters.

## Why this score

A missed quitter is more costly than a stay offer sent to someone who would have stayed. Recall asks how many of the real quitters the model catches. That matches the cost.

The tuned model does catch more quitters, and it also flags more people who stay. Precision on the test customers fell from 0.6572 to 0.5034. False positives rose from 109 to 289. That trade-off is real. Accuracy also fell, from 0.8055 to 0.7374, because accuracy gives the large "staying" group more influence. Accuracy was not used to choose the model.

## How the test was run

- The table has 7,043 customers: 1,869 quitters and 5,174 stayers. It was read from the project URL.
- `customerID` was not used as a feature. Eleven blank `TotalCharges` values were turned into missing numbers inside the model pipeline and filled from the practice customers only.
- Seed 42. 5,634 practice customers (80 percent) and 1,409 locked test customers (20 percent). Both piles have the same share of quitters, 0.2654.
- Both searches used 5 shuffled stratified folds and saw only the 5,634 practice customers.
- The test customers were scored twice: once for the factory model, once for the final model. The same scorecard was used both times.
- Library versions for this run: pandas 3.1.1 and scikit-learn 1.9.1.
- `max_iter=1000` and `random_state=42` are set on the factory model and the tuned model so the fit can finish and the run can be repeated. They are not a claim that the factory regularization strength was changed.

The path was factory model, then 20 random settings, then a small grid around the promising region.

The broad search tried `C` from 0.001 to 100 and class weight either off or `balanced`. With class weight off, practice recall sat around 0.54 to 0.55. With `balanced`, it sat around 0.79 to 0.80. That gap is about 0.25, much larger than the fold-to-fold spread of about 0.04. Inside the balanced group, `C` values from about 0.37 to 14.5 were within about 0.002 of each other.

The tight grid therefore kept `balanced` and checked these `C` values: 0.3, 0.5, 1, 2, 3.5, 5, 10, and 15.

## Factory model versus tuned model

These numbers are the locked test customers. Quitters are the positive class.

| Metric | Factory | Tuned |
| --- | ---: | ---: |
| Recall (main score) | 0.5588 | 0.7834 |
| Precision | 0.6572 | 0.5034 |
| F1 | 0.6040 | 0.6130 |
| PR-AUC | 0.6338 | 0.6324 |
| Accuracy (context only) | 0.8055 | 0.7374 |
| True negatives | 926 | 746 |
| False positives | 109 | 289 |
| False negatives | 165 | 81 |
| True positives | 209 | 293 |

The tuned model missed 81 quitters instead of 165. It also sent the "might leave" signal to 289 stayers instead of 109. PR-AUC barely moved, from 0.6338 to 0.6324, so the ranking quality is about the same. The recall gain comes from the balanced class weight changing who gets flagged at the default cutoff. No cutoff was tuned.

## What the search showed, and why this model was kept

The highest practice-fold average on the tight grid was **0.8020** at `C=3.5`, with a fold spread of **0.0376**. The chosen settings, `C=1.0` and `balanced`, scored **0.8013** with a fold spread of **0.0379**. The gap is **0.0007**, which sits inside the fold spread. That is not a real win for `C=3.5`.

`C=2.0` had the same 0.8013 average and a slightly smaller spread, 0.0366. The difference in spread is about 0.001, so it is not a meaningful stability win either. The rule for a tie is: keep the simpler settings. Class weight off was not in the tie, because its recall was far lower. Among the tied balanced settings, `C=1.0` is the one closest to the factory value. That is the final model.

The search trained those chosen settings once on the full practice pile. They were not trained a second time by hand.

On the practice folds, the chosen model's training recall was about 0.8062 and its validation recall was 0.8013. The small gap says the model is not memorizing the practice folds.

## Limits

- The rows are a public phone-company table, not StreamLoop's own customers.
- The rows are not ordered by time, so this split does not show whether the pattern would hold next month.
- No dollar costs or offer limits were provided, so this report does not claim a money result.
- The cutoff was left at the default. Probability calibration, a fairness check, and production monitoring were not part of this pass.
