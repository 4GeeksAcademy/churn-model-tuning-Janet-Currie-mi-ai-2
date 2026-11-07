# StreamLoop churn model tuning

This project tunes a logistic regression model that flags customers who may cancel. The full experiment is in `streamloop_churn_tuning.ipynb`. The short decision note is in `tuning_report.md`.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name streamloop-churn --display-name "Python (streamloop churn)"
```

## Run the notebook

Open it:

```bash
jupyter notebook streamloop_churn_tuning.ipynb
```

Or run every cell from top to bottom:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=streamloop-churn streamloop_churn_tuning.ipynb
```

The notebook downloads the customer table from the internet when it runs. Run it from this folder so it can import `preprocess.py`.

## Check the cleaner

```bash
python -m unittest test_preprocess.py
```
