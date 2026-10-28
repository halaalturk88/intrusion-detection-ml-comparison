# Intrusion Detection: Machine Learning & Feature Selection Comparison

An academic machine learning project exploring **binary network intrusion detection** (normal vs. attack) with two benchmark datasets: **NSL-KDD** and **UNSW-NB15**.

> **Status:** Source code and notebook are provided. Training datasets and verified experiment results are **not included** in this repository. Model performance is not claimed until the experiments are run and validated.

## What the project explores

- Preprocessing numeric and categorical network features, including missing-value handling, scaling, and one-hot encoding.
- Comparing **SVM**, **Random Forest**, and **XGBoost** classifiers.
- Comparing feature selection approaches: **ANOVA F-test**, **Mutual Information**, **RFE**, **RFECV**, and a full-feature baseline.
- Tuning parameters with `GridSearchCV` and evaluating predictions using **Accuracy, Precision, Recall, and F1-score**.
- Saving selected features, model configuration, classification reports, and a summary of results when datasets are available.

## Repository contents

| File | Description |
| --- | --- |
| `ids_assignment2_implementation.py` | Main Python implementation |
| `IDS_Assignment2_Implementation.ipynb` | Jupyter Notebook version |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes local datasets, environment files, and generated artifacts |

## Setup

1. Create a Python environment (recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `data/` folder and obtain the datasets from their respective legitimate sources. Place the following files inside it:
   - NSL-KDD: `KDDTrain+.txt` and `KDDTest+.txt` (or `KDDTrain+_20Percent.txt` for faster training).
   - UNSW-NB15: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
4. Run:
   ```bash
   python ids_assignment2_implementation.py --data-dir data --output-dir outputs --sample-size 15000
   ```

For a full run, specify `--sample-size 0`. Some model and feature-selection combinations are computationally expensive.

## Outputs (generated upon successful execution)

- `outputs/results_summary.csv`: metrics for each experimental combination.
- `outputs/interpretation.txt`: automated comparison notes.
- `outputs/best_params_*.json`: selected hyperparameters.
- `outputs/classification_report_*.txt`: per-experiment classification reports.
- `outputs/selected_features/*.txt`: selected input features.

## Limitations and next steps

The repository does not currently contain datasets or executed results, so performance comparisons cannot yet be independently verified here. After a successful run, add a concise results table, explain differences between datasets, and document limitations. Keep private or restricted data out of the public repository.

## Project context

Prepared as a university coursework project. Contributors and individual responsibilities should be acknowledged according to actual participation.
