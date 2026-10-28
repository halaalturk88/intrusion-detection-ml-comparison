"""
Assignment #2 Implementation
Comparative Analysis of Intrusion Detection Datasets using Feature Selection and Machine Learning

Datasets expected in --data-dir:
NSL-KDD:
    KDDTrain+.txt and KDDTest+.txt
    or KDDTrain+_20Percent.txt and KDDTest+.txt for a faster run
UNSW-NB15:
    UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv

Run example:
    python ids_assignment2_implementation.py --data-dir data --output-dir outputs --sample-size 15000

For a full run, use --sample-size 0. A full run can take a long time because SVM and wrapper
feature selection are computationally expensive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty"
]


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatible OneHotEncoder for multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_sample(df: pd.DataFrame, sample_size: int, label_col: str = "label") -> pd.DataFrame:
    if sample_size and sample_size > 0 and len(df) > sample_size:
        # Stratified sample when possible.
        return (
            df.groupby(label_col, group_keys=False)
              .apply(lambda x: x.sample(max(1, int(sample_size * len(x) / len(df))), random_state=RANDOM_STATE))
              .sample(frac=1, random_state=RANDOM_STATE)
              .reset_index(drop=True)
        )
    return df.reset_index(drop=True)


def find_file(data_dir: Path, candidates: List[str]) -> Optional[Path]:
    lower_map = {p.name.lower(): p for p in data_dir.glob("*") if p.is_file()}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_nsl_kdd(data_dir: Path, sample_size: int = 0) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_file = find_file(data_dir, ["KDDTrain+.txt", "KDDTrain+_20Percent.txt", "KDDTrain+.TXT", "KDDTrain+_20Percent.TXT"])
    test_file = find_file(data_dir, ["KDDTest+.txt", "KDDTest+.TXT", "KDDTest-21.txt", "KDDTest-21.TXT"])
    if train_file is None or test_file is None:
        raise FileNotFoundError(
            "NSL-KDD files were not found. Place KDDTrain+.txt and KDDTest+.txt in the data directory."
        )

    train = pd.read_csv(train_file, names=NSL_COLUMNS)
    test = pd.read_csv(test_file, names=NSL_COLUMNS)

    for df in (train, test):
        df["label"] = np.where(df["attack_type"].astype(str).str.lower().eq("normal"), 0, 1)
        df.drop(columns=["attack_type", "difficulty"], inplace=True, errors="ignore")

    train = safe_sample(train, sample_size, "label")
    test = safe_sample(test, max(1, sample_size // 2) if sample_size else 0, "label")

    X_train = train.drop(columns=["label"])
    y_train = train["label"].astype(int)
    X_test = test.drop(columns=["label"])
    y_test = test["label"].astype(int)
    return X_train, y_train, X_test, y_test


def load_unsw_nb15(data_dir: Path, sample_size: int = 0) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_file = find_file(data_dir, ["UNSW_NB15_training-set.csv", "UNSW-NB15_training-set.csv"])
    test_file = find_file(data_dir, ["UNSW_NB15_testing-set.csv", "UNSW-NB15_testing-set.csv"])
    if train_file is None or test_file is None:
        raise FileNotFoundError(
            "UNSW-NB15 files were not found. Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in the data directory."
        )

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Standard binary target: 0 = normal, 1 = attack.
    if "label" not in train.columns:
        if "attack_cat" in train.columns:
            train["label"] = np.where(train["attack_cat"].astype(str).str.lower().eq("normal"), 0, 1)
            test["label"] = np.where(test["attack_cat"].astype(str).str.lower().eq("normal"), 0, 1)
        else:
            raise ValueError("UNSW-NB15 must contain either 'label' or 'attack_cat'.")

    drop_cols = [col for col in ["id", "attack_cat"] if col in train.columns]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    train = safe_sample(train, sample_size, "label")
    test = safe_sample(test, max(1, sample_size // 2) if sample_size else 0, "label")

    X_train = train.drop(columns=["label"])
    y_train = train["label"].astype(int)
    X_test = test.drop(columns=["label"])
    y_test = test["label"].astype(int)
    return X_train, y_train, X_test, y_test


def split_if_needed(df: pd.DataFrame, target_col: str = "label") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_models() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    models = {
        "SVM": (
            SVC(random_state=RANDOM_STATE),
            {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
                "clf__subsample": [0.8, 1.0],
            },
        )
    else:
        print("WARNING: xgboost is not installed. Install it with: pip install xgboost")
    return models


def get_feature_selectors() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    # Filter methods and wrapper methods required by the assignment.
    lr_for_wrapper = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    return {
        "Full_Features": ("passthrough", {}),
        "Filter_ANOVA_F": (SelectKBest(score_func=f_classif), {"selector__k": [10, 20, 30]}),
        "Filter_Mutual_Info": (
            SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE)),
            {"selector__k": [10, 20, 30]},
        ),
        "Wrapper_RFE": (
            RFE(estimator=lr_for_wrapper, step=0.25),
            {"selector__n_features_to_select": [10, 20, 30]},
        ),
        "Wrapper_RFECV": (
            RFECV(
                estimator=lr_for_wrapper,
                step=0.25,
                min_features_to_select=10,
                cv=3,
                scoring="f1",
                n_jobs=-1,
            ),
            {},
        ),
    }


def get_selected_feature_names(best_pipeline: Pipeline, X_train: pd.DataFrame) -> List[str]:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    selector = best_pipeline.named_steps["selector"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(best_pipeline[:-1].transform(X_train).shape[1])])

    if selector == "passthrough" or not hasattr(selector, "get_support"):
        return feature_names.tolist()
    mask = selector.get_support()
    return np.array(feature_names)[mask].tolist()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def run_experiment(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    cv: int = 3,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_features"
    selected_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(X_train)
    models = get_models()
    selectors = get_feature_selectors()
    rows = []
    params_store = {}

    for selector_name, (selector, selector_grid) in selectors.items():
        for model_name, (model, model_grid) in models.items():
            print(f"\n=== {dataset_name} | {selector_name} | {model_name} ===")

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("selector", selector),
                ("clf", model),
            ])

            param_grid = {}
            param_grid.update(selector_grid)
            param_grid.update(model_grid)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=1,
                error_score="raise",
            )
            grid.fit(X_train, y_train)
            preds = grid.predict(X_test)
            metrics = evaluate_predictions(y_test, preds)

            selected_features = get_selected_feature_names(grid.best_estimator_, X_train)
            selected_file = selected_dir / f"{dataset_name}_{selector_name}_{model_name}.txt"
            selected_file.write_text("\n".join(selected_features), encoding="utf-8")

            report = classification_report(y_test, preds, zero_division=0)
            report_file = output_dir / f"classification_report_{dataset_name}_{selector_name}_{model_name}.txt"
            report_file.write_text(report, encoding="utf-8")

            row = {
                "dataset": dataset_name,
                "feature_selection": selector_name,
                "model": model_name,
                "n_selected_features": len(selected_features),
                **metrics,
            }
            rows.append(row)
            params_store[f"{dataset_name}_{selector_name}_{model_name}"] = grid.best_params_
            print(row)
            print("Best parameters:", grid.best_params_)

    params_file = output_dir / f"best_params_{dataset_name}.json"
    params_file.write_text(json.dumps(params_store, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def create_interpretation(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    best_overall = results.sort_values("f1_score", ascending=False).iloc[0]
    by_dataset = results.groupby("dataset")["f1_score"].max().sort_values(ascending=False)
    by_model = results.groupby("model")["f1_score"].mean().sort_values(ascending=False)
    by_selector = results.groupby("feature_selection")["f1_score"].mean().sort_values(ascending=False)

    lines = [
        "Automatic Interpretation of IDS Experiment Results",
        "================================================",
        "",
        f"Best overall combination: {best_overall['dataset']} | {best_overall['feature_selection']} | {best_overall['model']} with F1 = {best_overall['f1_score']:.4f}.",
        "",
        "Best F1 by dataset:",
        by_dataset.to_string(),
        "",
        "Average F1 by model:",
        by_model.to_string(),
        "",
        "Average F1 by feature-selection strategy:",
        by_selector.to_string(),
        "",
        "General reading:",
        "- If selected-feature runs are close to or better than full-feature runs, feature selection improved efficiency without losing detection quality.",
        "- Wrapper methods may produce stronger subsets but require more computation than filter methods.",
        "- Large differences between training/cross-validation and testing results indicate overfitting.",
        "- Compare UNSW-NB15 and NSL-KDD using F1-score rather than accuracy alone because IDS data can be imbalanced.",
    ]
    (output_dir / "interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="IDS dataset comparison using feature selection and machine learning.")
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing NSL-KDD and UNSW-NB15 files.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Folder where results will be saved.")
    parser.add_argument("--sample-size", type=int, default=15000, help="Rows sampled from each training set. Use 0 for full data.")
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds for GridSearchCV.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    loaders = {
        "NSL-KDD": load_nsl_kdd,
        "UNSW-NB15": load_unsw_nb15,
    }

    for dataset_name, loader in loaders.items():
        try:
            X_train, y_train, X_test, y_test = loader(data_dir, sample_size=args.sample_size)
            print(f"Loaded {dataset_name}: train={X_train.shape}, test={X_test.shape}")
            result_df = run_experiment(dataset_name, X_train, y_train, X_test, y_test, output_dir, cv=args.cv)
            all_results.append(result_df)
        except FileNotFoundError as e:
            print(f"Skipping {dataset_name}: {e}")
        except Exception as e:
            print(f"Error while processing {dataset_name}: {e}")
            raise

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(output_dir / "results_summary.csv", index=False)
        create_interpretation(final_results, output_dir)
        print(f"\nDone. Results saved in: {output_dir.resolve()}")
    else:
        print("No datasets were processed. Please place the required dataset files in the data directory.")


if __name__ == "__main__":
    main()
