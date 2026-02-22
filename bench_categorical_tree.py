"""Benchmark categorical split performance on the Amazon Employee dataset.

Usage:
    # Native categorical handling:
    python bench_categorical_tree.py --mode categorical --label cat --output results_cat.csv

    # Ordinal (treat integers as numeric):
    python bench_categorical_tree.py --mode ordinal --label ord --output results_ord.csv

    # One-hot encoding:
    python bench_categorical_tree.py --mode onehot --label ohe --output results_ohe.csv

    # Plot all three:
    python plot_categorical_bench.py results_cat.csv results_ord.csv results_ohe.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

MODELS = {
    "tree": DecisionTreeClassifier,
    "forest": RandomForestClassifier,
}

MAX_CAT_VALUES = 64  # max distinct category values per feature


def encode_rare_categories(X_cat, cat_features, max_categories=MAX_CAT_VALUES):
    """Remap each categorical feature to have at most max_categories values.

    Keeps the (max_categories - 1) most frequent categories remapped to
    0..max_categories-2, and maps all remaining rare categories to
    max_categories - 1.
    """
    X_out = X_cat.copy()
    for col in cat_features:
        vals = X_out[:, col].astype(int)
        unique, counts = np.unique(vals, return_counts=True)
        order = np.argsort(-counts)
        top = unique[order[: max_categories - 1]]
        mapping = {v: i for i, v in enumerate(top)}
        rare_code = max_categories - 1
        X_out[:, col] = np.array(
            [mapping.get(v, rare_code) for v in vals], dtype=np.float64
        )
    return X_out


def run_benchmark(
    model_name="forest",
    mode="categorical",
    k_folds=3,
    n_repeats=5,
    label="unknown",
    output="results.csv",
):
    ModelClass = MODELS[model_name]

    # Amazon Employee Access dataset (OpenML ID 4135)
    data = fetch_openml(data_id=4135, as_frame=True, parser="auto")
    X = data.data
    y = (data.target == "1").astype(int).values

    # Identify categorical columns (all features in this dataset are categorical)
    cat_features = list(range(X.shape[1]))

    # Convert to integer codes and cap rare categories
    X_cat = X.values.astype(np.float64)
    X_cat = encode_rare_categories(X_cat, cat_features)

    # Apply one-hot encoding if requested
    if mode == "onehot":
        ohe = OneHotEncoder(sparse_output=False, categories="auto")
        X_cat = ohe.fit_transform(X_cat)
        use_categorical = False
    elif mode == "categorical":
        use_categorical = True
    else:  # ordinal
        use_categorical = False

    results = []
    rkf = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=42)

    # Base kwargs shared by both models
    base_kwargs = {"random_state": 42, "max_features": None}
    if model_name == "forest":
        base_kwargs["n_estimators"] = 100
        base_kwargs["n_jobs"] = -1

    for repeat_fold, (train_idx, test_idx) in enumerate(rkf.split(X_cat)):
        repeat = repeat_fold // k_folds
        fold = repeat_fold % k_folds

        X_train, X_test = X_cat[train_idx], X_cat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_categorical:
            clf = ModelClass(categorical_features=cat_features, **base_kwargs)
        else:
            clf = ModelClass(**base_kwargs)

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = clf.predict(X_test)
        predict_time = time.perf_counter() - t0

        y_proba = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Collect tree stats
        if model_name == "tree":
            n_nodes = clf.tree_.node_count
            depth = clf.get_depth()
        else:
            n_nodes = np.mean([t.tree_.node_count for t in clf.estimators_])
            depth = np.mean([t.get_depth() for t in clf.estimators_])

        results.append(
            {
                "label": label,
                "model": model_name,
                "mode": mode,
                "repeat": repeat,
                "fold": fold,
                "fit_time": fit_time,
                "predict_time": predict_time,
                "accuracy": acc,
                "roc_auc": auc,
                "n_nodes": n_nodes,
                "depth": depth,
            }
        )

        print(
            f"  repeat={repeat} fold={fold}: "
            f"acc={acc:.4f} auc={auc:.4f} "
            f"fit={fit_time:.3f}s nodes={n_nodes:.0f} depth={depth:.1f}"
        )

    df = pd.DataFrame(results)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} rows to {output}")
    print(
        df.groupby("label")[
            ["accuracy", "roc_auc", "fit_time", "n_nodes", "depth"]
        ].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="forest", choices=["tree", "forest"]
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="categorical",
        choices=["categorical", "ordinal", "onehot"],
    )
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--label", type=str, default="unknown")
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()
    run_benchmark(
        args.model, args.mode, args.k_folds, args.n_repeats, args.label, args.output
    )