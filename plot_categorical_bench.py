"""Plot benchmark results from bench_categorical_tree.py.

Usage:
    python plot_categorical_bench.py \
        results_cat.csv results_ord.csv results_ohe.csv
"""
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_context('paper', font_scale=1.2)


def load_and_concat(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _boxplot_by_label(ax, df, column, ylabel, title):
    """Helper to draw side-by-side boxplots grouped by label."""
    labels = list(df["label"].unique())
    for label in labels:
        grp = df[df["label"] == label]
        idx = labels.index(label)
        ax.boxplot(
            grp[column],
            positions=[idx],
            widths=0.4,
            patch_artist=True,
            showmeans=True,
            boxprops=dict(facecolor=f"C{idx}"),
            medianprops=dict(color="black", linewidth=1.2),
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=5,
            ),
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_comparison(df):
    """Overview boxplots comparing all labels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Performance (ROC AUC)
    _boxplot_by_label(axes[0, 0], df, "roc_auc", "ROC AUC", "Performance")

    # 2. Fit time
    _boxplot_by_label(axes[0, 1], df, "fit_time", "Fit time (s)", "Runtime")

    # 3. Runtime vs Performance scatter
    ax = axes[1, 0]
    for label, grp in df.groupby("label"):
        ax.scatter(grp["fit_time"], grp["roc_auc"], label=label, alpha=0.6)
    ax.set_xlabel("Fit time (s)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Runtime vs Performance")
    ax.legend()

    # 4. Average tree depth
    _boxplot_by_label(axes[1, 1], df, "depth", "Avg Tree Depth", "Tree Depth")

    plt.tight_layout()
    plt.savefig("categorical_benchmark.png", dpi=150)
    print("Saved categorical_benchmark.png")


def plot_paired(df):
    """Paired scatter plots comparing metrics across labels fold-by-fold.

    Each point represents the same (repeat, fold) evaluated under two
    different modes.  Points above the diagonal mean the y-axis label
    scored higher; below means the x-axis label scored higher.
    """
    labels = sorted(df["label"].unique())
    if len(labels) < 2:
        print("Need at least 2 labels for paired comparison")
        return

    pairs = [
        (labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    metrics = [
        ("roc_auc", "ROC AUC"),
        ("fit_time", "Fit Time (s)"),
        ("depth", "Tree Depth"),
    ]

    n_rows = len(metrics)
    n_cols = len(pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Ensure axes is always 2-d
    if n_cols == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    elif n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, (label_a, label_b) in enumerate(pairs):
        df_a = df[df["label"] == label_a].set_index(["repeat", "fold"])
        df_b = df[df["label"] == label_b].set_index(["repeat", "fold"])
        merged = df_a.join(df_b, lsuffix="_a", rsuffix="_b")

        for row, (metric, metric_name) in enumerate(metrics):
            ax = axes[row, col]
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            ax.scatter(merged[col_a], merged[col_b], alpha=0.6, edgecolors="k", lw=0.3)

            # Diagonal reference line
            lo = min(merged[col_a].min(), merged[col_b].min())
            hi = max(merged[col_a].max(), merged[col_b].max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.3)

            ax.set_xlabel(label_a)
            ax.set_ylabel(label_b)
            ax.set_title(f"{metric_name}")
            ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Paired Comparison (each point = same fold)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("categorical_benchmark_paired.png", dpi=150, bbox_inches="tight")
    print("Saved categorical_benchmark_paired.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="+", help="CSV files from bench_categorical_tree.py"
    )
    args = parser.parse_args()
    df = load_and_concat(args.files)
    plot_comparison(df)
    plot_paired(df)
    plt.show()
