import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "thesis-mlflow-demo"
MODEL_NAME = "iris-decision-tree"

PROMOTION_RULE = (
    "Select the highest validation macro-F1; "
    "break ties by lower max_depth. "
    "Assign the champion alias only if validation macro-F1 >= 0.90. "
    "This is a classroom validation rule, not production approval."
)


def git_metadata():
    """Record the source revision and working-tree status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_DIR,
            text=True,
        ).strip()

        return {
            "git_commit": commit,
            "git_tree": "dirty" if status else "clean",
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": "unavailable",
            "git_tree": "uncommitted or Git unavailable",
        }


def train_run(config, dataset, dataset_hash, source_run_id=None):
    """Train and log one reproducible configuration."""
    X = dataset.drop(columns="target")
    y = dataset["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["validation_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        random_state=config["random_state"],
    )

    with mlflow.start_run(
        run_name=f"decision-tree-depth-{config['max_depth']}"
    ) as run:
        mlflow.log_params({
            **config,
            "model_family": "DecisionTreeClassifier",
            "preprocessing": "none; tree models do not require scaling",
            "split_strategy": "stratified",
            "dataset": "sklearn.datasets.load_iris",
            "dataset_sha256": dataset_hash,
            "sklearn_version": sklearn.__version__,
        })

        mlflow.set_tags(git_metadata())
        if source_run_id:
            mlflow.set_tag("reproduced_from_run_id", source_run_id)

        mlflow.log_artifact(str(Path(__file__).resolve()), "source")
        mlflow.log_artifact(str(PROJECT_DIR / "requirements.txt"), "source")
        mlflow.log_text(
            dataset.to_csv(index=False, lineterminator="\n"),
            "data/iris.csv",
        )
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict(
            {
                "source_url": "https://archive.ics.uci.edu/dataset/53/iris",
                "loader": "sklearn.datasets.load_iris",
                "version": sklearn.__version__,
                "sha256": dataset_hash,
                "train_indices": X_train.index.tolist(),
                "validation_indices": X_val.index.tolist(),
            },
            "dataset_metadata.json",
        )

        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "train_f1_macro": f1_score(
                y_train, train_predictions, average="macro"
            ),
            "val_accuracy": accuracy_score(y_val, val_predictions),
            "val_f1_macro": f1_score(
                y_val, val_predictions, average="macro"
            ),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "tree_depth": model.get_depth(),
            "tree_leaves": model.get_n_leaves(),
        }
        mlflow.log_metrics(metrics)

        display = ConfusionMatrixDisplay.from_predictions(
            y_val,
            val_predictions,
            display_labels=load_iris().target_names,
            cmap="Blues",
        )
        display.ax_.set_title("Validation confusion matrix")
        display.figure_.tight_layout()
        mlflow.log_figure(display.figure_, "confusion_matrix.png")
        plt.close(display.figure_)

        predictions = X_val.copy()
        predictions["actual"] = y_val
        predictions["predicted"] = val_predictions
        mlflow.log_text(
            predictions.to_csv(index=True),
            "validation_predictions.csv",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(X_train, train_predictions),
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        result = {
            "run_id": run.info.run_id,
            "max_depth": config["max_depth"],
            **metrics,
            "model_uri": model_info.model_uri,
        }

        print(
            f"Run: {result['run_id']} | "
            f"max_depth={config['max_depth']} | "
            f"Validation F1={metrics['val_f1_macro']:.4f}"
        )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-from-run",
        help="Re-train using parameters from an existing MLflow run.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris(as_frame=True)
    dataset = iris.frame
    dataset_hash = hashlib.sha256(
        dataset.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()

    if args.config_from_run:
        previous = mlflow.get_run(args.config_from_run)
        params = previous.data.params

        if params["dataset_sha256"] != dataset_hash:
            raise ValueError("Dataset hash differs from the original run.")

        if params["sklearn_version"] != sklearn.__version__:
            raise ValueError("Use the original scikit-learn version.")

        config = {
            "max_depth": int(params["max_depth"]),
            "random_state": int(params["random_state"]),
            "validation_size": float(params["validation_size"]),
        }

        result = train_run(
            config, dataset, dataset_hash, args.config_from_run
        )

        for metric in ("val_accuracy", "val_f1_macro"):
            if abs(result[metric] - previous.data.metrics[metric]) > 1e-12:
                raise RuntimeError(f"Reproduction mismatch: {metric}")

        print("Reproduction successful: validation metrics match.")
        return

    results = []
    for depth in (1, 3):
        config = {
            "max_depth": depth,
            "random_state": 42,
            "validation_size": 0.20,
        }
        results.append(train_run(config, dataset, dataset_hash))

    comparison = pd.DataFrame(results).sort_values(
        ["val_f1_macro", "max_depth"],
        ascending=[False, True],
    )

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    comparison.to_csv(reports_dir / "comparison.csv", index=False)

    winner = comparison.iloc[0]
    version = mlflow.register_model(
        model_uri=winner["model_uri"],
        name=MODEL_NAME,
    )

    client = MlflowClient()
    client.update_model_version(
        name=MODEL_NAME,
        version=version.version,
        description=PROMOTION_RULE,
    )

    promoted = bool(winner["val_f1_macro"] >= 0.90)
    if promoted:
        client.set_registered_model_alias(
            MODEL_NAME, "champion", version.version
        )

    summary = {
        "run_id": winner["run_id"],
        "model_name": MODEL_NAME,
        "model_version": str(version.version),
        "alias": "champion" if promoted else None,
        "validation_f1_macro": float(winner["val_f1_macro"]),
        "promotion_rule": PROMOTION_RULE,
        "selection_reason": (
            "Highest validation macro-F1; lower max_depth breaks ties."
        ),
        "reproduce_command": (
            f"python train.py --config-from-run {winner['run_id']}"
        ),
    }

    (reports_dir / "winner.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    client.log_artifact(
        winner["run_id"], str(reports_dir / "comparison.csv"), "selection"
    )
    client.log_artifact(
        winner["run_id"], str(reports_dir / "winner.json"), "selection"
    )

    print("\nComparison:")
    print(comparison[
        ["run_id", "max_depth", "train_f1_macro", "val_f1_macro"]
    ].to_string(index=False))
    print("\nWinner:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()