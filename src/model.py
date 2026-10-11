"""
Module for Machine Learning model training, evaluation, comparison, serialization, and inference.

Multi-Layer Phishing Detection & Risk Assessment.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction import FEATURE_COLUMNS


def get_model_candidates(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Instantiate and return the suite of classification algorithms for comparison.

    All models are configured with fixed random_state for reproducibility and
    balanced class weighting where applicable.

    Algorithms:
    1. Logistic Regression (with StandardScaler in Scikit-Learn Pipeline)
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. Support Vector Machine (Linear SVM with probability calibration via CalibratedClassifierCV)
    5. Gradient Boosting Classifier (HistGradientBoostingClassifier for fast, scalable boosting)

    Args:
        random_state: Random seed for reproducible training.

    Returns:
        Dictionary mapping model names to un-fitted estimator / pipeline objects.
    """
    models: Dict[str, BaseEstimator] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs"
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(
                    dual=False,
                    C=1.0,
                    max_iter=2000,
                    random_state=random_state,
                    class_weight="balanced"
                ),
                method="sigmoid",
                cv=3
            )),
        ]),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_depth=12,
            min_samples_leaf=20,
            random_state=random_state,
            class_weight="balanced"
        ),
    }
    return models


def train_model(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = True
) -> Tuple[BaseEstimator, float]:
    """
    Fit a model estimator on training data and measure execution time.

    Args:
        name: Name identifier for the model.
        model: Sklearn estimator or Pipeline instance.
        X_train: Training feature matrix.
        y_train: Training labels.
        verbose: Whether to print progress.

    Returns:
        Tuple of (fitted_model, elapsed_training_time_seconds).
    """
    if verbose:
        print(f"  Training '{name}' on {len(X_train):,} samples...")

    start_time = time.time()
    # Flatten y_train to 1D array if needed
    y_vec = y_train.values.ravel() if hasattr(y_train, "values") else np.ravel(y_train)
    model.fit(X_train, y_vec)
    elapsed = time.time() - start_time

    if verbose:
        print(f"    Completed '{name}' in {elapsed:.2f}s.")
    return model, elapsed


def evaluate_model(
    name: str,
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_time_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on held-out test data across standard cybersecurity metrics.

    Pos-class is 1 (Phishing), Neg-class is 0 (Legitimate).

    Metrics calculated:
    - Accuracy
    - Precision
    - Recall (critical for phishing detection)
    - F1-Score
    - ROC-AUC
    - Confusion Matrix (TN, FP, FN, TP, FPR, FNR)

    Args:
        name: Model identifier name.
        model: Fitted estimator or Pipeline.
        X_test: Test features.
        y_test: Ground truth labels.

    Returns:
        Dictionary containing all metrics and predictions.
    """
    y_true = y_test.values.ravel() if hasattr(y_test, "values") else np.ravel(y_test)
    y_pred = model.predict(X_test)

    # Estimate positive class probability
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
    elif hasattr(model, "decision_function"):
        try:
            df_vals = model.decision_function(X_test)
            # Min-max scale or sigmoid convert for probability approximation
            y_prob = 1.0 / (1.0 + np.exp(-df_vals))
        except Exception:
            y_prob = None

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    eval_result = {
        "model_name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    if train_time_sec is not None:
        eval_result["training_time_seconds"] = round(train_time_sec, 2)
    return eval_result


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Union[str, Path]
) -> Path:
    """
    Generate and save a publication-quality confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        model_name: Model name for plot title.
        output_path: Destination image path.

    Returns:
        Path of the saved plot.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    total_samples = cm_arr.sum()
    percentages = cm_arr / total_samples * 100

    labels = [
        [f"True Neg (Legit)\n{cm_arr[0, 0]:,}\n({percentages[0, 0]:.1f}%)",
         f"False Pos (Alarm)\n{cm_arr[0, 1]:,}\n({percentages[0, 1]:.1f}%)"],
        [f"False Neg (Miss)\n{cm_arr[1, 0]:,}\n({percentages[1, 0]:.1f}%)",
         f"True Pos (Phish)\n{cm_arr[1, 1]:,}\n({percentages[1, 1]:.1f}%)"]
    ]

    plt.figure(figsize=(7, 5.5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Legitimate (0)", "Phishing (1)"],
        yticklabels=["Legitimate (0)", "Phishing (1)"],
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("Actual Label (Ground Truth)", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()

    return out_p


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Union[str, Path]
) -> Path:
    """
    Create a multi-metric comparative bar chart for all evaluated models.

    Args:
        comparison_df: DataFrame containing model metrics.
        output_path: Destination path for image.

    Returns:
        Path of the saved chart.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall (Phishing)", "F1-Score", "ROC-AUC"]

    # Melt dataframe for seaborn grouping
    plot_df = comparison_df[["model_name"] + metrics_to_plot].copy()
    melted = plot_df.melt(
        id_vars=["model_name"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )
    metric_map = dict(zip(metrics_to_plot, metric_labels))
    melted["Metric"] = melted["Metric"].map(metric_map)

    plt.figure(figsize=(12, 6.5), dpi=300)
    ax = sns.barplot(
        data=melted,
        x="model_name",
        y="Score",
        hue="Metric",
        palette="viridis"
    )

    plt.title("Phishing Detection Model Performance Comparison", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Algorithm", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylim(0.80, 1.02)  # Focus on top discriminatory range
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    # Annotate bars with numeric values
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2., h),
                ha="center", va="bottom",
                fontsize=7.5,
                rotation=45,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()

    return out_p


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Union[str, Path]
) -> Path:
    """
    Plot and overlay the ROC curves for all models on a single figure.

    Args:
        roc_data: Dict mapping model_name -> (fpr_array, tpr_array, auc_score).
        output_path: Destination path for image.

    Returns:
        Path of the saved chart.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 7), dpi=300)

    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    for idx, (name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        label_text = f"{name} (AUC = {auc_score:.4f})" if not np.isnan(auc_score) else f"{name} (AUC = N/A)"
        plt.plot(fpr, tpr, label=label_text, color=color, linewidth=2.2)

    # Reference diagonal
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Phishing Recall)", fontsize=11, labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()

    return out_p


def save_model_artifacts(
    model: BaseEstimator,
    model_name: str,
    feature_names: List[str],
    metrics: Dict[str, Any],
    output_dir: Union[str, Path] = "models",
    filename_prefix: str = "best_model"
) -> Tuple[Path, Path]:
    """
    Serialize the trained model pipeline and write comprehensive metadata.

    Args:
        model: Fitted estimator or Pipeline.
        model_name: Name identifier for the best model.
        feature_names: Exact list and order of feature columns.
        metrics: Evaluation metrics dictionary.
        output_dir: Directory where models are stored.
        filename_prefix: Prefix for saved artifact files.

    Returns:
        Tuple of (model_file_path, metadata_file_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save serialized model
    model_path = out_dir / f"{filename_prefix}.joblib"
    joblib.dump(model, model_path, compress=3)

    # 2. Save metadata JSON
    meta_path = out_dir / f"{filename_prefix}_metadata.json"
    cleaned_metrics = {
        k: v for k, v in metrics.items()
        if k not in ("y_pred", "y_prob")
    }

    metadata = {
        "model_name": model_name,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_columns": feature_names,
        "num_features": len(feature_names),
        "target_classes": {"0": "Legitimate", "1": "Phishing"},
        "positive_class": 1,
        "evaluation_metrics": cleaned_metrics,
        "model_architecture": str(type(model)),
        "risk_thresholds": {
            "low_max": 0.35,
            "medium_max": 0.70,
            "high_min": 0.70
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, meta_path


def load_model(
    model_path: Union[str, Path] = "models/best_model.joblib",
    metadata_path: Optional[Union[str, Path]] = None
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Load a trained model artifact and its metadata.

    Args:
        model_path: Path to serialized .joblib file.
        metadata_path: Optional path to metadata JSON (defaults to matching name).

    Returns:
        Tuple of (loaded_model, metadata_dict).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If model file is corrupted or features mismatch.
    """
    m_path = Path(model_path)
    if not m_path.exists() or not m_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at: '{m_path.resolve()}'. "
            "Please train the model first by running 'python main.py --train'."
        )

    try:
        model = joblib.load(m_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from '{m_path.resolve()}': {e}")

    # Resolve metadata path
    if metadata_path is None:
        cand_meta = m_path.parent / f"{m_path.stem}_metadata.json"
        if not cand_meta.exists():
            cand_meta = m_path.parent / "best_model_metadata.json"
        meta_path = cand_meta
    else:
        meta_path = Path(metadata_path)

    metadata: Dict[str, Any] = {}
    if meta_path.exists() and meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return model, metadata


def predict_single_url_features(
    model: BaseEstimator,
    features_dict: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run inference on a single URL's feature dictionary using the trained model.

    Guarantees:
    - Features are aligned strictly in feature_names order.
    - Handles estimators with and without predict_proba gracefully.

    Args:
        model: Trained model estimator or Pipeline.
        features_dict: Extracted feature dictionary.
        feature_names: Expected feature column ordering (defaults to FEATURE_COLUMNS).

    Returns:
        Dictionary containing predicted class (0/1), class label string,
        and phishing probability estimate.
    """
    cols = feature_names if feature_names is not None else FEATURE_COLUMNS

    # Build 1-row DataFrame in exact expected column order
    row_data = {col: [features_dict.get(col, 0)] for col in cols}
    df_row = pd.DataFrame(row_data, columns=cols)

    pred_class = int(model.predict(df_row)[0])

    phishing_prob = None
    if hasattr(model, "predict_proba"):
        try:
            proba_arr = model.predict_proba(df_row)[0]
            # Assumes class 1 is the second index in classes_
            if hasattr(model, "classes_"):
                classes_list = list(model.classes_)
                if 1 in classes_list:
                    idx_1 = classes_list.index(1)
                    phishing_prob = float(proba_arr[idx_1])
                else:
                    phishing_prob = float(proba_arr[-1])
            else:
                phishing_prob = float(proba_arr[1])
        except Exception:
            phishing_prob = 1.0 if pred_class == 1 else 0.0
    elif hasattr(model, "decision_function"):
        try:
            decision = float(model.decision_function(df_row)[0])
            phishing_prob = float(1.0 / (1.0 + np.exp(-decision)))
        except Exception:
            phishing_prob = 1.0 if pred_class == 1 else 0.0
    else:
        phishing_prob = 1.0 if pred_class == 1 else 0.0

    class_name = "PHISHING" if pred_class == 1 else "LEGITIMATE"

    return {
        "predicted_class": pred_class,
        "classification": class_name,
        "phishing_probability": phishing_prob,
        "confidence": phishing_prob if pred_class == 1 else (1.0 - phishing_prob)
    }


def run_training_pipeline(
    splits_dir: Union[str, Path] = "dataset/processed/splits",
    results_dir: Union[str, Path] = "results",
    models_dir: Union[str, Path] = "models",
    sample_size: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute the complete Model Training, Evaluation, and Comparison Pipeline.

    Workflow:
    1. Load held-out train and test splits (X_train, y_train, X_test, y_test).
    2. Instantiate all 5 classification algorithms.
    3. Train each model on training data.
    4. Evaluate each model on test set.
    5. Rank models according to cybersecurity priority (F1 and Phishing Recall).
    6. Generate comparison tables, individual confusion matrices, and ROC curves.
    7. Save best-performing model artifact and full evaluation report.

    Args:
        splits_dir: Directory containing split CSVs.
        results_dir: Directory to save evaluation charts and reports.
        models_dir: Directory to save best model artifact.
        sample_size: Optional row sample for fast dry-run testing.
        random_state: Random state for reproducibility.
        verbose: Verbosity logging flag.

    Returns:
        Summary dictionary of results, metrics, and file paths.
    """
    start_total_time = time.time()
    s_dir = Path(splits_dir)
    res_dir = Path(results_dir)
    m_dir = Path(models_dir)

    vis_dir = res_dir / "visualizations"
    reports_dir = res_dir / "reports"
    vis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("  MULTI-LAYER PHISHING DETECTION — MODEL TRAINING & EVALUATION PIPELINE")
        print("=" * 80)

    # 1. Load splits
    x_train_path = s_dir / "X_train.csv"
    y_train_path = s_dir / "y_train.csv"
    x_test_path = s_dir / "X_test.csv"
    y_test_path = s_dir / "y_test.csv"

    for p in (x_train_path, y_train_path, x_test_path, y_test_path):
        if not p.exists():
            raise FileNotFoundError(f"Required split file missing: '{p.resolve()}'. Prepare dataset splits first.")

    if verbose:
        print("\n[Step 1/5] Loading preprocessed train and test splits...")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)["label"]
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    if sample_size is not None and sample_size < len(X_train):
        if verbose:
            print(f"  Sampling {sample_size:,} rows for rapid dry-run...")
        X_train = X_train.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        y_train = y_train.iloc[X_train.index].reset_index(drop=True)
        test_sample = min(len(X_test), max(1000, sample_size // 4))
        X_test = X_test.sample(n=test_sample, random_state=random_state).reset_index(drop=True)
        y_test = y_test.iloc[X_test.index].reset_index(drop=True)

    if verbose:
        print(f"  Train samples: {len(X_train):,} ({y_train.value_counts().to_dict()})")
        print(f"  Test samples:  {len(X_test):,} ({y_test.value_counts().to_dict()})")
        print(f"  Feature count: {X_train.shape[1]} features")

    # 2. Instantiate models
    candidates = get_model_candidates(random_state=random_state)
    trained_models: Dict[str, BaseEstimator] = {}
    training_times: Dict[str, float] = {}
    evaluation_results: List[Dict[str, Any]] = []
    roc_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    # 3. Train and Evaluate each candidate
    if verbose:
        print("\n[Step 2/5] Training and evaluating candidate classifiers...")

    for name, candidate_model in candidates.items():
        if verbose:
            print(f"\n--- Model: {name} ---")
        fitted_model, t_train = train_model(name, candidate_model, X_train, y_train, verbose=verbose)
        trained_models[name] = fitted_model
        training_times[name] = t_train

        eval_res = evaluate_model(name, fitted_model, X_test, y_test)
        eval_res["training_time_seconds"] = round(t_train, 2)
        evaluation_results.append(eval_res)

        if verbose:
            print(f"    Accuracy:  {eval_res['accuracy'] * 100:.2f}%")
            print(f"    Precision: {eval_res['precision'] * 100:.2f}%")
            print(f"    Recall:    {eval_res['recall'] * 100:.2f}% (Phishing detection rate)")
            print(f"    F1-Score:  {eval_res['f1_score'] * 100:.2f}%")
            print(f"    ROC-AUC:   {eval_res['roc_auc']:.4f}")
            print(f"    False Pos: {eval_res['false_positives']:,} | False Neg: {eval_res['false_negatives']:,}")

        # Compute ROC curve points if probabilities exist
        if eval_res["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, eval_res["y_prob"], pos_label=1)
            roc_curves_data[name] = (fpr, tpr, eval_res["roc_auc"])

    # 4. Generate comparison table & select best model
    if verbose:
        print("\n[Step 3/5] Ranking models and selecting optimal classifier...")

    records = []
    for r in evaluation_results:
        records.append({
            "model_name": r["model_name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "roc_auc": r["roc_auc"],
            "false_positive_rate": r["false_positive_rate"],
            "false_negative_rate": r["false_negative_rate"],
            "training_time_seconds": r["training_time_seconds"],
            "false_positives": r["false_positives"],
            "false_negatives": r["false_negatives"],
            "true_positives": r["true_positives"],
            "true_negatives": r["true_negatives"],
        })

    comp_df = pd.DataFrame(records)
    # Primary ranking: Cybersecurity priority uses F1-score + Phishing Recall
    comp_df["cybersecurity_score"] = (comp_df["f1_score"] * 0.6) + (comp_df["recall"] * 0.4)
    comp_df = comp_df.sort_values(by="cybersecurity_score", ascending=False).reset_index(drop=True)

    best_record = comp_df.iloc[0].to_dict()
    best_model_name = best_record["model_name"]
    best_model = trained_models[best_model_name]

    if verbose:
        print(f"\nOptimal Model Selected: '{best_model_name}'")
        print(f"  F1-Score: {best_record['f1_score'] * 100:.2f}% | Recall: {best_record['recall'] * 100:.2f}% | Accuracy: {best_record['accuracy'] * 100:.2f}%")

    # 5. Visualizations & Reports
    if verbose:
        print("\n[Step 4/5] Generating evaluation charts and comparison reports...")

    # A. Confusion matrix for each model
    generated_cm_plots = []
    for r in evaluation_results:
        clean_name = r["model_name"].lower().replace(" ", "_")
        cm_path = vis_dir / f"confusion_matrix_{clean_name}.png"
        plot_confusion_matrix(r["confusion_matrix"], r["model_name"], cm_path)
        generated_cm_plots.append(str(cm_path))

    # B. Model comparison bar chart
    comp_chart_path = vis_dir / "model_comparison.png"
    plot_model_comparison(comp_df, comp_chart_path)

    # C. ROC Curves overlay
    roc_chart_path = vis_dir / "roc_curves.png"
    if roc_curves_data:
        plot_roc_curves(roc_curves_data, roc_chart_path)

    # D. Save CSV comparison
    comp_csv_path = reports_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    # E. Save JSON evaluation report
    report_json_path = reports_dir / "model_evaluation_report.json"
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_split_dir": str(s_dir),
        "test_sample_count": len(X_test),
        "train_sample_count": len(X_train),
        "feature_count": len(FEATURE_COLUMNS),
        "best_model": best_model_name,
        "best_model_metrics": best_record,
        "all_models_ranked": comp_df.to_dict(orient="records"),
        "visualizations": {
            "model_comparison": str(comp_chart_path),
            "roc_curves": str(roc_chart_path),
            "confusion_matrices": generated_cm_plots,
        }
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # 6. Save best model artifact
    if verbose:
        print("\n[Step 5/5] Saving best model artifact and inference metadata...")

    best_eval_dict = next(r for r in evaluation_results if r["model_name"] == best_model_name)
    best_model_file, best_meta_file = save_model_artifacts(
        model=best_model,
        model_name=best_model_name,
        feature_names=FEATURE_COLUMNS,
        metrics=best_eval_dict,
        output_dir=m_dir,
        filename_prefix="best_model"
    )

    # Also save with alternative standard name for convenience
    joblib.dump(best_model, m_dir / "phishing_model.joblib", compress=3)

    total_elapsed = time.time() - start_total_time

    if verbose:
        print("\n" + "=" * 80)
        print("  MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total Pipeline Time: {total_elapsed:.2f} seconds")
        print("\nGenerated Evaluation Outputs:")
        print(f"  - Best Model Saved:        {best_model_file.resolve()}")
        print(f"  - Model Metadata:          {best_meta_file.resolve()}")
        print(f"  - Model Comparison CSV:    {comp_csv_path.resolve()}")
        print(f"  - Evaluation Report JSON:  {report_json_path.resolve()}")
        print(f"  - Comparison Bar Chart:    {comp_chart_path.resolve()}")
        print(f"  - ROC Curves Chart:        {roc_chart_path.resolve()}")
        print(f"  - Confusion Matrices:      {len(generated_cm_plots)} plots in '{vis_dir.resolve()}'")
        print("-" * 80)
        print("Model Comparison Summary Table:")
        print(comp_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "training_time_seconds"]].to_string(index=False))
        print("=" * 80)

    return {
        "best_model_name": best_model_name,
        "best_model_path": str(best_model_file),
        "metadata_path": str(best_meta_file),
        "comparison_df": comp_df,
        "comparison_csv_path": str(comp_csv_path),
        "report_json_path": str(report_json_path),
        "comparison_chart_path": str(comp_chart_path),
        "roc_chart_path": str(roc_chart_path),
        "confusion_matrix_plots": generated_cm_plots,
        "total_elapsed_seconds": total_elapsed,
    }
