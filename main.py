#!/usr/bin/env python
"""
Main entry point and execution script for:
Multi-Layer Phishing Detection & Risk Assessment.

Contributors:
- Siddhi Shetkar & Sankeerthana Thoviti

Modes:
- URL Prediction:        python main.py --url "https://example.com"
- Interactive Shell:     python main.py --interactive
- Train & Compare ML:    python main.py --train
- Data Pipeline / ETL:   python main.py --prepare-data
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data Engineering Modules
from src.data_loader import inspect_dataset, load_dataset
from src.preprocessing import clean_dataset, save_cleaned_dataset
from src.feature_extraction import FEATURE_COLUMNS, extract_features_from_dataset, extract_url_features
from src.data_preparation import generate_feature_descriptions, prepare_features_and_splits
from src.eda import generate_eda_report, generate_eda_visualizations

# Machine Learning & Risk Assessment Modules
from src.model import load_model, predict_single_url_features, run_training_pipeline
from src.risk_assessment import DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, RiskAssessmentEngine


# ANSI terminal colors for readable CLI output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def parse_arguments():
    """Parse unified command-line arguments for data preparation and ML workflows."""
    parser = argparse.ArgumentParser(
        description="Multi-Layer Phishing Detection & Risk Assessment System"
    )

    # Operational Mode Selection
    mode_group = parser.add_argument_group("Operational Modes")
    mode_group.add_argument(
        "--url", "--predict", "-u",
        type=str,
        default=None,
        help="Input URL to inspect, classify (Legitimate vs Phishing), and assess risk."
    )
    mode_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal prompt to analyze multiple URLs continuously."
    )
    mode_group.add_argument(
        "--train", "-t",
        action="store_true",
        help="Train all 5 ML models, compare metrics, generate plots, and save best model."
    )
    mode_group.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Start the SENTINEL web security dashboard and REST API server."
    )
    mode_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the SENTINEL web server (default: 8000)."
    )
    mode_group.add_argument(
        "--prepare-data", "--run-data-pipeline", "--run-part1",
        dest="prepare_data",
        action="store_true",
        help="Run data cleaning, 28-feature extraction, EDA generation, and train/test splits."
    )

    # Model and Risk Parameters
    ml_group = parser.add_argument_group("ML & Risk Assessment Options")
    ml_group.add_argument(
        "--model-path", "-m",
        type=str,
        default="models/best_model.joblib",
        help="Path to serialized trained model artifact (default: models/best_model.joblib)."
    )
    ml_group.add_argument(
        "--threshold-low",
        type=float,
        default=DEFAULT_LOW_THRESHOLD,
        help=f"Probability cutoff for LOW risk classification (default: {DEFAULT_LOW_THRESHOLD})."
    )
    ml_group.add_argument(
        "--threshold-high",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD,
        help=f"Probability cutoff for HIGH risk classification (default: {DEFAULT_HIGH_THRESHOLD})."
    )

    # Dataset and Directory Arguments
    data_group = parser.add_argument_group("Dataset & Directory Options")
    data_group.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Path to raw CSV dataset file. If omitted, searches default locations."
    )
    data_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default="dataset/processed",
        help="Directory to save processed datasets and splits (default: dataset/processed)."
    )
    data_group.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory to save visualization charts and reports (default: results)."
    )
    data_group.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store trained model artifacts (default: models)."
    )
    data_group.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Optional sample size for quick dry-run training or processing."
    )
    data_group.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="Batch size for progress reporting (default: 10,000)."
    )
    data_group.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to reserve for test split (default: 0.2)."
    )
    data_group.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible execution (default: 42)."
    )

    return parser.parse_args()


def find_dataset_path(provided_path=None) -> Path:
    """Resolve the dataset path from argument or default workspace locations."""
    if provided_path:
        p = Path(provided_path)
        if p.exists() and p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"Specified dataset file not found: '{p.resolve()}'")

    candidate_paths = [
        PROJECT_ROOT / "dataset" / "raw" / "phishing_url_dataset.csv",
        PROJECT_ROOT / "phishing_url_dataset.csv",
    ]

    for cand in candidate_paths:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not find default dataset 'phishing_url_dataset.csv' in root or 'dataset/raw/'. "
        "Please specify a path using --data-path."
    )


def run_data_pipeline(args):
    """Execute Data Engineering Pipeline."""
    start_time = time.time()

    print("=" * 80)
    print("  MULTI-LAYER PHISHING DETECTION & RISK ASSESSMENT - DATA PIPELINE")
    print("=" * 80)

    # Step 1: Load and Inspect Dataset
    print("\n[STEP 1/7] Loading and Inspecting Raw Dataset...")
    data_path = find_dataset_path(args.data_path)
    print(f"Using dataset file: '{data_path}'")

    raw_df = load_dataset(data_path, standardize_col_names=False)
    inspection_summary = inspect_dataset(raw_df, verbose=True)

    if args.sample_size is not None and args.sample_size < len(raw_df):
        print(f"\n[Note] Sampling {args.sample_size:,} records from full dataset as requested...")
        raw_df = raw_df.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)

    # Step 2: Preprocess and Clean Data
    print("\n[STEP 2/7] Preprocessing and Cleaning Data...")
    cleaned_df, cleaning_stats = clean_dataset(raw_df, verbose=True)

    out_dir = Path(args.output_dir)
    cleaned_csv_path = out_dir / "cleaned_dataset.csv"
    save_cleaned_dataset(cleaned_df, cleaned_csv_path)

    # Step 3: Extract URL Features
    print("\n[STEP 3/7] Extracting URL Features...")
    t_feat_start = time.time()
    X_features = extract_features_from_dataset(
        cleaned_df,
        url_col='url',
        batch_size=args.batch_size,
        verbose=True
    )
    t_feat_elapsed = time.time() - t_feat_start
    print(f"Feature extraction completed in {t_feat_elapsed:.2f} seconds ({len(X_features):,} URLs, {len(FEATURE_COLUMNS)} features).")

    # Assemble full features dataframe with 'url' and 'label'
    full_df = X_features.copy()
    full_df.insert(0, 'url', cleaned_df['url'].values)
    full_df['label'] = cleaned_df['label'].values.astype(int)

    # Step 4: Generate EDA Visualizations and Report
    print("\n[STEP 4/7] Generating EDA Visualizations & Report...")
    vis_dir = Path(args.results_dir) / "visualizations"
    reports_dir = Path(args.results_dir) / "reports"

    generated_plots = generate_eda_visualizations(
        df_features=full_df,
        cleaning_stats=cleaning_stats,
        output_dir=vis_dir,
        verbose=True
    )

    report_path = reports_dir / "eda_report.md"
    generate_eda_report(
        df_raw_meta=inspection_summary,
        df_cleaned=cleaned_df,
        df_features=full_df,
        cleaning_stats=cleaning_stats,
        output_path=report_path,
        verbose=True
    )

    # Step 5 & 6: Prepare Feature Dataset and Create Splits
    print("\n[STEP 5 & 6/7] Preparing Feature Dataset & Train/Test Splits...")
    split_summary = prepare_features_and_splits(
        cleaned_df=cleaned_df,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        batch_size=args.batch_size,
        precomputed_features=X_features,
        verbose=True
    )

    # Step 7: Final Summary and Record Counts
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  DATA ENGINEERING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total Execution Time: {total_elapsed:.2f} seconds")
    print("\nGenerated Datasets & Files:")
    print(f"  - Cleaned Dataset:        {cleaned_csv_path.resolve()} ({len(cleaned_df):,} records)")
    print(f"  - Full Feature Dataset:   {Path(split_summary['full_csv_path']).resolve()} ({split_summary['total_records']:,} records)")
    print(f"  - Feature Descriptions:   {Path(split_summary['feature_descriptions_path']).resolve()}")
    print(f"  - Training Features (X):  {Path(split_summary['x_train_path']).resolve()} ({split_summary['train_records']:,} records)")
    print(f"  - Training Labels (y):    {Path(split_summary['y_train_path']).resolve()} ({split_summary['train_records']:,} records)")
    print(f"  - Testing Features (X):   {Path(split_summary['x_test_path']).resolve()} ({split_summary['test_records']:,} records)")
    print(f"  - Testing Labels (y):     {Path(split_summary['y_test_path']).resolve()} ({split_summary['test_records']:,} records)")
    print(f"  - EDA Report:             {report_path.resolve()}")
    print(f"  - EDA Visualizations:     {vis_dir.resolve()} ({len(generated_plots)} charts)")
    print("=" * 80)


def predict_and_assess_url(
    url: str,
    model: Any,
    risk_engine: RiskAssessmentEngine,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute full inference and risk assessment pipeline for a single URL string.

    Guarantees:
    - Never executes or makes HTTP network requests to the target URL.
    - Uses exact same 28 features in identical order expected by the model.
    - Produces structured diagnostic evaluation.

    Args:
        url: Raw URL string.
        model: Loaded model estimator or Pipeline.
        risk_engine: RiskAssessmentEngine instance.
        verbose: Whether to print formatted CLI card.

    Returns:
        Structured result dictionary.
    """
    clean_url = str(url).strip() if url is not None else ""
    if not clean_url:
        raise ValueError("Cannot predict on an empty or blank URL.")

    # 1. Feature Extraction (strictly local offline parsing)
    features = extract_url_features(clean_url)

    # 2. Machine Learning Model Inference
    model_pred = predict_single_url_features(model, features, FEATURE_COLUMNS)

    # 3. Multi-layer Risk Assessment & Explainability
    assessment = risk_engine.assess_risk(clean_url, features, model_pred)

    if verbose:
        display_prediction_report(assessment, features)

    return assessment


def display_prediction_report(res: Dict[str, Any], features: Dict[str, Any]):
    """Print an aesthetic, structured terminal report for the URL assessment."""
    risk = res["risk_level"]
    cls_name = res["classification"]
    prob = res["phishing_probability"]
    score = res["risk_score"]

    if risk == "HIGH":
        risk_color = RED
        risk_badge = f"{RED}[HIGH RISK - MALICIOUS THREAT]{RESET}"
    elif risk == "MEDIUM":
        risk_color = YELLOW
        risk_badge = f"{YELLOW}[MEDIUM RISK - SUSPICIOUS]{RESET}"
    else:
        risk_color = GREEN
        risk_badge = f"{GREEN}[LOW RISK - LIKELY LEGITIMATE]{RESET}"

    cls_color = RED if cls_name == "PHISHING" else GREEN

    # Visual probability bar
    bar_len = 24
    filled = int(round(prob * bar_len))
    bar_str = ("#" * filled) + ("-" * (bar_len - filled))

    print("\n" + "=" * 78)
    print(f" {BOLD}MULTI-LAYER PHISHING DETECTION & RISK ASSESSMENT REPORT{RESET}")
    print("=" * 78)
    print(f" {BOLD}Analyzed URL:{RESET}     {CYAN}{res['url']}{RESET}")
    print(f" {BOLD}Classification:{RESET}   {cls_color}{BOLD}{cls_name}{RESET} (Confidence: {res['confidence']*100:.1f}%)")
    print(f" {BOLD}Risk Level:{RESET}       {risk_badge}")
    print(f" {BOLD}Risk Score:{RESET}       {risk_color}{score:.1f} / 100.0{RESET}")
    print(f" {BOLD}Phishing Prob:{RESET}    [{risk_color}{bar_str}{RESET}] {prob*100:.2f}%")
    print("-" * 78)

    print(f" {BOLD}Detection Reasoning:{RESET}")
    print(f"   {GRAY}*{RESET} {res['reason_summary']}")

    if res["critical_flags"]:
        print(f"\n {RED}{BOLD}Critical Security Flags:{RESET}")
        for flag in res["critical_flags"]:
            print(f"   {RED}!{RESET} {flag}")

    if res["warnings"]:
        print(f"\n {YELLOW}{BOLD}Observed Suspicious Indicators ({len(res['warnings'])}):{RESET}")
        for w in res["warnings"]:
            print(f"   {YELLOW}-{RESET} {w}")

    if res["safe_signals"]:
        print(f"\n {GREEN}{BOLD}Positive Baseline Indicators ({len(res['safe_signals'])}):{RESET}")
        for s in res["safe_signals"]:
            print(f"   {GREEN}+{RESET} {s}")

    print("-" * 78)
    print(f" {GRAY}{BOLD}Key Extracted Features:{RESET}")
    print(
        f"   Length: {features.get('url_length')} | Host Length: {features.get('hostname_length')} | "
        f"Dots: {features.get('num_dots')} | Subdomains: {features.get('num_subdomains')} | "
        f"Digits: {features.get('num_digits')} | Special Chars: {features.get('num_special_characters')}"
    )
    print("-" * 78)
    print(f" {GRAY}{res['disclaimer']}{RESET}")
    print("=" * 78 + "\n")


def run_interactive_mode(model: Any, risk_engine: RiskAssessmentEngine):
    """Launch interactive shell for rapid ad-hoc URL evaluation."""
    print("\n" + "=" * 78)
    print("  INTERACTIVE URL PHISHING DETECTOR & RISK ASSESSMENT CONSOLE")
    print("  Type any URL to inspect, or type 'exit' or 'quit' to exit.")
    print("=" * 78)

    sample_urls = [
        "https://www.google.com",
        "http://login.secure-bank.update.com.attacker.net/verify/account?id=99",
        "http://192.168.1.100:8080/admin/login",
        "https://bit.ly/3xYz123",
        "https://github.com/explore"
    ]
    print("\nQuick sample URLs to test:")
    for idx, s in enumerate(sample_urls, 1):
        print(f"  {idx}. {s}")
    print("-" * 78)

    while True:
        try:
            user_input = input(f"\n{CYAN}Enter URL > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive console. Goodbye!")
                break

            predict_and_assess_url(user_input, model, risk_engine, verbose=True)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive console.")
            break
        except Exception as e:
            print(f"{RED}[Error]{RESET} Could not analyze URL: {e}")


def main():
    """Main program entry point orchestrating data engineering & ML prediction."""
    args = parse_arguments()

    # Mode 0: Start SENTINEL Web & API Server
    if getattr(args, "serve", False):
        from server import run_server
        run_server(port=getattr(args, "port", 8000))
        return

    # Mode 1: Run Data Pipeline
    if args.prepare_data:
        run_data_pipeline(args)
        return

    # Mode 2: Run Model Training & Evaluation Pipeline
    if args.train:
        splits_dir = Path(args.output_dir) / "splits"
        if not (splits_dir / "X_train.csv").exists():
            print(f"[Notice] Splits not found in '{splits_dir}'. Preparing dataset first...")
            run_data_pipeline(args)

        results = run_training_pipeline(
            splits_dir=splits_dir,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
            sample_size=args.sample_size,
            random_state=args.random_state,
            verbose=True
        )
        return

    # Mode 3: Single URL prediction or Interactive Mode
    model_path = Path(args.model_path)
    risk_engine = RiskAssessmentEngine(
        low_threshold=args.threshold_low,
        high_threshold=args.threshold_high
    )

    # Auto-train if model doesn't exist yet and user requested inference
    if not model_path.exists():
        print(f"[Notice] Trained model not found at '{model_path.resolve()}'.")
        splits_dir = Path(args.output_dir) / "splits"
        if not (splits_dir / "X_train.csv").exists():
            print(f"[Notice] Splits missing. Executing data pipeline first...")
            run_data_pipeline(args)

        print("[Notice] Executing model training pipeline to generate artifact...")
        run_training_pipeline(
            splits_dir=splits_dir,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
            sample_size=args.sample_size,
            random_state=args.random_state,
            verbose=True
        )

    # Load model
    model, metadata = load_model(model_path)

    if args.url:
        predict_and_assess_url(args.url, model, risk_engine, verbose=True)
    elif args.interactive:
        run_interactive_mode(model, risk_engine)
    else:
        # Default behavior: If no argument specified, present a help prompt or run interactive demo
        print("\n" + "=" * 78)
        print("  MULTI-LAYER PHISHING DETECTION & RISK ASSESSMENT")
        print("=" * 78)
        print("Usage Examples:")
        print("  * Web Dashboard:      python main.py --serve")
        print("  * Check a URL:        python main.py --url 'https://example.com'")
        print("  * Interactive prompt: python main.py --interactive")
        print("  * Train ML models:    python main.py --train")
        print("  * Prepare dataset:    python main.py --prepare-data")
        print("-" * 78)
        print("Launching Interactive URL Prediction console by default...")
        run_interactive_mode(model, risk_engine)


if __name__ == "__main__":
    main()
