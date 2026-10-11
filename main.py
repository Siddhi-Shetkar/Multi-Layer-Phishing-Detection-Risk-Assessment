#!/usr/bin/env python
"""
Main pipeline execution script for Part 1:
Multi-Layer Phishing Detection & Risk Assessment.

Pipeline steps:
1. Load and inspect the CSV.
2. Preprocess and clean the data.
3. Extract URL features.
4. Generate EDA visualizations and the report.
5. Prepare the feature dataset.
6. Create training and testing splits.
7. Print a summary of the generated files and record counts.
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import inspect_dataset, load_dataset
from src.preprocessing import clean_dataset, save_cleaned_dataset
from src.feature_extraction import FEATURE_COLUMNS, extract_features_from_dataset
from src.data_preparation import generate_feature_descriptions, prepare_features_and_splits
from src.eda import generate_eda_report, generate_eda_visualizations


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Layer Phishing Detection & Risk Assessment - Pipeline Part 1 Runner"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default=None,
        help="Path to the raw CSV dataset file. If not provided, searches default workspace locations."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="dataset/processed",
        help="Directory to save processed datasets and splits (default: dataset/processed)."
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory to save visualization charts and reports (default: results)."
    )
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Optional sample size for quick dry-runs (default: process entire dataset)."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10000,
        help="Batch size for feature extraction progress reporting (default: 10,000)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splits (default: 42)."
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


def run_pipeline(args):
    """Execute the complete Part 1 machine learning data engineering pipeline."""
    start_time = time.time()

    print("=" * 80)
    print("  MULTI-LAYER PHISHING DETECTION & RISK ASSESSMENT - PART 1 PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Load and Inspect Dataset
    # -------------------------------------------------------------
    print("\n[STEP 1/7] Loading and Inspecting Raw Dataset...")
    data_path = find_dataset_path(args.data_path)
    print(f"Using dataset file: '{data_path}'")

    raw_df = load_dataset(data_path, standardize_col_names=False)
    inspection_summary = inspect_dataset(raw_df, verbose=True)

    if args.sample_size is not None and args.sample_size < len(raw_df):
        print(f"\n[Note] Sampling {args.sample_size:,} records from full dataset as requested...")
        raw_df = raw_df.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)

    # -------------------------------------------------------------
    # Step 2: Preprocess and Clean Data
    # -------------------------------------------------------------
    print("\n[STEP 2/7] Preprocessing and Cleaning Data...")
    cleaned_df, cleaning_stats = clean_dataset(raw_df, verbose=True)

    out_dir = Path(args.output_dir)
    cleaned_csv_path = out_dir / "cleaned_dataset.csv"
    save_cleaned_dataset(cleaned_df, cleaned_csv_path)

    # -------------------------------------------------------------
    # Step 3: Extract URL Features
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 4: Generate EDA Visualizations and Report
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 5 & 6: Prepare Feature Dataset and Create Splits
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Step 7: Final Summary and Record Counts
    # -------------------------------------------------------------
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  PART 1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
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
    print("-" * 80)
    print("Dataset Distribution Summary:")
    print(f"  - Training Set: Legitimate = {split_summary['train_legit_count']:,}, Phishing = {split_summary['train_phish_count']:,}")
    print(f"  - Testing Set:  Legitimate = {split_summary['test_legit_count']:,}, Phishing = {split_summary['test_phish_count']:,}")
    print("=" * 80)


def main():
    """Entry point with top-level error handling."""
    args = parse_arguments()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n[PIPELINE FAILURE] An error occurred during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
