#!/usr/bin/env python3
"""
ISV Partner Propensity Model - Main Pipeline v2.0

Industry-standard implementation featuring:
- OpenAI Batch API for 50% cost savings
- Optuna Bayesian hyperparameter optimization
- Probability calibration (isotonic/Platt scaling)
- Confidence-based label quality filtering
- Comprehensive logging and monitoring
- Incremental caching with atomic writes

Pipeline Steps:
1. Load and enrich Shopify app data (11,951 apps)
2. Engineer 40+ features from raw data
3. Stratified sampling (2,500 apps)
4. Review sentiment analysis (GPT-4o)
5. Partner propensity scoring (GPT-4o via Batch API)
6. Train ML model with Optuna + calibration
7. Score all apps
8. Output ranked CSV with tiers

Usage:
    python main.py                        # Full pipeline
    python main.py --sample-only          # Create sample only
    python main.py --skip-llm             # Use cached LLM scores
    python main.py --skip-sentiment       # Skip sentiment analysis
    python main.py --use-batch-api        # Use Batch API (recommended)
    python main.py --optuna-trials 100    # More hyperparameter trials
    python main.py --sample-size 500      # Smaller sample for testing

References:
- OpenAI Batch API: https://cookbook.openai.com/examples/batch_processing
- Optuna: https://optuna.org/
- Probability Calibration: https://scikit-learn.org/stable/modules/calibration.html
- LLM Distillation: https://snorkel.ai/blog/llm-distillation-demystified-a-complete-guide/
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import OUTPUT_DIR, CACHE_DIR, SAMPLE_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    sample_only: bool = False,
    skip_llm: bool = False,
    skip_sentiment: bool = False,
    use_batch_api: bool = True,
    sample_size: int = SAMPLE_SIZE,
    optuna_trials: int = 50,
    resume_batch_job: str = None
):
    """
    Run the full ISV partner propensity pipeline.

    Args:
        sample_only: Only create stratified sample
        skip_llm: Skip LLM scoring, use cached results
        skip_sentiment: Skip sentiment analysis
        use_batch_api: Use OpenAI Batch API (50% savings)
        sample_size: Number of apps to sample for LLM
        optuna_trials: Number of Optuna hyperparameter trials
        resume_batch_job: Resume a previous batch job by ID
    """

    start_time = datetime.now()

    print("\n" + "="*70)
    print("  ISV PARTNER PROPENSITY MODEL PIPELINE v2.0")
    print("  CustomGPT.ai RAG API Partner Identification")
    print("="*70)
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sample Size: {sample_size:,}")
    print(f"  Batch API: {'Enabled (50% savings)' if use_batch_api else 'Disabled'}")
    print(f"  Optuna Trials: {optuna_trials}")
    print("="*70 + "\n")

    # Import modules (deferred to allow --help to work without deps)
    from src.data_loader import load_enriched_apps
    from src.feature_engineering import extract_features
    from src.sampler import create_stratified_sample, ensure_high_value_coverage
    from src.sentiment_analyzer import get_sentiment_for_apps
    from src.llm_scorer import (
        score_apps_with_llm, score_apps_with_batch_api,
        merge_llm_scores, load_llm_cache
    )
    from src.model_trainer import train_ensemble, score_all_apps
    from src.output_generator import (
        generate_ranked_csv, generate_report,
        generate_outreach_lists, print_summary
    )

    # =========================================================================
    # PHASE 1: Data Loading and Feature Engineering
    # =========================================================================
    logger.info("PHASE 1: Data Loading and Feature Engineering")
    logger.info("-" * 50)

    df = load_enriched_apps()
    logger.info(f"Loaded {len(df):,} apps with enriched data")

    df = extract_features(df)
    logger.info(f"Engineered features, total columns: {len(df.columns)}")

    # =========================================================================
    # PHASE 2: Stratified Sampling
    # =========================================================================
    logger.info("\nPHASE 2: Stratified Sampling")
    logger.info("-" * 50)

    sample_df = create_stratified_sample(df, sample_size=sample_size)
    sample_df = ensure_high_value_coverage(sample_df, df, min_high_value=min(500, sample_size // 5))
    logger.info(f"Final sample size: {len(sample_df):,} apps")

    if sample_only:
        sample_path = OUTPUT_DIR / "sample_for_llm.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Sample saved to {sample_path}")
        logger.info("Exiting (--sample-only mode)")
        return

    # =========================================================================
    # PHASE 3: Sentiment Analysis (Optional)
    # =========================================================================
    if not skip_sentiment:
        logger.info("\nPHASE 3: Review Sentiment Analysis")
        logger.info("-" * 50)

        sample_ids = sample_df['id'].tolist()
        sentiment_df = get_sentiment_for_apps(sample_ids)
        logger.info(f"Sentiment analyzed for {len(sentiment_df):,} apps")
    else:
        logger.info("\nPHASE 3: Skipping sentiment analysis (--skip-sentiment)")

    # =========================================================================
    # PHASE 4: LLM Scoring
    # =========================================================================
    if not skip_llm:
        logger.info("\nPHASE 4: LLM Scoring with GPT-4o")
        logger.info("-" * 50)

        if use_batch_api:
            logger.info("Using OpenAI Batch API (50% cost savings)")
            logger.info(f"Estimated cost: ~${len(sample_df) * 0.004:.2f}")

            llm_scores_df = score_apps_with_batch_api(
                sample_df,
                use_cache=True,
                resume_job_id=resume_batch_job
            )
        else:
            logger.info("Using synchronous API (consider --use-batch-api for savings)")
            llm_scores_df = score_apps_with_llm(sample_df, use_cache=True)

        logger.info(f"LLM scored {len(llm_scores_df):,} apps")

        # Merge LLM scores with sample
        sample_df = merge_llm_scores(sample_df, llm_scores_df)
    else:
        logger.info("\nPHASE 4: Skipping LLM scoring (--skip-llm)")
        logger.info("Loading cached scores...")

        cache = load_llm_cache()
        if cache:
            llm_scores_df = pd.DataFrame(list(cache.values()))
            sample_df = merge_llm_scores(sample_df, llm_scores_df)
            logger.info(f"Loaded {len(cache):,} cached scores")
        else:
            logger.error("No cached LLM scores found! Run without --skip-llm first.")
            return

    # =========================================================================
    # PHASE 5: ML Model Training
    # =========================================================================
    logger.info("\nPHASE 5: ML Model Training with Optuna")
    logger.info("-" * 50)

    model, metrics, importance_df = train_ensemble(
        sample_df,
        test_size=0.2,
        use_optuna=True,
        optuna_trials=optuna_trials,
        calibrate=True,
        filter_confidence=True
    )

    logger.info(f"Model trained - RMSE: {metrics.get('rmse', 'N/A'):.4f}, R²: {metrics.get('r2', 'N/A'):.4f}")

    # =========================================================================
    # PHASE 6: Score All Apps
    # =========================================================================
    logger.info("\nPHASE 6: Scoring All Apps")
    logger.info("-" * 50)

    all_df = df.copy()

    # Merge available LLM scores
    cache = load_llm_cache()
    if cache:
        llm_df = pd.DataFrame(list(cache.values()))
        all_df = merge_llm_scores(all_df, llm_df)
        logger.info(f"Merged {len(cache):,} LLM scores with full dataset")

    # Fill missing LLM features with neutral values
    llm_fill_values = {
        'llm_partner_probability': 0.35,
        'llm_company_maturity': 2,
        'llm_technical_sophistication': 3,
        'llm_business_model': 2,
        'llm_merchant_segment': 2,
        'llm_confidence': 1,
        'llm_fit_signal_count': 0,
        'llm_risk_flag_count': 0,
        'llm_rag_use_case_count': 0,
    }

    for col, val in llm_fill_values.items():
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(val)
        else:
            all_df[col] = val

    scored_df = score_all_apps(all_df, model)
    logger.info(f"Scored {len(scored_df):,} apps")

    # =========================================================================
    # PHASE 7: Generate Outputs
    # =========================================================================
    logger.info("\nPHASE 7: Generating Outputs")
    logger.info("-" * 50)

    # Merge back LLM reasoning for output
    if cache:
        reasoning_df = pd.DataFrame([
            {
                'id': k,
                'reasoning': v.get('reasoning', ''),
                'rag_use_cases': v.get('rag_use_cases', []),
                'partnership_fit_signals': v.get('partnership_fit_signals', []),
                'partnership_risk_flags': v.get('partnership_risk_flags', []),
                'inferred_company_maturity': v.get('inferred_company_maturity', ''),
                'inferred_technical_sophistication': v.get('inferred_technical_sophistication', 0),
                'inferred_business_model': v.get('inferred_business_model', ''),
                'inferred_merchant_segment': v.get('inferred_merchant_segment', '')
            }
            for k, v in cache.items()
        ])
        scored_df = scored_df.merge(reasoning_df, on='id', how='left', suffixes=('', '_llm'))

    # Generate outputs
    csv_path = generate_ranked_csv(scored_df)
    report_path = generate_report(scored_df, importance_df, metrics)
    tier_paths = generate_outreach_lists(scored_df)

    # Print summary
    print_summary(scored_df)

    # =========================================================================
    # DONE
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"  Duration: {duration}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Main output: {csv_path}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ISV Partner Propensity Model Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Full pipeline with defaults
  python main.py --sample-size 500         # Quick test with 500 samples
  python main.py --skip-llm                # Use cached LLM scores
  python main.py --use-batch-api           # Use Batch API (50% savings)
  python main.py --optuna-trials 100       # More thorough optimization

Cost Estimates (GPT-4o):
  500 samples:  ~$2-3 (sync) or ~$1-1.50 (batch)
  2500 samples: ~$10-15 (sync) or ~$5-8 (batch)
        """
    )

    # Execution modes
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Only create stratified sample (no LLM/ML)'
    )
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM scoring, use cached results'
    )
    parser.add_argument(
        '--skip-sentiment',
        action='store_true',
        help='Skip sentiment analysis'
    )

    # API options
    parser.add_argument(
        '--use-batch-api',
        action='store_true',
        default=True,
        help='Use OpenAI Batch API for 50%% cost savings (default: True)'
    )
    parser.add_argument(
        '--no-batch-api',
        action='store_true',
        help='Disable Batch API, use synchronous calls'
    )
    parser.add_argument(
        '--resume-batch-job',
        type=str,
        default=None,
        help='Resume a previous batch job by ID'
    )

    # Configuration
    parser.add_argument(
        '--sample-size',
        type=int,
        default=SAMPLE_SIZE,
        help=f'Number of apps to sample for LLM scoring (default: {SAMPLE_SIZE})'
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=50,
        help='Number of Optuna hyperparameter optimization trials (default: 50)'
    )

    args = parser.parse_args()

    # Handle batch API flag
    use_batch = args.use_batch_api and not args.no_batch_api

    run_pipeline(
        sample_only=args.sample_only,
        skip_llm=args.skip_llm,
        skip_sentiment=args.skip_sentiment,
        use_batch_api=use_batch,
        sample_size=args.sample_size,
        optuna_trials=args.optuna_trials,
        resume_batch_job=args.resume_batch_job
    )


if __name__ == "__main__":
    main()
