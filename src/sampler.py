"""Stratified sampling for LLM scoring."""

import pandas as pd
import numpy as np
from typing import Optional
from .config import SAMPLE_SIZE, RANDOM_STATE


def create_stratified_sample(
    df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Create a stratified sample ensuring coverage across key dimensions."""

    print(f"Creating stratified sample of {sample_size} from {len(df)} apps...")

    # Create stratification bins
    df = df.copy()

    # Rating bands
    df['rating_band'] = pd.cut(
        df['rating'],
        bins=[-0.1, 0, 3.0, 4.0, 4.5, 5.1],
        labels=['no_rating', 'low', 'medium', 'high', 'excellent']
    )

    # Review count bands (log scale)
    df['review_band'] = pd.cut(
        df['reviews_count'],
        bins=[-1, 0, 10, 50, 200, 1000, float('inf')],
        labels=['none', 'few', 'moderate', 'good', 'high', 'very_high']
    )

    # Pricing bands
    df['price_band'] = pd.cut(
        df['max_price'].fillna(0),
        bins=[-1, 0, 10, 50, 200, float('inf')],
        labels=['free', 'low', 'mid', 'high', 'enterprise']
    )

    # Create composite stratum
    df['stratum'] = (
        df['rating_band'].astype(str) + '_' +
        df['review_band'].astype(str) + '_' +
        df['price_band'].astype(str)
    )

    # Count apps per stratum
    stratum_counts = df['stratum'].value_counts()
    print(f"  Found {len(stratum_counts)} unique strata")

    # Calculate proportional sample size per stratum
    # But ensure minimum representation for small strata
    min_per_stratum = 2
    total_strata = len(stratum_counts)

    # Reserve some samples for minimum representation
    reserved = min(min_per_stratum * total_strata, sample_size // 2)
    proportional_budget = sample_size - reserved

    # Calculate samples per stratum
    stratum_samples = {}
    for stratum, count in stratum_counts.items():
        # Proportional allocation + minimum
        prop_samples = int((count / len(df)) * proportional_budget)
        stratum_samples[stratum] = min(count, max(min_per_stratum, prop_samples))

    # Adjust to hit target sample size
    current_total = sum(stratum_samples.values())
    if current_total < sample_size:
        # Add more from largest strata
        deficit = sample_size - current_total
        large_strata = stratum_counts.head(20).index.tolist()
        for stratum in large_strata:
            if deficit <= 0:
                break
            available = stratum_counts[stratum] - stratum_samples[stratum]
            add = min(available, deficit // len(large_strata) + 1)
            stratum_samples[stratum] += add
            deficit -= add

    # Sample from each stratum
    samples = []
    for stratum, n_samples in stratum_samples.items():
        stratum_df = df[df['stratum'] == stratum]
        if len(stratum_df) <= n_samples:
            samples.append(stratum_df)
        else:
            samples.append(stratum_df.sample(n=n_samples, random_state=random_state))

    sample_df = pd.concat(samples, ignore_index=True)

    # If still under target, add random samples
    if len(sample_df) < sample_size:
        remaining_ids = set(df['id']) - set(sample_df['id'])
        remaining_df = df[df['id'].isin(remaining_ids)]
        additional = remaining_df.sample(
            n=min(sample_size - len(sample_df), len(remaining_df)),
            random_state=random_state
        )
        sample_df = pd.concat([sample_df, additional], ignore_index=True)

    # Trim if over
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=random_state)

    print(f"  Final sample size: {len(sample_df)}")

    # Print distribution summary
    print("\n  Sample distribution:")
    print(f"    Rating bands: {sample_df['rating_band'].value_counts().to_dict()}")
    print(f"    Review bands: {sample_df['review_band'].value_counts().to_dict()}")
    print(f"    Price bands: {sample_df['price_band'].value_counts().to_dict()}")

    # Drop stratification columns
    sample_df = sample_df.drop(columns=['rating_band', 'review_band', 'price_band', 'stratum'])

    return sample_df


def ensure_high_value_coverage(
    sample_df: pd.DataFrame,
    full_df: pd.DataFrame,
    min_high_value: int = 500
) -> pd.DataFrame:
    """Ensure sample includes sufficient high-value candidates."""

    # Define high-value: rating >= 4.0, reviews >= 20, has paid plans
    high_value_mask = (
        (full_df['rating'] >= 4.0) &
        (full_df['reviews_count'] >= 20) &
        (full_df['max_price'] > 0)
    )
    high_value_df = full_df[high_value_mask]

    # Check current high-value count in sample
    sample_ids = set(sample_df['id'])
    current_high_value = len(high_value_df[high_value_df['id'].isin(sample_ids)])

    if current_high_value >= min_high_value:
        print(f"  High-value coverage OK: {current_high_value} apps")
        return sample_df

    # Add more high-value apps
    needed = min_high_value - current_high_value
    missing_hv = high_value_df[~high_value_df['id'].isin(sample_ids)]

    if len(missing_hv) > 0:
        additional = missing_hv.sample(n=min(needed, len(missing_hv)), random_state=RANDOM_STATE)
        sample_df = pd.concat([sample_df, additional], ignore_index=True)
        print(f"  Added {len(additional)} high-value apps to sample")

    return sample_df


if __name__ == "__main__":
    from .data_loader import load_enriched_apps
    from .feature_engineering import extract_features

    df = load_enriched_apps()
    df = extract_features(df)
    sample = create_stratified_sample(df)
    sample = ensure_high_value_coverage(sample, df)
    print(f"\nFinal sample: {len(sample)} apps")
