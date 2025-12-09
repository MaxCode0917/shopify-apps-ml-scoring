"""Generate final output files."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import OUTPUT_DIR


def generate_ranked_csv(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Generate the primary ranked output CSV."""

    if output_path is None:
        output_path = OUTPUT_DIR / f"partner_candidates_ranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Select and order columns
    output_cols = [
        # Identifiers
        'id', 'title', 'developer', 'url',

        # Scores and ranking
        'final_partner_probability', 'rank', 'tier',
        'llm_partner_probability', 'ml_partner_probability',

        # Key metrics
        'rating', 'reviews_count', 'pricing_hint',

        # LLM insights
        'reasoning', 'rag_use_cases', 'partnership_fit_signals', 'partnership_risk_flags',
        'inferred_company_maturity', 'inferred_technical_sophistication',
        'inferred_business_model', 'inferred_merchant_segment',

        # Additional context
        'categories', 'key_benefits', 'developer_app_count',
        'lastmod', 'max_price',
    ]

    # Filter to available columns
    available_cols = [c for c in output_cols if c in df.columns]

    output_df = df[available_cols].copy()

    # Sort by rank
    output_df = output_df.sort_values('rank')

    # Format list columns
    for col in ['rag_use_cases', 'partnership_fit_signals', 'partnership_risk_flags']:
        if col in output_df.columns:
            output_df[col] = output_df[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ''
            )

    output_df.to_csv(output_path, index=False)
    print(f"Ranked CSV saved to {output_path}")

    return output_path


def generate_tier_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics by tier."""

    summary = {}

    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier]

        summary[f'tier_{tier}'] = {
            'count': len(tier_df),
            'avg_probability': round(tier_df['final_partner_probability'].mean(), 3),
            'avg_rating': round(tier_df['rating'].mean(), 2),
            'avg_reviews': round(tier_df['reviews_count'].mean(), 0),
            'pct_with_paid_plans': round((tier_df['max_price'] > 0).mean() * 100, 1),
            'top_categories': tier_df['categories'].value_counts().head(5).to_dict(),
        }

    return summary


def generate_report(
    df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_metrics: dict
) -> Path:
    """Generate comprehensive JSON report."""

    report = {
        'generated_at': datetime.now().isoformat(),
        'total_apps_scored': len(df),

        'tier_summary': generate_tier_summary(df),

        'model_performance': model_metrics,

        'top_features': feature_importance.head(15).to_dict('records'),

        'score_distribution': {
            'mean': round(df['final_partner_probability'].mean(), 3),
            'median': round(df['final_partner_probability'].median(), 3),
            'std': round(df['final_partner_probability'].std(), 3),
            'min': round(df['final_partner_probability'].min(), 3),
            'max': round(df['final_partner_probability'].max(), 3),
            'percentiles': {
                '25': round(df['final_partner_probability'].quantile(0.25), 3),
                '50': round(df['final_partner_probability'].quantile(0.50), 3),
                '75': round(df['final_partner_probability'].quantile(0.75), 3),
                '90': round(df['final_partner_probability'].quantile(0.90), 3),
                '95': round(df['final_partner_probability'].quantile(0.95), 3),
            }
        },

        'top_10_candidates': df.nsmallest(10, 'rank')[[
            'title', 'developer', 'final_partner_probability', 'rating',
            'reviews_count', 'reasoning'
        ]].to_dict('records'),
    }

    report_path = OUTPUT_DIR / f"partner_propensity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Report saved to {report_path}")

    return report_path


def generate_outreach_lists(df: pd.DataFrame) -> dict:
    """Generate separate CSV files per tier for outreach."""

    paths = {}

    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier].sort_values('rank')

        cols = ['rank', 'title', 'developer', 'url', 'final_partner_probability',
                'rating', 'reviews_count', 'pricing_hint', 'reasoning', 'rag_use_cases']
        cols = [c for c in cols if c in tier_df.columns]

        tier_path = OUTPUT_DIR / f"tier_{tier}_candidates.csv"
        tier_df[cols].to_csv(tier_path, index=False)
        paths[f'tier_{tier}'] = str(tier_path)

        print(f"Tier {tier}: {len(tier_df)} candidates saved to {tier_path}")

    return paths


def print_summary(df: pd.DataFrame):
    """Print summary to console."""

    print("\n" + "="*60)
    print("ISV PARTNER PROPENSITY MODEL - RESULTS SUMMARY")
    print("="*60)

    print(f"\nTotal apps scored: {len(df):,}")

    print("\nTier Distribution:")
    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier]
        print(f"  Tier {tier}: {len(tier_df):,} apps "
              f"(avg prob: {tier_df['final_partner_probability'].mean():.2f})")

    print("\nTop 10 Partner Candidates:")
    print("-" * 60)
    for i, row in df.nsmallest(10, 'rank').iterrows():
        print(f"  #{int(row['rank']):>4} | {row['final_partner_probability']:.2f} | "
              f"{row['title'][:35]:<35} | ★{row['rating']:.1f} ({int(row['reviews_count'])} reviews)")

    print("\nScore Distribution:")
    print(f"  Mean:   {df['final_partner_probability'].mean():.3f}")
    print(f"  Median: {df['final_partner_probability'].median():.3f}")
    print(f"  Std:    {df['final_partner_probability'].std():.3f}")
    print(f"  90th %: {df['final_partner_probability'].quantile(0.9):.3f}")

    print("\n" + "="*60)
