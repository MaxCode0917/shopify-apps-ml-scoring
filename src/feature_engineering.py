"""Feature engineering for ML model."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from .config import (
    INTEGRATION_KEYWORDS, SUPPORT_KEYWORDS, ENTERPRISE_KEYWORDS,
    RAG_RELEVANT_KEYWORDS, HIGH_VALUE_CATEGORIES
)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all features for ML model."""
    features = df.copy()

    print("Engineering features...")

    # === Numeric Features ===
    features['log_reviews'] = np.log1p(features['reviews_count'])
    features['log_rating_reviews'] = features['rating'] * features['log_reviews']

    # Days since last update
    today = datetime.now()
    features['days_since_lastmod'] = (today - features['lastmod']).dt.days
    features['is_recently_updated'] = (features['days_since_lastmod'] < 90).astype(int)

    # === Pricing Features ===
    features['has_free_plan'] = (features['min_price'] == 0).astype(int)
    features['has_paid_plan'] = (features['max_price'] > 0).astype(int)
    features['is_freemium'] = ((features['min_price'] == 0) & (features['max_price'] > 0)).astype(int)
    features['is_paid_only'] = ((features['min_price'] > 0) & (features['max_price'] > 0)).astype(int)

    # Price tier
    def get_price_tier(row):
        max_price = row['max_price']
        if pd.isna(max_price) or max_price == 0:
            return 'free'
        elif max_price < 10:
            return 'low'
        elif max_price < 50:
            return 'mid'
        elif max_price < 200:
            return 'high'
        else:
            return 'enterprise'

    features['price_tier'] = features.apply(get_price_tier, axis=1)

    # === Text-Based Features ===
    desc_lower = features['description'].fillna('').str.lower()
    benefits_lower = features['key_benefits'].fillna('').str.lower()
    combined_text = desc_lower + ' ' + benefits_lower

    # Keyword indicators
    features['has_integration_keywords'] = combined_text.apply(
        lambda x: any(kw in x for kw in INTEGRATION_KEYWORDS)
    ).astype(int)

    features['has_support_keywords'] = combined_text.apply(
        lambda x: any(kw in x for kw in SUPPORT_KEYWORDS)
    ).astype(int)

    features['has_enterprise_keywords'] = combined_text.apply(
        lambda x: any(kw in x for kw in ENTERPRISE_KEYWORDS)
    ).astype(int)

    features['has_rag_relevant_keywords'] = combined_text.apply(
        lambda x: any(kw in x for kw in RAG_RELEVANT_KEYWORDS)
    ).astype(int)

    # Keyword counts (more nuanced)
    features['integration_keyword_count'] = combined_text.apply(
        lambda x: sum(1 for kw in INTEGRATION_KEYWORDS if kw in x)
    )

    features['rag_keyword_count'] = combined_text.apply(
        lambda x: sum(1 for kw in RAG_RELEVANT_KEYWORDS if kw in x)
    )

    # Description quality indicators
    features['description_length'] = features['description'].fillna('').str.len()
    features['benefits_count'] = features['key_benefits'].fillna('').str.count(r'\|') + 1
    features['benefits_count'] = features['benefits_count'].where(
        features['key_benefits'].fillna('') != '', 0
    )

    # === Category Features ===
    cat_lower = features['categories'].fillna('').str.lower()
    features['is_high_value_category'] = cat_lower.apply(
        lambda x: any(cat in x for cat in HIGH_VALUE_CATEGORIES)
    ).astype(int)

    # === Developer Features ===
    features['is_multi_app_developer'] = (features['developer_app_count'] > 1).astype(int)
    features['is_prolific_developer'] = (features['developer_app_count'] >= 5).astype(int)

    # === Quality Composite Scores ===
    # Traction score (normalized 0-1)
    features['traction_score'] = (
        (features['log_reviews'] / features['log_reviews'].max()) * 0.6 +
        (features['rating'] / 5.0) * 0.4
    )

    # Technical indicator score
    features['technical_score'] = (
        features['has_integration_keywords'] * 0.3 +
        features['has_rag_relevant_keywords'] * 0.3 +
        features['is_recently_updated'].astype(int) * 0.2 +
        (features['integration_keyword_count'] > 2).astype(int) * 0.2
    )

    # Business maturity score
    features['maturity_score'] = (
        features['has_paid_plan'] * 0.3 +
        features['has_enterprise_keywords'] * 0.2 +
        features['is_multi_app_developer'].astype(int) * 0.2 +
        (features['rating'] >= 4.0).astype(int) * 0.15 +
        (features['reviews_count'] >= 50).astype(int) * 0.15
    )

    print(f"  Engineered {len([c for c in features.columns if c not in df.columns])} new features")

    return features


def get_feature_columns() -> List[str]:
    """Return list of feature columns for ML model."""
    return [
        # Numeric
        'rating', 'reviews_count', 'log_reviews', 'log_rating_reviews',
        'days_since_lastmod', 'is_recently_updated',

        # Pricing
        'min_price', 'max_price', 'avg_price', 'plan_count',
        'has_free_plan', 'has_paid_plan', 'is_freemium', 'is_paid_only',

        # Text-derived
        'has_integration_keywords', 'has_support_keywords',
        'has_enterprise_keywords', 'has_rag_relevant_keywords',
        'integration_keyword_count', 'rag_keyword_count',
        'description_length', 'benefits_count',

        # Category
        'category_count', 'is_high_value_category',

        # Developer
        'developer_app_count', 'developer_avg_rating', 'developer_total_reviews',
        'is_multi_app_developer', 'is_prolific_developer',

        # Composite
        'traction_score', 'technical_score', 'maturity_score',
    ]


def get_llm_feature_columns() -> List[str]:
    """Return list of LLM-inferred feature columns."""
    return [
        'llm_partner_probability',
        'llm_company_maturity',  # encoded
        'llm_technical_sophistication',
        'llm_business_model',  # encoded
        'llm_merchant_segment',  # encoded
        'llm_confidence',  # encoded
        'llm_fit_signal_count',
        'llm_risk_flag_count',
        'llm_rag_use_case_count',
    ]


if __name__ == "__main__":
    from .data_loader import load_enriched_apps

    df = load_enriched_apps()
    df = extract_features(df)
    print("\nFeature columns:")
    for col in get_feature_columns():
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
