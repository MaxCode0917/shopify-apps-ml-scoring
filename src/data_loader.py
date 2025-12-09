"""Data loading and merging utilities."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from .config import DATA_DIR


def load_apps() -> pd.DataFrame:
    """Load main apps dataset."""
    df = pd.read_csv(DATA_DIR / "apps.csv")
    df['lastmod'] = pd.to_datetime(df['lastmod'])
    return df


def load_categories() -> pd.DataFrame:
    """Load categories lookup table."""
    return pd.read_csv(DATA_DIR / "categories.csv")


def load_apps_categories() -> pd.DataFrame:
    """Load app-category mappings."""
    return pd.read_csv(DATA_DIR / "apps_categories.csv")


def load_pricing_plans() -> pd.DataFrame:
    """Load pricing plans."""
    return pd.read_csv(DATA_DIR / "pricing_plans.csv")


def load_reviews() -> pd.DataFrame:
    """Load reviews dataset."""
    return pd.read_csv(DATA_DIR / "reviews.csv")


def load_key_benefits() -> pd.DataFrame:
    """Load key benefits."""
    return pd.read_csv(DATA_DIR / "key_benefits.csv")


def get_app_categories(apps_df: pd.DataFrame) -> pd.DataFrame:
    """Merge category names with apps."""
    categories = load_categories()
    apps_cats = load_apps_categories()

    # Join to get category names
    apps_cats_named = apps_cats.merge(
        categories,
        left_on='category_id',
        right_on='id',
        suffixes=('', '_cat')
    )

    # Aggregate categories per app
    cat_agg = apps_cats_named.groupby('app_id').agg({
        'title': lambda x: ' | '.join(x.dropna().unique()),
        'category_id': 'count'
    }).rename(columns={
        'title': 'categories',
        'category_id': 'category_count'
    }).reset_index()

    return apps_df.merge(cat_agg, left_on='id', right_on='app_id', how='left')


def get_app_pricing_details(apps_df: pd.DataFrame) -> pd.DataFrame:
    """Extract detailed pricing information per app."""
    pricing = load_pricing_plans()

    def parse_price(price_str: str) -> Optional[float]:
        """Extract numeric price from string."""
        if pd.isna(price_str):
            return None
        price_str = str(price_str).lower()
        if 'free' in price_str:
            return 0.0
        # Extract first number
        import re
        matches = re.findall(r'[\d,]+\.?\d*', price_str.replace(',', ''))
        if matches:
            try:
                return float(matches[0])
            except:
                return None
        return None

    pricing['price_numeric'] = pricing['price'].apply(parse_price)

    # Aggregate per app
    pricing_agg = pricing.groupby('app_id').agg({
        'price_numeric': ['min', 'max', 'mean'],
        'title': lambda x: ' | '.join(x.dropna().unique()[:3]),  # First 3 plan names
        'id': 'count'
    })
    pricing_agg.columns = ['min_price', 'max_price', 'avg_price', 'plan_names', 'plan_count']
    pricing_agg = pricing_agg.reset_index()

    return apps_df.merge(pricing_agg, left_on='id', right_on='app_id', how='left')


def get_app_benefits(apps_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key benefits per app."""
    benefits = load_key_benefits()

    benefits_agg = benefits.groupby('app_id').agg({
        'description': lambda x: ' | '.join(x.dropna().unique()[:5])  # First 5 benefits
    }).rename(columns={'description': 'key_benefits'}).reset_index()

    return apps_df.merge(benefits_agg, left_on='id', right_on='app_id', how='left')


def get_developer_stats(apps_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate developer-level statistics."""
    dev_stats = apps_df.groupby('developer').agg({
        'id': 'count',
        'rating': 'mean',
        'reviews_count': 'sum'
    }).rename(columns={
        'id': 'developer_app_count',
        'rating': 'developer_avg_rating',
        'reviews_count': 'developer_total_reviews'
    }).reset_index()

    return apps_df.merge(dev_stats, on='developer', how='left')


def load_enriched_apps() -> pd.DataFrame:
    """Load and enrich apps with all related data."""
    print("Loading apps data...")
    apps = load_apps()
    print(f"  Loaded {len(apps):,} apps")

    print("Merging categories...")
    apps = get_app_categories(apps)

    print("Merging pricing details...")
    apps = get_app_pricing_details(apps)

    print("Merging key benefits...")
    apps = get_app_benefits(apps)

    print("Calculating developer stats...")
    apps = get_developer_stats(apps)

    # Clean up duplicate columns from merges
    cols_to_drop = [c for c in apps.columns if c.endswith('_x') or c.endswith('_y') or c == 'app_id']
    apps = apps.drop(columns=cols_to_drop, errors='ignore')

    # Fill NaN categories
    apps['categories'] = apps['categories'].fillna('Uncategorized')
    apps['category_count'] = apps['category_count'].fillna(0).astype(int)
    apps['key_benefits'] = apps['key_benefits'].fillna('')

    print(f"  Final dataset: {len(apps):,} apps with {len(apps.columns)} columns")
    return apps


if __name__ == "__main__":
    df = load_enriched_apps()
    print("\nSample row:")
    print(df.iloc[0])
