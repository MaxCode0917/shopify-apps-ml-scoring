"""Review sentiment analysis using OpenAI."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI

from .config import OPENAI_API_KEY, CACHE_DIR, OPENAI_MODEL
from .data_loader import load_reviews


SENTIMENT_CACHE_FILE = CACHE_DIR / "sentiment_cache.json"


def load_sentiment_cache() -> Dict:
    """Load cached sentiment results."""
    if SENTIMENT_CACHE_FILE.exists():
        with open(SENTIMENT_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_sentiment_cache(cache: Dict):
    """Save sentiment cache."""
    with open(SENTIMENT_CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def aggregate_reviews_for_app(reviews_df: pd.DataFrame, app_id: str, max_reviews: int = 20) -> str:
    """Aggregate review text for a single app."""
    app_reviews = reviews_df[reviews_df['app_id'] == app_id]

    if len(app_reviews) == 0:
        return ""

    # Sort by helpful_count and recency, take top reviews
    app_reviews = app_reviews.copy()
    app_reviews['helpful_count'] = app_reviews['helpful_count'].fillna(0)
    app_reviews = app_reviews.sort_values(
        ['helpful_count', 'rating'],
        ascending=[False, False]
    ).head(max_reviews)

    # Format reviews
    review_texts = []
    for _, row in app_reviews.iterrows():
        rating = row.get('rating', 'N/A')
        body = str(row.get('body', ''))[:300]  # Truncate long reviews
        review_texts.append(f"[{rating}★] {body}")

    return "\n".join(review_texts)


def analyze_sentiment_batch(
    app_ids: List[str],
    reviews_df: pd.DataFrame,
    client: OpenAI,
    cache: Dict
) -> Dict[str, Dict]:
    """Analyze sentiment for a batch of apps."""

    results = {}

    for app_id in tqdm(app_ids, desc="Analyzing sentiment"):
        # Check cache first
        if app_id in cache:
            results[app_id] = cache[app_id]
            continue

        review_text = aggregate_reviews_for_app(reviews_df, app_id)

        if not review_text:
            results[app_id] = {
                "summary": "No reviews available",
                "overall_sentiment": "neutral",
                "sentiment_score": 0.5,
                "key_positives": [],
                "key_negatives": [],
                "review_count_analyzed": 0
            }
            cache[app_id] = results[app_id]
            continue

        prompt = f"""Analyze these Shopify app reviews and provide a brief sentiment summary.

Reviews:
{review_text}

Respond ONLY with valid JSON:
{{
  "summary": "<1-2 sentence summary of overall customer sentiment>",
  "overall_sentiment": "<positive|neutral|negative|mixed>",
  "sentiment_score": <float 0.0-1.0, where 1.0 is very positive>,
  "key_positives": ["<positive_1>", "<positive_2>"],
  "key_negatives": ["<negative_1>", "<negative_2>"]
}}"""

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            # Clean JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            result = json.loads(content)
            result["review_count_analyzed"] = len(reviews_df[reviews_df['app_id'] == app_id])
            results[app_id] = result
            cache[app_id] = result

        except Exception as e:
            print(f"  Error for {app_id}: {e}")
            results[app_id] = {
                "summary": "Analysis failed",
                "overall_sentiment": "unknown",
                "sentiment_score": 0.5,
                "key_positives": [],
                "key_negatives": [],
                "review_count_analyzed": 0,
                "error": str(e)
            }

        # Save cache periodically
        if len(cache) % 50 == 0:
            save_sentiment_cache(cache)

    return results


def get_sentiment_for_apps(app_ids: List[str], use_cache: bool = True) -> pd.DataFrame:
    """Get sentiment analysis for a list of apps."""

    print(f"Analyzing sentiment for {len(app_ids)} apps...")

    client = OpenAI(api_key=OPENAI_API_KEY)
    reviews_df = load_reviews()

    print(f"  Loaded {len(reviews_df):,} reviews")

    cache = load_sentiment_cache() if use_cache else {}
    print(f"  Cache has {len(cache)} entries")

    # Find apps that need analysis
    uncached = [aid for aid in app_ids if aid not in cache]
    print(f"  Need to analyze {len(uncached)} new apps")

    if uncached:
        analyze_sentiment_batch(uncached, reviews_df, client, cache)
        save_sentiment_cache(cache)

    # Build results DataFrame
    sentiment_data = []
    for app_id in app_ids:
        if app_id in cache:
            row = {"app_id": app_id, **cache[app_id]}
        else:
            row = {
                "app_id": app_id,
                "summary": "No analysis",
                "overall_sentiment": "unknown",
                "sentiment_score": 0.5,
                "key_positives": [],
                "key_negatives": [],
                "review_count_analyzed": 0
            }
        sentiment_data.append(row)

    df = pd.DataFrame(sentiment_data)

    # Convert lists to strings for easier handling
    df['key_positives_str'] = df['key_positives'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )
    df['key_negatives_str'] = df['key_negatives'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else str(x)
    )

    return df


def get_sentiment_summary_for_prompt(app_id: str, cache: Optional[Dict] = None) -> str:
    """Get a formatted sentiment summary for LLM prompt."""
    if cache is None:
        cache = load_sentiment_cache()

    if app_id not in cache:
        return "No review sentiment data available."

    s = cache[app_id]
    summary = s.get('summary', 'No summary')
    sentiment = s.get('overall_sentiment', 'unknown')
    score = s.get('sentiment_score', 0.5)
    positives = s.get('key_positives', [])
    negatives = s.get('key_negatives', [])

    result = f"Overall: {sentiment} (score: {score:.2f}). {summary}"
    if positives:
        result += f"\nPositives: {', '.join(positives[:3])}"
    if negatives:
        result += f"\nNegatives: {', '.join(negatives[:3])}"

    return result


if __name__ == "__main__":
    # Test with a few apps
    from .data_loader import load_apps

    apps = load_apps()
    test_ids = apps[apps['reviews_count'] > 10]['id'].head(5).tolist()

    sentiment_df = get_sentiment_for_apps(test_ids)
    print("\nSentiment results:")
    print(sentiment_df[['app_id', 'overall_sentiment', 'sentiment_score', 'summary']])
