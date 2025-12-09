"""
LLM-based partner scoring using OpenAI Batch API.

Industry best practices implemented:
- OpenAI Batch API for 50% cost reduction
- Confidence-based quality filtering
- Proper JSON schema validation with Pydantic
- Comprehensive error handling and retry logic
- Caching with atomic writes
- Progress tracking and cost estimation
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from .config import (
    OPENAI_API_KEY, OPENAI_MODEL, CACHE_DIR, OUTPUT_DIR,
    LLM_PROMPT_TEMPLATE, PARTNER_CONTEXT,
    MAX_TOKENS, TEMPERATURE
)
from .sentiment_analyzer import get_sentiment_summary_for_prompt, load_sentiment_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Response Schema (Pydantic for validation)
# ============================================================================

class PartnerAssessment(BaseModel):
    """Validated response schema for LLM partner assessment."""
    partner_probability: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10, max_length=1000)
    rag_use_cases: List[str] = Field(default_factory=list, max_length=5)
    inferred_company_maturity: str = Field(pattern=r'^(hobbyist|small_team|established_business|agency|enterprise|unclear)$')
    inferred_technical_sophistication: int = Field(ge=1, le=5)
    inferred_business_model: str = Field(pattern=r'^(free_only|freemium|paid|enterprise|unclear)$')
    inferred_merchant_segment: str = Field(pattern=r'^(smb|mid_market|enterprise|all|unclear)$')
    partnership_fit_signals: List[str] = Field(default_factory=list, max_length=10)
    partnership_risk_flags: List[str] = Field(default_factory=list, max_length=10)
    confidence: str = Field(pattern=r'^(low|medium|high)$')


@dataclass
class BatchJobStatus:
    """Track batch job status and results."""
    job_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None


# ============================================================================
# Cache Management
# ============================================================================

LLM_CACHE_FILE = CACHE_DIR / "llm_scores_cache.jsonl"
BATCH_JOBS_FILE = CACHE_DIR / "batch_jobs.json"


def load_llm_cache() -> Dict[str, Dict]:
    """Load cached LLM responses with validation."""
    cache = {}
    if LLM_CACHE_FILE.exists():
        with open(LLM_CACHE_FILE, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if 'id' in entry and 'partner_probability' in entry:
                        cache[entry['id']] = entry
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid cache line {line_num}: {e}")
    logger.info(f"Loaded {len(cache)} entries from LLM cache")
    return cache


def append_to_llm_cache(entries: List[Dict]):
    """Append entries to cache with atomic write."""
    if not entries:
        return

    # Write to temp file first, then rename (atomic)
    temp_file = LLM_CACHE_FILE.with_suffix('.tmp')

    # Read existing entries
    existing = []
    if LLM_CACHE_FILE.exists():
        with open(LLM_CACHE_FILE, 'r') as f:
            existing = f.readlines()

    # Append new entries
    with open(temp_file, 'w') as f:
        for line in existing:
            f.write(line)
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    # Atomic rename
    temp_file.rename(LLM_CACHE_FILE)
    logger.info(f"Saved {len(entries)} entries to cache")


def save_batch_job_status(job: BatchJobStatus):
    """Save batch job status for recovery."""
    jobs = {}
    if BATCH_JOBS_FILE.exists():
        with open(BATCH_JOBS_FILE, 'r') as f:
            jobs = json.load(f)

    jobs[job.job_id] = {
        'job_id': job.job_id,
        'status': job.status,
        'created_at': job.created_at.isoformat(),
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'total_requests': job.total_requests,
        'completed_requests': job.completed_requests,
        'failed_requests': job.failed_requests,
        'output_file_id': job.output_file_id,
        'error_file_id': job.error_file_id
    }

    with open(BATCH_JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2)


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_prompt_for_app(row: pd.Series, sentiment_cache: Dict) -> str:
    """Format the LLM prompt for a single app with all context."""

    sentiment_summary = get_sentiment_summary_for_prompt(row['id'], sentiment_cache)

    # Safe extraction with defaults
    def safe_get(key, default=''):
        val = row.get(key, default)
        return default if pd.isna(val) else val

    categories = safe_get('categories', 'Uncategorized')
    key_benefits = safe_get('key_benefits', 'Not specified')

    # Truncate description intelligently (keep first and last parts)
    description = safe_get('description', 'No description available')
    if len(description) > 2000:
        description = description[:1500] + '\n...[truncated]...\n' + description[-400:]

    # Format lastmod
    lastmod = row.get('lastmod', 'Unknown')
    if isinstance(lastmod, pd.Timestamp):
        lastmod = lastmod.strftime('%Y-%m-%d')

    prompt = LLM_PROMPT_TEMPLATE.format(
        partner_context=PARTNER_CONTEXT,
        title=safe_get('title', 'Unknown App'),
        developer=safe_get('developer', 'Unknown Developer'),
        rating=row.get('rating', 0),
        reviews_count=row.get('reviews_count', 0),
        pricing_hint=safe_get('pricing_hint', 'Not specified'),
        lastmod=lastmod,
        categories=categories,
        description=description,
        key_benefits=key_benefits if key_benefits else 'Not specified',
        sentiment_summary=sentiment_summary
    )

    return prompt


# ============================================================================
# Response Parsing with Validation
# ============================================================================

def parse_llm_response(content: str, app_id: str) -> Optional[Dict]:
    """Parse and validate LLM JSON response with robust error handling."""
    try:
        # Clean response - handle various markdown formats
        content = content.strip()

        # Remove markdown code blocks
        if '```' in content:
            parts = content.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    part = part[4:].strip()
                if part.startswith('{'):
                    content = part
                    break

        # Find JSON object boundaries
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]

        data = json.loads(content)

        # Validate with Pydantic
        assessment = PartnerAssessment(**data)
        result = assessment.model_dump()
        result['id'] = app_id
        result['parsed_at'] = datetime.now().isoformat()

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {app_id}: {e}")
        return None
    except ValidationError as e:
        logger.warning(f"Validation error for {app_id}: {e}")
        # Try to salvage what we can
        try:
            data = json.loads(content)
            # Apply defaults for missing/invalid fields
            return {
                'id': app_id,
                'partner_probability': float(data.get('partner_probability', 0.3)),
                'reasoning': str(data.get('reasoning', 'Partial parse'))[:1000],
                'rag_use_cases': data.get('rag_use_cases', [])[:5],
                'inferred_company_maturity': 'unclear',
                'inferred_technical_sophistication': 3,
                'inferred_business_model': 'unclear',
                'inferred_merchant_segment': 'unclear',
                'partnership_fit_signals': data.get('partnership_fit_signals', [])[:10],
                'partnership_risk_flags': data.get('partnership_risk_flags', [])[:10],
                'confidence': 'low',
                'parsed_at': datetime.now().isoformat(),
                'partial_parse': True
            }
        except:
            return None
    except Exception as e:
        logger.error(f"Unexpected error parsing response for {app_id}: {e}")
        return None


# ============================================================================
# OpenAI Batch API Implementation
# ============================================================================

def create_batch_file(df: pd.DataFrame, sentiment_cache: Dict) -> Path:
    """Create JSONL file for OpenAI Batch API."""

    batch_file_path = CACHE_DIR / f"batch_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    tasks = []
    for _, row in df.iterrows():
        prompt = format_prompt_for_app(row, sentiment_cache)

        task = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert B2B partnership analyst. Always respond with valid JSON matching the exact schema requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        }
        tasks.append(task)

    with open(batch_file_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    logger.info(f"Created batch file with {len(tasks)} tasks: {batch_file_path}")
    return batch_file_path


def submit_batch_job(client: OpenAI, batch_file_path: Path) -> BatchJobStatus:
    """Submit batch job to OpenAI and return job status."""

    # Upload file
    logger.info("Uploading batch file to OpenAI...")
    with open(batch_file_path, 'rb') as f:
        batch_file = client.files.create(file=f, purpose="batch")

    logger.info(f"File uploaded: {batch_file.id}")

    # Create batch job
    logger.info("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "ISV Partner Propensity Scoring",
            "created_by": "partner_propensity_pipeline"
        }
    )

    job_status = BatchJobStatus(
        job_id=batch_job.id,
        status=batch_job.status,
        created_at=datetime.now(),
        total_requests=batch_job.request_counts.total if batch_job.request_counts else 0
    )

    save_batch_job_status(job_status)
    logger.info(f"Batch job created: {batch_job.id}")

    return job_status


def wait_for_batch_completion(
    client: OpenAI,
    job_id: str,
    poll_interval: int = 60,
    max_wait_hours: float = 24
) -> BatchJobStatus:
    """Poll for batch job completion with progress updates."""

    max_wait_seconds = max_wait_hours * 3600
    elapsed = 0

    logger.info(f"Waiting for batch job {job_id} to complete...")

    with tqdm(desc="Batch Progress", unit="req") as pbar:
        last_completed = 0

        while elapsed < max_wait_seconds:
            batch_job = client.batches.retrieve(job_id)

            status = BatchJobStatus(
                job_id=batch_job.id,
                status=batch_job.status,
                created_at=datetime.fromtimestamp(batch_job.created_at) if batch_job.created_at else datetime.now(),
                total_requests=batch_job.request_counts.total if batch_job.request_counts else 0,
                completed_requests=batch_job.request_counts.completed if batch_job.request_counts else 0,
                failed_requests=batch_job.request_counts.failed if batch_job.request_counts else 0,
                output_file_id=batch_job.output_file_id,
                error_file_id=batch_job.error_file_id
            )

            # Update progress bar
            if status.total_requests > 0:
                pbar.total = status.total_requests
                new_completed = status.completed_requests - last_completed
                if new_completed > 0:
                    pbar.update(new_completed)
                last_completed = status.completed_requests

            if batch_job.status in ['completed', 'failed', 'expired', 'cancelled']:
                status.completed_at = datetime.now()
                save_batch_job_status(status)

                if batch_job.status == 'completed':
                    logger.info(f"Batch job completed! {status.completed_requests}/{status.total_requests} successful")
                else:
                    logger.error(f"Batch job {batch_job.status}: {status.failed_requests} failed")

                return status

            save_batch_job_status(status)
            time.sleep(poll_interval)
            elapsed += poll_interval

    raise TimeoutError(f"Batch job {job_id} did not complete within {max_wait_hours} hours")


def download_batch_results(client: OpenAI, status: BatchJobStatus) -> List[Dict]:
    """Download and parse batch results."""

    if not status.output_file_id:
        logger.error("No output file available")
        return []

    logger.info(f"Downloading results from {status.output_file_id}...")

    result_content = client.files.content(status.output_file_id).content

    # Save raw results
    results_file = CACHE_DIR / f"batch_results_{status.job_id}.jsonl"
    with open(results_file, 'wb') as f:
        f.write(result_content)

    # Parse results
    results = []
    failed_parses = []

    for line in result_content.decode('utf-8').strip().split('\n'):
        try:
            result = json.loads(line)
            custom_id = result.get('custom_id')

            if result.get('error'):
                logger.warning(f"API error for {custom_id}: {result['error']}")
                failed_parses.append({'id': custom_id, 'error': result['error']})
                continue

            response_body = result.get('response', {}).get('body', {})
            content = response_body.get('choices', [{}])[0].get('message', {}).get('content', '')

            parsed = parse_llm_response(content, custom_id)
            if parsed:
                results.append(parsed)
            else:
                failed_parses.append({'id': custom_id, 'error': 'Parse failed', 'raw': content[:500]})

        except Exception as e:
            logger.error(f"Error processing result line: {e}")

    logger.info(f"Parsed {len(results)} results, {len(failed_parses)} failures")

    # Save failures for debugging
    if failed_parses:
        failures_file = CACHE_DIR / f"batch_failures_{status.job_id}.json"
        with open(failures_file, 'w') as f:
            json.dump(failed_parses, f, indent=2)

    return results


def score_apps_with_batch_api(
    df: pd.DataFrame,
    use_cache: bool = True,
    resume_job_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Score apps using OpenAI Batch API.

    Benefits:
    - 50% cost reduction
    - Higher rate limits
    - Automatic retries
    - Results within 24 hours
    """

    logger.info(f"Starting batch scoring for {len(df)} apps...")

    client = OpenAI(api_key=OPENAI_API_KEY)
    sentiment_cache = load_sentiment_cache()
    llm_cache = load_llm_cache() if use_cache else {}

    # Find uncached apps
    uncached_mask = ~df['id'].isin(llm_cache.keys())
    uncached_df = df[uncached_mask]

    logger.info(f"Cache: {len(llm_cache)} entries, Need to score: {len(uncached_df)} apps")

    if len(uncached_df) == 0:
        logger.info("All apps already cached!")
    elif resume_job_id:
        # Resume existing job
        logger.info(f"Resuming job {resume_job_id}...")
        status = wait_for_batch_completion(client, resume_job_id)
        results = download_batch_results(client, status)
        append_to_llm_cache(results)
        for r in results:
            llm_cache[r['id']] = r
    else:
        # Create and submit new batch
        batch_file = create_batch_file(uncached_df, sentiment_cache)
        status = submit_batch_job(client, batch_file)

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch job submitted: {status.job_id}")
        logger.info(f"Estimated completion: Within 24 hours (usually much faster)")
        logger.info(f"Cost estimate: ~${len(uncached_df) * 0.004:.2f} (50% batch discount)")
        logger.info(f"{'='*60}\n")

        # Wait for completion
        status = wait_for_batch_completion(client, status.job_id)

        if status.status == 'completed':
            results = download_batch_results(client, status)
            append_to_llm_cache(results)
            for r in results:
                llm_cache[r['id']] = r

    # Build results DataFrame
    return _build_results_dataframe(df, llm_cache)


# ============================================================================
# Synchronous Fallback (for small batches or testing)
# ============================================================================

def score_apps_with_llm(
    df: pd.DataFrame,
    use_cache: bool = True,
    max_retries: int = 3,
    rate_limit_delay: float = 0.5
) -> pd.DataFrame:
    """
    Score apps using synchronous API calls.
    Use for small batches (<100) or testing.
    For larger batches, use score_apps_with_batch_api().
    """

    logger.info(f"Scoring {len(df)} apps with synchronous API ({OPENAI_MODEL})...")

    client = OpenAI(api_key=OPENAI_API_KEY)
    sentiment_cache = load_sentiment_cache()
    llm_cache = load_llm_cache() if use_cache else {}

    logger.info(f"Sentiment cache: {len(sentiment_cache)} entries")
    logger.info(f"LLM cache: {len(llm_cache)} entries")

    # Find uncached apps
    uncached_mask = ~df['id'].isin(llm_cache.keys())
    uncached_df = df[uncached_mask]
    logger.info(f"Need to score {len(uncached_df)} new apps")

    if len(uncached_df) > 100:
        logger.warning(f"Large batch ({len(uncached_df)} apps) - consider using score_apps_with_batch_api() for 50% cost savings")

    # Score uncached apps
    results = []
    errors = []

    for idx, row in tqdm(uncached_df.iterrows(), total=len(uncached_df), desc="LLM Scoring"):
        app_id = row['id']
        prompt = format_prompt_for_app(row, sentiment_cache)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert B2B partnership analyst. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                parsed = parse_llm_response(content, app_id)

                if parsed:
                    results.append(parsed)
                    llm_cache[app_id] = parsed
                    break
                elif attempt == max_retries - 1:
                    errors.append({
                        'id': app_id,
                        'error': 'Failed to parse response',
                        'raw_response': content[:500]
                    })

            except Exception as e:
                if attempt == max_retries - 1:
                    errors.append({'id': app_id, 'error': str(e)})
                    logger.error(f"Failed to score {app_id} after {max_retries} attempts: {e}")
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff

        time.sleep(rate_limit_delay)

        # Periodic cache saves
        if len(results) % 50 == 0 and results:
            append_to_llm_cache(results[-50:])

    # Final cache save
    if results:
        append_to_llm_cache(results)

    logger.info(f"Successfully scored: {len(results)}, Errors: {len(errors)}")

    if errors:
        error_file = CACHE_DIR / "llm_errors.json"
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.info(f"Errors saved to {error_file}")

    return _build_results_dataframe(df, llm_cache)


# ============================================================================
# Results Processing
# ============================================================================

def _build_results_dataframe(df: pd.DataFrame, llm_cache: Dict) -> pd.DataFrame:
    """Build results DataFrame with encoded features."""

    # Encoding maps for categorical features
    maturity_map = {
        'hobbyist': 1, 'small_team': 2, 'established_business': 3,
        'agency': 4, 'enterprise': 5, 'unclear': 2
    }
    business_model_map = {
        'free_only': 1, 'freemium': 2, 'paid': 3, 'enterprise': 4, 'unclear': 2
    }
    segment_map = {
        'smb': 1, 'mid_market': 2, 'enterprise': 3, 'all': 2, 'unclear': 2
    }
    confidence_map = {'low': 1, 'medium': 2, 'high': 3}

    all_results = []
    for app_id in df['id']:
        if app_id in llm_cache:
            entry = llm_cache[app_id]
            all_results.append({
                'id': app_id,
                'llm_partner_probability': entry.get('partner_probability', 0.3),
                'llm_company_maturity': maturity_map.get(entry.get('inferred_company_maturity', 'unclear'), 2),
                'llm_technical_sophistication': entry.get('inferred_technical_sophistication', 3),
                'llm_business_model': business_model_map.get(entry.get('inferred_business_model', 'unclear'), 2),
                'llm_merchant_segment': segment_map.get(entry.get('inferred_merchant_segment', 'unclear'), 2),
                'llm_confidence': confidence_map.get(entry.get('confidence', 'low'), 1),
                'llm_fit_signal_count': len(entry.get('partnership_fit_signals', [])),
                'llm_risk_flag_count': len(entry.get('partnership_risk_flags', [])),
                'llm_rag_use_case_count': len(entry.get('rag_use_cases', [])),
                'reasoning': entry.get('reasoning', ''),
                'rag_use_cases': entry.get('rag_use_cases', []),
                'partnership_fit_signals': entry.get('partnership_fit_signals', []),
                'partnership_risk_flags': entry.get('partnership_risk_flags', []),
                'inferred_company_maturity': entry.get('inferred_company_maturity', 'unclear'),
                'inferred_technical_sophistication': entry.get('inferred_technical_sophistication', 3),
                'inferred_business_model': entry.get('inferred_business_model', 'unclear'),
                'inferred_merchant_segment': entry.get('inferred_merchant_segment', 'unclear'),
            })
        else:
            # Default values for unscored apps
            all_results.append({
                'id': app_id,
                'llm_partner_probability': 0.3,
                'llm_company_maturity': 2,
                'llm_technical_sophistication': 3,
                'llm_business_model': 2,
                'llm_merchant_segment': 2,
                'llm_confidence': 1,
                'llm_fit_signal_count': 0,
                'llm_risk_flag_count': 1,
                'llm_rag_use_case_count': 0,
                'reasoning': 'Not scored',
                'rag_use_cases': [],
                'partnership_fit_signals': [],
                'partnership_risk_flags': ['not_scored'],
                'inferred_company_maturity': 'unclear',
                'inferred_technical_sophistication': 3,
                'inferred_business_model': 'unclear',
                'inferred_merchant_segment': 'unclear',
            })

    return pd.DataFrame(all_results)


def merge_llm_scores(df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    """Merge LLM scores with main DataFrame."""

    # Columns to merge
    merge_cols = [
        'id', 'llm_partner_probability', 'llm_company_maturity',
        'llm_technical_sophistication', 'llm_business_model',
        'llm_merchant_segment', 'llm_confidence',
        'llm_fit_signal_count', 'llm_risk_flag_count', 'llm_rag_use_case_count',
        'reasoning', 'rag_use_cases', 'partnership_fit_signals', 'partnership_risk_flags',
        'inferred_company_maturity', 'inferred_technical_sophistication',
        'inferred_business_model', 'inferred_merchant_segment'
    ]

    scores_subset = scores_df[[c for c in merge_cols if c in scores_df.columns]]
    merged = df.merge(scores_subset, on='id', how='left')

    # Fill missing values
    fill_values = {
        'llm_partner_probability': 0.3,
        'llm_company_maturity': 2,
        'llm_technical_sophistication': 3,
        'llm_business_model': 2,
        'llm_merchant_segment': 2,
        'llm_confidence': 1,
        'llm_fit_signal_count': 0,
        'llm_risk_flag_count': 0,
        'llm_rag_use_case_count': 0,
    }

    for col, val in fill_values.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)

    return merged


# ============================================================================
# Quality Filtering
# ============================================================================

def filter_high_confidence_labels(
    scores_df: pd.DataFrame,
    min_confidence: str = 'medium',
    min_probability_variance: float = 0.1
) -> pd.DataFrame:
    """
    Filter LLM labels by confidence for higher quality training data.

    Best practice: Use confidence thresholds to improve label quality.
    Reference: https://www.refuel.ai/blog-posts/llm-labeling-technical-report
    """

    confidence_order = {'low': 1, 'medium': 2, 'high': 3}
    min_conf_value = confidence_order.get(min_confidence, 2)

    # Filter by confidence
    mask = scores_df['llm_confidence'] >= min_conf_value

    # Also filter out extreme probabilities that might indicate uncertainty
    mask &= (scores_df['llm_partner_probability'] >= min_probability_variance)
    mask &= (scores_df['llm_partner_probability'] <= 1 - min_probability_variance)

    filtered = scores_df[mask].copy()

    logger.info(f"Filtered from {len(scores_df)} to {len(filtered)} high-confidence labels "
                f"(min_confidence={min_confidence})")

    return filtered


if __name__ == "__main__":
    # Test with small sample
    from .data_loader import load_enriched_apps
    from .feature_engineering import extract_features

    df = load_enriched_apps()
    df = extract_features(df)

    # Test with 3 high-value apps
    test_df = df[(df['rating'] >= 4.5) & (df['reviews_count'] >= 100)].head(3)
    print(f"Testing with {len(test_df)} apps: {test_df['title'].tolist()}")

    scores = score_apps_with_llm(test_df, use_cache=True)
    print("\nResults:")
    print(scores[['id', 'llm_partner_probability', 'llm_confidence', 'reasoning']].head())
