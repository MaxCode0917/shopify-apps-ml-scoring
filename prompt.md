# ISV Partner Propensity Model: Requirements Document

## Executive Summary

### Why

We have 12,000 Shopify app companies. We want to identify which would be high-value ISV partner candidates for direct outreach and campaigns. No historical partner conversion data exists to train on.

We'll use an LLM to score companies based on signals in our data (app quality, reviews, pricing, description), then distill into a fast ML model for ranking all 12K.

### How

1. **Sample**: Take 2,000-2,500 companies from the 12K
2. **Score with LLM**: Claude analyzes each app's profile and estimates ISV partner fit
3. **Train ML model**: Use LLM outputs as labels to train XGBoost/LogReg
4. **Score all**: Run trained model on full 12K (instant, free)
5. **Validate & iterate**: As real partner signups happen, retrain with actual labels

### What

A Python pipeline that:
- Scores a sample via Anthropic API
- Trains a scikit-learn model on synthetic labels
- Outputs ranked list of all 12K companies by partner probability
- Segments into tiers for outreach prioritization

---

## Data Schema

| Column | Description | ML Signal Value |
|--------|-------------|-----------------|
| `id` | Unique identifier | Key only |
| `url` | App store URL | Could extract app age/category |
| `title` | App name | LLM context |
| `developer` | Company/developer name | LLM context |
| `developer_link` | Developer website/profile | Could enrich |
| `rating` | App rating (likely 1-5) | **Strong signal** — quality indicator |
| `reviews_count` | Number of reviews | **Strong signal** — scale/traction |
| `description_raw` | Raw description text | LLM analysis |
| `description` | Cleaned description | **Rich signal** — LLM analyzes for sophistication |
| `pricing_hint` | Pricing model/tier | **Strong signal** — business model maturity |
| `lastmod` | Last modified date | **Signal** — active development |

**Key insight**: This dataset is richer than typical firmographic data. The `description`, `rating`, `reviews_count`, and `pricing_hint` give the LLM substantial signal to reason about partner fit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                INPUT: 12K Shopify App Companies             │
│    (title, developer, rating, reviews, description, etc.)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               STEP 1: Sample Selection                      │
│                                                             │
│  - Stratified sample of 2,000-2,500 companies               │
│  - Stratify by: rating bands, review count bands            │
│  - Ensure coverage across pricing tiers                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               STEP 2: LLM Scoring                           │
│                                                             │
│  For each company in sample:                                │
│  - Send: title, developer, rating, reviews, description,    │
│          pricing_hint, lastmod                              │
│  - Get: partner_probability, reasoning, inferred attributes │
│                                                             │
│  Cost estimate: ~$8-15 for 2.5K (descriptions add tokens)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               STEP 3: Model Training                        │
│                                                             │
│  Features:                                                  │
│  - Numeric: rating, reviews_count, days_since_lastmod       │
│  - Categorical: pricing_hint (encoded)                      │
│  - LLM-inferred: company_maturity, technical_depth, etc.    │
│                                                             │
│  Labels: LLM partner probability scores                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               STEP 4: Full Scoring + Segmentation           │
│                                                             │
│  - Score all 12K companies                                  │
│  - Segment into tiers:                                      │
│    • Tier 1 (Top 500): Direct outreach                      │
│    • Tier 2 (500-2000): Campaign target                     │
│    • Tier 3 (2000+): Nurture/passive                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Specification

### 1. LLM Scoring Module

#### Prompt Template

```
You are an expert at evaluating software companies for B2B partnership potential.

## Context
We run an ISV (Independent Software Vendor) Partner Program for [YOUR PRODUCT].
We're looking for Shopify app developers who would be strong partners for:
- Technical integrations
- Co-marketing opportunities  
- Joint customer success

## Ideal ISV Partner Profile
[FILL IN: What makes a great partner for you?]
- Scale: [e.g., established apps with real traction, not hobbyist projects]
- Quality: [e.g., well-maintained, good reviews, professional]
- Business model: [e.g., paid apps or freemium with clear monetization]
- Technical depth: [e.g., sophisticated functionality, not simple widgets]
- Alignment: [e.g., serves similar merchant segments as us]

## App to Evaluate

**App Title**: {title}
**Developer**: {developer}
**Rating**: {rating}/5
**Reviews**: {reviews_count}
**Pricing**: {pricing_hint}
**Last Updated**: {lastmod}

**Description**:
{description}

## Task

Analyze this Shopify app and its developer as a potential ISV partner candidate.

Consider:
1. Does this appear to be a serious business or a side project?
2. Does the app quality/traction suggest a mature organization?
3. Would this developer likely have resources for partnership activities?
4. Is there potential strategic alignment with our product?

Respond ONLY with valid JSON:
{
  "partner_probability": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences explaining your assessment>",
  "inferred_company_maturity": "<hobbyist|small_team|established_business|agency|enterprise>",
  "inferred_technical_sophistication": <int 1-5>,
  "inferred_business_model": "<free_only|freemium|paid|enterprise|unclear>",
  "inferred_merchant_segment": "<smb|mid_market|enterprise|all|unclear>",
  "partnership_fit_signals": ["<signal1>", "<signal2>"],
  "partnership_risk_flags": ["<flag1>", "<flag2>"],
  "confidence": "<low|medium|high>"
}
```

#### Why This Prompt Works

The LLM can infer a lot from app store data:
- **Description text** → technical sophistication, target market, feature depth
- **Rating + reviews** → quality and traction
- **Pricing hint** → business model maturity
- **Last modified** → active maintenance
- **Developer name** → agency vs indie vs company

These inferences become features for the ML model.

#### API Configuration

```python
MODEL = "claude-opus-4-5-20251101"  # Highest quality reasoning
MAX_TOKENS = 400  # Slightly higher due to richer output
TEMPERATURE = 0.3

# Rate limiting
REQUESTS_PER_MINUTE = 50
BATCH_SIZE = 50

# Token estimate per request
# ~500 tokens input (prompt + description) + ~300 tokens output = ~800 tokens
# 2,500 companies × 500 input tokens × $5/1M ≈ $6.25 input
# 2,500 companies × 300 output tokens × $25/1M ≈ $18.75 output
# Total: ~$25 per run
```

### 2. Feature Engineering

#### From Raw Data

```python
features = {
    # Direct numeric
    'rating': float,  # Already numeric
    'reviews_count': int,  # Already numeric
    
    # Derived
    'days_since_lastmod': int,  # Today - lastmod
    'has_pricing': bool,  # pricing_hint is not null/empty
    'log_reviews': float,  # log(reviews_count + 1) for scale normalization
    
    # From pricing_hint (categorical encoding)
    'pricing_tier': str,  # Parse into: free, freemium, paid, enterprise
    
    # Text-derived (optional, simple)
    'description_length': int,
    'description_has_integration_keywords': bool,  # "API", "integrate", etc.
}
```

#### From LLM Inference

```python
llm_features = {
    'inferred_company_maturity': str,  # Ordinal encode: hobbyist=1, ..., enterprise=5
    'inferred_technical_sophistication': int,  # Already 1-5
    'inferred_business_model': str,  # One-hot or ordinal
    'inferred_merchant_segment': str,  # One-hot or ordinal
    'confidence': str,  # Ordinal: low=1, medium=2, high=3
    'num_fit_signals': int,  # len(partnership_fit_signals)
    'num_risk_flags': int,  # len(partnership_risk_flags)
}
```

### 3. Model Training

#### Recommended Model

```python
from sklearn.ensemble import GradientBoostingRegressor

# Use regression since we want probability scores, not just binary
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Alternative: Classification with probability output
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(...)
# Use model.predict_proba() for scores
```

#### Training Split

```python
# Stratify by LLM score bands to ensure good coverage
score_bands = pd.cut(llm_scores['partner_probability'], bins=[0, 0.3, 0.6, 1.0])
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=score_bands,
    random_state=42
)
```

### 4. Output Specification

#### Primary Output: Ranked Companies CSV

```csv
id,developer,title,partner_probability,rank,tier,rating,reviews_count,key_signals
abc123,Acme Apps,Super Inventory,0.84,12,1,4.8,892,"established_business,high_reviews,paid_model"
def456,Beta Dev,Quick Reports,0.71,156,1,4.5,234,"technical_depth,active_development"
ghi789,Solo Builder,Simple Widget,0.23,8934,3,3.2,12,"hobbyist,low_traction"
...
```

#### Tier Definitions

| Tier | Probability Range | Count (approx) | Action |
|------|-------------------|----------------|--------|
| 1 | > 0.7 | ~500-800 | Direct sales outreach |
| 2 | 0.4 - 0.7 | ~1500-2500 | Targeted campaigns |
| 3 | < 0.4 | Remainder | Passive/nurture |

Adjust thresholds based on your outreach capacity.

#### Secondary Output: Feature Importance

```json
{
  "feature_importance": {
    "inferred_company_maturity": 0.24,
    "log_reviews": 0.18,
    "inferred_technical_sophistication": 0.16,
    "rating": 0.12,
    "days_since_lastmod": 0.09,
    ...
  }
}
```

This tells you which signals matter most — useful for refining your ICP.

---

## Next Steps

- [ ] Load and clean 12K company CSV
- [ ] Implement LLM scoring with rate limiting and error handling
- [ ] Basic feature engineering (numeric + LLM-inferred)
- [ ] Train Gradient Boosting model
- [ ] Score all 12K, output ranked CSV with tiers

---

## Configuration

```yaml
# ISV Partner Program context
partner_program:
  name: "Your ISV Partner Program"
  product_description: "Brief description of your product"
  ideal_partner:
    scale: "Established apps with real traction"
    quality: "4+ rating, professional presentation"
    business_model: "Paid or freemium with monetization"
    alignment: "Serves ecommerce merchants in X segment"

# Scoring settings
scoring:
  sample_size: 2500
  model: "claude-opus-4-5-20251101"  # Highest quality reasoning
  # Alternative: "claude-sonnet-4-5-20250929" (~$15, still excellent)
  
# Tier thresholds
tiers:
  tier_1_min: 0.70
  tier_2_min: 0.40

# Paths
paths:
  input: "data/shopify_apps.csv"
  output: "output/partner_candidates_ranked.csv"
  llm_cache: "cache/llm_responses.jsonl"
```

---

## Cost Estimate

| Item | Calculation | Cost |
|------|-------------|------|
| LLM scoring (2.5K apps) | 2,500 × ~500 input tokens × $5/1M + 2,500 × ~300 output tokens × $25/1M | ~$25 |
| With 2x runs for stability | $25 × 2 | ~$50 |
| Compute | Negligible | $0 |
| **Total (single run)** | | **~$25** |
| **Total (2x averaging)** | | **~$50** |

*Using Claude Opus 4.5 for highest quality reasoning on company evaluations.*

---

## Key Differences from Previous Project

| Aspect | Stripe Project | Shopify ISV Project |
|--------|---------------|---------------------|
| Dataset size | 27K | 12K |
| Key features | Category only | Rating, reviews, description, pricing |
| Target | Trial → Paid conversion | ISV partner signup |
| LLM signal richness | Low (little context) | **High** (descriptions are goldmine) |
| Expected accuracy | Moderate | Higher (richer signals) |

The app descriptions give the LLM much more to work with. Expect better quality scores.

---

## Open Questions

1. **Partner program details**: What does the ISV program offer? What do you need from partners? (Helps LLM assess fit)

2. **Existing partners**: Do you have any current ISV partners? Even 10-20 examples would help validate the model's rankings.

3. **Outreach capacity**: How many companies can you realistically contact? (Determines tier thresholds)

4. **Enrichment appetite**: Want to enrich with developer website data? Could add signal but adds complexity.

5. **Description quality**: Are the descriptions generally good, or lots of garbage/duplicates?

---

## Appendix: Sample LLM Analysis

**Input:**
```
Title: InventorySync Pro
Developer: ScaleOps Inc
Rating: 4.7
Reviews: 1,247
Pricing: From $49/month
Last Updated: 2024-01-10
Description: "Enterprise-grade inventory synchronization for multi-location 
retailers. Real-time sync across Shopify, Amazon, and warehouse systems. 
REST API included. SOC 2 compliant. Trusted by 500+ brands including..."
```

**LLM Output:**
```json
{
  "partner_probability": 0.82,
  "reasoning": "ScaleOps appears to be an established B2B SaaS company targeting mid-market/enterprise merchants. High review count, strong rating, and enterprise positioning (SOC 2, API) suggest a mature organization with partnership capacity. Multi-channel focus indicates potential integration synergies.",
  "inferred_company_maturity": "established_business",
  "inferred_technical_sophistication": 5,
  "inferred_business_model": "paid",
  "inferred_merchant_segment": "mid_market",
  "partnership_fit_signals": ["enterprise_positioning", "api_available", "high_traction", "multi_channel"],
  "partnership_risk_flags": [],
  "confidence": "high"
}
```

This is the kind of nuanced analysis that makes the approach work.
