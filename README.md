# ISV Partner Propensity Model

Identify potential RAG API partners from Shopify App Store companies using LLM-to-ML knowledge distillation.

## Overview

This pipeline uses GPT-4o to score a stratified sample of Shopify apps for partnership potential, then trains a LightGBM model to score the entire catalog at scale.

```
12,000 Shopify Apps
       ↓
Stratified Sample (1,000)
       ↓
GPT-4o Scoring (Teacher)
       ↓
LightGBM Training (Student)
       ↓
Score All 12,000 Apps
       ↓
Ranked Partner Candidates
```

## Results

- **11,951 apps scored** with partner probability
- **AUC-ROC: 0.9575** (excellent classification)
- **85.6% knowledge retention** from LLM to ML model
- **1,196 Tier-1 candidates** identified for outreach

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/yourusername/shopify-partner-propensity.git
cd shopify-partner-propensity

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit with your OpenAI API key
nano .env
```

### 3. Add Data

Place your Shopify app data in `Shopify App Store Data/` directory:
- `apps_data.csv` - Main app metadata
- `apps_categories.csv` - App category mappings
- `categories.csv` - Category definitions

### 4. Run

```bash
# Basic run (1000 samples, 30 optimization trials)
python3 main.py --sample-size 1000 --optuna-trials 30

# Skip LLM scoring (use cached results)
python3 main.py --skip-llm --optuna-trials 30

# Full options
python3 main.py --sample-size 2000 --optuna-trials 50 --calibration sigmoid
```

## Output Files

After running, find results in `output/`:

| File | Description |
|------|-------------|
| `partner_candidates_ranked_*.csv` | All apps ranked by probability |
| `tier_1_candidates.csv` | Top priority outreach list |
| `tier_2_candidates.csv` | Mid-tier candidates |
| `tier_3_candidates.csv` | Lower priority |
| `partner_propensity_report_*.json` | Full report with metrics |
| `feature_importance.csv` | SHAP feature importance |
| `training_metrics.json` | Model performance metrics |

## Architecture

```
src/
├── model_trainer.py    # LightGBM training with SHAP, Optuna, calibration
├── llm_scorer.py       # GPT-4o scoring with structured outputs
├── feature_engineer.py # 40 features from app metadata
├── data_loader.py      # Data loading and preprocessing
└── report_generator.py # Output generation
```

### Key Features

- **SHAP TreeExplainer** for interpretable feature importance
- **Optuna Bayesian optimization** with TPE sampler
- **Probability calibration** (Isotonic, Sigmoid, Venn-ABERS)
- **Dual-signal confidence filtering** (LLKD-inspired)
- **Knowledge Retention Score (KRS)** for distillation quality
- **5-fold stratified cross-validation**

## Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| AUC-ROC | 0.9575 | Excellent ranking |
| Brier Score | 0.095 | Good calibration |
| ECE | 0.070 | Well-calibrated |
| KRS | 0.856 | 85.6% knowledge retention |

## Top Feature Drivers

| Feature | SHAP % |
|---------|--------|
| `llm_technical_sophistication` | 35.0% |
| `llm_rag_use_case_count` | 10.6% |
| `llm_merchant_segment` | 6.2% |
| `developer_avg_rating` | 5.3% |
| `traction_score` | 5.0% |

## CLI Options

```
--sample-size N       Number of apps to score with LLM (default: 500)
--optuna-trials N     Hyperparameter optimization trials (default: 50)
--calibration METHOD  Probability calibration: isotonic|sigmoid|venn_abers
--skip-llm            Skip LLM scoring, use cached results
--skip-sentiment      Skip sentiment analysis
--no-batch-api        Use synchronous API instead of batch
```

## Cost

| Sample Size | LLM Cost | Time |
|-------------|----------|------|
| 500 | ~$5-8 | ~15 min |
| 1,000 | ~$10-15 | ~30 min |
| 2,000 | ~$20-30 | ~60 min |

ML inference for all 12,000 apps takes <1 second.

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o access)
- ~8GB RAM for training

## License

MIT

## See Also

- [SUMMARY.md](SUMMARY.md) - Detailed implementation summary
