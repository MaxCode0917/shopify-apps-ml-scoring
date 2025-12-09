# ISV Partner Propensity Model v3.0 - Implementation Summary

## Project Overview

**Objective**: Identify potential CustomGPT.ai RAG API partners from ~12,000 Shopify app companies using LLM-to-ML knowledge distillation.

**Approach**: Use GPT-4o to score a stratified sample of apps, train a LightGBM model on those labels, then score all 12,000 apps at scale.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Flow                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load 11,951 Shopify Apps                                    │
│         ↓                                                        │
│  2. Stratified Sampling (1,018 apps)                            │
│         ↓                                                        │
│  3. GPT-4o Scoring (LLM Teacher)                                │
│     • partner_probability (0-1)                                  │
│     • technical_sophistication (1-5)                            │
│     • rag_use_case_count (0-10)                                 │
│     • merchant_segment, confidence, reasoning                    │
│         ↓                                                        │
│  4. Feature Engineering (40 features)                           │
│         ↓                                                        │
│  5. LightGBM Training (ML Student)                              │
│     • Optuna Bayesian optimization (30 trials)                  │
│     • 5-fold cross-validation                                    │
│     • Sigmoid probability calibration                           │
│         ↓                                                        │
│  6. Score All 11,951 Apps                                       │
│         ↓                                                        │
│  7. Tier Classification & Export                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research & Best Practices Implemented

### 2025 Industry Standards Applied

| Area | Implementation | Source/Rationale |
|------|----------------|------------------|
| **Knowledge Distillation** | LLKD-inspired dual-signal confidence filtering | Latest LLM-to-ML distillation research |
| **Feature Importance** | SHAP TreeExplainer | Gold standard for interpretable ML |
| **Calibration** | Isotonic, Sigmoid (Platt), Venn-ABERS options | Probability calibration best practices |
| **Hyperparameter Tuning** | Optuna with TPE sampler | State-of-the-art Bayesian optimization |
| **Evaluation Metrics** | KRS, ECE, SmoothECE, Brier Score | Comprehensive distillation quality metrics |

### Key Metrics Explained

| Metric | Purpose | Target |
|--------|---------|--------|
| **AUC-ROC** | Classification ranking quality | >0.9 excellent |
| **KRS (Knowledge Retention Score)** | How much LLM knowledge ML retained | >0.8 good |
| **ECE (Expected Calibration Error)** | Probability accuracy | <0.1 well-calibrated |
| **Brier Score** | Overall probability error | <0.15 good |

---

## Code Changes Made

### 1. Model Trainer Rewrite (`src/model_trainer.py`)

**Version**: 3.0.0 with 2025 best practices

#### Major Additions:

```python
# SHAP Feature Importance
def compute_shap_importance(model, X, feature_names):
    """TreeExplainer for interpretable feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Returns both SHAP and native importance with percentages

# Dual-Signal Confidence Filtering (LLKD-inspired)
def filter_by_dual_signal(df, llm_confidence_col, prediction_col,
                          confidence_threshold, agreement_threshold):
    """Filter training data by LLM confidence + prediction agreement."""

# Knowledge Retention Score
def compute_krs(y_true, y_pred, threshold=0.5):
    """Measure how well ML model retains LLM's classification knowledge."""

# Venn-ABERS Calibration
def venn_abers_calibration(model, X_cal, y_cal, X_test):
    """Conservative probability estimates with validity guarantees."""

# Robust Stratification
def create_stratification_bins(y, n_bins=5, min_samples_per_bin=2):
    """Handle edge cases: few unique values, bins with <2 samples."""
```

#### Bug Fixes Applied:

1. **SHAP percentage display** - Fixed `nan%` when SHAP values sum to zero
2. **Stratification bins** - Fixed `ValueError` when bins have only 1 sample
3. **Zero-sum handling** - Graceful fallback to native importance

### 2. Requirements Update (`requirements.txt`)

```
# Added
shap>=0.42.0
```

---

## Final Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | 0.9575 | Excellent classification |
| **Brier Score** | 0.0949 | Good calibration |
| **ECE** | 0.070 | Well-calibrated probabilities |
| **KRS** | 0.856 | 85.6% knowledge retention |
| **CV RMSE** | 0.080 ± 0.004 | Stable across folds |

### Tier Distribution (11,951 apps scored)

| Tier | Count | Avg Probability | Avg Rating | % Paid Plans |
|------|-------|-----------------|------------|--------------|
| **Tier 1** | 1,196 | 0.281 | 4.81 | 96.8% |
| **Tier 2** | 3,585 | 0.135 | 4.39 | 71.0% |
| **Tier 3** | 7,170 | 0.108 | 1.89 | 55.8% |

### Top 10 Partner Candidates

| Rank | App | Developer | Probability | Rating | Reviews |
|------|-----|-----------|-------------|--------|---------|
| 1 | SmartBot: Leading AI Chatbot | BestChat | 62.0% | 4.8 | 219 |
| 2 | HelpCenter: AI FAQ+Help Center | Shark Byte, Inc | 60.8% | 4.7 | 1,336 |
| 3 | SearchPie SEO & Speed optimize | PieLab | 60.4% | 4.9 | 4,664 |
| 4 | Searchanise Search & Filter | Searchanise | 58.8% | 4.7 | 1,686 |
| 5 | AG Order Printer, PDF Invoices | Avada | 58.7% | 4.9 | 909 |
| 6 | AfterShip Personalization | AfterShip | 57.2% | 4.9 | 144 |
| 7 | Boost AI Search & Filter | Boost Commerce | 56.9% | 4.7 | 1,967 |
| 8 | StoreSEO: AI for SEO & Image | Storeware | 56.5% | 4.9 | 494 |
| 9 | OnePixel Facebook Pixel,TikTok | OneCommerce | 55.6% | 4.7 | 224 |
| 10 | Zotabox: Promote & Convert 20+ | Zotabox | 55.4% | 4.8 | 482 |

### Top Feature Drivers (SHAP Importance)

| Feature | SHAP % | Cumulative % |
|---------|--------|--------------|
| `llm_technical_sophistication` | 35.0% | 35.0% |
| `llm_rag_use_case_count` | 10.6% | 45.6% |
| `llm_merchant_segment` | 6.2% | 51.8% |
| `developer_avg_rating` | 5.3% | 57.1% |
| `traction_score` | 5.0% | 62.1% |
| `llm_confidence` | 4.8% | 66.9% |
| `rating` | 4.3% | 71.3% |

**Insight**: LLM-extracted features dominate (51.8% cumulative from top 3), validating the distillation approach.

---

## Output Files Generated

```
output/
├── ranked_apps_20251210_024518.csv      # All 11,951 apps ranked
├── tier_1_apps_20251210_024518.csv      # 1,196 top candidates
├── tier_2_apps_20251210_024518.csv      # 3,585 mid-tier apps
├── tier_3_apps_20251210_024518.csv      # 7,170 lower priority
├── partner_propensity_report_*.json     # Full report with metrics
├── feature_importance.csv               # SHAP + native importance
├── training_metrics.json                # Model performance metrics
└── lgbm_model_*.joblib                  # Trained model (reusable)
```

---

## Usage

### Basic Run
```bash
python3 main.py --sample-size 1000 --optuna-trials 30
```

### With Batch API (50% cost savings)
```bash
python3 main.py --sample-size 2000 --optuna-trials 50
```

### Skip LLM (use cached scores)
```bash
python3 main.py --skip-llm --optuna-trials 30
```

### Full Options
```bash
python3 main.py \
  --sample-size 1000 \
  --optuna-trials 30 \
  --calibration sigmoid \
  --no-batch-api \
  --skip-sentiment
```

---

## Cost Analysis

| Component | Cost per 1,000 apps | Your Run (1,018) |
|-----------|--------------------:|------------------:|
| GPT-4o scoring | ~$10-15 | ~$10-15 |
| ML inference | ~$0.01 | ~$0.01 |
| **Total** | ~$10-15 | ~$10-15 |

**ROI**: Score 12,000 apps for the cost of scoring 1,000 with LLM alone.

---

## Technical Specifications

### Train/Test Split
- **Total samples**: 1,018 LLM-labeled apps
- **Test set**: 20% (~204 samples)
- **Train set**: 80% (~814 samples)
  - Validation: 20% (~163 samples) for early stopping
  - Training: ~651 samples
- **Features**: 40
- **Samples/features ratio**: 25:1 (healthy)

### Model Configuration
- **Algorithm**: LightGBM (gradient boosting)
- **Optimization**: Optuna TPE sampler, 30 trials
- **Calibration**: Sigmoid (Platt scaling)
- **Cross-validation**: 5-fold stratified

---

## Validation Checklist

- [x] SHAP feature importance analysis
- [x] Dual-signal confidence filtering (LLKD-inspired)
- [x] Venn-ABERS calibration option
- [x] KRS and SmoothECE metrics
- [x] Robust stratification handling
- [x] Optuna Bayesian optimization
- [x] Probability calibration
- [x] 5-fold cross-validation
- [x] Comprehensive evaluation metrics

---

## Next Steps (Optional)

1. **Outreach**: Contact Tier 1 candidates starting with top 10
2. **Refresh**: Re-run monthly to catch new apps
3. **Feedback loop**: Track conversion rates to improve model
4. **Scale**: Increase sample size to 2,000+ for marginal gains

---

*Generated: 2025-12-10*
*Model Version: 3.0.0*
*Pipeline: ISV Partner Propensity Model*
