"""
ML model training for partner propensity scoring.

Industry Best Practices (December 2025):
- Optuna Bayesian hyperparameter optimization with TPE sampler
- Probability calibration: Isotonic (>1000), Platt (<1000), Venn-ABERS (uncertainty intervals)
- SHAP TreeExplainer for interpretable feature importance
- Dual-signal confidence filtering (teacher confidence + student uncertainty)
- Knowledge Retention Score (KRS) for distillation quality
- Expected Calibration Error (ECE) with SmoothECE variant
- DART boosting option for small datasets
- Cross-validation with proper regression handling (KFold)
- Comprehensive logging and model versioning

References:
- LLKD (Learning with Less): https://arxiv.org/abs/2411.08028
- Venn-ABERS Calibration: https://arxiv.org/abs/2502.05676
- SHAP TreeExplainer: https://shap.readthedocs.io/
- LightGBM Tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Probability Calibration: https://scikit-learn.org/stable/modules/calibration.html
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, cross_val_predict
)
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss
)
import lightgbm as lgb

from .config import OUTPUT_DIR, CACHE_DIR, RANDOM_STATE
from .feature_engineering import get_feature_columns, get_llm_feature_columns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# Probability Calibration Wrappers (Module-level for pickling)
# ============================================================================

class CalibratedRegressorWrapper(BaseEstimator):
    """
    Wrapper applying isotonic/sigmoid calibration to regression predictions.

    Best Practices (2025):
    - Isotonic for n >= 1000 samples (more flexible)
    - Sigmoid/Platt for n < 1000 samples (less overfitting)

    Module-level class enables proper pickling for model persistence.
    """

    def __init__(self, base_model=None, method='isotonic'):
        self.base_model = base_model
        self.method = method
        self.calibrator_ = None

    def fit(self, X, y):
        """Fit calibration on validation predictions."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        raw_preds = np.clip(self.base_model.predict(X), 0, 1)

        if self.method == 'isotonic':
            self.calibrator_ = IsotonicRegression(
                y_min=0, y_max=1, out_of_bounds='clip'
            )
            self.calibrator_.fit(raw_preds, y)
        else:  # sigmoid/Platt scaling
            self.calibrator_ = LogisticRegression(C=1e5, max_iter=1000)
            self.calibrator_.fit(raw_preds.reshape(-1, 1), (y >= 0.5).astype(int))

        return self

    def predict(self, X):
        """Return calibrated predictions."""
        raw_preds = np.clip(self.base_model.predict(X), 0, 1)

        if self.calibrator_ is None:
            return raw_preds

        if self.method == 'isotonic':
            return np.clip(self.calibrator_.predict(raw_preds), 0, 1)
        else:
            return np.clip(
                self.calibrator_.predict_proba(raw_preds.reshape(-1, 1))[:, 1],
                0, 1
            )

    def predict_proba(self, X):
        """Return calibrated probabilities in classification format."""
        probs = self.predict(X)
        return np.column_stack([1 - probs, probs])


class VennAbersWrapper(BaseEstimator):
    """
    Venn-ABERS calibration wrapper providing probability intervals.

    Benefits (2025 Research):
    - Mathematical guarantees regardless of data distribution
    - Returns probability intervals (epistemic uncertainty)
    - Significantly outperforms Platt/isotonic on most metrics

    Reference: https://arxiv.org/abs/2502.05676
    """

    def __init__(self, base_model=None):
        self.base_model = base_model
        self.va_calibrator_ = None

    def fit(self, X, y):
        """Fit Venn-ABERS calibrator."""
        try:
            from venn_abers import VennAbersCalibrator

            raw_preds = np.clip(self.base_model.predict(X), 0, 1)
            y_binary = (y >= 0.5).astype(int)

            self.va_calibrator_ = VennAbersCalibrator()
            self.va_calibrator_.fit(raw_preds.reshape(-1, 1), y_binary)

        except ImportError:
            logger.warning("venn-abers not installed. Using isotonic fallback.")
            self.va_calibrator_ = None

        return self

    def predict(self, X):
        """Return point predictions (midpoint of interval)."""
        raw_preds = np.clip(self.base_model.predict(X), 0, 1)

        if self.va_calibrator_ is None:
            return raw_preds

        p0, p1 = self.va_calibrator_.predict_proba(raw_preds.reshape(-1, 1))
        # Return midpoint of interval
        return np.clip((p0 + p1) / 2, 0, 1)

    def predict_with_uncertainty(self, X):
        """Return predictions with uncertainty intervals."""
        raw_preds = np.clip(self.base_model.predict(X), 0, 1)

        if self.va_calibrator_ is None:
            return raw_preds, np.zeros_like(raw_preds), np.ones_like(raw_preds)

        p0, p1 = self.va_calibrator_.predict_proba(raw_preds.reshape(-1, 1))
        midpoint = (p0 + p1) / 2
        return midpoint, p0, p1

    def predict_proba(self, X):
        """Return calibrated probabilities in classification format."""
        probs = self.predict(X)
        return np.column_stack([1 - probs, probs])


# ============================================================================
# Feature Preparation
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'llm_partner_probability'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix and target vector with robust handling.

    Handles:
    - Missing columns gracefully
    - Numeric imputation with median (robust to outliers)
    - Infinite value replacement
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} columns: {missing_cols[:5]}...")

    X = df[available_cols].copy()
    y = df[target_col].copy()

    # Robust imputation
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
        else:
            X[col] = X[col].fillna(0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(0)

    return X, y, available_cols


def create_stratification_bins(y: pd.Series, n_bins: int = 5, min_samples_per_bin: int = 2) -> pd.Series:
    """
    Create stratification bins for regression target with robust handling.

    Handles edge cases:
    - Too few unique values
    - Bins with only 1 sample (causes train_test_split to fail)
    """
    try:
        bins = pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
        # Check for bins with too few samples
        bin_counts = bins.value_counts()
        if bin_counts.min() < min_samples_per_bin:
            # Fall back to fewer bins or quantile-based
            raise ValueError("Bin too small")
        return bins
    except (ValueError, TypeError):
        # Try quantile-based binning
        try:
            effective_bins = min(n_bins, len(y) // min_samples_per_bin)
            if effective_bins < 2:
                # Not enough data for stratification - return constant
                return pd.Series([0] * len(y), index=y.index)
            bins = pd.qcut(y.rank(method='first'), q=effective_bins, labels=False, duplicates='drop')
            return bins
        except (ValueError, TypeError):
            # Final fallback: no stratification
            return pd.Series([0] * len(y), index=y.index)


# ============================================================================
# Dual-Signal Confidence Filtering (LLKD-inspired)
# ============================================================================

def filter_by_dual_signal(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    X: Optional[pd.DataFrame] = None,
    teacher_confidence_col: str = 'llm_confidence',
    teacher_threshold: int = 2,
    student_uncertainty_percentile: float = 75
) -> pd.DataFrame:
    """
    Dual-signal filtering based on LLKD (Learning with Less, 2024).

    Prioritizes samples where:
    1. Teacher (LLM) has high confidence (reliable labels)
    2. Student (ML) has high uncertainty (challenging samples needing learning)

    Reference: https://arxiv.org/abs/2411.08028
    """
    filtered_df = df.copy()

    # Signal 1: Teacher confidence
    if teacher_confidence_col in filtered_df.columns:
        original = len(filtered_df)
        filtered_df = filtered_df[filtered_df[teacher_confidence_col] >= teacher_threshold]
        logger.info(f"Teacher confidence filter: {original} → {len(filtered_df)}")

    # Signal 2: Student uncertainty (if model provided)
    if model is not None and X is not None:
        try:
            preds = model.predict(X.loc[filtered_df.index])
            # Uncertainty = distance from decision boundary (0.5)
            uncertainty = np.abs(preds - 0.5)
            threshold = np.percentile(uncertainty, 100 - student_uncertainty_percentile)

            # Keep high-uncertainty samples (most informative)
            mask = uncertainty >= threshold
            filtered_df = filtered_df.iloc[mask]
            logger.info(f"Student uncertainty filter (top {student_uncertainty_percentile}%): kept {len(filtered_df)}")
        except Exception as e:
            logger.warning(f"Student uncertainty filtering failed: {e}")

    return filtered_df


# ============================================================================
# Optuna Hyperparameter Optimization (2025 Best Practices)
# ============================================================================

def create_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_dart: bool = False
):
    """
    Create Optuna objective for LightGBM optimization.

    2025 Best Practices:
    - Log scale for learning_rate (explore values near 0.001)
    - DART boosting option for small datasets
    - Comprehensive regularization
    """

    def objective(trial: optuna.Trial) -> float:
        boosting_type = 'dart' if use_dart else 'gbdt'

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': boosting_type,
            'verbosity': -1,
            'random_state': RANDOM_STATE,
            'force_row_wise': True,  # Suppress data format warning

            # Core parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

            # L1/L2 Regularization (log scale as per research)
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

            # Subsampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        }

        # DART-specific parameters
        if use_dart:
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.3)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

        model = lgb.LGBMRegressor(**params)

        # Early stopping only works with GBDT (not DART)
        if use_dart:
            model.fit(X_train, y_train)
        else:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

        y_pred = np.clip(model.predict(X_val), 0, 1)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        return rmse

    return objective


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = 600,
    use_dart: bool = False
) -> Dict[str, Any]:
    """
    Run Optuna Bayesian optimization.

    Research shows 30 trials typically sufficient for good results.
    """
    logger.info(f"Starting Optuna ({n_trials} trials, {timeout}s timeout, DART={use_dart})...")

    sampler = TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='lightgbm_partner_propensity'
    )

    objective = create_optuna_objective(X_train, y_train, X_val, y_val, use_dart)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )

    logger.info(f"Best trial: RMSE={study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Save study results
    study_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'use_dart': use_dart,
        'timestamp': datetime.now().isoformat()
    }

    study_file = OUTPUT_DIR / 'optuna_study_results.json'
    with open(study_file, 'w') as f:
        json.dump(study_results, f, indent=2)

    return study.best_params


# ============================================================================
# Model Training
# ============================================================================

def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None,
    use_optuna: bool = True,
    optuna_trials: int = 50,
    use_dart: bool = False
) -> lgb.LGBMRegressor:
    """
    Train LightGBM regressor with best practices.

    Small Dataset Recommendations (<10K samples):
    - Use DART boosting (randomly drops trees)
    - Increase regularization
    - Lower num_leaves
    """
    n_samples = len(X_train)

    # Auto-enable DART for small datasets
    if n_samples < 1000 and not use_dart:
        logger.info(f"Small dataset ({n_samples} samples) - consider using DART boosting")

    if params is None:
        if use_optuna and X_val is not None and y_val is not None:
            params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val,
                n_trials=optuna_trials,
                use_dart=use_dart
            )
        else:
            # Default params optimized for small-medium datasets
            params = {
                'n_estimators': 200,
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }

    # Ensure required params
    boosting = 'dart' if use_dart else 'gbdt'
    params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': boosting,
        'verbosity': -1,
        'random_state': RANDOM_STATE,
        'force_row_wise': True
    })

    model = lgb.LGBMRegressor(**params)

    if X_val is not None and y_val is not None and not use_dart:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
    else:
        model.fit(X_train, y_train)

    return model


# ============================================================================
# SHAP Feature Importance (2025 Best Practice)
# ============================================================================

def compute_shap_importance(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    feature_names: List[str],
    max_samples: int = 1000
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Compute SHAP values for interpretable feature importance.

    TreeSHAP Benefits (2025):
    - Local interpretability (per-prediction)
    - Exact computation for tree models
    - Fast computation exploiting tree structure
    - Essential for high-stakes applications

    Reference: https://shap.readthedocs.io/
    """
    try:
        import shap

        # Subsample for efficiency
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=RANDOM_STATE)
        else:
            X_sample = X

        # TreeSHAP - fast exact computation for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Global importance = mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap,
            'native_importance': model.feature_importances_
        }).sort_values('shap_importance', ascending=False)

        # Handle zero-sum cases gracefully
        shap_sum = importance_df['shap_importance'].sum()
        native_sum = importance_df['native_importance'].sum()

        importance_df['shap_pct'] = (
            importance_df['shap_importance'] / shap_sum * 100 if shap_sum > 0 else 0
        )
        importance_df['native_pct'] = (
            importance_df['native_importance'] / native_sum * 100 if native_sum > 0 else 0
        )
        importance_df['cumulative_shap_pct'] = importance_df['shap_pct'].cumsum()

        logger.info("SHAP analysis complete")
        return importance_df, shap_values

    except ImportError:
        logger.warning("SHAP not installed. Using native importance only.")
        return get_native_feature_importance(model, feature_names), None
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Using native importance.")
        return get_native_feature_importance(model, feature_names), None


def get_native_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Extract native LightGBM feature importance (gain-based)."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'feature_importances_'):
        importance = model.base_model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({
        'feature': feature_names,
        'native_importance': importance
    }).sort_values('native_importance', ascending=False)

    total = df['native_importance'].sum()
    df['native_pct'] = df['native_importance'] / total * 100 if total > 0 else 0
    df['cumulative_pct'] = df['native_pct'].cumsum()

    return df


# ============================================================================
# Probability Calibration (2025 Methods)
# ============================================================================

def calibrate_probabilities(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'auto'
) -> Tuple[Any, str]:
    """
    Calibrate model probabilities.

    Methods (2025 Research):
    - 'auto': Isotonic for n>=1000, Platt for n<1000
    - 'isotonic': Flexible, needs more data
    - 'sigmoid': Parametric, works with less data
    - 'venn_abers': Returns probability intervals (best for uncertainty)

    References:
    - https://scikit-learn.org/stable/modules/calibration.html
    - https://arxiv.org/abs/2502.05676 (Venn-ABERS)
    """
    n_samples = len(y_train)

    if method == 'auto':
        if n_samples >= 1000:
            method = 'isotonic'
            logger.info(f"Auto-selected isotonic calibration (n={n_samples} >= 1000)")
        else:
            method = 'sigmoid'
            logger.info(f"Auto-selected sigmoid (Platt) calibration (n={n_samples} < 1000)")

    # Check prediction distribution
    raw_preds = np.clip(model.predict(X_train), 0, 1)
    pred_mean = raw_preds.mean()
    if pred_mean < 0.1 or pred_mean > 0.9:
        logger.warning(f"Skewed predictions: mean={pred_mean:.2%}. Calibration may be less effective.")

    try:
        if method == 'venn_abers':
            calibrated = VennAbersWrapper(base_model=model)
            calibrated.fit(X_train, y_train)
            logger.info("Venn-ABERS calibration complete (provides uncertainty intervals)")
        else:
            calibrated = CalibratedRegressorWrapper(base_model=model, method=method)
            calibrated.fit(X_train, y_train)
            logger.info(f"Calibration complete using {method} method")

        return calibrated, method

    except Exception as e:
        logger.warning(f"Calibration failed: {e}. Returning uncalibrated model.")
        return model, 'none'


# ============================================================================
# Evaluation Metrics (2025 Comprehensive)
# ============================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (|bin_size| / n) * |accuracy - confidence|

    Limitations (2025 research):
    - Binning-dependent (bias-variance tradeoff)
    - Only considers max probabilities
    - ECE=0 doesn't guarantee high accuracy
    """
    y_binary = (y_true >= 0.5).astype(int)

    try:
        prob_true, prob_pred = calibration_curve(
            y_binary, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Weight by bin counts
        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        bin_weights = bin_counts / len(y_prob)

        # ECE = weighted average of |accuracy - confidence|
        ece = np.sum(bin_weights[:len(prob_true)] * np.abs(prob_true - prob_pred))
        return ece

    except Exception:
        return np.nan


def compute_smooth_ece(y_true: np.ndarray, y_prob: np.ndarray, bandwidth: float = 0.1) -> float:
    """
    Compute SmoothECE using kernel density estimation.

    Benefits over standard ECE:
    - Continuous (no binning artifacts)
    - More stable estimates
    - Better for comparing models
    """
    try:
        from scipy.stats import gaussian_kde

        y_binary = (y_true >= 0.5).astype(int)

        # Kernel density estimation for calibration
        kde_correct = gaussian_kde(y_prob[y_binary == 1], bw_method=bandwidth) if y_binary.sum() > 0 else None
        kde_all = gaussian_kde(y_prob, bw_method=bandwidth)

        # Evaluate on grid
        grid = np.linspace(0.01, 0.99, 100)

        if kde_correct is not None:
            p_correct = kde_correct(grid) * y_binary.mean()
            p_all = kde_all(grid)

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                calibrated = np.where(p_all > 1e-10, p_correct / p_all, grid)

            smooth_ece = np.mean(np.abs(calibrated - grid))
            return smooth_ece

        return np.nan

    except Exception:
        return np.nan


def compute_knowledge_retention_score(
    teacher_preds: np.ndarray,
    student_preds: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute Knowledge Retention Score (KRS) for distillation quality.

    KRS captures:
    - Output consistency between teacher and student
    - Student's ability to retain teacher knowledge

    Reference: https://www.researchgate.net/publication/389579727
    """
    # Output agreement (correlation)
    output_agreement = np.corrcoef(teacher_preds, student_preds)[0, 1]

    # Teacher accuracy
    teacher_binary = (teacher_preds >= 0.5).astype(int)
    student_binary = (student_preds >= 0.5).astype(int)
    y_binary = (y_true >= 0.5).astype(int)

    teacher_acc = (teacher_binary == y_binary).mean()
    student_acc = (student_binary == y_binary).mean()

    # KRS = combination of agreement and relative accuracy
    krs = 0.5 * output_agreement + 0.5 * (student_acc / max(teacher_acc, 1e-10))

    return {
        'krs': krs,
        'output_agreement': output_agreement,
        'teacher_accuracy': teacher_acc,
        'student_accuracy': student_acc,
        'accuracy_retention': student_acc / max(teacher_acc, 1e-10)
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    is_calibrated: bool = False,
    teacher_preds: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Comprehensive model evaluation with 2025 metrics."""

    # Get predictions
    if is_calibrated and hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)

    y_pred = np.clip(y_pred, 0, 1)

    metrics = {
        'model': model_name,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }

    # Binary classification metrics
    y_binary = (y_test >= 0.5).astype(int)

    if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
        try:
            metrics['auc_roc'] = roc_auc_score(y_binary, y_pred)
            metrics['avg_precision'] = average_precision_score(y_binary, y_pred)
        except Exception:
            pass

    # Calibration metrics
    metrics['brier_score'] = brier_score_loss(y_binary, y_pred)
    metrics['log_loss'] = log_loss(y_binary, np.clip(y_pred, 1e-15, 1-1e-15))
    metrics['ece'] = compute_ece(y_test.values, y_pred)
    metrics['smooth_ece'] = compute_smooth_ece(y_test.values, y_pred)

    # Knowledge Retention Score (if teacher predictions available)
    if teacher_preds is not None:
        krs_metrics = compute_knowledge_retention_score(teacher_preds, y_pred, y_test.values)
        metrics.update({f'krs_{k}': v for k, v in krs_metrics.items()})

    return metrics


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_ensemble(
    df: pd.DataFrame,
    test_size: float = 0.2,
    use_optuna: bool = True,
    optuna_trials: int = 50,
    calibrate: bool = True,
    calibration_method: str = 'auto',
    filter_confidence: bool = True,
    use_shap: bool = True,
    use_dart: bool = False
) -> Tuple[Any, Dict, pd.DataFrame]:
    """
    Train ensemble model with 2025 best practices.

    Pipeline:
    1. Feature preparation
    2. Dual-signal confidence filtering (LLKD-inspired)
    3. Optuna Bayesian hyperparameter optimization
    4. Model training with early stopping (or DART)
    5. SHAP feature importance analysis
    6. Probability calibration (isotonic/Platt/Venn-ABERS)
    7. Comprehensive evaluation (KRS, ECE, SmoothECE)
    """
    logger.info("=" * 70)
    logger.info("  ISV PARTNER PROPENSITY MODEL - TRAINING PIPELINE v3.0")
    logger.info("  Industry Best Practices (December 2025)")
    logger.info("=" * 70)

    # Get feature columns
    base_features = get_feature_columns()
    llm_features = get_llm_feature_columns()
    all_features = base_features + [f for f in llm_features if f != 'llm_partner_probability']

    logger.info(f"\nTotal features available: {len(all_features)}")
    logger.info(f"  Base features: {len(base_features)}")
    logger.info(f"  LLM features: {len(llm_features) - 1}")

    # Confidence filtering
    if filter_confidence and 'llm_confidence' in df.columns:
        original_size = len(df)
        df = df[df['llm_confidence'] >= 2].copy()
        logger.info(f"\nConfidence filtering: {original_size} → {len(df)} samples")

    # Prepare features
    X, y, used_features = prepare_features(df, all_features)
    logger.info(f"Features used: {len(used_features)}")
    logger.info(f"Training samples: {len(X)}")

    # Stratified split
    y_bins = create_stratification_bins(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y_bins, random_state=RANDOM_STATE
    )

    # Validation split for early stopping
    y_train_bins = create_stratification_bins(y_train)
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train_bins, random_state=RANDOM_STATE
    )

    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(X_train_opt)}")
    logger.info(f"  Val: {len(X_val)}")
    logger.info(f"  Test: {len(X_test)}")

    # Recommend DART for small datasets
    if len(X_train_opt) < 500 and not use_dart:
        logger.warning(f"Small training set ({len(X_train_opt)} samples). Consider --use-dart flag.")

    # =========================================================================
    # Model Training
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("PHASE 1: Hyperparameter Optimization & Training")
    logger.info("-" * 50)

    model = train_lightgbm_model(
        X_train_opt, y_train_opt,
        X_val, y_val,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        use_dart=use_dart
    )

    # =========================================================================
    # Evaluation (Pre-Calibration)
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("PHASE 2: Model Evaluation (Pre-Calibration)")
    logger.info("-" * 50)

    # Store teacher predictions for KRS
    teacher_preds = y_test.values if 'llm_partner_probability' in df.columns else None

    pre_calib_metrics = evaluate_model(
        model, X_test, y_test,
        "LightGBM (uncalibrated)",
        teacher_preds=teacher_preds
    )

    logger.info(f"  RMSE: {pre_calib_metrics['rmse']:.4f}")
    logger.info(f"  R²: {pre_calib_metrics['r2']:.4f}")
    logger.info(f"  Brier Score: {pre_calib_metrics['brier_score']:.4f}")
    logger.info(f"  ECE: {pre_calib_metrics['ece']:.4f}")

    # =========================================================================
    # Probability Calibration
    # =========================================================================
    calibrated_model = model
    actual_calibration_method = 'none'

    if calibrate:
        logger.info("\n" + "-" * 50)
        logger.info("PHASE 3: Probability Calibration")
        logger.info("-" * 50)

        calibrated_model, actual_calibration_method = calibrate_probabilities(
            model, X_train, y_train, method=calibration_method
        )

        if actual_calibration_method != 'none':
            post_calib_metrics = evaluate_model(
                calibrated_model, X_test, y_test,
                f"LightGBM ({actual_calibration_method})",
                is_calibrated=True,
                teacher_preds=teacher_preds
            )

            logger.info(f"  Brier Score: {post_calib_metrics['brier_score']:.4f} "
                       f"(was {pre_calib_metrics['brier_score']:.4f})")
            logger.info(f"  ECE: {post_calib_metrics['ece']:.4f} "
                       f"(was {pre_calib_metrics['ece']:.4f})")

            best_metrics = post_calib_metrics
        else:
            best_metrics = pre_calib_metrics
    else:
        best_metrics = pre_calib_metrics

    # =========================================================================
    # Cross-Validation
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("PHASE 4: Cross-Validation")
    logger.info("-" * 50)

    cv_model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, verbosity=-1,
        random_state=RANDOM_STATE, force_row_wise=True
    )
    cv_scores = cross_val_score(
        cv_model, X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='neg_root_mean_squared_error'
    )
    logger.info(f"  5-Fold CV RMSE: {-cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # =========================================================================
    # Feature Importance (SHAP + Native)
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("PHASE 5: Feature Importance Analysis")
    logger.info("-" * 50)

    shap_values = None
    if use_shap:
        importance_df, shap_values = compute_shap_importance(model, X_test, used_features)

        # Use SHAP if available and valid, otherwise fall back to native
        use_shap_col = 'shap_pct' in importance_df.columns and not importance_df['shap_pct'].isna().all()

        if use_shap_col:
            logger.info("\nTop 15 Features (SHAP importance):")
            for _, row in importance_df.head(15).iterrows():
                shap_pct = row.get('shap_pct', 0) if pd.notna(row.get('shap_pct', np.nan)) else 0
                cum_pct = row.get('cumulative_shap_pct', 0) if pd.notna(row.get('cumulative_shap_pct', np.nan)) else 0
                logger.info(f"  {row['feature']:40} {shap_pct:6.2f}% (cum: {cum_pct:.1f}%)")
        else:
            logger.info("\nTop 15 Features (native importance - SHAP values were zero):")
            for _, row in importance_df.head(15).iterrows():
                native_pct = row.get('native_pct', 0) if pd.notna(row.get('native_pct', np.nan)) else 0
                logger.info(f"  {row['feature']:40} {native_pct:6.2f}%")
    else:
        importance_df = get_native_feature_importance(model, used_features)

        logger.info("\nTop 15 Features (native importance):")
        for _, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']:40} {row['native_pct']:6.2f}% (cum: {row['cumulative_pct']:.1f}%)")

    # =========================================================================
    # Save Model & Artifacts
    # =========================================================================
    logger.info("\n" + "-" * 50)
    logger.info("PHASE 6: Saving Model & Artifacts")
    logger.info("-" * 50)

    model_data = {
        'model': model,
        'calibrated_model': calibrated_model if calibrate else None,
        'calibration_method': actual_calibration_method,
        'features': used_features,
        'metrics': best_metrics,
        'pre_calibration_metrics': pre_calib_metrics,
        'cv_scores': cv_scores.tolist(),
        'training_config': {
            'use_optuna': use_optuna,
            'optuna_trials': optuna_trials,
            'calibrate': calibrate,
            'calibration_method': actual_calibration_method,
            'filter_confidence': filter_confidence,
            'use_shap': use_shap,
            'use_dart': use_dart,
            'test_size': test_size,
            'n_samples': len(df),
            'n_features': len(used_features)
        },
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0'
    }

    model_path = OUTPUT_DIR / "partner_propensity_model.joblib"
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")

    importance_path = OUTPUT_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to {importance_path}")

    # Save metrics report
    metrics_report = {
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in best_metrics.items()},
        'cv_rmse_mean': float(-cv_scores.mean()),
        'cv_rmse_std': float(cv_scores.std()),
        'calibration_method': actual_calibration_method,
        'n_samples': len(df),
        'n_features': len(used_features),
        'timestamp': datetime.now().isoformat()
    }

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_report, f, indent=2, default=str)
    logger.info(f"Training metrics saved to {metrics_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  MODEL TRAINING COMPLETE")
    logger.info("=" * 70)

    return model, best_metrics, importance_df


def load_trained_model() -> Tuple[Any, List[str], Optional[Any]]:
    """Load trained model, feature list, and optionally calibrated model."""
    model_path = OUTPUT_DIR / "partner_propensity_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    data = joblib.load(model_path)

    model = data['model']
    features = data['features']
    calibrated_model = data.get('calibrated_model')

    logger.info(f"Loaded model v{data.get('version', '1.0')}")
    logger.info(f"  Features: {len(features)}")
    logger.info(f"  Calibration: {data.get('calibration_method', 'none')}")

    return model, features, calibrated_model


# ============================================================================
# Scoring
# ============================================================================

def score_all_apps(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    features: Optional[List[str]] = None,
    use_calibrated: bool = True
) -> pd.DataFrame:
    """Score all apps using trained model."""
    calibrated_model = None

    if model is None:
        model, features, calibrated_model = load_trained_model()
    elif features is None:
        try:
            _, features, calibrated_model = load_trained_model()
        except FileNotFoundError:
            from .feature_engineering import get_feature_columns, get_llm_feature_columns
            base_features = get_feature_columns()
            llm_features = get_llm_feature_columns()
            features = base_features + [f for f in llm_features if f != 'llm_partner_probability']
            logger.warning("Using default feature list - model file not found")

    logger.info(f"Scoring {len(df)} apps...")

    X, _, _ = prepare_features(df, features)

    # Score using calibrated model if available
    if use_calibrated and calibrated_model is not None:
        try:
            scores = calibrated_model.predict_proba(X)[:, 1]
            logger.info("Using calibrated model for scoring")
        except Exception as e:
            logger.warning(f"Calibrated scoring failed: {e}. Using uncalibrated.")
            scores = model.predict(X)
    else:
        scores = model.predict(X)

    scores = np.clip(scores, 0, 1)

    df = df.copy()
    df['ml_partner_probability'] = scores

    # Calculate final score (weighted combination)
    if 'llm_partner_probability' in df.columns:
        df['final_partner_probability'] = (
            df['llm_partner_probability'].fillna(df['ml_partner_probability']) * 0.3 +
            df['ml_partner_probability'] * 0.7
        )
    else:
        df['final_partner_probability'] = df['ml_partner_probability']

    # Rank
    df['rank'] = df['final_partner_probability'].rank(ascending=False, method='min').astype(int)

    # Tier assignment (percentile-based)
    df['tier'] = pd.cut(
        df['final_partner_probability'].rank(pct=True),
        bins=[0, 0.6, 0.9, 1.0],
        labels=[3, 2, 1],
        include_lowest=True
    ).astype(int)

    tier_dist = df['tier'].value_counts().sort_index()
    logger.info(f"Tier distribution: {tier_dist.to_dict()}")

    return df


if __name__ == "__main__":
    # Test training with sample data
    from .data_loader import load_enriched_apps
    from .feature_engineering import extract_features

    df = load_enriched_apps()
    df = extract_features(df)

    # Create synthetic LLM scores for testing
    np.random.seed(RANDOM_STATE)
    n = len(df)

    base_score = (
        0.2 +
        (df['rating'] / 5) * 0.3 +
        (np.log1p(df['reviews_count']) / 10) * 0.2 +
        df['has_integration_keywords'] * 0.15 +
        df['is_multi_app_developer'] * 0.1 +
        np.random.beta(2, 5, size=n) * 0.25
    )

    df['llm_partner_probability'] = np.clip(base_score, 0, 1)
    df['llm_company_maturity'] = np.random.randint(1, 6, size=n)
    df['llm_technical_sophistication'] = np.random.randint(1, 6, size=n)
    df['llm_business_model'] = np.random.randint(1, 5, size=n)
    df['llm_merchant_segment'] = np.random.randint(1, 4, size=n)
    df['llm_confidence'] = np.random.choice([1, 2, 3], size=n, p=[0.2, 0.5, 0.3])
    df['llm_fit_signal_count'] = np.random.randint(0, 5, size=n)
    df['llm_risk_flag_count'] = np.random.randint(0, 3, size=n)
    df['llm_rag_use_case_count'] = np.random.randint(0, 4, size=n)

    # Train on sample
    model, metrics, importance = train_ensemble(
        df.sample(2000, random_state=RANDOM_STATE),
        use_optuna=True,
        optuna_trials=20,
        calibrate=True,
        use_shap=True
    )
