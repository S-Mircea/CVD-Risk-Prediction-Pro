"""
Model Comparison Script
Compares multiple ML models for CVD risk prediction and selects the best one.
Run with: python compare_models.py
"""
import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost - it's optional
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

from data_processor import CVDDataProcessor

MODELS = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=2000, class_weight='balanced', C=0.1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, max_depth=15,
        min_samples_split=3, class_weight='balanced', min_samples_leaf=2
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, random_state=42, max_depth=5, learning_rate=0.1
    ),
    'SVM (RBF)': SVC(
        random_state=42, probability=True, class_weight='balanced', C=1.0
    ),
}

if XGB_AVAILABLE:
    MODELS['XGBoost'] = XGBClassifier(
        n_estimators=200, random_state=42, max_depth=10,
        learning_rate=0.1, eval_metric='logloss', use_label_encoder=False
    )


def load_and_prepare_data():
    """Load data and prepare features/target."""
    print("📥 Loading data...")
    processor = CVDDataProcessor()
    raw_data = processor.load_data()
    X, y, processed = processor.preprocess_data(raw_data)
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target distribution: {np.bincount(y.astype(int))}")
    return X, y, processor


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    print(f"\n{'='*50}")
    print(f"🔍 Evaluating: {model_name}")
    print('='*50)

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    # Metrics
    results = {
        'model': model_name,
        'train_time_s': round(train_time, 3),
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        'cv_auc_mean': round(cv_scores.mean(), 4),
        'cv_auc_std': round(cv_scores.std(), 4),
    }

    print(f"   ⏱  Train time: {train_time:.2f}s")
    print(f"   ✅ Accuracy:   {results['accuracy']:.4f}")
    print(f"   🎯 Precision:  {results['precision']:.4f}")
    print(f"   🔍 Recall:     {results['recall']:.4f}")
    print(f"   📊 F1 Score:   {results['f1_score']:.4f}")
    print(f"   📈 ROC AUC:    {results['roc_auc']:.4f}")
    print(f"   🔄 CV AUC:     {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")

    return results, model


def print_summary_table(all_results):
    """Print a clean comparison table."""
    print(f"\n{'='*70}")
    print("📊 MODEL COMPARISON SUMMARY")
    print('='*70)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC AUC':>8} {'CV AUC':>8} {'Time(s)':>8}")
    print('-'*70)
    for r in sorted(all_results, key=lambda x: x['f1_score'], reverse=True):
        print(f"{r['model']:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1_score']:>8.4f} {r['roc_auc']:>8.4f} {r['cv_auc_mean']:>8.4f} {r['train_time_s']:>8.3f}")
    print('='*70)


def save_best_model(best_results, best_model, processor):
    """Save the best performing model."""
    model_data = {
        'model': best_model,
        'data_processor': processor,
        'model_type': best_results['model'],
        'performance': {k: v for k, v in best_results.items() if k != 'model'}
    }
    filename = 'cvd_risk_model.pkl'
    joblib.dump(model_data, filename)
    print(f"\n💾 Best model saved as '{filename}'")
    print(f"   🏆 Champion: {best_results['model']}")
    print(f"   📊 F1 Score: {best_results['f1_score']:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("🏥 CVD Risk Prediction - Model Comparison")
    print("=" * 60)

    X, y, processor = load_and_prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Evaluate all models
    all_results = []
    best_f1 = 0
    best_model = None
    best_results = None

    for name, model in MODELS.items():
        results, trained_model = evaluate_model(model, name, X_train, X_test, y_train, y_test)
        all_results.append(results)
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_model = trained_model
            best_results = results

    print_summary_table(all_results)
    save_best_model(best_results, best_model, processor)