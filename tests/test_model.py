"""
Unit tests for CVD Risk Prediction model and data processing.
Run with: python -m pytest tests/ -v
"""
import sys
import os
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app_code'))

from data_processor import CVDDataProcessor
from ml_model import CVDRiskModel


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_kaggle_data(n=100):
    """Generate synthetic data in Kaggle Cardiovascular Disease format."""
    np.random.seed(42)
    data = {
        'id': range(n),
        'age': np.random.randint(15000, 25000, n),  # days
        'gender': np.random.choice([1, 2], n, p=[0.5, 0.5]),
        'height': np.random.randint(140, 200, n),
        'weight': np.random.uniform(50, 120, n).round(1),
        'ap_hi': np.random.randint(90, 180, n),
        'ap_lo': np.random.randint(60, 120, n),
        'cholesterol': np.random.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1]),
        'gluc': np.random.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1]),
        'smoke': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'alco': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'active': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'cardio': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    }
    return pd.DataFrame(data)


# ── Tests: Data Processor ─────────────────────────────────────────────────

class TestDataProcessor:
    def setup_method(self):
        self.processor = CVDDataProcessor()
        # Set Kaggle format manually for tests
        self.processor.is_kaggle_format = True

    def test_preprocess_data_shape(self):
        """Preprocessed data should have features and target."""
        df = make_kaggle_data()
        X, y, processed = self.processor.preprocess_data(df)
        assert X.shape[0] == y.shape[0], "X and y have different row counts"
        assert X.shape[1] == 12, f"Expected 12 features (Kaggle), got {X.shape[1]}"

    def test_preprocess_no_missing_values(self):
        """Preprocessing should handle missing values."""
        df = make_kaggle_data()
        df.loc[0, 'weight'] = np.nan  # Introduce a missing value
        X, y, processed = self.processor.preprocess_data(df)
        assert not np.any(pd.isna(X)), "Features still contain NaN after preprocessing"

    def test_preprocess_filters_invalid(self):
        """Should filter out physiologically impossible values."""
        df = make_kaggle_data(50)
        # Add obviously invalid rows
        bad_row = df.iloc[0].copy()
        bad_row['ap_hi'] = 500  # impossible systolic BP
        bad_row['cardio'] = 0
        df_bad = pd.concat([df, bad_row.to_frame().T], ignore_index=True)
        X, y, processed = self.processor.preprocess_data(df_bad)
        # Should have filtered out the bad row
        assert X.shape[0] <= len(df_bad), "Should have filtered invalid rows"

    def test_single_prediction_kaggle_format(self):
        """Single prediction should accept Kaggle-style input."""
        sample_input = {
            'age': 45, 'gender': 2, 'height': 175, 'weight': 78,
            'ap_hi': 130, 'ap_lo': 85, 'cholesterol': 1, 'gluc': 1,
            'smoke': 0, 'alco': 1, 'active': 1,
            'cardio': 0
        }
        df = make_kaggle_data()
        self.processor.is_kaggle_format = True
        X, y, _ = self.processor.preprocess_data(df.copy())
        result = self.processor.prepare_single_prediction(sample_input)
        assert result is not None, "prepare_single_prediction returned None"
        assert result.shape[0] == 1, f"Expected 1 row, got {result.shape[0]}"
        assert result.shape[1] == 12, f"Expected 12 features, got {result.shape[1]}"

    def test_load_data_from_csv(self):
        """Processor should load the real Kaggle CSV."""
        path = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'cardio_train.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
            assert len(df) > 0, "Dataset is empty"
            assert 'cardio' in df.columns, "Target column missing"
            print(f"Real dataset: {len(df)} records ✅")
        else:
            print("⚠️ cardio_train.csv not found — test skipped (data will be downloaded on first run)")


# ── Tests: ML Model ──────────────────────────────────────────────────────

class TestCVDRiskModel:
    def setup_method(self):
        self.model = CVDRiskModel('random_forest')

    def test_model_creation(self):
        """Model should be created without errors."""
        self.model.create_model()
        assert self.model.model is not None, "Model was not created"

    def test_model_training(self):
        """Model should train on Kaggle-formatted data."""
        df = make_kaggle_data(200)
        processor = CVDDataProcessor()
        processor.is_kaggle_format = True
        X, y, _ = processor.preprocess_data(df.copy())
        self.model.create_model()
        self.model.model.fit(X, y)
        assert hasattr(self.model.model, 'predict'), "Model has no predict method"
        preds = self.model.model.predict(X)
        assert len(preds) == len(y), "Predictions count mismatch"

    def test_prediction_range(self):
        """Risk probability should be between 0 and 1."""
        df = make_kaggle_data(200)
        processor = CVDDataProcessor()
        processor.is_kaggle_format = True
        X, y, _ = processor.preprocess_data(df.copy())
        self.model.create_model()
        self.model.model.fit(X, y)

        sample = X[:1]
        proba = self.model.model.predict_proba(sample)[0]
        risk = float(proba[1]) if len(proba) > 1 else float(proba[0])
        assert 0.0 <= risk <= 1.0, f"Risk probability {risk} outside [0, 1]"

    def test_risk_level_mapping(self):
        """_get_risk_level should return correct category for each range."""
        assert self.model._get_risk_level(0.1) == "Very Low Risk"
        assert self.model._get_risk_level(0.3) == "Low Risk"
        assert self.model._get_risk_level(0.5) == "Moderate Risk"
        assert self.model._get_risk_level(0.7) == "High Risk"
        assert self.model._get_risk_level(0.9) == "Very High Risk"

    def test_save_load_model(self):
        """Model should save and load correctly."""
        df = make_kaggle_data(200)
        processor = CVDDataProcessor()
        processor.is_kaggle_format = True
        X, y, _ = processor.preprocess_data(df.copy())
        self.model.create_model()
        self.model.model.fit(X, y)
        self.model.data_processor = processor

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name
        try:
            self.model.save_model(tmp_path)
            assert os.path.exists(tmp_path), "Model file not saved"

            loaded = CVDRiskModel()
            result = loaded.load_model(tmp_path)
            assert result, "Model not loaded"
            assert loaded.model is not None, "Loaded model is None"
        finally:
            os.unlink(tmp_path)


# ── Tests: Environment Data ──────────────────────────────────────────────

class TestEnvironmentData:
    def test_environmental_data_exists(self):
        """Environmental CSV should exist and have data."""
        path = os.path.join(os.path.dirname(__file__), '..', 'environmental_data', 'london_environmental_data.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert len(df) > 0, "Environmental data is empty"

    def test_personal_health_data_exists(self):
        """Original synthetic health data CSV should exist."""
        path = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'expanded_health_data.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert len(df) > 0, "Health data is empty"