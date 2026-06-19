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

def make_sample_data(n=50):
    """Generate a small synthetic dataset for testing."""
    np.random.seed(42)
    data = {
        'age': np.random.randint(25, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'bmi': np.random.uniform(18, 38, n).round(1),
        'smoking': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'alcohol': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'physical_activity': np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3]),
        'diabetes': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'hypertension': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'family_history': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'cholesterol': np.random.uniform(120, 280, n).round(1),
        'systolic_bp': np.random.uniform(100, 180, n).round(1),
        'diastolic_bp': np.random.uniform(60, 120, n).round(1),
        'stress_level': np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3]),
        'sleep_hours': np.random.uniform(4, 10, n).round(1),
        'borough': np.random.choice(['Westminster', 'Camden', 'Islington', 'Tower Hamlets'], n),
        'cvd_risk': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    }
    return pd.DataFrame(data)


# ── Tests: Data Processor ─────────────────────────────────────────────────

class TestDataProcessor:
    def setup_method(self):
        self.processor = CVDDataProcessor()

    def test_load_data_from_csv(self):
        """Processor should load CSV files."""
        df = self.processor.load_data()
        assert df is not None, "load_data() returned None"
        assert len(df) > 0, "Dataset is empty"

    def test_preprocess_data_shape(self):
        """Preprocessed data should have features and target."""
        df = make_sample_data()
        X, y, processed = self.processor.preprocess_data(df)
        assert X.shape[0] == y.shape[0], "X and y have different row counts"
        assert X.shape[1] >= 15, f"Expected at least 15 features, got {X.shape[1]}"

    def test_preprocess_no_missing_values(self):
        """Preprocessing should handle missing values."""
        df = make_sample_data()
        df.loc[0, 'bmi'] = np.nan  # Introduce a missing value
        X, y, processed = self.processor.preprocess_data(df)
        assert not np.any(pd.isna(X)), "Features still contain NaN after preprocessing"

    def test_single_prediction_input_format(self):
        """Single prediction should accept a dict with expected keys."""
        sample_input = {
            'age': 45, 'gender': 'Male', 'bmi': 27.5,
            'smoking': 0, 'alcohol': 1, 'physical_activity': 1,
            'diabetes': 0, 'hypertension': 1, 'family_history': 0,
            'cholesterol': 220, 'systolic_bp': 130, 'diastolic_bp': 85,
            'stress_level': 1, 'sleep_hours': 7,
            'borough': 'Camden'
        }
        df = make_sample_data()
        self.processor.preprocess_data(df)  # Fit encoders/scalers
        X = self.processor.prepare_single_prediction(sample_input)
        assert X is not None, "prepare_single_prediction returned None"
        assert X.shape[0] == 1, f"Expected 1 row, got {X.shape[0]}"


# ── Tests: ML Model ──────────────────────────────────────────────────────

class TestCVDRiskModel:
    def setup_method(self):
        self.model = CVDRiskModel('random_forest')

    def test_model_creation(self):
        """Model should be created without errors."""
        self.model.create_model()
        assert self.model.model is not None, "Model was not created"

    def test_model_training(self):
        """Model should train and return accuracy metrics."""
        df = make_sample_data()
        processor = CVDDataProcessor()
        X, y, _ = processor.preprocess_data(df)
        self.model.create_model()
        self.model.model.fit(X, y)
        assert hasattr(self.model.model, 'predict'), "Model has no predict method"
        preds = self.model.model.predict(X)
        assert len(preds) == len(y), "Predictions count mismatch"

    def test_prediction_range(self):
        """Risk probability should be between 0 and 1."""
        df = make_sample_data()
        processor = CVDDataProcessor()
        X, y, _ = processor.preprocess_data(df)
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
        df = make_sample_data()
        processor = CVDDataProcessor()
        X, y, _ = processor.preprocess_data(df)
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


# ── Tests: Environment Integration ────────────────────────────────────────

class TestEnvironmentData:
    def test_environmental_data_exists(self):
        """Environmental CSV should exist and have data."""
        path = os.path.join(os.path.dirname(__file__), '..', 'environmental_data', 'london_environmental_data.csv')
        assert os.path.exists(path), "Environmental data file not found"
        df = pd.read_csv(path)
        assert len(df) > 0, "Environmental data is empty"

    def test_personal_health_data_exists(self):
        """Personal health data CSV should exist."""
        path = os.path.join(os.path.dirname(__file__), '..', 'user_data', 'personal_health_data.csv')
        assert os.path.exists(path), "Personal health data file not found"
