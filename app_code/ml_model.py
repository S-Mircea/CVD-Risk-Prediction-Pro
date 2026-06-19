import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from data_processor import CVDDataProcessor

class CVDRiskModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.data_processor = CVDDataProcessor()
        
    def create_model(self):
        """Create the ML model based on specified type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=15,
                min_samples_split=3,
                class_weight='balanced',
                min_samples_leaf=2
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            )
        else:
            raise ValueError("Unsupported model type")
    
    def train_model(self):
        """Train the CVD risk prediction model"""
        # Load and preprocess data
        raw_data = self.data_processor.load_data()
        X, y, processed_data = self.data_processor.preprocess_data(raw_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        self.create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        print(f"Model: {self.model_type}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, cv_scores
    
    def save_model(self, filename='cvd_risk_model.pkl'):
        """Save the trained model and data processor"""
        model_data = {
            'model': self.model,
            'data_processor': self.data_processor,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='cvd_risk_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filename):
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.data_processor = model_data['data_processor']
            self.model_type = model_data['model_type']
            print(f"Model loaded from {filename}")
            return True
        else:
            print(f"Model file {filename} not found")
            return False
    
    def predict_risk(self, user_input):
        """Predict CVD risk for a single user"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess input
        X_input = self.data_processor.prepare_single_prediction(user_input)
        
        # Make prediction
        risk_probability = self.model.predict_proba(X_input)[0]
        risk_prediction = self.model.predict(X_input)[0]
        
        return {
            'risk_prediction': int(risk_prediction),
            'risk_probability': float(risk_probability[1]),  # Probability of CVD risk
            'risk_level': self._get_risk_level(risk_probability[1])
        }
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level with more granular thresholds"""
        if probability < 0.2:
            return "Very Low Risk"
        elif probability < 0.4:
            return "Low Risk"
        elif probability < 0.6:
            return "Moderate Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"