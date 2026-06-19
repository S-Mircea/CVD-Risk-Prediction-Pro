import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class CVDDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Load and merge health and environmental data"""
        # Load expanded health data
        health_data = pd.read_csv('../user_data/expanded_health_data.csv')
        
        # Load expanded environmental data
        env_data = pd.read_csv('../environmental_data/expanded_environmental_data.csv')
        
        # Merge data on Borough
        merged_data = health_data.merge(env_data, on='Borough', how='left')
        
        return merged_data
    
    def preprocess_data(self, data):
        """Clean and preprocess the data for ML model"""
        # Handle missing values
        data = data.fillna(data.median(numeric_only=True))
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Smoker', 'FamilyHistoryCVD', 'Diabetes', 
                             'HighBloodPressure', 'PhysicalActivityLevel', 'AlcoholConsumption',
                             'StressLevel', 'Borough']
        
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features for model
        feature_columns = ['Age', 'Gender_encoded', 'Smoker_encoded', 'FamilyHistoryCVD_encoded',
                          'Diabetes_encoded', 'HighBloodPressure_encoded', 'PhysicalActivityLevel_encoded',
                          'AlcoholConsumption_encoded', 'StressLevel_encoded', 'Borough_encoded',
                          'BMI', 'TotalCholesterol', 'SystolicBP', 'DiastolicBP', 'SleepHours',
                          'Avg_PM25', 'Avg_NO2', 'NoiseLevel_dB', 'GreenSpacePercent', 
                          'WalkabilityScore', 'UrbanHeatIncrease']
        
        self.feature_columns = feature_columns
        X = data[feature_columns]
        y = data['CVD_Risk']
        
        # Scale numerical features with robust scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, data
    
    def prepare_single_prediction(self, user_input):
        """Prepare single user input for prediction"""
        # Create DataFrame from user input
        input_df = pd.DataFrame([user_input])
        
        # Encode categorical variables using fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col + '_encoded'] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    input_df[col + '_encoded'] = 0
        
        # Select and scale features
        X_input = input_df[self.feature_columns]
        X_scaled = self.scaler.transform(X_input)
        
        return X_scaled