"""
CVD Data Processor — handles both synthetic and Kaggle dataset formats.
Auto-detects which format is being used.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class CVDDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_kaggle_format = False

    def load_data(self):
        """Load data — auto-detect format."""
        # Prefer Kaggle real dataset if available
        kaggle_path = '../user_data/cardio_train.csv'
        synthetic_path = '../user_data/expanded_health_data.csv'
        env_path = '../environmental_data/expanded_environmental_data.csv'

        if os.path.exists(kaggle_path):
            print(f"📊 Loading Kaggle dataset ({kaggle_path})")
            df = pd.read_csv(kaggle_path, sep=';')
            self.is_kaggle_format = True
            return self._preprocess_kaggle(df)
        elif os.path.exists(synthetic_path):
            print(f"📊 Loading synthetic dataset ({synthetic_path})")
            health = pd.read_csv(synthetic_path)
            if os.path.exists(env_path):
                env = pd.read_csv(env_path)
                df = health.merge(env, on='Borough', how='left')
            else:
                df = health
            self.is_kaggle_format = False
            return df
        else:
            raise FileNotFoundError("No dataset found. Place cardio_train.csv in user_data/")

    def _preprocess_kaggle(self, df):
        """Convert Kaggle format to match expected feature set."""
        # Age is in days → convert to years
        df['Age'] = (df['age'] / 365.25).round(0).astype(int)

        # Gender: 1=female, 2=male → 0/1
        df['Gender_encoded'] = df['gender'] - 1

        # Create BMI from height (cm) and weight (kg)
        df['height_m'] = df['height'] / 100.0
        df['BMI'] = (df['weight'] / (df['height_m'] ** 2)).round(1)
        df = df.drop(columns=['height_m'])

        # Blood pressure
        df['SystolicBP'] = df['ap_hi']
        df['DiastolicBP'] = df['ap_lo']

        # Cholesterol: 1=normal, 2=above, 3=well above
        df['Cholesterol_encoded'] = df['cholesterol']

        # Glucose
        df['Glucose_encoded'] = df['gluc']

        # Binary flags
        df['Smoker_encoded'] = df['smoke']
        df['Alcohol_encoded'] = df['alco']
        df['Active_encoded'] = df['active']

        # Target
        df['CVD_Risk'] = df['cardio']

        # Derived features
        df['BP_Ratio'] = (df['SystolicBP'] / df['DiastolicBP'].replace(0, 1)).round(2)
        df['Age_BMI'] = (df['Age'] * df['BMI'] / 100).round(1)

        return df

    def _get_features_kaggle(self, df):
        """Feature selection for Kaggle dataset."""
        return [
            'Age', 'Gender_encoded', 'BMI', 'SystolicBP', 'DiastolicBP',
            'Cholesterol_encoded', 'Glucose_encoded',
            'Smoker_encoded', 'Alcohol_encoded', 'Active_encoded',
            'BP_Ratio', 'Age_BMI'
        ]

    def _get_features_synthetic(self, df):
        """Feature selection for original synthetic dataset."""
        categorical = ['Gender', 'Smoker', 'FamilyHistoryCVD', 'Diabetes',
                       'HighBloodPressure', 'PhysicalActivityLevel', 'AlcoholConsumption',
                       'StressLevel', 'Borough']
        for col in categorical:
            if col in df.columns and col + '_encoded' not in df.columns:
                df[col + '_encoded'] = pd.factorize(df[col].astype(str))[0]

        return ['Age', 'Gender_encoded', 'Smoker_encoded', 'FamilyHistoryCVD_encoded',
                'Diabetes_encoded', 'HighBloodPressure_encoded', 'PhysicalActivityLevel_encoded',
                'AlcoholConsumption_encoded', 'StressLevel_encoded', 'Borough_encoded',
                'BMI', 'TotalCholesterol', 'SystolicBP', 'DiastolicBP', 'SleepHours',
                'Avg_PM25', 'Avg_NO2', 'NoiseLevel_dB', 'GreenSpacePercent',
                'WalkabilityScore', 'UrbanHeatIncrease']

    def preprocess_data(self, data):
        """Clean, engineer features, and scale."""
        data = data.copy()

        # Remove invalid values
        if self.is_kaggle_format:
            # Filter physiologically impossible values
            data = data[data['SystolicBP'] > 50]
            data = data[data['SystolicBP'] < 250]
            data = data[data['DiastolicBP'] > 30]
            data = data[data['DiastolicBP'] < 200]
            data = data[data['BMI'] > 12]
            data = data[data['BMI'] < 55]
            data = data[data['Age'] >= 18]

        # Handle missing
        data = data.fillna(data.median(numeric_only=True))

        # Select features
        if self.is_kaggle_format:
            self.feature_columns = self._get_features_kaggle(data)
        else:
            self.feature_columns = self._get_features_synthetic(data)

        X = data[self.feature_columns]
        y = data['CVD_Risk']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, data

    def prepare_single_prediction(self, user_input):
        """Convert single user input dict to scaled features."""
        import pandas as pd
        input_df = pd.DataFrame([user_input])

        if self.is_kaggle_format or 'smoke' in input_df.columns:
            # Kaggle-style input
            input_df['BMI'] = input_df['weight'] / ((input_df['height'] / 100.0) ** 2)
            input_df['Gender_encoded'] = input_df['gender'] - 1
            input_df['SystolicBP'] = input_df['ap_hi']
            input_df['DiastolicBP'] = input_df['ap_lo']
            input_df['Cholesterol_encoded'] = input_df['cholesterol']
            input_df['Glucose_encoded'] = input_df.get('gluc', pd.Series([1]))
            input_df['Smoker_encoded'] = input_df['smoke']
            input_df['Alcohol_encoded'] = input_df.get('alco', pd.Series([0]))
            input_df['Active_encoded'] = input_df.get('active', pd.Series([1]))
            input_df['Age'] = input_df['age'] / 365.25 if input_df['age'].iloc[0] > 100 else input_df['age']
            input_df['BP_Ratio'] = input_df['SystolicBP'] / input_df['DiastolicBP'].replace(0, 1)
            input_df['Age_BMI'] = input_df['Age'] * input_df['BMI'] / 100
        else:
            # Original format
            for col in self.label_encoders:
                if col in input_df.columns:
                    try:
                        input_df[col + '_encoded'] = self.label_encoders[col].transform(
                            input_df[col].astype(str))
                    except ValueError:
                        input_df[col + '_encoded'] = 0

        X_input = input_df[self.feature_columns]
        return self.scaler.transform(X_input)