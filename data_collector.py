import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class CVDDataCollector:
    def __init__(self):
        self.london_boroughs = [
            'Camden', 'Westminster', 'Greenwich', 'Hackney', 'Tower Hamlets',
            'Southwark', 'Lambeth', 'Islington', 'Kensington and Chelsea',
            'Hammersmith and Fulham', 'Wandsworth', 'Richmond upon Thames',
            'Kingston upon Thames', 'Merton', 'Sutton', 'Croydon', 'Bromley',
            'Lewisham', 'Bexley', 'Havering', 'Barking and Dagenham',
            'Redbridge', 'Newham', 'Waltham Forest', 'Haringey', 'Enfield',
            'Barnet', 'Harrow', 'Hillingdon', 'Ealing', 'Hounslow', 'Brent',
            'City of London'
        ]
    
    def generate_expanded_health_data(self, num_records=500):
        """Generate expanded health data with additional cardiovascular risk factors"""
        np.random.seed(42)
        data = []
        
        for _ in range(num_records):
            age = np.random.randint(18, 85)
            gender = random.choice(['Male', 'Female'])
            
            # Age-based risk adjustments
            age_risk_factor = 1.0 if age < 45 else (1.2 if age < 65 else 1.5)
            
            # Generate correlated health factors
            smoker = random.choices(['Yes', 'No'], weights=[0.2, 0.8])[0]
            family_history = random.choices(['Yes', 'No'], weights=[0.3, 0.7])[0]
            
            # Age-correlated conditions
            diabetes_prob = min(0.15 * age_risk_factor, 0.4)
            diabetes = random.choices(['Yes', 'No'], weights=[diabetes_prob, 1-diabetes_prob])[0]
            
            hypertension_prob = min(0.2 * age_risk_factor, 0.5)
            hypertension = random.choices(['Yes', 'No'], weights=[hypertension_prob, 1-hypertension_prob])[0]
            
            # Additional health metrics
            bmi = np.random.normal(26, 4)  # BMI with realistic distribution
            bmi = max(18, min(45, bmi))  # Constrain to reasonable range
            
            cholesterol = np.random.normal(200, 40)  # Total cholesterol
            cholesterol = max(120, min(350, cholesterol))
            
            systolic_bp = np.random.normal(130, 20)
            systolic_bp = max(90, min(200, systolic_bp))
            
            diastolic_bp = np.random.normal(80, 15)
            diastolic_bp = max(60, min(120, diastolic_bp))
            
            # Physical activity (inversely correlated with age)
            activity_weights = [0.2, 0.5, 0.3] if age < 50 else [0.4, 0.4, 0.2]
            physical_activity = random.choices(['Low', 'Moderate', 'High'], weights=activity_weights)[0]
            
            # Lifestyle factors
            alcohol_consumption = random.choices(['None', 'Light', 'Moderate', 'Heavy'], 
                                               weights=[0.2, 0.4, 0.3, 0.1])[0]
            stress_level = random.choices(['Low', 'Moderate', 'High'], weights=[0.3, 0.5, 0.2])[0]
            sleep_hours = np.random.normal(7, 1.5)
            sleep_hours = max(4, min(12, sleep_hours))
            
            # Geographic factors
            borough = random.choice(self.london_boroughs)
            
            # Calculate CVD risk based on multiple factors
            risk_score = 0
            risk_score += 0.02 * (age - 18)  # Age factor
            risk_score += 0.15 if smoker == 'Yes' else 0
            risk_score += 0.1 if family_history == 'Yes' else 0
            risk_score += 0.12 if diabetes == 'Yes' else 0
            risk_score += 0.1 if hypertension == 'Yes' else 0
            risk_score += max(0, (bmi - 25) * 0.02)  # BMI over 25
            risk_score += max(0, (cholesterol - 200) * 0.001)  # High cholesterol
            risk_score += max(0, (systolic_bp - 120) * 0.002)  # High BP
            risk_score -= 0.05 if physical_activity == 'High' else 0
            risk_score += 0.05 if alcohol_consumption == 'Heavy' else 0
            risk_score += 0.03 if stress_level == 'High' else 0
            risk_score -= max(0, (sleep_hours - 6) * 0.01)  # Good sleep
            
            # Add some randomness
            risk_score += np.random.normal(0, 0.1)
            
            # Convert to binary classification
            cvd_risk = 1 if risk_score > 0.3 else 0
            
            record = {
                'Age': age,
                'Gender': gender,
                'Smoker': smoker,
                'FamilyHistoryCVD': family_history,
                'Diabetes': diabetes,
                'HighBloodPressure': hypertension,
                'BMI': round(bmi, 1),
                'TotalCholesterol': round(cholesterol, 0),
                'SystolicBP': round(systolic_bp, 0),
                'DiastolicBP': round(diastolic_bp, 0),
                'PhysicalActivityLevel': physical_activity,
                'AlcoholConsumption': alcohol_consumption,
                'StressLevel': stress_level,
                'SleepHours': round(sleep_hours, 1),
                'Borough': borough,
                'CVD_Risk': cvd_risk
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def collect_additional_environmental_data(self):
        """Generate additional environmental factors"""
        env_data = []
        
        for borough in self.london_boroughs:
            # Base pollution levels (realistic for London)
            base_pm25 = np.random.normal(12, 3)
            base_no2 = np.random.normal(45, 10)
            
            # Add seasonal variations
            seasonal_factor = np.random.uniform(0.8, 1.2)
            
            # Additional environmental factors
            noise_level = np.random.normal(55, 8)  # dB
            green_space_pct = np.random.uniform(10, 40)  # % green space
            walkability_score = np.random.uniform(30, 90)
            
            # Urban heat island effect
            temperature_increase = np.random.uniform(1, 4)  # degrees above rural
            
            record = {
                'Borough': borough,
                'Avg_PM25': round(max(5, base_pm25 * seasonal_factor), 1),
                'Avg_NO2': round(max(20, base_no2 * seasonal_factor), 1),
                'NoiseLevel_dB': round(noise_level, 1),
                'GreenSpacePercent': round(green_space_pct, 1),
                'WalkabilityScore': round(walkability_score, 1),
                'UrbanHeatIncrease': round(temperature_increase, 1)
            }
            
            env_data.append(record)
        
        return pd.DataFrame(env_data)
    
    def save_expanded_datasets(self, health_records=500):
        """Generate and save expanded datasets"""
        print(f"Generating {health_records} health records...")
        health_df = self.generate_expanded_health_data(health_records)
        
        print("Generating expanded environmental data...")
        env_df = self.collect_additional_environmental_data()
        
        # Save to new files
        health_df.to_csv('user_data/expanded_health_data.csv', index=False)
        env_df.to_csv('environmental_data/expanded_environmental_data.csv', index=False)
        
        print(f"✓ Saved expanded health data: {len(health_df)} records")
        print(f"✓ Saved expanded environmental data: {len(env_df)} records")
        
        return health_df, env_df

if __name__ == "__main__":
    collector = CVDDataCollector()
    health_data, env_data = collector.save_expanded_datasets(1000)
    
    print("\nDataset Summary:")
    print(f"Health Records: {len(health_data)}")
    print(f"Environmental Records: {len(env_data)}")
    print(f"CVD Risk Distribution: {health_data['CVD_Risk'].value_counts().to_dict()}")