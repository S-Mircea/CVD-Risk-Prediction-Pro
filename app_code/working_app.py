from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model import CVDRiskModel
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import LLM advisor
try:
    from llm_advisor import CVDLlamaAdvisor
    LLM_AVAILABLE = True
    print("✓ LLM advisor imported successfully")
except Exception as e:
    print(f"⚠ LLM advisor not available: {e}")
    LLM_AVAILABLE = False

app = Flask(__name__)

# Initialize model
model = CVDRiskModel()
if not model.load_model():
    print("Warning: Model not found. Please train the model first.")

# Initialize LLM advisor if available
ollama_status = False
if LLM_AVAILABLE:
    try:
        llm_advisor = CVDLlamaAdvisor()
        ollama_status = llm_advisor.check_ollama_availability()
        if ollama_status:
            print("✓ Ollama LLM advisor connected successfully")
        else:
            print("⚠ Ollama not available - using fallback advice system")
    except Exception as e:
        print(f"⚠ LLM advisor initialization failed: {e}")
        LLM_AVAILABLE = False
        ollama_status = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
        # Get form data
        user_data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Smoker': request.form['smoker'],
            'FamilyHistoryCVD': request.form['family_history'],
            'Diabetes': request.form['diabetes'],
            'HighBloodPressure': request.form['high_bp'],
            'PhysicalActivityLevel': request.form['activity'],
            'BMI': float(request.form['bmi']),
            'TotalCholesterol': float(request.form['cholesterol']),
            'SystolicBP': float(request.form['systolic_bp']),
            'DiastolicBP': float(request.form['diastolic_bp']),
            'AlcoholConsumption': request.form['alcohol'],
            'StressLevel': request.form['stress'],
            'SleepHours': float(request.form['sleep_hours']),
            'Borough': request.form['borough']
        }
        
        # Load environmental data for the borough
        env_data = pd.read_csv('../environmental_data/expanded_environmental_data.csv')
        borough_env = env_data[env_data['Borough'] == user_data['Borough']]
        
        if not borough_env.empty:
            user_data['Avg_PM25'] = borough_env['Avg_PM25'].iloc[0]
            user_data['Avg_NO2'] = borough_env['Avg_NO2'].iloc[0]
            user_data['NoiseLevel_dB'] = borough_env['NoiseLevel_dB'].iloc[0]
            user_data['GreenSpacePercent'] = borough_env['GreenSpacePercent'].iloc[0]
            user_data['WalkabilityScore'] = borough_env['WalkabilityScore'].iloc[0]
            user_data['UrbanHeatIncrease'] = borough_env['UrbanHeatIncrease'].iloc[0]
        else:
            user_data['Avg_PM25'] = env_data['Avg_PM25'].mean()
            user_data['Avg_NO2'] = env_data['Avg_NO2'].mean()
            user_data['NoiseLevel_dB'] = env_data['NoiseLevel_dB'].mean()
            user_data['GreenSpacePercent'] = env_data['GreenSpacePercent'].mean()
            user_data['WalkabilityScore'] = env_data['WalkabilityScore'].mean()
            user_data['UrbanHeatIncrease'] = env_data['UrbanHeatIncrease'].mean()
        
        # Make prediction
        result = model.predict_risk(user_data)
        
        # Add environmental data to result
        result['environmental_data'] = {
            'pm25': user_data['Avg_PM25'],
            'no2': user_data['Avg_NO2'],
            'borough': user_data['Borough']
        }
        
        # Add recommendations
        result['recommendations'] = get_recommendations(result['risk_level'])
        
        # Generate LLM advice if available
        if LLM_AVAILABLE and ollama_status:
            try:
                llm_advice = llm_advisor.get_environmental_advice(
                    result['risk_level'],
                    result['environmental_data'],
                    user_data
                )
                result['llm_advice'] = llm_advice
                result['llm_available'] = True
                print(f"✓ Generated LLM advice for {result['risk_level']} risk in {user_data['Borough']}")
            except Exception as e:
                print(f"⚠ LLM advice generation failed: {str(e)}")
                result['llm_advice'] = get_fallback_advice(result['risk_level'], user_data['Borough'])
                result['llm_available'] = False
        else:
            result['llm_advice'] = get_fallback_advice(result['risk_level'], user_data['Borough'])
            result['llm_available'] = False
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        print(f"Error in assess_risk: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_recommendations(risk_level):
    recommendations = {
        'Low Risk': [
            "Continue your healthy lifestyle",
            "Schedule regular check-ups",
            "Monitor environmental exposure",
            "Maintain current activity levels"
        ],
        'Moderate Risk': [
            "Increase physical activity to 150+ minutes/week",
            "Consider lifestyle modifications",
            "Consult with your healthcare provider",
            "Monitor air quality in your area",
            "Consider dietary improvements"
        ],
        'High Risk': [
            "Seek immediate medical consultation",
            "Comprehensive cardiovascular assessment needed",
            "Urgent lifestyle intervention required",
            "Consider relocation if air quality is poor",
            "Regular monitoring and follow-up essential"
        ]
    }
    return recommendations.get(risk_level, ["Consult with healthcare provider"])

def get_fallback_advice(risk_level, borough):
    """Generate fallback advice when LLM is not available"""
    borough_advice = {
        'Tower Hamlets': 'High pollution area - exercise in Mile End Park, avoid busy roads during peak hours',
        'Camden': 'Urban environment - use Regent\'s Park for exercise, check air quality app',
        'Westminster': 'Very high traffic - exercise early morning in St James\'s Park',
        'Hackney': 'Above-average pollution - use Victoria Park, avoid main roads',
        'Richmond upon Thames': 'Excellent air quality - take advantage of Richmond Park',
        'Kingston upon Thames': 'Good air quality - riverside walks ideal for exercise',
        'Barking and Dagenham': 'High pollution - limit outdoor exercise during peak hours, use indoor alternatives'
    }
    
    risk_advice = {
        'Low Risk': 'Maintain healthy lifestyle while monitoring air quality.',
        'Moderate Risk': 'Increase exercise in green spaces, monitor pollution levels.',
        'High Risk': 'Prioritize indoor exercise on high pollution days, consult healthcare provider.'
    }
    
    borough_tip = borough_advice.get(borough, f'Monitor air quality in {borough}')
    risk_tip = risk_advice.get(risk_level, 'Consult healthcare provider')
    
    return f"{risk_tip} {borough_tip}"

@app.route('/llm-status', methods=['GET'])
def llm_status():
    """Get LLM advisor status"""
    return jsonify({
        'success': True,
        'status': {
            'llm_available': LLM_AVAILABLE,
            'ollama_running': ollama_status,
            'model_name': 'llama3.2:3b' if ollama_status else 'Not Available'
        }
    })

if __name__ == '__main__':
    print("\n🫀 Starting CVD Risk Assessment with Ollama Integration")
    print("=" * 60)
    print(f"✓ ML Model: {'Loaded' if model.model else 'Not Found'}")
    print(f"✓ LLM Available: {LLM_AVAILABLE}")
    print(f"✓ Ollama Connected: {ollama_status}")
    print("=" * 60)
    print("🌐 Server URLs:")
    print("   • http://localhost:5000")
    print("   • http://127.0.0.1:5000")
    print("=" * 60)
    print("📋 Features:")
    print("   • CVD Risk Assessment (93% accuracy)")
    print("   • London Environmental Data")
    print("   • Animated Heart Visualization")
    if ollama_status:
        print("   • 🦙 Local Llama Health Advice")
    else:
        print("   • 📝 Fallback Health Advice")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative configuration...")
        app.run(host='localhost', port=5000, debug=False)