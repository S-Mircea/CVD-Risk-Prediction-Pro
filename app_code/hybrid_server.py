#!/usr/bin/env python3
"""
Hybrid server that runs Flask backend on one port and serves frontend on another
This approach ensures the interface works while providing full backend functionality
"""
import threading
import time
import webbrowser
import http.server
import socketserver
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from ml_model import CVDRiskModel

# Flask backend setup
app = Flask(__name__)

# Initialize model
model = CVDRiskModel()
if not model.load_model():
    print("Warning: Model not found. Please train the model first.")

# Try to import LLM advisor
try:
    from llm_advisor import CVDLlamaAdvisor
    LLM_AVAILABLE = True
    print("✓ LLM advisor imported successfully")
except Exception as e:
    print(f"⚠ LLM advisor not available: {e}")
    LLM_AVAILABLE = False

# Initialize LLM advisor if available
ollama_status = False
if LLM_AVAILABLE:
    try:
        llm_advisor = CVDLlamaAdvisor()
        ollama_status = llm_advisor.check_ollama_availability()
        if ollama_status:
            print("✓ Ollama connected successfully")
        else:
            print("⚠ Ollama not available - using fallback")
    except Exception as e:
        print(f"⚠ LLM advisor failed: {e}")
        LLM_AVAILABLE = False

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
        
        # Load environmental data
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
        
        # Add environmental data
        result['environmental_data'] = {
            'pm25': user_data['Avg_PM25'],
            'no2': user_data['Avg_NO2'],
            'borough': user_data['Borough']
        }
        
        # Add recommendations
        result['recommendations'] = get_recommendations(result['risk_level'])
        
        # Generate advice
        if LLM_AVAILABLE and ollama_status:
            try:
                llm_advice = llm_advisor.get_environmental_advice(
                    result['risk_level'], result['environmental_data'], user_data)
                result['llm_advice'] = llm_advice
                result['llm_available'] = True
                print(f"✓ Generated Ollama advice for {result['risk_level']} risk")
            except Exception as e:
                print(f"⚠ Ollama failed: {e}")
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
            "Monitor air quality in your area"
        ],
        'High Risk': [
            "Seek immediate medical consultation",
            "Comprehensive cardiovascular assessment needed",
            "Urgent lifestyle intervention required",
            "Consider relocation if air quality is poor"
        ]
    }
    return recommendations.get(risk_level, ["Consult with healthcare provider"])

def get_fallback_advice(risk_level, borough):
    borough_advice = {
        'Tower Hamlets': 'High pollution - exercise in Mile End Park, avoid busy roads',
        'Camden': 'Urban environment - use Regent\'s Park, check air quality',
        'Westminster': 'Very high traffic - exercise early morning in St James\'s Park',
        'Hackney': 'Above-average pollution - use Victoria Park',
        'Richmond upon Thames': 'Excellent air quality - enjoy Richmond Park',
        'Kingston upon Thames': 'Good air quality - riverside walks ideal'
    }
    
    risk_advice = {
        'Low Risk': 'Maintain healthy lifestyle while monitoring air quality.',
        'Moderate Risk': 'Increase exercise in green spaces, monitor pollution.',
        'High Risk': 'Prioritize indoor exercise on high pollution days.'
    }
    
    borough_tip = borough_advice.get(borough, f'Monitor air quality in {borough}')
    risk_tip = risk_advice.get(risk_level, 'Consult healthcare provider')
    
    return f"{risk_tip} {borough_tip}"

# Frontend server class
class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/templates/index.html'
        return super().do_GET()
    
    def do_POST(self):
        # Proxy POST requests to Flask backend
        import urllib.request
        import urllib.parse
        import json
        
        if self.path == '/assess_risk':
            try:
                # Get form data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                print(f"Proxying request to backend: {len(post_data)} bytes")
                
                # Forward to Flask backend
                req = urllib.request.Request(
                    'http://127.0.0.1:5001/assess_risk',
                    data=post_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = response.read()
                    print(f"Backend responded: {response.status}")
                
                # Send response back
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(result)
                
            except Exception as e:
                print(f"⚠ Backend request failed: {e}")
                self.send_response(200)  # Send 200 to avoid browser errors
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = json.dumps({
                    "success": False, 
                    "error": f"Backend connection failed: {str(e)}"
                })
                self.wfile.write(error_response.encode())
        else:
            super().do_POST()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_flask_backend():
    """Run Flask backend on port 5001"""
    try:
        print("🔧 Starting Flask backend on port 5001...")
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"⚠ Flask backend failed: {e}")

def run_frontend_server():
    """Run frontend server on port 8080"""
    try:
        os.chdir('/Users/mirceaserban/Desktop/Cardio_project_test1/app_code')
        with socketserver.TCPServer(("", 8080), FrontendHandler) as httpd:
            print("🌐 Frontend server running on http://localhost:8080")
            httpd.serve_forever()
    except Exception as e:
        print(f"⚠ Frontend server failed: {e}")

def open_browser():
    """Open browser after servers start"""
    time.sleep(3)
    webbrowser.open('http://localhost:8080')

if __name__ == "__main__":
    print("\n🫀 CVD Assessment with Hybrid Server Architecture")
    print("=" * 55)
    print(f"✓ ML Model: {'Loaded' if model.model else 'Not Found'}")
    print(f"✓ LLM Available: {LLM_AVAILABLE}")
    print(f"✓ Ollama Connected: {ollama_status}")
    print("=" * 55)
    print("🚀 Starting servers:")
    print("   • Frontend: http://localhost:8080")
    print("   • Backend: http://127.0.0.1:5001")
    print("=" * 55)
    
    # Start Flask backend in thread
    backend_thread = threading.Thread(target=run_flask_backend, daemon=True)
    backend_thread.start()
    
    # Start browser opener in thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run frontend server in main thread
    try:
        run_frontend_server()
    except KeyboardInterrupt:
        print("\n✓ Servers stopped")