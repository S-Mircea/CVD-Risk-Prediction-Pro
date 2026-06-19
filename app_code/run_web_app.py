import os
import sys
import argparse
from train_model import main as train_model
from web_app import app

def main():
    parser = argparse.ArgumentParser(description='CVD Risk Assessment Web Application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    args = parser.parse_args()
    
    print("CVD Risk Assessment Web Application")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists('cvd_risk_model.pkl'):
        print("Model not found. Training new model...")
        train_model()
        print("\nModel training completed!")
    
    print("\nStarting web application...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main()