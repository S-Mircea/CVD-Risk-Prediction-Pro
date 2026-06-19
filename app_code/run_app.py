import os
import sys
from train_model import main as train_model
from gui_app import main as run_gui

def main():
    print("CVD Risk Assessment Application")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists('cvd_risk_model.pkl'):
        print("Model not found. Training new model...")
        train_model()
        print("\nModel training completed!")
    
    print("\nStarting GUI application...")
    run_gui()

if __name__ == "__main__":
    main()