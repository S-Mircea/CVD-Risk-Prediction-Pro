"""
Verify the trained model file loads correctly.
Used by GitHub Actions CI pipeline.
"""
import os
import sys

try:
    import joblib
except ImportError:
    print("❌ joblib not installed")
    sys.exit(1)

model_path = os.path.join(os.path.dirname(__file__), 'cvd_risk_model.pkl')

if not os.path.exists(model_path):
    print(f"❌ Model file not found at {model_path}")
    sys.exit(1)

data = joblib.load(model_path)

model = data.get('model')
processor = data.get('data_processor')
model_type = data.get('model_type', 'unknown')

if model is None:
    print("❌ Model object is None")
    sys.exit(1)

print(f"✅ Model type: {type(model).__name__}")
print(f"✅ Data processor: {type(processor).__name__}")
print(f"✅ Model type label: {model_type}")
print(f"✅ Model file size: {os.path.getsize(model_path):,} bytes")
print("✅ Model verification passed!")
sys.exit(0)