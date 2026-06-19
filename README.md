# CVD Risk Prediction Pro 🏥

> **ML-powered cardiovascular disease risk assessment** — improved edition with CI/CD, Docker, model comparison, and production-ready tooling.

[![ML Pipeline](https://github.com/S-Mircea/CVD-Risk-Prediction-Pro/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/S-Mircea/CVD-Risk-Prediction-Pro/actions/workflows/ml_pipeline.yml)

A machine learning application that assesses cardiovascular disease risk by combining personal health data with London-specific environmental factors (air quality, pollution, noise). Built with a production-grade toolchain from day one.

## ✨ What's New (vs. Original)

| Feature | Original | Pro Edition |
|---------|----------|-------------|
| 🐳 Docker | ❌ | ✅ One-command deploy |
| 🔄 CI/CD Pipeline | ❌ | ✅ GitHub Actions auto-test & train |
| 🧪 Unit Tests | ❌ | ✅ 10+ tests (model, data, environment) |
| 📊 Model Comparison | ❌ | ✅ RF vs XGBoost vs SVM vs Logistic Regression |
| 🚀 Quick Start | Manual pip | `docker compose up` |

## 🚀 Quick Start

### Option 1: Docker (recommended)

```bash
docker compose up
# Open http://localhost:5002
```

### Option 2: Manual

```bash
pip install -r app_code/requirements.txt
cd app_code && python web_app.py
# Open http://127.0.0.1:5002
```

## 📊 Model Comparison

Run the comparison script to see which model performs best:

```bash
cd app_code && python compare_models.py
```

This evaluates **Random Forest**, **Gradient Boosting**, **Logistic Regression**, **SVM (RBF)**, and optionally **XGBoost** — with a summary table of accuracy, precision, recall, F1, ROC AUC, and training time.

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## 🐳 Docker

| Command | What it does |
|---------|-------------|
| `docker compose up` | Start the app |
| `docker compose up -d` | Start in background |
| `docker compose down` | Stop and clean up |

## 🏗️ Architecture

```
CVD-Risk-Prediction-Pro/
├── app_code/               # Application code
│   ├── web_app.py          # Flask web server
│   ├── ml_model.py         # ML model class
│   ├── data_processor.py   # Data preprocessing
│   ├── train_model.py      # Model training
│   ├── compare_models.py   # 🔥 NEW: Multi-model comparison
│   └── requirements.txt    # Dependencies
├── tests/                  # 🔥 NEW: Unit tests
│   └── test_model.py
├── .github/workflows/      # 🔥 NEW: CI/CD pipeline
│   └── ml_pipeline.yml
├── Dockerfile              # 🔥 NEW: Container image
├── docker-compose.yml      # 🔥 NEW: Orchestration
└── environmental_data/     # London borough data
```

## 📈 Model Performance (Original)

- **Algorithm**: Random Forest (200 estimators)
- **Cross-Validation Accuracy**: 93.0% (±3.8%)
- **Features**: 21 total (15 health + 6 environmental)
- **Coverage**: All 33 London boroughs

> ⚠️ Current model trained on synthetic data. Real-world performance will differ. See [Future Enhancements](#-future-enhancements) for real data roadmap.

## 🗺️ London Environmental Data

The app integrates borough-specific data:
- **Air Quality**: PM2.5 and NO2 pollution levels
- **Urban Factors**: Noise levels, green space percentage
- **Lifestyle**: Walkability scores, urban heat island effects

## 🎯 Future Enhancements

- [ ] **Real-world data**: NHANES / UK Biobank integration (see `data/download_real_data.py`)
- [ ] Real-time London air quality via LAQN API
- [ ] Genetic profiling markers
- [ ] User accounts & risk tracking over time
- [ ] Mobile app (React Native)
- [ ] Model explainability (SHAP/LIME)

## 📋 Requirements

- Python 3.8+ (manual)
- Docker & Docker Compose (containerized)
- 500MB free disk space

## ⚕️ Medical Disclaimer

**For educational and research purposes only.** Not a substitute for professional medical advice. Always consult a qualified healthcare provider.

## 👤 Author

**Mircea Serban** — [@S-Mircea](https://github.com/S-Mircea)

---

*Built with love for better cardiovascular health awareness ❤️*
