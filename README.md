# UAE Car Price Predictor

A machine learning project that predicts used car prices in the United Arab Emirates using the Dubizzle dataset. This comprehensive system includes data exploration, multiple predictive models, and a production-ready Flask web application with an intuitive user interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Key Features](#key-features)
- [Results & Performance](#results--performance)
- [Team & Responsibilities](#team--responsibilities)

---

## Project Overview

This project aims to accurately predict used car prices in the UAE market using machine learning techniques. The dataset is sourced from Dubizzle, the leading online classifieds platform in the Middle East. The system supports multiple state-of-the-art regression models and deploys the best-performing model (XGBoost) through a REST API with a user-friendly web interface.

**Key Goals:**
- Clean and preprocess automotive market data
- Develop and compare multiple regression models
- Interpret model predictions using explainability techniques (SHAP)
- Deploy a production-ready prediction service
- Provide an intuitive UI for non-technical users

---

## Dataset

### Source
[Dubizzle Dataset (~10,000 cars)](https://www.kaggle.com/datasets/alihassankp/dubizzle-used-car-sale-data)

### Data Characteristics
- **Records:** 10,000 used vehicles
- **Features:** Car make, model, year, mileage, transmission, fuel type, body type, condition, regional specifications, and more
- **Target Variable:** Price (AED)

### Data Files
- `data/data.csv` - Raw dataset (unprocessed)
- `data/dubizzle_cleaned.csv` - Cleaned and preprocessed dataset (ready for modeling)

### Preprocessing Steps
- Handling missing values
- Outlier detection and removal
- Feature encoding (One-Hot Encoding for categorical variables)
- Feature normalization
- Feature engineering (e.g., age calculation, km/year ratio)

---

## Project Structure

```
uae-car-price-predictor/
│
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
│
├── 📁 app/                                # Flask Application & Web UI
│   ├── 🐍 app.py                          # Flask REST API server
│   ├── 📁 templates/
│   │   └── 📄 index.html                  # Frontend HTML interface
│   ├── 📁 static/
│   │   ├── 📁 css/
│   │   │   └── 📄 style.css               # Web UI styling
│   │   └── 📁 js/
│   │       └── 📄 script.js               # Frontend logic & API calls
│   └── 📁 models/                         # Saved trained models
│       ├── xgboost_model.pkl              # Deployed XGBoost model
│       └── feature_columns.pkl            # Feature column metadata
│
├── 📁 data/                               # Datasets
│   ├── 📄 data.csv
│   └── 📄 dubizzle_cleaned.csv            # Primary cleaned dataset
│
├── 📁 notebooks/                          # Model Development & Analysis
│   ├── 📄 01_eda_preprocessing_linear_regression.ipynb
│   │   └── EDA, data cleaning, baseline linear regression
│   ├── 📄 02_random_forest.ipynb
│   │   └── Random Forest model & feature importance analysis
│   ├── 📄 03_xgboost.ipynb
│   │   └── XGBoost model (primary) & SHAP explainability
│   ├── 📄 04_Ridge_SVR.ipynb
│   │   └── Ridge Regression & Support Vector Regressor
│   └── 📄 05_neural_network.ipynb
│       └── Deep learning with TensorFlow/Keras MLP
│
├── 📁 models/                             # Serialized model artifacts
│   └── [Trained model files saved here]
│
└── 📁 plots/                              # Generated visualizations
    └── [Analysis plots and charts]

```

---

## Models Implemented

### 1. **Linear Regression** (Baseline)
- Simple linear model for baseline performance
- Used to establish performance benchmarks
- Fast training and inference

### 2. **Random Forest Regressor**
- Ensemble method with multiple decision trees
- Feature importance analysis
- Good generalization on non-linear relationships

### 3. **XGBoost** ⭐ (Primary Model)
- Gradient boosting framework
- Best overall performance
- Deployed in production
- SHAP analysis for model interpretability

### 4. **Ridge Regression**
- L2 regularization to prevent overfitting
- Comparison with other regularization techniques

### 5. **Support Vector Regressor (SVR)**
- Kernel-based regression
- Excellent for high-dimensional data
- Non-linear decision boundaries

### 6. **Neural Network (Deep Learning)**
- Multi-Layer Perceptron (MLP) with TensorFlow/Keras
- Non-linear activation functions
- Batch normalization and dropout for regularization

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd uae-car-price-predictor
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas              # Data manipulation
numpy               # Numerical computing
matplotlib          # Plotting
seaborn             # Statistical visualization
scikit-learn        # Machine learning
xgboost             # Gradient boosting
shap                # Model explainability
tensorflow          # Deep learning
keras               # Neural network API
flask               # Web framework
joblib              # Model serialization
```

---

## Running the Application

### Option 1: Run Flask Server
```bash
cd app
flask run --port 5000
```

Or directly:
```bash
cd app
python app.py
```

The application will be available at: `http://localhost:5000`

### Option 2: Using Python
```bash
python app/app.py
```

---

## API Documentation

### Endpoint: `POST /predict`

**Description:** Predict car price based on vehicle specifications

**Base URL:** `http://localhost:5000`

### Request Format

```json
{
  "make": "toyota",
  "model": "land-cruiser",
  "year": 2020,
  "mileage": 65000,
  "city": "Dubai",
  "transmission": "Automatic",
  "fuel_type": "Gasoline",
  "body_type": "SUV",
  "cylinders": 8,
  "horsepower": 400,
  "regional_specs": "GCC Specs",
  "color": "White",
  "seller_type": "Owner",
  "body_condition": "Perfect inside and out",
  "mechanical_condition": "Perfect inside and out"
}
```

### Response Format

```json
{
  "predicted_price": 125000,
  "currency": "AED",
  "confidence": 0.92,
  "model": "XGBoost"
}
```

### cURL Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "toyota",
    "model": "land-cruiser",
    "year": 2020,
    "mileage": 65000,
    "city": "Dubai",
    "transmission": "Automatic",
    "fuel_type": "Gasoline",
    "body_type": "SUV",
    "cylinders": 8,
    "horsepower": 400,
    "regional_specs": "GCC Specs",
    "color": "White",
    "seller_type": "Owner",
    "body_condition": "Perfect inside and out",
    "mechanical_condition": "Perfect inside and out"
  }'
```

### Supported Categories

**Emirates/Cities:**
- Abu Dhabi, Dubai, Sharjah, Ajman, Ras Al Khaimah, Fujairah, Umm Al Quwain

**Body Types:**
- SUV, Sedan, Hatchback, Crossover, Wagon, Van, Pick Up Truck, Sports Car, Convertible, Utility Truck

**Fuel Types:**
- Gasoline, Electric, Hybrid

**Colors:**
- Black, White, Silver, Grey, Red, Blue, Brown, Gold, Green, Orange, Yellow, Purple, Burgundy, Tan, Teal

**Regional Specs:**
- GCC Specs, North American Specs, Japanese Specs, Other

---

## Key Features

✅ **Multiple ML Models** - Compare Linear, Tree-based, and Deep Learning approaches
✅ **Model Explainability** - SHAP analysis for feature importance
✅ **Production-Ready** - Flask REST API with error handling
✅ **Intuitive UI** - Clean web interface for price predictions
✅ **Feature Engineering** - Advanced preprocessing pipeline
✅ **Data Validation** - Graceful handling of unknown categories
✅ **Type Hints** - Python type annotations throughout
✅ **Comprehensive Documentation** - Detailed notebooks and code comments

---

## Results & Performance

### Model Comparison

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 49,001.59 | 32,063.14 | 0.7538 |
| Random Forest | 39,322.55 | 22,906.33 | 0.8415 |
| **XGBoost** ⭐ | 29,388 | 17,566 | 0.905 |
| Ridge Regression | 41,029 | 24,238 | 0.8257 |
| SVR | 47,248 | 27,895 | 0.7689 |
| Neural Network | 31,524 | 18,423 | 0.8981 |

*Note: Specific metrics can be found in individual notebook files (01-05)*

### Top Predictive Features (XGBoost)
Features are identified through:
- Feature importance analysis
- SHAP values for local explanations
- Correlation analysis

Common high-impact features:
- Vehicle make and model
- Age of vehicle
- Mileage (kilometers)
- Regional specifications
- Fuel type and engine specifications

---

## Team & Responsibilities

| Member | Role | Notebooks/Files |
|--------|------|-----------------|
| Member 1 | Data Science Lead | `01_eda_preprocessing_linear_regression.ipynb` - EDA & Baseline |
| Member 2 | ML Engineer | `02_random_forest.ipynb` - Random Forest & Feature Analysis |
| Member 3 | Backend Developer | `03_xgboost.ipynb` + `app/app.py` - XGBoost & API |
| Member 4 | Model Specialist | `04_Ridge_SVR.ipynb` - Regularization & SVR |
| Member 5 | Frontend Developer | `05_neural_network.ipynb` + `app/templates/` - Neural Network & UI |

---

## Development Workflow

### Adding New Models
1. Create a new notebook: `06_model_name.ipynb`
2. Follow the preprocessing pipeline from notebook 01
3. Train and evaluate the model
4. Save the model using `joblib.dump()` or `model.save()`

### Updating the Frontend
1. Modify `app/templates/index.html` for structure
2. Update `app/static/css/style.css` for styling
3. Modify `app/static/js/script.js` for API calls and interactions

### Deploying to Production
1. Update the model in `app/models/`
2. Ensure `feature_columns.pkl` is up-to-date
3. Test the API thoroughly
4. Deploy Flask app to production server

---

## Notes

- **Primary Dataset:** Use `data/dubizzle_cleaned.csv` for all model training
- **Model Serialization:** Use `joblib` for sklearn models and `model.save()` for Keras
- **Feature Alignment:** Always ensure predictions use the same features as training
- **Missing Values:** Handle gracefully by defaulting to reference categories

---

## Contributing

When contributing to this project:
1. Keep notebooks well-organized and documented
2. Add comments explaining key decisions
3. Update this README if adding new features
4. Test API endpoints before committing
5. Maintain consistent code style

---



