## This README outlines our project's directory structure so everyone knows exactly where to find and save their work. Please adhere to this structure to ensure our final submission aligns perfectly with the grading rubric.

### Repository Structure 

UAE-CAR-PRICE-PREDICTOR/
│
├── app/                        # Frontend UI and Backend API
│   ├── app.py                  # (Member 3) Flask REST API serving the prediction model.
│   ├── index.html              # (Member 5) The main .html page.
│   ├── script.js               # (Member 5) Handles API calls to Flask and dynamic DOM updates.
│   └── style.css               # (Member 5) Styling for the web application.
│
├── data/                       # Datasets, Member 1 must save cleaned dataset here which everyone must use to train/test the models. 
│   └── DriveArabia_All_uae_updated.csv # The raw dataset.
│
│
├── models/                     # Saved Model Files
│   # NOTE: Save your trained models here using joblib or keras.save()
│   # Expected files to be added: xgboost_model.pkl, rf_model.pkl, etc.
│
├── notebooks/                  # Model Development & Training
│   ├── 01_eda_preprocessing_linear_regression.ipynb  # (Member 1) Data cleaning pipeline & Baseline Linear Regression.
│   ├── 02_random_forest.ipynb                        # (Member 2) Random Forest Regressor & Feature Importance analysis.
│   ├── 03_xgboost.ipynb                              # (Member 3) XGBoost Regressor (Lead Model) & SHAP analysis.
│   ├── 04_svr_ridge.ipynb                            # (Member 4) SVR and Ridge Regression models for regularisation comparison.
│   └── 05_neural_network.ipynb                       # (Member 5) Deep Learning (MLP) Neural Network built with Keras.
│
├── README.md                   # This documentation file.
└── requirements.txt            # Project dependencies (pandas, scikit-learn, xgboost, flask, etc.)


