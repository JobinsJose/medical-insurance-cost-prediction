from flask import Blueprint, render_template
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

dashboard_bp = Blueprint('dashboard', __name__)
model = joblib.load('best_insurance_model.pkl')

# --------Assume test data is available
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').values.ravel()

@dashboard_bp.route('/dashboard')
def show_dashboard():
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocess']
    X_test_transformed = preprocessor.transform(X_test)

    y_pred = regressor.predict(X_test_transformed)
    residuals = y_test - y_pred

    #-------- Metrics
    r2 = round(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2), 4)
    rmse = round(np.sqrt(np.mean((y_test - y_pred) ** 2)), 2)
    mae = round(np.mean(np.abs(y_test - y_pred)), 2)

    # --------Plot 1: Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig("static/img/actual_vs_pred.png")
    plt.close()

    # --------Plot 2: Residual plot
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig("static/img/residual_plot.png")
    plt.close()

    # --------Plot 3: SHAP summary
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_transformed)
    shap.summary_plot(shap_values, X_test_transformed, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.savefig("static/img/shap_summary.png", bbox_inches="tight")
    plt.close()

    # --------Plot 4: Histogram of residuals
    plt.figure()
    plt.hist(residuals, bins=30, color='gray', edgecolor='black')
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.savefig("static/img/residual_hist.png")
    plt.close()

    return render_template('dashboard.html', r2=r2, rmse=rmse, mae=mae)