import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import joblib
from flask import Blueprint, render_template

dashboard_bp = Blueprint('dashboard', __name__)

# =========================
# BASE PATH FIX (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# SAFE PATHS
# =========================
model_path = os.path.join(BASE_DIR, "best_insurance_model.pkl")
x_test_path = os.path.join(BASE_DIR, "X_test.csv")
y_test_path = os.path.join(BASE_DIR, "y_test.csv")

model = joblib.load(model_path)

X_test = pd.read_csv(x_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()


@dashboard_bp.route('/dashboard')
def show_dashboard():

    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocess']

    X_test_transformed = preprocessor.transform(X_test)

    y_pred = regressor.predict(X_test_transformed)
    residuals = y_test - y_pred

    # =========================
    # METRICS
    # =========================
    r2 = round(1 - np.sum((y_test - y_pred) ** 2) /
               np.sum((y_test - np.mean(y_test)) ** 2), 4)

    rmse = round(np.sqrt(np.mean((y_test - y_pred) ** 2)), 2)
    mae = round(np.mean(np.abs(y_test - y_pred)), 2)

    # =========================
    # SAFE STATIC PATH FIX
    # =========================
    img_dir = os.path.join(BASE_DIR, "static", "img")
    os.makedirs(img_dir, exist_ok=True)

    # =========================
    # PLOT 1: ACTUAL VS PRED
    # =========================
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')

    plt.savefig(os.path.join(img_dir, "actual_vs_pred.png"))
    plt.close()

    # =========================
    # PLOT 2: RESIDUALS
    # =========================
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.savefig(os.path.join(img_dir, "residual_plot.png"))
    plt.close()

    # =========================
    # PLOT 3: SHAP SUMMARY
    # =========================
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_transformed)

    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=preprocessor.get_feature_names_out(),
        show=False
    )

    plt.savefig(os.path.join(img_dir, "shap_summary.png"), bbox_inches="tight")
    plt.close()

    # =========================
    # PLOT 4: HISTOGRAM
    # =========================
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(img_dir, "residual_hist.png"))
    plt.close()

    return render_template(
        'dashboard.html',
        r2=r2,
        rmse=rmse,
        mae=mae
    )
