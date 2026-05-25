from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import uuid
import os
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dashboard import dashboard_bp

app = Flask(__name__)

# =========================
# BASE PATH (IMPORTANT FIX)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('about.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/project')
def project():
    return render_template('my_project.html')


# =========================
# ML FUNCTION
# =========================
def PredictpreprocessData(age, bmi, children, sex, smoker, region, n_bootstraps=1000):

    input_df = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }])

    # ------------------------
    # FIX MODEL PATH
    # ------------------------
    model_path = os.path.join(BASE_DIR, "best_insurance_model.pkl")
    model = joblib.load(model_path)

    preprocessor = model.named_steps['preprocess']
    regressor = model.named_steps['regressor']

    # ------------------------
    # TRANSFORM INPUT
    # ------------------------
    transformed_input = preprocessor.transform(input_df)

    # ------------------------
    # PREDICTION
    # ------------------------
    prediction = regressor.predict(transformed_input)[0]

    # ------------------------
    # BOOTSTRAP CI
    # ------------------------
    bootstrap_preds = []

    for _ in range(n_bootstraps):
        noise = np.random.normal(0, 0.05, transformed_input.shape)
        sample_input = transformed_input + noise
        pred = regressor.predict(sample_input)[0]
        bootstrap_preds.append(pred)

    lower_bound = np.percentile(bootstrap_preds, 2.5)
    upper_bound = np.percentile(bootstrap_preds, 97.5)

    # ------------------------
    # SHAP EXPLAINABILITY
    # ------------------------
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_input)

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed_input, "toarray"):
        transformed_df = pd.DataFrame(
            transformed_input.toarray(),
            columns=feature_names
        )
    else:
        transformed_df = pd.DataFrame(
            transformed_input,
            columns=feature_names
        )

    expected_value = explainer.expected_value

    # ------------------------
    # SAFE STATIC FOLDER FIX
    # ------------------------
    img_dir = os.path.join(BASE_DIR, "static", "img")
    os.makedirs(img_dir, exist_ok=True)

    image_id = str(uuid.uuid4())[:8]
    shap_path = os.path.join(img_dir, f"shap_{image_id}.png")

    # ------------------------
    # SHAP PLOT
    # ------------------------
    shap.initjs()
    plt.figure()

    shap.plots._waterfall.waterfall_legacy(
        expected_value=expected_value,
        shap_values=shap_values[0],
        features=transformed_df.iloc[0],
        feature_names=feature_names,
        show=False
    )

    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    filename = f"shap_{image_id}.png"

    shap_image = f"img/{filename}"

    return prediction, shap_image, lower_bound, upper_bound


# =========================
# HTML FORM PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = request.form['sex']
        smoker = request.form['smoker']
        region = request.form['region']

        prediction, shap_image, lower, upper = PredictpreprocessData(
            age, bmi, children, sex, smoker, region
        )

        return render_template(
            'prediction.html',
            predictions=round(prediction, 2),
            lower=round(lower, 2),
            upper=round(upper, 2),
            shap_image=shap_image
        )

    except Exception as e:
        return str(e)


# =========================
# BLUEPRINT
# =========================
app.register_blueprint(dashboard_bp)


# =========================
# API ENDPOINT
# =========================
@app.route('/api/predict', methods=['POST'])
def api_predict():

    data = request.get_json()

    prediction, shap_img_path, lower, upper = PredictpreprocessData(
        data['age'],
        data['bmi'],
        data['children'],
        data['sex'],
        data['smoker'],
        data['region']
    )

    return jsonify({
        "prediction": float(prediction),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "shap_image": shap_img_path
    })


# =========================
# RUN APP (RENDER SAFE)
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)