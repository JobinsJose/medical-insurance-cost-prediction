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

#----Defining the endpoints for the application
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


def PredictpreprocessData(age, bmi, children, sex, smoker, region, n_bootstraps=1000):
    input_df = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }])

    # ----Load model pipeline
    model = joblib.load('best_insurance_model.pkl')
    preprocessor = model.named_steps['preprocess']
    regressor = model.named_steps['regressor']

    # ----Transform input
    transformed_input = preprocessor.transform(input_df)

    # -----Predict base value
    prediction = regressor.predict(transformed_input)[0]

    # -----Bootstrapped Confidence Interval
    bootstrap_preds = []
    for _ in range(n_bootstraps):
        #-----Sample with replacement (simulate input uncertainty)
        noise = np.random.normal(0, 0.05, transformed_input.shape)
        sample_input = transformed_input + noise
        pred = regressor.predict(sample_input)[0]
        bootstrap_preds.append(pred)

    lower_bound = np.percentile(bootstrap_preds, 2.5)
    upper_bound = np.percentile(bootstrap_preds, 97.5)

    # ----SHAP explanation
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(transformed_input)

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(transformed_input, "toarray"):
        transformed_df = pd.DataFrame(transformed_input.toarray(), columns=feature_names)
    else:
        transformed_df = pd.DataFrame(transformed_input, columns=feature_names)

    expected_value = explainer.expected_value

    # ----SHAP Plot
    shap.initjs()
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value=expected_value,
        shap_values=shap_values[0],
        features=transformed_df.iloc[0],
        feature_names=feature_names,
        show=False
    )
     #------ saving the plot into folder
    image_id = str(uuid.uuid4())[:8]
    shap_path = f"static/img/shap_{image_id}.png"
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()

    return prediction, shap_path, lower_bound, upper_bound


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    sex = request.form['sex']
    smoker = request.form['smoker']
    region = request.form['region']

    prediction, shap_img_path, lower, upper = PredictpreprocessData(age, bmi, children, sex, smoker, region)

    return render_template(
        'prediction.html',
        predictions=round(prediction, 2),
        lower=round(lower, 2),
        upper=round(upper, 2),
        shap_image=shap_img_path
    )

app.register_blueprint(dashboard_bp)


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
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)