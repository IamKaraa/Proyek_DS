from flask import Flask, render_template, request, jsonify
from model_util import generate_random_features, get_model, load_model, set_features
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard_view.html')

@app.route('/model')
def get_model_new():
    # model = get_model()
    # if model:
    #     print("Model downloaded and saved successfully.")
    # else:
    #     print("Failed to download and save the model.")
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    features_names = set_features()
    model = load_model()
    
    if request.method == "POST":
        # Ambil data dari form
        features = {name: int(request.form.get(name, 0)) for name in features_names}
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]
        prediction = "Yes" if pred == 1 else "No"
        
        # Balas dengan JSON (untuk Modal)
        return jsonify({"prediction": prediction, "features": features})
    
    # Untuk GET (tampilan awal)
    features = generate_random_features()
    return render_template("predict_view.html", features=features, prediction=None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    features = {
        'MonthlyIncome': data.get('MonthlyIncome'),
        'Age': data.get('Age'),
        'TotalWorkingYears': data.get('TotalWorkingYears'),
        'OverTime': data.get('OverTime'),
        'MonthlyRate': data.get('MonthlyRate'),
        'DailyRate': data.get('DailyRate'),
        'DistanceFromHome': data.get('DistanceFromHome'),
        'HourlyRate': data.get('HourlyRate'),
        'NumCompaniesWorked': data.get('NumCompaniesWorked')
    }
    # Real prediction using model
    model = load_model()
    import pandas as pd
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    prediction = "Yes" if pred == 1 else "No" if pred == 0 else str(pred)
    return jsonify({
        'prediction': prediction,
        'features': features
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)