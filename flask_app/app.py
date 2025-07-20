from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Prometheus Metrics Setup
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# Load the model from local path
model_path = os.path.join("models", "model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)
print(f"✅ Loaded model from: {model_path}")

# Features expected from form
input_features = ['Passport', 'MaritalStatus', 'Age', 'ProductPitched',
                  'MonthlyIncome', 'NumberOfFollowups', 'Designation',
                  'PreferredPropertyStar']

categorical_features = ['MaritalStatus', 'ProductPitched', 'Designation']

@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        # Get input values from form
        data_dict = {feature: request.form.get(feature) for feature in input_features}

        # Convert to DataFrame
        df = pd.DataFrame([data_dict])

        # Convert numerical columns
        df['Age'] = pd.to_numeric(df['Age'])
        df['MonthlyIncome'] = pd.to_numeric(df['MonthlyIncome'])
        df['NumberOfFollowups'] = pd.to_numeric(df['NumberOfFollowups'])
        df['Passport'] = pd.to_numeric(df['Passport'])
        df['PreferredPropertyStar'] = pd.to_numeric(df['PreferredPropertyStar'])

        # Encode categorical columns using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Predict
        prediction = model.predict(df)[0]

        # Update metrics
        label = f"Class_{prediction}"
        PREDICTION_COUNT.labels(prediction=label).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        # Render result
        result = f"Prediction: {'ProdTaken-Yes' if prediction == 1 else 'ProdTaken-No'}"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error occurred: {e}")

# Prometheus metrics endpoint
@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True)
