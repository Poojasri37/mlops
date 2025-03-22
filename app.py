from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained model (with preprocessing)
try:
    model = joblib.load("model.pkl")  # Ensure this path is correct
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from frontend
        data = request.json
        logging.debug(f"Received data: {data}")

        features = np.array(data["features"]).reshape(1, -1)

        # Ensure there are 34 features by filling missing ones with zeros
        if features.shape[1] < 34:
            missing_features = 34 - features.shape[1]
            features = np.hstack([features, np.zeros((1, missing_features))])
        logging.debug(f"Features after padding (should be 34): {features}")

        # Make prediction
        prediction = model.predict(features)
        logging.debug(f"Model prediction: {prediction}")

        # Make probability prediction
        probability = model.predict_proba(features)[0][1]  # Probability for Attrition class (1)
        logging.debug(f"Prediction probability: {probability}")

        return jsonify({"prediction": int(prediction[0]), "probability": float(probability)})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
return true
else:
  return index.html
