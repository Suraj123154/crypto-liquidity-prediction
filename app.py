from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("models/best_model.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from HTML form
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_input)[0]

        # Return result to web page
        return render_template('index.html',
                               prediction_text=f"Predicted Liquidity Ratio: {prediction:.4f}")
    except Exception as e:
        # Handle any errors gracefully
        return render_template('index.html',
                               prediction_text=f"Error: {e}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)