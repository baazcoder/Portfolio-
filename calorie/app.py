from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import joblib
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyFloatEncoder, self).default(obj)

app.json_encoder = NumpyFloatEncoder

# Load the trained XGBoost model
try:
    model = joblib.load('model/calorie_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Validate request contains JSON
    if not request.is_json:
        raise BadRequest("Request must be JSON")
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['gender', 'age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp']
        if not all(field in data for field in required_fields):
            raise ValueError("Missing required fields")
        
        # Create input DataFrame with strict type conversion
        input_data = pd.DataFrame([{
            'Gender': int(data['gender']),  # 0=Male, 1=Female
            'Age': float(data['age']),
            'Height': float(data['height']),
            'Weight': float(data['weight']),
            'Duration': float(data['duration']),
            'Heart_Rate': float(data['heart_rate']),
            'Body_Temp': float(data['body_temp'])
        }])
        
        # Validate feature names match model
        if list(input_data.columns) != model.get_booster().feature_names:
            raise ValueError(f"Features don't match model. Expected: {model.get_booster().feature_names}")
        
        # Make prediction and ensure proper float conversion
        prediction = model.predict(input_data)[0]
        calories = float(prediction) if isinstance(prediction, (np.floating, float)) else float(prediction.item())
        
        return jsonify({
            'status': 'success',
            'prediction': round(calories, 2),
            'units': 'kcal'
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f"Invalid input: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)