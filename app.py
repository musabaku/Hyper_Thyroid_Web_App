from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import io
import os
from error_handling import InvalidUsage, register_error_handlers

app = Flask(__name__)

# Register error handlers
register_error_handlers(app)

# Load models and their corresponding training data
models = {
    'rf': {
        'model': joblib.load('models/rf_model.pkl'),
        'X_train': np.load(os.path.join('data', 'rf_X_train.npy')),
    },
    'xgb': {
        'model': joblib.load('models/xgb_model.pkl'),
        'X_train': np.load(os.path.join('data', 'xgb_X_train.npy')),
    },
    'lgbm': {
        'model': joblib.load('models/lgbm_model.pkl'),
        'X_train': np.load(os.path.join('data', 'lgbm_X_train.npy')),
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return handle_prediction(request, 'predict')

@app.route('/test', methods=['POST'])
def test():
    return handle_prediction(request, 'test')

def create_multiclass_target(y):
    return np.where((y[:, 0] == 0) & (y[:, 1] == 0), 0,
                    np.where((y[:, 0] == 1) & (y[:, 1] == 1), 1,
                             np.where((y[:, 0] == 0) & (y[:, 1] == 1), 2, 3)))

def handle_prediction(request, mode):
    try:
        if 'csv_file' not in request.files:
            raise InvalidUsage('No file uploaded')

        model_name = request.form.get('model_name')
        if model_name not in models:
            raise InvalidUsage('Invalid model name selected')

        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            raise InvalidUsage('No file selected')

        df = pd.read_csv(csv_file)
        X_input = df.iloc[:, 2:].values
        y_input = df.iloc[:, :2].values

        X_train = models[model_name]['X_train']
        if X_input.shape[1] != X_train.shape[1]:
            raise InvalidUsage(f"Input data has {X_input.shape[1]} features, but expected {X_train.shape[1]} features.")

        y_input_multiclass = create_multiclass_target(y_input)
        model = models[model_name]['model']

        y_pred_input = model.predict(X_input)

        reverse_mapping = {
            0: (0, 0),
            1: (1, 1),
            2: (0, 1),
            3: (1, 0)
        }

        decoded_predictions = [reverse_mapping[pred] for pred in y_pred_input]

        input_accuracy = accuracy_score(y_input_multiclass, y_pred_input)
        input_precision = precision_score(y_input_multiclass, y_pred_input, average='macro')
        input_recall = recall_score(y_input_multiclass, y_pred_input, average='macro')
        input_f1 = f1_score(y_input_multiclass, y_pred_input, average='macro')

        return jsonify({
            'model_name': model_name,
            'decoded_predictions': decoded_predictions,
            'accuracy': input_accuracy,
            'precision': input_precision,
            'recall': input_recall,
            'f1_score': input_f1
        })
    except InvalidUsage as e:
        return handle_invalid_usage(e)
    except Exception as e:
        return handle_exception(e)

if __name__ == '__main__':
    app.run(debug=True)
