import numpy as np
import joblib

# Load model
model = joblib.load('../models/model.pkl')

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Example usage
# if __name__ == "__main__":
#     sample_input = np.array([[value1, value2, value3, ...]])  # Replace with actual values
#     print(predict(sample_input))
