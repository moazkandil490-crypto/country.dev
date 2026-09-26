
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained scaler, KMeans model, and feature columns
scaler = joblib.load('scaler.joblib')
kmeans_model = joblib.load('kmeans_model.joblib')
feature_columns = joblib.load('feature_columns.joblib') # This list includes original features + country_dummies

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        
        # Create a DataFrame from the input data
        input_data_df = pd.DataFrame([data])

        # Prepare a DataFrame with all expected feature columns, initialized to 0
        # This handles missing columns in input and maintains order
        processed_df = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Populate numeric columns from input_data_df
        numeric_cols_in_input = [col for col in input_data_df.columns if col in feature_columns and not col.startswith('country_')]
        for col in numeric_cols_in_input:
            if col in input_data_df.columns:
                processed_df[col] = input_data_df[col]

        # Handle 'country' column for one-hot encoding
        if 'country' in input_data_df.columns:
            country_name = input_data_df['country'].iloc[0]
            one_hot_col_name = f'country_{country_name}'
            if one_hot_col_name in feature_columns:
                processed_df[one_hot_col_name] = 1
            # If the country is not seen during training, its one-hot encoded column will remain 0,
            # which is the correct way to handle unseen categories for one-hot encoding.

        # The processed_df now has the correct columns, order, and values (including one-hot encoding)
        # It's ready for scaling

        # Scale the features
        scaled_features = scaler.transform(processed_df)

        # Predict the cluster
        cluster_prediction = kmeans_model.predict(scaled_features)[0]

        return jsonify({'cluster': int(cluster_prediction)})
    
    return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
