from flask import Flask, render_template, request
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model dan scaler
model = load_model('lstm_calories_model_no_metric.h5')

scaler = joblib.load('scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# Halaman Utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        proteins = float(request.form['proteins'])
        fat = float(request.form['fat'])
        carbohydrate = float(request.form['carbohydrate'])

        # Buat DataFrame untuk data baru
        new_data = pd.DataFrame({
            'proteins': [proteins],
            'fat': [fat],
            'carbohydrate': [carbohydrate]
        })

        # Normalisasi data baru
        new_data_scaled = scaler.transform(new_data)

        # Reshape data untuk LSTM ([samples, timesteps, features])
        new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

        # Prediksi
        predicted_scaled = model.predict(new_data_reshaped)
        predicted_calories = y_scaler.inverse_transform(predicted_scaled)

        # Hasil prediksi
        result = float(predicted_calories[0][0])

        # Tampilkan hasil di halaman result.html
        return render_template('result.html', calories=result)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
