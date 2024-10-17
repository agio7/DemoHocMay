from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load các mô hình đã lưu
models = {
    'linear_regression': joblib.load('linear_model.pkl'),
    'lasso': joblib.load('lasso_model.pkl'),
    'neural_network': joblib.load('mlp_model.pkl'),
    'stacking': joblib.load('stacking_model.pkl')
}

@app.route('/')
def index():
    return render_template('index.html', salary='')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age', type=float)
    education_level = request.form.get('education_level')
    experience = request.form.get('experience', type=float)
    model_name = request.form.get('model')

    # Kiểm tra tính hợp lệ của dữ liệu đầu vào
    if age < 0 or experience < 0:
        salary = "Invalid input: Age and years of experience must be non-negative."
        return render_template('index.html', salary=salary)

    # Mã hóa trình độ học vấn
    education_level_mapping = {
        "Bachelor's": 0,
        "Master's": 1,
        "PhD": 2
    }
    education_encoded = education_level_mapping[education_level]

    # Tạo mảng đầu vào cho mô hình
    input_features = np.array([[age, education_encoded, experience]])

    # Chọn mô hình để dự đoán
    model = models[model_name]
    
    # Dự đoán lương
    predicted_salary = model.predict(input_features)[0]

    # Định dạng số với khoảng cách
    formatted_salary = f"{predicted_salary:,.2f}".replace(',', ' ')

    return render_template('index.html', salary=f"Predicted Salary: {formatted_salary}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
