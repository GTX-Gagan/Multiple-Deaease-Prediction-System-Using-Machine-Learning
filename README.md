🚀 Multiple Disease Prediction System (Machine Learning)

This project provides ML-based prediction for Diabetes, Heart Disease, and Parkinson’s Disease, using trained classification models deployed through a lightweight Python application.

📌 Overview

The repository includes Jupyter notebooks for model training, CSV datasets, and serialized .sav model files used by a demo application (app.py). Each notebook contains data preprocessing, exploratory analysis, model training, and evaluation.

Supported predictions:

Diabetes

Heart Disease

Parkinson’s Disease

📂 Project Structure
├── app.py                         # Main application
├── requirements.txt               # Dependencies
├── diabetes.csv                   # Dataset
├── heart.csv
├── parkinsons.csv
├── diabetes_model.sav             # Trained models
├── heart_disease_model.sav
├── parkinsons_model.sav
├── Multiple disease prediction system - diabetes.ipynb
├── Multiple disease prediction system - heart.ipynb
└── Multiple disease prediction system - Parkinsons.ipynb

⚙️ Quick Start
1️⃣ Clone the repository
git clone https://github.com/GTX-Gagan/Multiple-Deaease-Prediction-System-Using-Machine-Learning.git
cd Multiple-Deaease-Prediction-System-Using-Machine-Learning

2️⃣ Install dependencies
python -m venv .venv
source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

3️⃣ Run the application
python app.py


If the app uses Streamlit, run:

streamlit run app.py

4️⃣ Open notebooks for model training

Run any notebook in Jupyter or VS Code to reproduce training results.

🧠 Model Training & Files

Each notebook:

Loads dataset

Performs EDA

Applies preprocessing

Trains ML models (Logistic Regression, SVM, Random Forest, etc.)

Evaluates metrics (Accuracy, Precision, Recall, F1-score)

Models are saved as .sav using pickle/joblib.

Loading example:

import joblib
model = joblib.load("diabetes_model.sav")

📊 Evaluation

Recommended metrics:

Accuracy

Precision / Recall / F1

ROC–AUC

Confusion Matrix

You can extend the notebooks to add cross-validation or calibration checks.

🌐 Deployment
Docker (optional)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]


Build & run:

docker build -t disease-predictor .
docker run -p 5000:5000 disease-predictor

🔌 API / Usage (Suggested)
POST /predict/diabetes
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BMI": 32.5,
  ...
}


Response:

{
  "prediction": 1,
  "probability": 0.87
}


Implement similarly for heart and Parkinson’s predictions.

🔒 Security Notes

Never load pickle .sav files from untrusted sources.

Validate all user inputs on API endpoints.

Pin dependency versions before production deployment.

🚧 Future Enhancements

Add FastAPI backend for structured REST endpoints

Better UI for multi-disease prediction

Add model versioning and experiment tracking

Export models to ONNX for portable serving

🤝 Contributing

Fork the repo

Create a feature branch

Submit a pull request with clear description

📜 License

Add an MIT or Apache-2.0 license if you plan to make this project open-source friendly.
