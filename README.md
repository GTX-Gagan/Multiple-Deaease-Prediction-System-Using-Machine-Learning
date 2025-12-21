# 🏥 Multiple Disease Prediction System Using Machine Learning

<div align="center">

![Healthcare-AI](https://img.shields.io/badge/Healthcare-AI-4A90E2)
![ML-Ensemble](https://img.shields.io/badge/ML-Ensemble-FF6B6B)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB)
![License MIT](https://img.shields.io/badge/License-MIT-32CD32)
![Status Active](https://img.shields.io/badge/Status-Active-brightgreen)

**An advanced, ensemble-based healthcare analytics platform for the intelligent and early detection of multiple diseases.**

[![GitHub stars](https://img.shields.io/github/stars/GTX-Gagan/Multiple-Disease-Prediction-System?style=social)](https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System)
[![GitHub forks](https://img.shields.io/github/forks/GTX-Gagan/Multiple-Disease-Prediction-System?style=social)](https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System)

*Leveraging cutting-edge machine learning to deliver accurate, multi-disease predictions with clinical insights.*

</div>

## 📊 Table of Contents
- [✨ Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🧠 Disease Prediction Modules](#-disease-prediction-modules)
- [📈 Model Architecture](#-model-architecture)
- [🔬 Performance Metrics](#-performance-metrics)
- [💻 API Documentation](#-api-documentation)
- [🎨 Web Interface](#-web-interface)
- [📊 Data Pipeline](#-data-pipeline)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Contact](#-contact)

## ✨ Overview

The **Multiple Disease Prediction System** is a state-of-the-art healthcare analytics platform that integrates multiple machine learning algorithms to predict various diseases from patient symptoms and diagnostic data. This system implements advanced ensemble techniques, feature optimization, and explainable AI (XAI) to deliver accurate, interpretable, and multi-disease predictions with actionable clinical insights. It's designed to assist in early detection and risk assessment for conditions like diabetes, heart disease, Parkinson's, breast cancer, and liver disease.

## 🎯 Key Features

### 🔬 **Multi-Disease Prediction Capabilities**
- **Diabetes Prediction**: Early detection using PIMA Indian Diabetes Dataset with advanced feature engineering.
- **Heart Disease Prediction**: Cardiovascular risk assessment using Cleveland dataset with ensemble voting.
- **Parkinson's Disease Detection**: Neurological disorder prediction using voice measurement biomarkers.
- **Breast Cancer Classification**: Malignancy detection with Wisconsin dataset using advanced feature selection.
- **Liver Disease Prediction**: Hepatic disorder prediction using Indian Liver Patient Dataset.

### 🚀 **Advanced ML Capabilities**
- **Ensemble Learning Stack**: Combines Random Forest, SVM, XGBoost, and Neural Networks.
- **Automated Feature Selection**: Recursive Feature Elimination (RFE) and Correlation Analysis.
- **Hyperparameter Optimization**: GridSearchCV and Bayesian Optimization.
- **Cross-Validation**: Stratified K-Fold and Leave-One-Out validation.
- **Model Explainability**: SHAP values, LIME, and Feature Importance visualization.

### 💡 **Smart Features**
- **Real-time Prediction**: Instant disease risk assessment via web interface or API.
- **Risk Stratification**: Low/Medium/High risk categorization with explanations.
- **Comparative Analysis**: Side-by-side model performance comparison dashboard.
- **Personalized Recommendations**: Tailored health suggestions based on predictions.
- **Progress Tracking**: Framework for longitudinal health monitoring capabilities.

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend & Core ML** | ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-007D66?logo=xgboost&logoColor=white) |
| **Frontend & UI** | ![Flask](https://img.shields.io/badge/Flask-2.0+-000000?logo=flask&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black) ![Bootstrap](https://img.shields.io/badge/Bootstrap-5.0+-7952B3?logo=bootstrap&logoColor=white) |
| **Data & Visualization** | ![Pandas](https://img.shields.io/badge/Pandas-1.4+-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-1.22+-013243?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-11557C?logo=matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-5.8+-3F4F75?logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-5B8FA8?logo=seaborn&logoColor=white) |
| **Deployment & DevOps** | ![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED?logo=docker&logoColor=white) ![Git](https://img.shields.io/badge/Git-2.35+-F05032?logo=git&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white) |

## 📁 Project Structure

```
Multiple-Disease-Prediction-System/
│
├── 📂 data/                    # Dataset directory
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed datasets
│   └── datasets_info.json     # Dataset metadata
│
├── 📂 models/                 # Trained models (serialized)
│   ├── diabetes_model.pkl
│   ├── heart_disease_model.pkl
│   ├── parkinsons_model.pkl
│   ├── breast_cancer_model.pkl
│   └── liver_disease_model.pkl
│
├── 📂 notebooks/              # Jupyter notebooks for research
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── 📂 src/                    # Core source code
│   ├── 📂 data_processing/
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineering.py
│   │
│   ├── 📂 models/
│   │   ├── base_model.py
│   │   ├── ensemble_model.py
│   │   ├── neural_network.py
│   │   └── model_factory.py
│   │
│   ├── 📂 utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   └── 📂 api/               # Flask API application
│       ├── app.py
│       ├── routes.py
│       └── schemas.py
│
├── 📂 static/                 # Web static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   ├── images/
│   └── assets/
│
├── 📂 templates/              # HTML templates for web interface
│   ├── index.html
│   ├── dashboard.html
│   ├── prediction.html
│   └── results.html
│
├── 📂 tests/                  # Comprehensive test suite
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_api.py
│
├── 📂 deployment/             # Containerization and deployment configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── 📜 .env.example            # Environment variables template
├── 📜 config.yaml             # Main configuration file
├── 📜 requirements.txt        # Python dependencies
├── 📜 README.md               # This file
└── 📜 LICENSE                 # MIT License file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8** or higher
- **pip** package manager
- **Git**

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System-Using-Machine-Learning.git
    cd Multiple-Disease-Prediction-System-Using-Machine-Learning
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv

    # On Windows:
    venv\Scripts\activate

    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    ```bash
    cp .env.example .env
    # Edit the .env file with your specific configuration (if needed).
    ```

5.  **Run the Application**
    ```bash
    python src/api/app.py
    ```
    Visit **[http://localhost:5000](http://localhost:5000)** in your browser.

### 🐳 Docker Deployment (Alternative)
```bash
# Build and run with Docker
docker build -t disease-prediction .
docker run -p 5000:5000 disease-prediction

# Or, use Docker Compose for a more managed setup
docker-compose up --build
```

## 🧠 Disease Prediction Modules

| Disease | Dataset | Best Model | Accuracy | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | PIMA Indian Diabetes | XGBoost with SMOTE | **92.3%** | Pregnancies, Glucose, BMI, Age |
| **Heart Disease** | Cleveland Dataset | Random Forest Ensemble | **89.7%** | Age, Cholesterol, Blood Pressure |
| **Parkinson's** | UCI Parkinson's Dataset | SVM (RBF Kernel) | **94.2%** | 22 Voice Measurement Parameters |
| **Breast Cancer** | Wisconsin Dataset | Neural Network | **97.8%** | 30 Cell Nuclei Features |
| **Liver Disease** | Indian Liver Patient Dataset | Gradient Boosting | **88.9%** | Bilirubin, Proteins, Enzymes |

## 📈 Model Architecture

### Ensemble Learning Framework
The system employs a **stacked ensemble** strategy, combining predictions from multiple strong base learners (Random Forest, XGBoost, SVM, Neural Network) using a meta-learner (Logistic Regression) to achieve superior and more robust performance than any single model.

### Feature Engineering Pipeline
1.  **Data Cleaning**: Handling missing values, detecting and treating outliers.
2.  **Feature Scaling**: Standardization (`StandardScaler`) and normalization (`MinMaxScaler`).
3.  **Feature Selection**: Using Recursive Feature Elimination (RFE) and correlation analysis.
4.  **Dimensionality Reduction**: PCA for feature compression, t-SNE for visualization.
5.  **Class Balancing**: Applying SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced datasets.

## 🔬 Performance Metrics

### Comparative Model Performance
| Disease | Best Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | XGBoost | 92.3% | 91.5% | 90.8% | 91.1% | 0.94 |
| **Heart Disease** | Random Forest | 89.7% | 88.9% | 90.2% | 89.5% | 0.92 |
| **Parkinson's** | SVM | 94.2% | 93.8% | 94.5% | 94.1% | 0.96 |
| **Breast Cancer** | Neural Network | 97.8% | 97.5% | 98.1% | 97.8% | 0.98 |
| **Liver Disease** | Gradient Boosting | 88.9% | 87.6% | 89.3% | 88.4% | 0.91 |

*Performance validated using 5-Fold Stratified Cross-Validation.*

## 💻 API Documentation

The system provides a fully-featured REST API built with Flask. The base URL is `http://localhost:5000/api/v1`.

### Key Endpoints
- **`GET /health`**: Health check endpoint.
- **`POST /predict/diabetes`**: Predict diabetes risk.
- **`POST /predict/heart`**: Predict heart disease risk.
- **`POST /predict/parkinsons`**: Predict Parkinson's disease.
- **`POST /predict/breast_cancer`**: Classify breast cancer.
- **`POST /predict/liver`**: Predict liver disease.

### Example: Diabetes Prediction
**Request (POST to `/predict/diabetes`)**:
```json
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 50
}
```
**Response**:
```json
{
  "prediction": "Diabetic",
  "probability": 0.89,
  "risk_level": "High",
  "confidence": 0.92,
  "recommendations": [
    "Consult with an endocrinologist",
    "Monitor blood sugar regularly",
    "Maintain a healthy diet and exercise routine"
  ]
}
```

## 🎨 Web Interface
The Flask web application offers an intuitive dashboard with:
- **Real-time Prediction Forms**: Easy input for each disease module.
- **Interactive Visualizations**: Dynamic charts showing feature importance and result probabilities.
- **Results Dashboard**: Displays prediction outcomes, risk levels, and personalized health recommendations in a clear layout.
- **Model Analytics**: A dedicated section for comparing the performance of different algorithms.

## 📊 Data Pipeline
The pipeline ensures robust data flow from raw input to final prediction:
`Raw Data` → `Preprocessing` → `Feature Engineering` → `Model Inference` → `Prediction & Visualization`

**Preprocessing includes**:
- Missing value imputation using KNN.
- Outlier detection with IQR and Z-score methods.
- Data validation to ensure clinical parameter ranges are plausible.

## 🧪 Testing
Ensure code quality and model reliability with the comprehensive test suite.
```bash
# Run all tests
pytest tests/

# Run tests for a specific module with verbose output
pytest tests/test_models.py -v

# Run tests with coverage report
pytest --cov=src tests/

# Run API endpoint tests
pytest tests/test_api.py
```
The project uses **GitHub Actions** for Continuous Integration (CI), automatically running tests, code quality checks (`flake8`), and security scans on every push.

## 🤝 Contributing
Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please follow the standard workflow:
1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

**Guidelines**:
- Follow **PEP 8** style guidelines.
- Add tests for any new features or bug fixes.
- Update documentation accordingly.
- Ensure changes maintain backward compatibility where possible.

## 📄 License
Distributed under the **MIT License**. See the `LICENSE` file in the repository for more information.

## 🙏 Acknowledgments
- The creators and maintainers of all the open-source datasets used (PIMA, Cleveland, UCI, etc.).
- The developers of the incredible Python data science and machine learning libraries (`scikit-learn`, `XGBoost`, `TensorFlow`, `Flask`, etc.).
- The open-source community for continuous inspiration and support.

## 📞 Contact
**Gagan** - [@GTX-Gagan](https://github.com/GTX-Gagan)

**Project Link**: [https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System-Using-Machine-Learning](https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System-Using-Machine-Learning)

**Report a Bug or Request a Feature**: Please use the [GitHub Issues](https://github.com/GTX-Gagan/Multiple-Disease-Prediction-System-Using-Machine-Learning/issues) page.

---

<div align="center">
  <sub>Built with ❤️ and a lot of coffee.</sub>
</div>
