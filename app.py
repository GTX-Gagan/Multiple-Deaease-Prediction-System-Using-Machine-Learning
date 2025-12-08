import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🧑‍⚕️",
    layout="wide"
)

st.markdown(
    """
    <style>
        .big-font { font-size:22px !important; font-weight:bold; }
        .prediction-box {
            padding: 20px;
            background: #f7f7f7;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOAD MODELS
# -------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model.sav', 'rb'))

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    selected = option_menu(
        "Disease Prediction AI",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart", "person"],
        menu_icon="hospital-fill",
        default_index=0
    )

# -------------------------
# Helper: Validate and Predict
# -------------------------
def run_prediction(model, inputs):
    try:
        float_inputs = [float(x) for x in inputs]
        prediction = model.predict([float_inputs])[0]
        return prediction
    except ValueError:
        st.error("❌ Invalid input! Please ensure all fields are filled with numbers.")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        return None

# -------------------------
# DIABETES PAGE
# -------------------------
if selected == "Diabetes Prediction":
    st.title("🩺 Diabetes Prediction")

    st.markdown("Enter the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0.0)
        SkinThickness = st.number_input("Skin Thickness value", min_value=0.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)

    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0.0)
        Insulin = st.number_input("Insulin Level", min_value=0.0)
        Age = st.number_input("Age", min_value=0.0)

    with col3:
        BloodPressure = st.number_input("Blood Pressure value", min_value=0.0)
        BMI = st.number_input("BMI value", min_value=0.0)

    if st.button("🔍 Get Diabetes Test Result"):
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        result = run_prediction(diabetes_model, inputs)

        if result is not None:
            if result == 1:
                st.success("🟡 The person **is diabetic**.")
            else:
                st.success("🟢 The person **is not diabetic**.")

# -------------------------
# HEART DISEASE PAGE
# -------------------------
if selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")

    st.markdown("Enter the following details:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0.0)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
        restecg = st.number_input("Resting ECG results", min_value=0.0)
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0)
        thal = st.number_input("Thal (0=Normal,1=Fixed,2=Reversible)", min_value=0.0)

    with col2:
        sex = st.number_input("Sex (1=Male,0=Female)", min_value=0.0)
        chol = st.number_input("Serum Cholesterol", min_value=0.0)
        thalach = st.number_input("Maximum Heart Rate", min_value=0.0)
        slope = st.number_input("Slope of ST Segment", min_value=0.0)

    with col3:
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0.0)
        fbs = st.number_input("Fasting Blood Sugar >120", min_value=0.0)
        exang = st.number_input("Exercise Induced Angina", min_value=0.0)
        ca = st.number_input("Number of Major Vessels", min_value=0.0)

    if st.button("🔍 Get Heart Disease Test Result"):
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]

        result = run_prediction(heart_disease_model, inputs)

        if result is not None:
            if result == 1:
                st.error("🔴 The person **has heart disease**.")
            else:
                st.success("🟢 The person **does not have heart disease**.")

# -------------------------
# PARKINSONS PAGE
# -------------------------
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")

    st.markdown("Enter the following voice/neurological parameters:")

    cols = st.columns(5)
    labels = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR",
        "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    # Create number inputs dynamically
    values = []
    col_index = 0
    for label in labels:
        with cols[col_index]:
            val = st.number_input(label, value=0.0)
            values.append(val)
        col_index = (col_index + 1) % 5

    if st.button("🔍 Get Parkinson's Test Result"):
        result = run_prediction(parkinsons_model, values)

        if result is not None:
            if result == 1:
                st.error("🔴 The person **has Parkinson’s disease**.")
            else:
                st.success("🟢 The person **does not have Parkinson’s disease**.")
