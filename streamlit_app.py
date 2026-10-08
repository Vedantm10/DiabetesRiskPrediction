import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with open("diabetes_model.pkl", "rb") as file:
        model_data = pickle.load(file)

    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    model_name = model_data["model_name"]
    model_accuracy = model_data["accuracy"]

except FileNotFoundError:

    st.error("❌ diabetes_model.pkl was not found.")

    st.info(
        "Make sure diabetes_model.pkl is in the same folder "
        "as streamlit_app.py."
    )

    st.stop()

except Exception as e:

    st.error("❌ Error loading the model.")

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩺 Diabetes Predictor")

    st.write(
        "Type 2 Diabetes Risk Prediction"
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write(
        model_name
    )

    st.subheader("🎯 Accuracy")

    st.write(
        f"{model_accuracy * 100:.2f}%"
    )

    st.divider()

    st.subheader("📊 Risk Levels")

    st.success("🟢 Low Risk")

    st.warning("🟡 Moderate Risk")

    st.error("🔴 High Risk")

    st.divider()

    st.caption(
        "Machine Learning Classification Project"
    )


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Type 2 Diabetes Risk Predictor")

st.write(
    "Enter the patient's information below to predict "
    "the diabetes risk category."
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🤖 Model Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Algorithm",
        model_name
    )

with info2:

    st.metric(
        "Accuracy",
        f"{model_accuracy * 100:.2f}%"
    )

with info3:

    st.metric(
        "Risk Categories",
        "3"
    )


st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=40
    )


with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )


with col3:

    city = st.selectbox(
        "City",
        [
            "Mumbai",
            "Thane",
            "Bengaluru",
            "Hyderabad",
            "Delhi",
            "Kolkata",
            "Ahmedabad",
            "Pune",
            "Chennai",
            "Kanpur",
            "Nagpur",
            "Indore",
            "Lucknow",
            "Bhopal",
            "Jaipur",
            "Surat",
            "Visakhapatnam",
            "Patna"
        ]
    )


# ============================================================
# DIABETES HISTORY
# ============================================================

st.subheader("🧬 Diabetes History")

col1, col2, col3 = st.columns(3)


with col1:

    family_history_diabetes = st.selectbox(
        "Family History of Diabetes",
        [
            "Yes",
            "No"
        ]
    )


with col2:

    physical_activity_level = st.selectbox(
        "Physical Activity Level",
        [
            "Sedentary",
            "Moderate",
            "Active"
        ]
    )


with col3:

    income_bracket = st.selectbox(
        "Income Bracket",
        [
            "Low",
            "Middle",
            "High"
        ]
    )


# ============================================================
# LIFESTYLE
# ============================================================

st.subheader("🏃 Lifestyle Information")

col1, col2, col3 = st.columns(3)


with col1:

    diet_type = st.selectbox(
        "Diet Type",
        [
            "Vegetarian",
            "Non-Vegetarian",
            "Pescatarian",
            "Vegan"
        ]
    )


with col2:

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "Never",
            "Current",
            "Former"
        ]
    )


with col3:

    alcohol_consumption = st.selectbox(
        "Alcohol Consumption",
        [
            "Never",
            "Occasional",
            "Regular"
        ]
    )


# ============================================================
# BODY MEASUREMENTS
# ============================================================

st.subheader("⚖️ Body Measurements")

col1, col2, col3 = st.columns(3)


with col1:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )


with col2:

    waist_circumference_cm = st.number_input(
        "Waist Circumference (cm)",
        min_value=40.0,
        max_value=200.0,
        value=90.0,
        step=0.1
    )


with col3:

    hours_sleep_per_night = st.number_input(
        "Hours of Sleep Per Night",
        min_value=1.0,
        max_value=15.0,
        value=7.0,
        step=0.1
    )


# ============================================================
# MEDICAL MEASUREMENTS
# ============================================================

st.subheader("🩸 Medical Measurements")

col1, col2, col3 = st.columns(3)


with col1:

    fasting_blood_sugar = st.number_input(
        "Fasting Blood Sugar",
        min_value=50,
        max_value=400,
        value=100
    )


with col2:

    hba1c_level = st.number_input(
        "HbA1c Level",
        min_value=3.0,
        max_value=15.0,
        value=5.5,
        step=0.1
    )


with col3:

    stress_level = st.slider(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5
    )


# ============================================================
# BLOOD PRESSURE
# ============================================================

st.subheader("❤️ Blood Pressure")

col1, col2 = st.columns(2)


with col1:

    blood_pressure_systolic = st.number_input(
        "Systolic Blood Pressure",
        min_value=70,
        max_value=250,
        value=120
    )


with col2:

    blood_pressure_diastolic = st.number_input(
        "Diastolic Blood Pressure",
        min_value=40,
        max_value=150,
        value=80
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.subheader("🔮 Diabetes Risk Prediction")

predict_button = st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "city": [city],

        "bmi": [bmi],

        "family_history_diabetes": [
            family_history_diabetes
        ],

        "physical_activity_level": [
            physical_activity_level
        ],

        "diet_type": [
            diet_type
        ],

        "smoking_status": [
            smoking_status
        ],

        "alcohol_consumption": [
            alcohol_consumption
        ],

        "hours_sleep_per_night": [
            hours_sleep_per_night
        ],

        "stress_level": [
            stress_level
        ],

        "fasting_blood_sugar": [
            fasting_blood_sugar
        ],

        "hba1c_level": [
            hba1c_level
        ],

        "blood_pressure_systolic": [
            blood_pressure_systolic
        ],

        "blood_pressure_diastolic": [
            blood_pressure_diastolic
        ],

        "waist_circumference_cm": [
            waist_circumference_cm
        ],

        "income_bracket": [
            income_bracket
        ]

    })


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = model.predict(
        input_data
    )


    risk = label_encoder.inverse_transform(
        prediction
    )[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if risk == "High":

        st.error(
            "🔴 HIGH RISK"
        )

        st.write(
            "The model predicts a HIGH diabetes risk category."
        )


    elif risk == "Moderate":

        st.warning(
            "🟡 MODERATE RISK"
        )

        st.write(
            "The model predicts a MODERATE diabetes risk category."
        )


    else:

        st.success(
            "🟢 LOW RISK"
        )

        st.write(
            "The model predicts a LOW diabetes risk category."
        )


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.subheader("📋 Prediction Summary")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Risk",
            risk
        )

    with result_col2:

        st.metric(
            "Model Used",
            model_name
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 Type 2 Diabetes Risk Prediction | "
    "Traditional Machine Learning Classification Project"
)