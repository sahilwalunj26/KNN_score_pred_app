
import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="KNN Exam Score Predictor",
    page_icon="📊",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("KNN_RE.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# App title
st.title("📊 KNN Exam Score Predictor")
st.write(
    "Enter the student's details below to predict "
    "the exam score using your trained KNN model."
)

st.info("Enter categorical values using the same numeric encoding used during model training.")

# Input form
with st.form("prediction_form"):
    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=20)
        gender = st.number_input("Gender (encoded)", min_value=0, step=1, value=2)
        course = st.number_input("Course (encoded)", min_value=0, step=1, value=0)
        study_hours = st.number_input("Study Hours", min_value=0.0, value=5.0)
        class_attendance = st.number_input(
            "Class Attendance (%)",
            min_value=0.0, max_value=100.0, value=75.0
        )
        internet_access = st.number_input(
            "Internet Access (encoded)", min_value=0, step=1, value=1
        )

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours", min_value=0.0, max_value=24.0, value=8.0
        )
        sleep_quality = st.number_input(
            "Sleep Quality (encoded)", min_value=0, step=1, value=1
        )
        study_method = st.number_input(
            "Study Method (encoded)", min_value=0, step=1, value=1
        )
        facility_rating = st.number_input(
            "Facility Rating (encoded)", min_value=0, step=1, value=2
        )
        exam_difficulty = st.number_input(
            "Exam Difficulty (encoded)", min_value=0, step=1, value=1
        )

    submitted = st.form_submit_button(
        "Predict Exam Score", use_container_width=True
    )

# Prediction
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric("Predicted Exam Score", f"{prediction:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
