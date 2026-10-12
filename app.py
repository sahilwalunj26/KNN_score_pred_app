import pickle
import pandas as pd
import streamlit as st

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="KNN Exam Score Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Student Exam Score Predictor")
st.write("Enter the student's details to predict the exam score.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("KNN_RE.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- Categorical Mappings ----------------
# IMPORTANT:
# These mappings are examples. Replace them with the exact
# numeric encodings used during model training.

gender_map = {
    "Female": 0,
    "Male": 1
}

course_map = {
    "Arts": 0,
    "Business": 1,
    "Computer Science": 2,
    "Engineering": 3,
    "Science": 4
}

internet_map = {
    "No": 0,
    "Yes": 1
}

sleep_quality_map = {
    "Good": 0,
    "Poor": 1,
    "Average": 2
}

study_method_map = {
    "Coaching": 0,
    "Group Study": 1,
    "Online Videos": 2,
    "Self Study": 3
}

facility_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

exam_difficulty_map = {
    "Easy": 0,
    "Hard": 1,
    "Moderate": 2
}

# ---------------- Input Form ----------------
st.subheader("Student Information")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=1, max_value=100, value=20
        )

        gender = st.selectbox(
            "Gender", list(gender_map.keys())
        )

        course = st.selectbox(
            "Course", list(course_map.keys())
        )

        study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        class_attendance = st.number_input(
            "Class Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        internet_access = st.selectbox(
            "Internet Access", list(internet_map.keys())
        )

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

        sleep_quality = st.selectbox(
            "Sleep Quality", list(sleep_quality_map.keys())
        )

        study_method = st.selectbox(
            "Study Method", list(study_method_map.keys())
        )

        facility_rating = st.selectbox(
            "Facility Rating", list(facility_map.keys())
        )

        exam_difficulty = st.selectbox(
            "Exam Difficulty", list(exam_difficulty_map.keys())
        )

    submitted = st.form_submit_button(
        "Predict Exam Score",
        use_container_width=True
    )

# ---------------- Prediction ----------------
if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender_map[gender],
        "course": course_map[course],
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_map[internet_access],
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality_map[sleep_quality],
        "study_method": study_method_map[study_method],
        "facility_rating": facility_map[facility_rating],
        "exam_difficulty": exam_difficulty_map[exam_difficulty]
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Exam Score",
            value=f"{prediction:.2f}"
        )

        with st.expander("View input values sent to the model"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
