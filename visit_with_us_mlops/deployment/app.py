
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
MODEL_REPO = "Bash18/tourism-package-model"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package Prediction")
st.write("Provide customer details to predict purchase likelihood.")

# ---------------- MANUAL ENCODING MAPS ----------------
contact_map = {"Company Invited": 0, "Self Inquiry": 1}
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
occupation_map = {
    "Salaried": 0,
    "Freelancer": 1,
    "Small Business": 2,
    "Large Business": 3
}
product_map = {
    "Basic": 0,
    "Standard": 1,
    "Deluxe": 2,
    "Super Deluxe": 3
}

# ---------------- INPUT FORM ----------------
with st.form("prediction_form"):

    Age = st.number_input("Age", 18, 80, 30)
    MonthlyIncome = st.number_input("Monthly Income", 1000, 1000000, 20000)
    NumberOfTrips = st.number_input("Number of Trips per Year", 0, 20, 2)
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 5, 0)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 1, 120, 30)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    TypeofContact = st.selectbox("Type of Contact", list(contact_map.keys()))
    Gender = st.selectbox("Gender", list(gender_map.keys()))
    MaritalStatus = st.selectbox("Marital Status", list(marital_map.keys()))
    Occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    ProductPitched = st.selectbox("Product Pitched", list(product_map.keys()))

    Passport = st.selectbox("Passport", [0, 1])
    OwnCar = st.selectbox("Own Car", [0, 1])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

    submit = st.form_submit_button("Predict")

# ---------------- PREDICTION ----------------
if submit:

    input_data = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfTrips": NumberOfTrips,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "DurationOfPitch": DurationOfPitch,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "TypeofContact": contact_map[TypeofContact],
        "Gender": gender_map[Gender],
        "MaritalStatus": marital_map[MaritalStatus],
        "Occupation": occupation_map[Occupation],
        "ProductPitched": product_map[ProductPitched],
        "Passport": Passport,
        "OwnCar": OwnCar,
        "CityTier": CityTier,
        "PreferredPropertyStar": PreferredPropertyStar
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase the package. (Probability: {probability:.2f})")
    else:
        st.error(f"Customer is unlikely to purchase the package. (Probability: {probability:.2f})")
