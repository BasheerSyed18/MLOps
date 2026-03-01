
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

# ---------------- EXACT CATEGORICAL MAPS ----------------
contact_map = {
    "Company Invited": 0,
    "Self Enquiry": 1
}

occupation_map = {
    "Free Lancer": 0,
    "Large Business": 1,
    "Salaried": 2,
    "Small Business": 3
}

gender_map = {
    "Female": 0,
    "Male": 1
}

marital_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2,
    "Unmarried": 3
}

product_map = {
    "Basic": 0,
    "Deluxe": 1,
    "King": 2,
    "Standard": 3,
    "Super Deluxe": 4
}

designation_map = {
    "AVP": 0,
    "Executive": 1,
    "Manager": 2,
    "Senior Manager": 3,
    "VP": 4
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
    NumberOfFollowups = st.number_input("Number of Followups After Pitch", 0, 10, 1)

    TypeofContact = st.selectbox("Type of Contact", list(contact_map.keys()))
    Occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    Gender = st.selectbox("Gender", list(gender_map.keys()))
    MaritalStatus = st.selectbox("Marital Status", list(marital_map.keys()))
    ProductPitched = st.selectbox("Product Pitched", list(product_map.keys()))
    Designation = st.selectbox("Designation", list(designation_map.keys()))

    Passport = st.selectbox("Passport", [0, 1])
    OwnCar = st.selectbox("Own Car", [0, 1])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])

    submit = st.form_submit_button("Predict")

# ---------------- PREDICTION ----------------
if submit:

    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": contact_map[TypeofContact],
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": occupation_map[Occupation],
        "Gender": gender_map[Gender],
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": product_map[ProductPitched],
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": marital_map[MaritalStatus],
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": designation_map[Designation],
        "MonthlyIncome": MonthlyIncome
    }])

    # 🔥 Force exact feature order
    input_data = input_data[model.feature_names_in_]

    # ---------------- DEBUG LOGGING ----------------
    st.subheader("Debug Information")

    st.write("Model expects features in this order:")
    st.write(list(model.feature_names_in_))

    st.write("Encoded Input DataFrame sent to model:")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("Raw Probability:", probability)
    
    if prediction == 1:
        st.success(f"Customer is likely to purchase the package. (Probability: {probability:.2f})")
    else:
        st.error(f"Customer is unlikely to purchase the package. (Probability: {probability:.2f})")
