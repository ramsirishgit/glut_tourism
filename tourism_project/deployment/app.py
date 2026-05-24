import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="RamSirish/tourism", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts potential buyers for the Wellness Tourism Package
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=61, value=18, step=1)
DurationOfPitch = st.number_input("Duration of Sales Pitch (delivered to customer)",  min_value=5, max_value=130, value=5, step=1)
NumberOfFollowups = st.number_input("Number of Followups (with the customer)",  min_value=1, max_value=6, value=1, step=1)
NumberOfTrips = st.number_input("Number of Trips (to the customer)",  min_value=1, max_value=22, value=1, step=1)
NumberOfPersonVisiting = st.number_input("Number of Persons visiting",  min_value=0, max_value=4, value=1, step=1)
NumberOfChildrenVisiting = st.number_input("Number of Children visiting",  min_value=0, max_value=3, value=1, step=1)
MonthlyIncome = st.number_input("Monthly Income",  min_value=1000, max_value=100000, value=10000, step=1)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3 Stars", "4 Stars", "5 Stars"])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried'])
Passport = st.selectbox("Passport", ["No", "Yes"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3", "4", "5"])
OwnCar = st.selectbox("Own Car", ["No", "Yes"])
Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])

TypeofContact_Self_Enquiry = 1 if TypeofContact == "Self Enquiry" else 0
CityTier_2 = 1 if CityTier == "Tier 2" else 0
CityTier_3 = 1 if CityTier == "Tier 3" else 0
Occupation_Large_Business = 1 if Occupation == "Large Business" else 0
Occupation_Salaried = 1 if Occupation == "Salaried" else 0
Occupation_Small_Business = 1 if Occupation == "Small Business" else 0
Gender_Male = 1 if Gender == "Male" else 0
ProductPitched_Deluxe = 1 if ProductPitched == "Deluxe" else 0
ProductPitched_King = 1 if ProductPitched == "King" else 0
ProductPitched_Standard = 1 if ProductPitched == "Standard" else 0
ProductPitched_Super_Deluxe = 1 if ProductPitched == "Super Deluxe" else 0
PreferredPropertyStar_4 = 1 if PreferredPropertyStar == "4 Stars" else 0
PreferredPropertyStar_5 = 1 if PreferredPropertyStar == "5 Stars" else 0
MaritalStatus_Married = 1 if MaritalStatus == "Married" else 0
MaritalStatus_Single = 1 if MaritalStatus == "Single" else 0
MaritalStatus_Unmarried = 1 if MaritalStatus == "Unmarried" else 0
Passport_1 = 1 if Passport == "Yes" else 0
PitchSatisfactionScore_2 = 1 if PitchSatisfactionScore == "2" else 0
PitchSatisfactionScore_3 = 1 if PitchSatisfactionScore == "3" else 0
PitchSatisfactionScore_4 = 1 if PitchSatisfactionScore == "4" else 0
PitchSatisfactionScore_5 = 1 if PitchSatisfactionScore == "5" else 0
OwnCar_1 = 1 if OwnCar == "Yes" else 0
Designation_Executive = 1 if Designation == "Executive" else 0
Designation_Manager = 1 if Designation == "Manager" else 0
Designation_Senior_Manager = 1 if Designation == "Senior Manager" else 0
Designation_VP = 1 if Designation == "VP" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact_Self Enquiry': TypeofContact_Self_Enquiry,
    'CityTier_2': CityTier_2,
    'CityTier_3': CityTier_3,
    'Occupation_Large Business': Occupation_Large_Business,
    'Occupation_Salaried': Occupation_Salaried,
    'Occupation_Small Business': Occupation_Small_Business,
    'Gender_Male': Gender_Male,
    'ProductPitched_Deluxe': ProductPitched_Deluxe,
    'ProductPitched_King': ProductPitched_King,
    'ProductPitched_Standard': ProductPitched_Standard,
    'ProductPitched_Super Deluxe': ProductPitched_Super_Deluxe,
    'PreferredPropertyStar_4.0': PreferredPropertyStar_4,
    'PreferredPropertyStar_5.0': PreferredPropertyStar_5,
    'MaritalStatus_Married': MaritalStatus_Married,
    'MaritalStatus_Single': MaritalStatus_Single,
    'MaritalStatus_Unmarried': MaritalStatus_Unmarried,
    'Passport_1': Passport_1,
    'PitchSatisfactionScore_2': PitchSatisfactionScore_2,
    'PitchSatisfactionScore_3': PitchSatisfactionScore_3,
    'PitchSatisfactionScore_4': PitchSatisfactionScore_4,
    'PitchSatisfactionScore_5': PitchSatisfactionScore_5,
    'OwnCar_1': OwnCar_1,
    'Designation_Executive': Designation_Executive,
    'Designation_Manager': Designation_Manager,
    'Designation_Senior Manager': Designation_Senior_Manager,
    'Designation_VP': Designation_VP
    }])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction Result for {input_data.iloc[0]}:")
    st.success(f"With a probability of {prediction}, this person is predicted to {'NOT' if prediction < 0.5 else ''} purchase the plan.")
