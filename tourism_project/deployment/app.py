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

# Define numeric and categorical features
# numeric_features = [
#     'Age', 'DurationOfPitch', 'NumberOfFollowups', 'NumberOfTrips',
#     'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'MonthlyIncome'
# ]

# categorical_features = [
#     'TypeofContact', 'CityTier', 'Occupation','Gender',
#     'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'Passport',
#     'PitchSatisfactionScore', 'OwnCar', 'Designation'
# ]


# User input
# app_category = st.selectbox("App Category", ["FAMILY", "TOOLS", "NEWS AND MAGAZINES", "GAME", "EDUCATION", "OTHERS"])
# app_category = st.selectbox("App Category", ["FAMILY", "TOOLS", "NEWS AND MAGAZINES", "GAME", "EDUCATION", "OTHERS"])
# free_or_paid = st.selectbox("Free or Paid", ["Free", "Paid"])
# content_rating = st.selectbox("Content Rating", ["Everyone", "Teen", "Mature 17+", "Adults only 18+"])
# screentime_category = st.selectbox("Screen Time Category", ["Low", "Medium", "High"])

Age = st.number_input("Age", min_value=18, max_value=61, value=18, step=1)
DurationOfPitch = st.number_input("Duration of Sales Pitch (delivered to customer)",  min_value=5, max_value=130, value=5, step=1)
NumberOfFollowups = st.number_input("Number of Followups (with the customer)",  min_value=1, max_value=6, value=1, step=1)
NumberOfTrips = st.number_input("Number of Trips (to the customer)",  min_value=1, max_value=22, value=1, step=1)
NumberOfPersonVisiting = st.number_input("Number of Persons visiting",  min_value=0, max_value=4, value=1, step=1)
NumberOfChildrenVisiting = st.number_input("Number of Children visiting",  min_value=0, max_value=3, value=1, step=1)
MonthlyIncome = st.number_input("Monthly Income",  min_value=1000, max_value=100000, value=10000, step=1)

# TypeofContact ['Self Enquiry' 'Company Invited']
# CityTier [3 1 2]
# Occupation ['Salaried' 'Free Lancer' 'Small Business' 'Large Business']
# Gender ['Female' 'Male' 'Fe Male']
# ProductPitched ['Deluxe' 'Basic' 'Standard' 'Super Deluxe' 'King']
# PreferredPropertyStar [3. 4. 5.]
# MaritalStatus ['Single' 'Divorced' 'Married' 'Unmarried']
# Passport [1 0]
# PitchSatisfactionScore [2 3 5 4 1]
# OwnCar [1 0]
# Designation ['Manager' 'Executive' 'Senior Manager' 'AVP' 'VP']

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

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'Designation': Designation
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Predicted: {prediction}")
