# Imports required for data manipulation
import pandas as pd
import sklearn
# Import required for creating a folder
import os
# Imports required for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# Imports required for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# Imports required for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/RamSirish/tourism/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(df.head(2))
# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation','Gender',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'Designation'
]

# Encode categorical columns
label_encoder = LabelEncoder()

for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Xtrain",Xtrain.head(2),sep="\n")
print("Xtest",Xtest.head(2),sep="\n")
print("ytrain",ytrain.head(2),sep="\n")
print("ytest",ytest.head(2),sep="\n")

# Write the train and test sets to csvs on local disk
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

#Upload the train and testsets into HF
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="RamSirish/tourism",
        repo_type="dataset",
    )
