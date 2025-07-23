# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load the dataset
df = pd.read_csv("employee_data.csv")

# Label Encoding for categorical features
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])

# Features and target
X = df.drop("income", axis=1)
y = df["income"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/salary_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
print("Model trained and saved.")
