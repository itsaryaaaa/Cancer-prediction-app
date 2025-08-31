import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_scaled, y)

# Streamlit App
st.title("🔬 Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign** or **Malignant** using breast cancer dataset features.")

# Input fields for all features
user_input = []
for feature in cancer.feature_names:
    val = st.number_input(f"Enter {feature}", float(X[feature].min()), float(X[feature].max()))
    user_input.append(val)

# Prediction
if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.error("🔴 The tumor is **Malignant** (Cancerous).")
    else:
        st.success("🟢 The tumor is **Benign** (Not Cancerous).")
