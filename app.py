import pandas as pd
import numpy as np
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import streamlit as st


# ============================================================
# WEB PAGE CODE
# ============================================================

st.title("HEALTH INSURANCE PREDICTION")


# Image URL
img_url = "https://imgs.search.brave.com/TFjx_njIOx9kqjKDS6hW0MHtdM52dYbdvYuqHBihhwk/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly90aHVt/YnMuZHJlYW1zdGlt/ZS5jb20vYi9oZWFs/dGgtaW5zdXJhbmNl/LWJ1c2luZXNzbWFu/LWRy[...]"

st.image(img_url)


# ============================================================
# LOAD DATA
# ============================================================

url = "https://raw.githubusercontent.com/ankitmisk/UIT-data/refs/heads/main/Insurance.csv"

df = pd.read_csv(url)


# ============================================================
# EDA
# ============================================================

st.subheader("Dataset Preview")

st.write(df.head())


# Remove Customer_ID
if "Customer_ID" in df.columns:
    df.drop("Customer_ID", axis=1, inplace=True)


# Convert Yes/No values to 1/0
if "Previous_Insurance" in df.columns:
    df["Previous_Insurance"] = df["Previous_Insurance"].map({
        "Yes": 1,
        "No": 0
    })


if "Insurance_Bought" in df.columns:
    df["Insurance_Bought"] = df["Insurance_Bought"].map({
        "Yes": 1,
        "No": 0
    })


# ============================================================
# DIVIDE DATA INTO FEATURES AND TARGET
# ============================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.3
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# ============================================================
# MODEL ACCURACY
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# SIDEBAR USER INPUT
# ============================================================

st.sidebar.title("Fill Customer Details")

st.sidebar.image(img_url)


# List to store user answers
all_ans = []


for col_name in X.columns:

    min_v = float(X[col_name].min())
    max_v = float(X[col_name].max())

    # Previous Insurance is binary
    if col_name == "Previous_Insurance":

        value = st.sidebar.number_input(
            f"Select value for {col_name}",
            min_value=int(min_v),
            max_value=int(max_v),
            value=int(min_v),
            step=1
        )

    else:

        value = st.sidebar.slider(
            f"Select value for {col_name}",
            min_value=min_v,
            max_value=max_v,
            value=min_v
        )

    all_ans.append(value)


# ============================================================
# DISPLAY USER INPUT
# ============================================================

user_data = {
    col_name: all_ans[i]
    for i, col_name in enumerate(X.columns)
}

user_df = pd.DataFrame(user_data, index=[0])

st.subheader("Customer Details")

st.write(user_df)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Click to Predict"):

    with st.spinner("Predicting..."):

        time.sleep(2)

        # Convert user input into DataFrame
        prediction = model.predict(user_df)[0]

        # Prediction probability
        probability = model.predict_proba(user_df)[0]

    if prediction == 0:

        st.info("❎ Customer will NOT buy the insurance ❎")

    else:

        st.success("✅ Customer WILL buy the insurance ✅")

    # Show probability
    st.subheader("Prediction Probability")

    st.write(
        f"Probability of NOT buying insurance: "
        f"{probability[0] * 100:.2f}%"
    )

    st.write(
        f"Probability of buying insurance: "
        f"{probability[1] * 100:.2f}%"
    )
