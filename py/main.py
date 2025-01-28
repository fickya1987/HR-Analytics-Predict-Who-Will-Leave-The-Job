"""
Streamlit web app for Smart HR
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score


# Main containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
takeaway = st.container()

with header:
    st.title('Welcome to Smart HR!')
    st.write(
        "Tell me about the candidates, and I will let you know if they are actually looking for a new job!"
        " This will save you considerable time in reaching out or interviewing the candidates!"
    )

    # Replace this with your actual data source
    df_temp = pd.DataFrame(
        {
            "Candidate": np.arange(200),
            "Training Hours": np.random.randint(20, 100, size=200),
            "Probability": np.random.rand(200),
            "Current company": np.random.choice(["Company A", "Company B"], size=200),
        }
    )

    c = (
        alt.Chart(df_temp)
        .mark_circle()
        .encode(
            x=alt.X("Candidate", axis=alt.Axis(labels=False)),
            y="Training Hours",
            size="Probability",
            color="Current company",
            tooltip=["Probability", "Candidate", "Training Hours"],
        )
        .properties(width=700, height=250)
    )
    st.altair_chart(c)

# Sidebar logic remains unchanged
with dataset:
    st.header("Candidate Statistics")

    # Simulated DataFrame
    df_train = pd.DataFrame(
        {
            "Gender": np.random.choice(["Male", "Female"], 100),
            "Major": np.random.choice(["STEM", "Humanities"], 100),
            "Education": np.random.choice(["Graduate", "PhD"], 100),
            "City Index": np.random.rand(100),
        }
    )
    st.write("Sample Dataset:", df_train.head())

with model_training:
    st.header("Model Training")

    # Simulated data for Logistic Regression
    X = np.random.rand(100, 3)
    y = np.random.choice([0, 1], 100)

    model = LogisticRegression()
    model.fit(X, y)

    st.write("Model Trained Successfully!")


