import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score

# Sidebar: Filtering Options
st.sidebar.header("Info of Candidates")
st.sidebar.markdown("### Select the feature(s) you want to use for filtering!")

specify = st.sidebar.selectbox("Features", ("All", "Specify"))

gender_filter = []
major_filter = []

if specify == "Specify":
    st.sidebar.markdown("### Current status")
    if st.sidebar.checkbox("Gender", key="gender"):
        option = st.sidebar.selectbox("Select gender", ("All", "Specify"))
        if option == "Specify":
            if st.sidebar.checkbox("Female", key="female"):
                gender_filter.append("Female")
            if st.sidebar.checkbox("Male", key="male"):
                gender_filter.append("Male")
            if st.sidebar.checkbox("Other", key="gender_other"):
                gender_filter.append("Other")
            if st.sidebar.checkbox("Secret", value=True, key="gender_secret"):
                gender_filter.append("unknown")

    st.sidebar.markdown("### Major")
    if st.sidebar.checkbox("Major", key="major"):
        option = st.sidebar.selectbox("Select major", ("All", "Specify"))
        if option == "Specify":
            if st.sidebar.checkbox("STEM", key="STEM"):
                major_filter.append("STEM")
            if st.sidebar.checkbox("Humanities", key="Humanities"):
                major_filter.append("Humanities")
            if st.sidebar.checkbox("Business", key="Business"):
                major_filter.append("Business Degree")

# Main App Header
st.title("Welcome to Smart HR!")
st.write(
    "Tell me about the candidates, and I will let you know if they are actually looking for a new job!"
    " This will save you considerable time in reaching out or interviewing the candidates!"
)

# Dummy Data for Visualization
df_temp = pd.DataFrame(
    {
        "Candidate": np.arange(200),
        "Training Hours": np.random.randint(20, 100, size=200),
        "Probability": np.random.rand(200),
        "Current company": np.random.choice(["Company A", "Company B", "NGO", "Public Sector"], size=200),
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

# Candidate Statistics
st.header("Candidate Statistics")
df_train = pd.DataFrame(
    {
        "Gender": np.random.choice(["Male", "Female"], 100),
        "Major": np.random.choice(["STEM", "Humanities"], 100),
        "Education": np.random.choice(["Graduate", "PhD"], 100),
        "City Index": np.random.rand(100),
    }
)
qualified_candidates = df_train.shape[0]
st.write(f"Qualified candidate count: {qualified_candidates}")
if st.checkbox("Show dataframe", key="filtered"):
    st.write(df_train.head())

# Model Training
st.header("Model Training")
X = np.random.rand(100, 3)
y = np.random.choice([0, 1], 100)
model = LogisticRegression()
model.fit(X, y)
st.write("Model Trained Successfully!")

# ROC Curve Plot
y_scores = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
st.pyplot(fig)

