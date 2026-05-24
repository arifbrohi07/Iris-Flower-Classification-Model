import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Iris Flower Classification",
    layout="wide"
)

st.title("🌸 Iris Flower Classification")
st.write("Machine Learning Classification using Random Forest")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    return df

df = load_data()

# =====================================
# DATA PREVIEW
# =====================================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📌 Dataset Shape")
st.write(df.shape)

# =====================================
# REMOVE ID COLUMN
# =====================================
df = df.drop("Id", axis=1)

# =====================================
# FEATURES & TARGET
# =====================================
X = df.drop("Species", axis=1)
y = df["Species"]

# =====================================
# LABEL ENCODING
# =====================================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# MODEL PIPELINE
# =====================================
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# =====================================
# TRAIN MODEL
# =====================================
model.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================
y_pred = model.predict(X_test)

# =====================================
# ACCURACY
# =====================================
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Accuracy")
st.metric("Accuracy", f"{accuracy:.2f}")

# =====================================
# CONFUSION MATRIX
# =====================================
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# =====================================
# CLASSIFICATION REPORT
# =====================================
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# =====================================
# USER INPUT SECTION
# =====================================
st.subheader("🌼 Predict Iris Flower Species")

sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    value=0.2
)

# =====================================
# PREDICTION BUTTON
# =====================================
if st.button("Predict Species"):

    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })

    prediction = model.predict(input_data)

    species = encoder.inverse_transform(prediction)

    st.success(
        f"🌸 Predicted Species: {species[0]}"
    )

# =====================================
# FEATURE IMPORTANCE
# =====================================
st.subheader(" Feature Importance")

classifier = model.named_steps['classifier']

importance = classifier.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.barh(
    feature_importance['Feature'],
    feature_importance['Importance']
)

ax2.set_xlabel("Importance")
ax2.set_ylabel("Features")
ax2.set_title("Feature Importance")

st.pyplot(fig2)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.write("Developed using Streamlit & Scikit-learn")