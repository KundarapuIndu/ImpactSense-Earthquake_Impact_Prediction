# Step 4: Streamlit Web Application for Iris Classification

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset for exploration
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Load trained model
model = joblib.load("iris_model.joblib")

# Streamlit app layout
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
st.title("🌸 Iris Flower Classification App")
st.markdown("""
This app predicts the **species of Iris flower** based on four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# Sidebar mode selection
mode = st.sidebar.selectbox("Choose Mode", ["🔍 Data Exploration", "🤖 Prediction"])

# -------------------- Prediction Mode --------------------
if mode == "🤖 Prediction":
    st.header("🌼 Predict Iris Species")

    # Input sliders
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    # Create input array
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0]

    # Display results
    st.subheader("Prediction Result")
    st.success(f"🌺 Predicted Species: **{iris.target_names[prediction].capitalize()}**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": prediction_proba
    })
    st.bar_chart(proba_df.set_index("Species"))

# -------------------- Data Exploration Mode --------------------
else:
    st.header("📊 Data Exploration")

    st.write("Explore the Iris dataset below:")
    st.dataframe(df.head())

    # Histogram
    feature = st.selectbox("Choose a feature for histogram:", iris.feature_names)
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=15, color='skyblue', edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Scatter Plot")
    x_feature = st.selectbox("X-axis", iris.feature_names, index=0)
    y_feature = st.selectbox("Y-axis", iris.feature_names, index=1)
    fig2, ax2 = plt.subplots()
    for i, species in enumerate(iris.target_names):
        subset = df[df['target_name'] == species]
        ax2.scatter(subset[x_feature], subset[y_feature], label=species)
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    ax2.legend()
    st.pyplot(fig2)
