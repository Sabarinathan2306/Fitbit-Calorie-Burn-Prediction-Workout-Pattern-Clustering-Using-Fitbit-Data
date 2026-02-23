# =====================================================
# FITBIT CALORIE PREDICTION & WORKOUT CLUSTERING APP
# FULLY FIXED VERSION
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Fitbit ML App", layout="wide")
st.title("🏋️ Fitbit Calorie Burn Prediction & Workout Clustering")

# =====================================================
# SAFE CSV LOADING (WORKS IN CODESPACES & LOCAL)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Fitbit_data.csv")

if not os.path.exists(file_path):
    st.error("❌ fitbit_data.csv not found in this folder!")
    st.write("Available files:", os.listdir(BASE_DIR))
    st.stop()

df = pd.read_csv(file_path)
st.success("✅ Dataset Loaded Successfully")

# =====================================================
# DATA PREPROCESSING
# =====================================================

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# Fill missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# =====================================================
# TRAIN MODEL (IF NOT SAVED)
# =====================================================

X = df.drop("Calories_Burned", axis=1)
y = df["Calories_Burned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =====================================================
# SIDEBAR MENU
# =====================================================

menu = st.sidebar.radio(
    "Select Option",
    ["Calorie Prediction", "Workout Clustering"]
)

# =====================================================
# 1️⃣ CALORIE PREDICTION
# =====================================================

if menu == "Calorie Prediction":

    st.header("🔥 Predict Calories Burned")

    input_data = []

    for col in X.columns:

        if col in num_cols:
            value = st.number_input(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
            input_data.append(value)

        else:
            original_values = le_dict[col].classes_
            selected = st.selectbox(f"{col}", original_values)
            encoded_value = le_dict[col].transform([selected])[0]
            input_data.append(encoded_value)

    if st.button("Predict Calories"):

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)

        st.success(f"🔥 Estimated Calories Burned: {round(prediction[0], 2)}")

# =====================================================
# 2️⃣ WORKOUT CLUSTERING
# =====================================================

elif menu == "Workout Clustering":

    st.header("📊 Workout Pattern Clustering")

    cluster_df = df.copy()

    if "Workout_Type" in cluster_df.columns:
        cluster_df = cluster_df.drop("Workout_Type", axis=1)

    scaler_cluster = StandardScaler()
    scaled_data = scaler_cluster.fit_transform(cluster_df)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_data)

    cluster_df["Cluster"] = clusters

    st.subheader("PCA Cluster Visualization")

    fig, ax = plt.subplots()
    ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.subheader("Cluster Summary")
    st.dataframe(cluster_df.groupby("Cluster").mean())
