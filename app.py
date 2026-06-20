import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("🍷 Wine Quality Prediction using ANN")
st.write("Enter wine properties below:")

# 1. Load Dataset
data = pd.read_csv("winequality.csv")

if 'Id' in data.columns:
    data = data.drop('Id', axis=1)

# 2. Binary classification
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 3. Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 6. ANN Model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict"):
    user_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    user_data = scaler.transform(user_data)
    prediction = model.predict(user_data)

    if prediction[0][0] > 0.5:
        st.success("✅ Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")
