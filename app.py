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
fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.slider("Chlorides", 0.000, 1.000, 0.076)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.slider("Density", 0.9000, 1.1000, 0.9978)
pH = st.slider("pH", 2.0, 5.0, 3.51)
sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.slider("Alcohol", 5.0, 20.0, 9.4)

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
