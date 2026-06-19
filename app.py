# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. Load Dataset
data = pd.read_csv("winequality.csv")

# 3. Drop unnecessary column (if present)
if 'Id' in data.columns:
    data = data.drop('Id', axis=1)

# 4. Convert target to binary classification
# Good wine = 1 (quality >= 7), else 0
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 5. Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Build ANN Model
model = Sequential()

model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 9. Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 10. Train Model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# 11. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

# 12. Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# 13. Performance Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
