import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("archive (3)/cardekho.csv")

# Features and Target
X = df[["year", "km_driven"]]
y = df["selling_price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Example Prediction
prediction = model.predict([[2020, 30000]])

print("\nPredicted Price:")
print(int(prediction[0]))
