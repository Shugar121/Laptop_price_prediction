import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('laptop_data.csv')


df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
df = df.dropna()

X = pd.get_dummies(df.drop("Price", axis=1), drop_first=True)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)
print("MAE:", mae)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

with open("metrics.pkl", "wb") as f:
    pickle.dump((r2, mae), f)

print("✅ model.pkl, scaler.pkl, columns.pkl, metrics.pkl сохранены")


if not os.path.exists("static"):
    os.makedirs("static")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Laptop Prices")

plt.savefig("static/plot.png")
plt.close()

print("📈 plot.png сохранён в static/")