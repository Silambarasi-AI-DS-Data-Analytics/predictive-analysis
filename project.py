import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'Month': range(1, 49),
    'Sales': [120,135,128,142,150,165,158,172,168,180,
              175,190,195,210,205,220,215,230,225,240,
              235,250,245,260,258,272,268,280,275,290,
              285,300,295,310,305,320,315,330,325,340,
              335,350,345,360,355,370,365,380]
}

df = pd.DataFrame(data)

# Split data
X = df[['Month']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
print(f"Model Accuracy: {r2*100:.2f}%")

# Future prediction
future_months = pd.DataFrame({'Month': [49, 50, 51, 52]})
future_sales = model.predict(future_months)
print("\nFuture Sales Prediction:")
for i, sale in zip([49,50,51,52], future_sales):
    print(f"Month {i}: {sale:.0f} units")

# Visualize
plt.figure(figsize=(10,6))
plt.plot(df['Month'], df['Sales'], label='Historical Sales', color='blue')
plt.plot([49,50,51,52], future_sales, label='Predicted Sales', 
         color='red', linestyle='--', marker='o')
plt.title('Sales Prediction Using Linear Regression')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()