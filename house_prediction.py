# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("house_prices.csv")
x = data["size_sqft"].values.astype(np.float64)
y = data["price_usd"].values.astype(np.float64)

# Declare Initial Weight & Bias (random for now)
w = np.random.randn() * 0.01
b = np.random.randn() * 0.01

# Declare Learning Rate (not too large or small) + number of times gradient descent will run
learning_rate = 0.01
epochs = 1000

# Will be used later to plot results
costs = []

# Training Loop
for epoch in range(epochs):
    # Prediction
    f = w * x + b

    # Cost (MSE)
    j = np.mean((f - y) ** 2) / 2
    costs.append(j)

    # Gradient Descent
    dw = np.mean((f - y) * x)
    db = np.mean(f - y)
    w = w - learning_rate * dw
    b = b - learning_rate * db

final_predictions = w * x + b

# Plotting Data
plt.plot(range(epochs), costs)
plt.xlabel("Epoch")
plt.ylabel("Cost (J)")
plt.title("Cost Function Convergence")
plt.show()

plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, final_predictions, color="red", label="Fitted line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (USD)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Print learned parameters
print(f"Learned w: {w}, Learned b: {b}")
