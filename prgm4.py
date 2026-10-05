import numpy as np
from matplotlib import pyplot as plt

X = np.array([10, 15, 20, 25, 30], dtype=float)

Y = np.array([50, 65, 78, 90, 105], dtype=float)
n = len(X)

b0 = 0.0
b1 = 0.0

alpha = 0.001

iterations = 10000

m = len(X)

for i in range(iterations):

    Y_pred = b0 + b1 * X

    db0 = (-2 / m) * np.sum(Y - Y_pred)
    db1 = (-2 / m) * np.sum(X * (Y - Y_pred))

    b0 = b0 - alpha * db0
    b1 = b1 - alpha * db1

print("Intercept (b0) =", b0)
print("Slope (b1) =", b1)

X_new = 22

predicted_sales = b0 + b1 * X_new

print("Regression Equation:")
print("Sales =", b0, "+", b1, "* Advertising")

print("\nAdvertising expenditure =", 22000)
print("Predicted Sales =", predicted_sales, "Rs.'000")
print("Predicted Sales = Rs.", predicted_sales * 1000)

absolute_errors = np.abs(Y - Y_pred)

MAE = np.sum(absolute_errors) / n
print("Mean Absolute Error (MAE) =", round(MAE, 4))

Y_mean = np.mean(Y)

SS_res = np.sum((Y - Y_pred) ** 2)
print("RMSE =", np.sqrt(SS_res/n))

SS_tot = np.sum((Y - Y_mean) ** 2)

R2 = 1 - (SS_res / SS_tot)
print("R-squared =", R2)

plt.scatter(X, Y, color="m", marker="o", s=30)
plt.plot(X, Y_pred, label="Gradient Descent Line")

plt.xlabel("Advertising Expenditure (Rs.'000)")
plt.ylabel("Sales (Rs.'000)")
plt.title("Linear Regression with Gradient Descent")

plt.legend()
plt.grid(True)

import os
os.makedirs("Figures", exist_ok=True)

plt.savefig("Figures/pgm4-GradientDescent.png")

plt.show()

plt.show()