import numpy as np
from matplotlib import pyplot as plt
X = np.array([2000, 3000, 4000, 5000, 6000])
Y = np.array([25000, 30000, 34000, 40000, 45000])
n=len(X)

X_mean = np.mean(X)
Y_mean = np.mean(Y)     
b1 = (n*np.sum(X*Y)-np.sum(X)*np.sum(Y))/(n*np.sum(X*X)-(np.sum(X)*np.sum(X)))
b0 = Y_mean - b1 * X_mean

print("Simple Linear Regression")
print("------------------------")
print("Slope (b1)     =", b1)
print("Intercept (b0) =", b0)
print("\nRegression Equation:")
print("Y =", b0, "+", b1, "X")
advertising = 7000
predicted_sales = b0 + b1 * advertising
print("\nAdvertising Expenditure = Rs.", advertising)
print("Predicted Sales = Rs.", predicted_sales)
Y_pred = b0 + b1 * X
print("\nActual Sales vs Predicted Sales")
print("--------------------------------")
for i in range(n):
    print("Advertising:", X[i],
          "Actual Sales:", Y[i],
          "Predicted Sales:", Y_pred[i])

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, 'g*',label="Regression Line",linestyle='-')
plt.scatter(
    advertising,
    predicted_sales,
    marker="*",
    s=150,
    label="Prediction at Rs. 7000"
)
plt.xlabel("Advertising Expenditure (Rs.)")
plt.ylabel("Sales (Rs.)")
plt.title("Simple Linear Regression: Advertising vs Sales")
plt.legend()
plt.grid(True)
plt.savefig("pgm2-fig1.png")
plt.show()
absolute_errors = np.abs(Y - Y_pred)
MAE = np.sum(absolute_errors) / n
print("\nMean Absolute Error (MAE) =", MAE)
Y_mean = np.mean(Y)
SS_res = np.sum((Y - Y_pred) ** 2)
print("RMSE", np.sqrt(SS_res/n))

SS_tot = np.sum((Y - Y_mean) ** 2)
R2 = 1 - (SS_res / SS_tot)
print("R-squared =", R2)