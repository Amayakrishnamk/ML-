import numpy as np

X1 = np.array([2, 3, 4, 5, 6])
X2 = np.array([72, 74, 82, 84, 91])
Y = np.array([50, 55, 62, 68, 75])
n = len(X1)
X = np.column_stack((np.ones(n, dtype=int), X1, X2))
print("Matrix X:")
print(X)
XT = X.T
beta = np.linalg.inv(XT @ X) @ XT @ Y
b0, b1, b2 = beta
print("\nRegression Coefficients:")
print("b0 =", b0)
print("b1 =", b1)
print("b2 =", b2)
print("\nMultiple Linear Regression Equation:")
print(f"Y = {b0:.4f} + ({b1:.4f})X1 + ({b2:.4f})X2")
X_new = np.array([1, 5, 88])
predicted_marks = X_new @ bet
print("\nPrediction:")
print("Hours Studied =", X_new[1])
print("Attendance =", X_new[2], "%")
print(f"Predicted Marks = {predicted_marks:.2f}")
Y_pred = X @ beta
print("\nActual and Predicted Marks:")
for actual, predicted in zip(Y, Y_pred):
    print(f"Actual = {actual:.2f}, Predicted = {predicted:.2f}")
absolute_errors = np.abs(Y - Y_pred)
MAE = np.sum(absolute_errors) / n
print("\nMean Absolute Error (MAE) =", round(MAE, 4))
Y_mean = np.mean(Y)
SS_res = np.sum((Y - Y_pred) ** 2)
print("Residual Sum of Squares (SS_res) =", round(SS_res, 4))
SS_tot = np.sum((Y - Y_mean) ** 2)
R2 = 1 - (SS_res / SS_tot)
print("R-squared =", round(R2, 4))