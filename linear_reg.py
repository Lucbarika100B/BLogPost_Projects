#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

#generate a syntehtic dataset
X, y = make_regression(n_samples=500, n_features=1, noise=10)

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

#Train our Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Plot the data
plt.scatter(X_test, y_test, color='blue', label='Actual prices')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted prices')
plt.xlabel('House Size (Square Feet)')
plt.ylabel('House Price ($ USD)')
plt.legend(loc='upper left')
plt.title('Boston Houses dataset - Price Prediction')
plt.show()