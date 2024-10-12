# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Data Preprocessing: Load the dataset, drop unnecessary columns, convert categorical data to numerical using label encoding.

2. Model Initialization: Initialize parameters, define sigmoid and loss functions, and implement gradient descent.

3. Training: Train the model using gradient descent with specified alpha and iterations.

4. Prediction and Evaluation: Predict labels, calculate accuracy, and test the model with new data.


## Program and Ourputs:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```
<br>

```
# import modules.
import pandas as pd
import numpy as np
```
<br>

```
dataset = pd.read_csv("Placement_Data_Full_Class.csv")
dataset.head()
```
<br>

![output1](/1.png)
<br>

```
dataset.tail()
```
<br>

![output2](/2.png)
<br>

```
dataset.info()
```
<br>

![output3](/3.png)
<br>

```
# Dropping the serial 
dataset = dataset.drop('sl_no',axis=1)
dataset.info()
```
<br>

![output4](/4.png)
<br>

```
# Categorizing column for further labelling.
dataset['gender'] = dataset['gender'].astype('category')
dataset['ssc_b'] = dataset['ssc_b'].astype('category')
dataset['hsc_b'] = dataset['hsc_b'].astype('category')
dataset['degree_t'] = dataset['degree_t'].astype('category')
dataset['workex'] = dataset['workex'].astype('category')
dataset['specialisation'] = dataset['specialisation'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset['hsc_s'] = dataset['hsc_s'].astype('category')

# Analysing the datatype of the datase. 
dataset.dtypes
```
<br>

![output5](/5.png)
<br>

```
# Labelling the columns.
dataset['gender']=dataset['gender'].cat.codes
dataset['ssc_b']=dataset['ssc_b'].cat.codes
dataset['hsc_b']=dataset['hsc_b'].cat.codes
dataset['degree_t']=dataset['degree_t'].cat.codes
dataset['workex']=dataset['workex'].cat.codes
dataset['specialisation']=dataset['specialisation'].cat.codes
dataset['status']=dataset['status'].cat.codes
dataset['hsc_s']=dataset['hsc_s'].cat.codes
```
<br>

```
# Display dataset.
dataset.head()
```
<br>

![output6](/6.png)
<br>

```
# Selecting the features and labels.
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
```
<br>

```
# Analyse the shape of independent variable.
X.shape
```
<br>

![output7](/7.png)
<br>

```
# Analyse the shape of dependent variable.
Y.shape
```
<br>

![output8](/8.png)
<br>

```
# Initialize the model parameters.
theta = np.random.randn(X.shape[1])
y=Y

# Define the sigmoid function.
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
<br>

```
# Define the loss function.
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```
<br>

```
# Define the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
```
<br>

```
# Train the model.
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
```
<br>

```
# Make Predictions.
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
```
<br>

```
# Predict the y value by calling predict function.
y_pred = predict(theta, X)
```
<br>

```
# Actual y values.
print(y)
```
<br>

![output9](/9.png)
<br>

```
# Predicted y values.
print(y_pred)
```
<br>

![output10](/10.png)
<br>

```
# Evaluate the model.
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy)
```
<br>

![output11](/11.png)
<br>

```
print(theta)
```
<br>

![output12](/12.png)
<br>

```
# Testing the model with own data.
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
<br>

![output13](/13.png)
<br>

```
# Testing the model with own data.
x_new = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
y_pred_new = predict(theta, x_new)
print(y_pred_new)
```
<br>

![output14](/14.png)
<br>

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

