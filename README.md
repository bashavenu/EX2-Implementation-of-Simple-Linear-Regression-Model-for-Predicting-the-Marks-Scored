# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```

Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: BASHA VENU
RegisterNumber: 2305001005

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/ex1.csv")
df.head()
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,pred)
print(f'Mean Squared Error(MSE): {mse}')
```

## Output:
![image](https://github.com/user-attachments/assets/af5b6c56-04f6-49c1-af97-639e11e244b5)
![image](https://github.com/user-attachments/assets/2e7ff699-6d2d-4871-a05d-8f265fa96443)
![image](https://github.com/user-attachments/assets/c70d2cb1-92ec-4d5d-a48d-f556c832a606)
![image](https://github.com/user-attachments/assets/1674f4f3-a1d6-4591-9a13-e952b9755f9b)
![image](https://github.com/user-attachments/assets/1734abf9-029e-40e1-97a4-1d84f68f011e)
![image](https://github.com/user-attachments/assets/3bf4d3a3-2275-4f5e-8ada-9905964d490e)
![image](https://github.com/user-attachments/assets/30cd640c-cfb7-4a2d-bd01-46d9ccf9c225)
![image](https://github.com/user-attachments/assets/e8340660-0d08-4ecb-9c3d-cfd9adbb21fa)




## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
