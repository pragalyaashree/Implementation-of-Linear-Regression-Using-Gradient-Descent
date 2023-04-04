# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard python libraries for Gradient design.

2.Introduce the variables needed to execute the function.

3.Use function for the representation of the graph.

4.Using for loop apply the concept using the formulae.

5.Execute the program and plot the graph.

6.Predict and execute the values for the given conditions. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: R.K Pragalyaa shree
RegisterNumber:  212221040125


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data=pd.read_csv('/content/ex1.txt',header=None)

plt.scatter(data[0],data[1])

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("population of city(10,000s")

plt.ylabel("profit($10,000")

plt.title("profit prediction")

plt.show()

import numpy as np

import matplotlib.pyplot as plt

# Generate some random data

np.random.seed(42)

X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)

# Define the cost function

def cost_function(theta, X, y):
     m = len(y)
     
     predictions = X.dot(theta)
    
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    
    return cost
    
def gradient_descent(X, y, theta, learning_rate, iterations):
    
    m = len(y)
    
    cost_history = np.zeros(iterations)
    
    theta_history = np.zeros((iterations, 2))
    
    for i in range(iterations):
        
        predictions = X.dot(theta)
        
        errors = predictions - y
        
        gradient = (1/m) * X.T.dot(errors)
        
        theta = theta - learning_rate * gradient
        
        cost = cost_function(theta, X, y)
        
        cost_history[i] = cost
        
        theta_history[i] = theta.reshape(2,)
        
    return theta, cost_history, theta_history

lr = 0.10

n_iter = 10

theta = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1)), X]

theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)

plt.plot(range(n_iter), cost_history)

plt.xlabel('Iterations')

plt.ylabel('j(heta)')

plt.title(' Cost function using Gradient Descent')

plt.show()

def computecost(x,y,theta):
  
  m=len(y)
  
  h=x.dot(theta)
  
  se=(h-y)**2
  
  return 1/(2*m)*np.sum(se)
  
  data_n=data.values

m=data_n[:,0].size

x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)

y=data_n[:,1].reshape(m,1)

theta=np.zeros((2,1))

computecost(x,y,theta)

def gradientd(x,y,theta,alpha,num_iters):
  
  m=len(y)
  
  j_history=[]
  
  for i in range(num_iters):
    
    predictions=x.dot(theta)
    
    error=np.dot(x.transpose(),(predictions-y))
    
    descent=alpha*1/m*error
    
    theta-=descent
    
    j_history.append(computecost(x,y,theta))
    
    return theta,j_history
    
    theta,j_history=gradientd(x,y,theta,0.01,1500)

print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)

plt.xlabel("iteration")

plt.ylabel("$j(\theta)$")

plt.title("cost function using gradient descent")

plt.scatter(data[0],data[1])

x_value=[x for x in range(25)]

y_value=[y*theta[1]+theta[0] for y in x_value]

plt.plot(x_value,y_value,color="purple")

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30,step=5))

plt.xlabel("population of city(10,000s)")

plt.ylabel("profit($10,000)")

plt.title("profit prediction")

def predict(x,theta):
  
  predictions=np.dot(theta.transpose(),x)
  
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000

print("for population=35000 we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000

print("for population=70000 we predict a profit of $"+str(round(predict2,0)))
/*

```
## Output:
## Profit prediction graph:
<img width="435" alt="ml 1" src="https://user-images.githubusercontent.com/128135934/229811645-f92c3d23-ad94-4f94-bbf3-074c78759fe7.png">

## Compute cost value:
<img width="132" alt="ml 2" src="https://user-images.githubusercontent.com/128135934/229812464-e37f4a86-4692-481a-ac54-e373a9f1585b.png">

## h(x) value:
<img width="165" alt="ml 3" src="https://user-images.githubusercontent.com/128135934/229812686-7c643987-0932-4f0d-9861-746158225f70.png">

## Cost function using Gradient Descent graph:
<img width="431" alt="ml 4" src="https://user-images.githubusercontent.com/128135934/229812909-97a6f87e-96e9-4f22-a9bb-38f16319fa91.png">

## Profit prediction graph:
<img width="626" alt="ml 5" src="https://user-images.githubusercontent.com/128135934/229813091-a12dab1e-b454-4d7a-93ff-46e5db5aff1a.png">

## Profit for the population 35,000:
<img width="346" alt="ml 6" src="https://user-images.githubusercontent.com/128135934/229813338-0594f2a2-2900-47ec-b70c-9605c5f16986.png">

## Profit for the population 70,000:
<img width="339" alt="ml 7" src="https://user-images.githubusercontent.com/128135934/229813599-1d5b5968-5161-490d-b677-4d5a4789c2c0.png">

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
