# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:58:19 2023

@author: u1422952
"""

import numpy as np
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
import pandas as pd

training = pd.read_csv('train.csv', header = None)
X_train = training.iloc[:, :-1]
y_train = training.iloc[:, -1:]
y_train = y_train.replace ({0:-1})
y_train = np.array(y_train)
X_train = np.array(X_train)

testing = pd.read_csv('test.csv', header = None)
X_test = testing.iloc[:,:-1]
y_test = testing.iloc[:,-1:]
y_test = y_test.replace({0:-1})
X_test = X_test.replace({0:-1})
y_test = np.array(y_test)
X_test = np.array(X_test)
X_tests = X_test
X_trains = X_train 
X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))

X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))


#%%
#SOLUTION TO QUESTION 2 A

def learning_rate_schedule(gamma_0, t, alpha):
    return gamma_0 / (1 + (gamma_0 * t / alpha))

def svm_primal_sgd(X_train, y_train, X_test, y_test, C, gamma_0, alpha, max_epochs=100):
    w = np.zeros(X_train.shape[1])
    
    
    
    
    for epoch in range(max_epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        

        for i, xi in enumerate(X_train):
           
            t = epoch * len(X_train) + i
            eta = learning_rate_schedule(gamma_0, t, alpha)
            #print(eta)

            margin = y_train[i] * (np.dot(w, xi))

            if margin < 1:
                w = w + eta * (C * y_train[i] * xi*len(X_train)) - eta*w
                
            
    return w  



# Assuming X_train, y_train, X_test, y_test are your training and testing data
for C in [100/873, 500/873, 700/873]:
    for gamma_0 in [0.01, 0.1, 1.0]:
        for alpha in [0.001, 0.01, 0.1]:
            #print(f"\nTraining with C={C}, gamma_0={gamma_0}, alpha={alpha}")
            w= svm_primal_sgd(X_train, y_train, X_test, y_test, C, gamma_0, alpha)
            
    
   
    trained_pre = (np.sign(np.dot(X_train, w)))
    test_pre = (np.sign(np.dot(X_test, w)))
   
        
    train_error = 1 - accuracy_score(y_train, np.sign(np.dot(X_train, w)))
    test_error = 1 - accuracy_score(y_test, np.sign(np.dot(X_test, w)))
   
   
    print(f'\nWhen C is {C: .4f}, training error is{train_error: .4f} and test error is {test_error: .4f}, The associated weight is {w[:-1]} and b is {w[-1]}')
   
#%%

#SOLUTION TO QUESTION 2 B
def learning_rate_schedules(gamma_0, t):
    return gamma_0 / (1 + t)

def svm_primal_sgd2(X_train, y_train, X_test, y_test, C, gamma_0, max_epochs=100):
    w2 = np.zeros(X_train.shape[1])
    
    
    
    
    for epoch in range(max_epochs):
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        

        for i, xi in enumerate(X_train):
           
            t = epoch * len(X_train) + i
            eta = learning_rate_schedules(gamma_0, t)
         

            margin = y_train[i] * (np.dot(w, xi))

            if margin < 1:
                w2 = w + eta * (C * y_train[i] * xi*len(X_train)) - eta*w
                
            
    return w2  




# Assuming X_train, y_train, X_test, y_test are your training and testing data
for C in [100/873, 500/873, 700/873]:
    for gamma_0 in [0.01, 0.1, 1.0]:
        
        #print(f"\nTraining with C={C}, gamma_0={gamma_0}, alpha={alpha}")
        w2= svm_primal_sgd2(X_train, y_train, X_test, y_test, C, gamma_0)
            
    
   
    
        
    train_error2 = 1 - accuracy_score(y_train, np.sign(np.dot(X_train, w2)))
    test_error2 = 1 - accuracy_score(y_test, np.sign(np.dot(X_test, w2)))
   
   
    print(f'\nWhen C is {C: .4f}, training error is{train_error2: .4f} and test error is {test_error2: .4f}, The associated weight is {w2[:-1]} and b is {w2[-1]}')
   





#%%
#SOLUTION TO QUESTION 3 A
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()
# Define the values of C
C_values = [100/873, 500/873, 700/873]

# Loop over different values of C
for C in C_values:
    # Create an instance of the SVC with a linear kernel
    clf = SVC(C=C, kernel='linear', tol=1e-5)

    # Fit the model to the training data
    clf.fit(X_trains, y_train_flat)

    # Make predictions on the training data
    y_train_pred = clf.predict(X_trains)

    # Make predictions on the test data
    y_test_pred = clf.predict(X_tests)

    # Calculate training and test accuracy
    train_accuracy = 1- accuracy_score(y_train_flat, y_train_pred)
    test_accuracy = 1- accuracy_score(y_test_flat, y_test_pred)

    # Print the results
    print(f"\nResults for C = {C: .4f}")
    print(f"Training Error: {train_accuracy: .5f}")
    print(f"Test Error: {test_accuracy: .5f}")
    print(f"Weights (w): {clf.coef_}")
    print(f"Bias (b): {clf.intercept_}")

#%%
#SOLUTION TO QUESTION 3 B
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

# Define the values of C and gamma
C_values = [100/873, 500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]

# Loop over different values of C and gamma
for C in C_values:
    for gamma in gamma_values:
        # Create an instance of the SVC with an RBF kernel
        clf = SVC(C=C, kernel='rbf', gamma=gamma, tol=1e-5)

        # Fit the model to the training data
        clf.fit(X_trains, y_train_flat)

        # Make predictions on the training data
        y_train_pred = clf.predict(X_trains)

        # Make predictions on the test data
        y_test_pred = clf.predict(X_tests)

        # Calculate training and test accuracy
        train_accuracy = 1 - accuracy_score(y_train_flat, y_train_pred)
        test_accuracy = 1 - accuracy_score(y_test_flat, y_test_pred)

        # Print the results
        print(f"\nResults for C = {C:.4f}, Gamma = {gamma}")
        print(f"Training Error: {train_accuracy:.5f} and Test Error: {test_accuracy:.5f}")
       

#%%

#SOLUTION TO QUESTION 3 C
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

# Define the values of C and gamma
C_values = [100/873, 500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]

# Loop over different values of C and gamma
for C in C_values:
    for gamma in gamma_values:
        # Create an instance of the SVC with an RBF kernel
        clf = SVC(C=C, kernel='rbf', gamma=gamma, tol=1e-5)

        # Fit the model to the training data
        clf.fit(X_trains, y_train_flat)

        # Make predictions on the training data
        y_train_pred = clf.predict(X_trains)

        # Make predictions on the test data
        y_test_pred = clf.predict(X_tests)
        num_support_vectors = len(clf.support_)
        # Calculate training and test accuracy
        train_accuracy = 1 - accuracy_score(y_train_flat, y_train_pred)
        test_accuracy = 1 - accuracy_score(y_test_flat, y_test_pred)

        print(f"\nResults for C = {C: .4f} and gamma = {gamma: .3f}")
        print(f"Number of Support Vectors: {num_support_vectors}")
        
#%%

# Define the values of C
C_values = [500/873]  # Only using C = 500/873 for this analysis
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]

# Dictionary to store support vectors for each gamma value
support_vectors_dict = {gamma: None for gamma in gamma_values}

# Loop over different values of gamma
for gamma in gamma_values:
    # Create an instance of the SVC with an RBF kernel
    clf = SVC(C=C_values[0], kernel='rbf', tol=1e-5, gamma=gamma)

    # Fit the model to the training data
    clf.fit(X_train, y_train.ravel())

    # Get the support vectors
    support_vectors = clf.support_

    # Store the support vectors for the current gamma value
    support_vectors_dict[gamma] = set(support_vectors)

# Compare support vectors between consecutive gamma values
for i in range(len(gamma_values) - 1):
    current_gamma = gamma_values[i]
    next_gamma = gamma_values[i + 1]

    # Calculate the number of overlapped support vectors
    overlap_count = len(support_vectors_dict[current_gamma].intersection(support_vectors_dict[next_gamma]))

    # Print the results
    print(f"\nNumber of overlapping support vectors between gamma = {current_gamma} and gamma = {next_gamma}: {overlap_count}")


