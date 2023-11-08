# -*- coding: utf-8 -*-
"""
Preliminaries
"""

import numpy as np
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


#%%
#STANDARD PERCEPTRON

import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.max_epochs):
            # Shuffle the data before each epoch
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            errors = 0
            for i in range(X.shape[0]):
                prediction = np.dot(X[i], self.weights) + self.bias
                if prediction * y[i] <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    errors += 1
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs.")
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias), self.bias




# Create and train the Perceptron
perceptron = Perceptron(learning_rate=0.1, max_epochs=10)
perceptron.train(X_train, y_train)

# Predict on the test dataset
predictions, bias = perceptron.predict(X_test)

# Calculate the average prediction error
error = 0
for i in range((len(predictions))):
    if predictions [i] != y_test[i]:
        error += 1


error_rate = error / len(y_test)

print("Learned Weight Vector:", perceptron.weights,"and Bias ", bias)
print("Average Prediction Error on Test Dataset:", error_rate)

#%%

#VOTED PERCEPTRON

class VotedPerceptron:
    def __init__(self, max_epochs=10):
        self.max_epochs = max_epochs
        self.weight_vectors = []
        self.vote_counts = []

    def train(self, X, y):
        num_features = X.shape[1]
        num_samples = X.shape[0]

        weights = np.zeros(num_features)
        vote_count = 0
        
        
        for epoch in range(self.max_epochs):
            for i in range(num_samples):
                
                
                prediction = np.sign(np.dot(weights, X[i]))
                
                if prediction * y[i] <= 0:
                    
                    self.weight_vectors.append(weights.copy())
                    self.vote_counts.append(vote_count)
                    weights = weights + y[i] * X[i]
                    vote_count = 1
                else:
                    
                    vote_count += 1

        # Append the last weight vector
        self.weight_vectors.append(weights.copy())
        self.vote_counts.append(vote_count)

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)
        
        for i in range(num_samples):
            
            weighted_predictions = sum(self.vote_counts[j] * np.sign(np.dot(self.weight_vectors[j], X[i])) for j in range(len(self.weight_vectors)))
            predictions[i] = np.sign(weighted_predictions)
            
        
        return predictions
    
    
                     

if __name__ == "__main__":
    # Training data (X_train) and labels (y_train)


    voted_perceptron = VotedPerceptron(max_epochs=10)
    voted_perceptron.train(X_train, y_train)

    # Get distinct weight vectors and their counts
    distinct_weight_vectors = voted_perceptron.weight_vectors
    vote_counts = voted_perceptron.vote_counts
    

    for i in range(len(distinct_weight_vectors)):

        print("Weight vector ", distinct_weight_vectors[i], 'has ', vote_counts[i], 'count(s)')
    
    

    # Predict on the test dataset
    test_predictions = voted_perceptron.predict(X_test)

    # Calculate the average test error
    voted_error = 0
    for i in range((len(predictions))):
        if predictions [i] != y_test[i]:
            voted_error += 1


    voted_error_rate = voted_error / len(y_test)
    
    
    print("Average Prediction Error on Test Dataset using Voted Perceptron:", voted_error_rate)
    

#%%

#AVERAGED PERCEPTRON

class AveragedPerceptron:
    def __init__(self, max_epochs=10):
        self.max_epochs = max_epochs
        self.weights = None
        self.accumulator = None
        self.num_updates = 0

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.accumulator = np.zeros(num_features)

        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(num_samples):
                prediction = np.sign(np.dot(self.weights, X[i]))
                if prediction * y[i] <= 0:
                    self.weights += y[i] * X[i]
                    self.accumulator += self.weights
                    self.num_updates += 1
                    errors += 1

            if errors == 0:
                print(f"Converged after {epoch + 1} epochs.")
                break

    def get_average_weight_vector(self):
        return self.accumulator / self.num_updates

    def predict(self, X, weight_vector=None):
        if weight_vector is None:
            weight_vector = self.weights
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)
        for i in range(num_samples):
            predictions[i] = np.sign(np.dot(weight_vector, X[i]))
        return predictions

if __name__ == "__main__":
    # Training data (X_train) and labels (y_train)
  
    # Create and train the Averaged Perceptron
    averaged_perceptron = AveragedPerceptron(max_epochs=10)
    averaged_perceptron.train(X_train, y_train)

    # Get the average weight vector
    average_weight_vector = averaged_perceptron.get_average_weight_vector()

    # Compare the average weight vector with the last weight vector
    last_weight_vector = averaged_perceptron.weights

    print("Average Weight Vector:", average_weight_vector)
    print("Last Weight Vector:", last_weight_vector)
    print("Weight Vector Comparison:", np.array_equal(average_weight_vector, last_weight_vector))

    # Predict on the test dataset using the average weight vector
    test_predictions = averaged_perceptron.predict(X_test, weight_vector=average_weight_vector)

       # Calculate the average test error
    Averaged_error = 0
    for i in range((len(predictions))):
        if predictions [i] != y_test[i]:
            Averaged_error += 1


    Averaged_error_rate = Averaged_error / len(y_test)
       
       
    print("Average Prediction Error on Test Dataset using Voted:", Averaged_error_rate)
   
