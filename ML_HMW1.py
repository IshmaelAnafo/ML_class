#QUESTION 2 A


import pandas as pd
import math
from collections import Counter
from sklearn.metrics import accuracy_score

# Define a class for decision tree nodes
class TreeNode:
    def __init__(self, attribute=None, children=None, label=None):
        self.attribute = attribute
        self.children = children or {}
        self.label = label

# Minimum number of samples required for splitting a node
min_samples_for_split = 20

# Define a class for the Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None, splitting_criterion='information_gain'):
        self.max_depth = max_depth
        self.splitting_criterion = splitting_criterion

    def train(self, features, labels):
        self.root = self._build_tree(features, labels, depth=0)

    def _entropy(self, labels):
        label_counts = Counter(labels)
        label_probs = [count / len(labels) for count in label_counts.values()]
        return -sum([p * math.log(p, 2) for p in label_probs if p > 0])

    def _information_gain(self, features, labels, attribute):
        parent_entropy = self._entropy(labels)
        unique_values = set(features[attribute])
        weighted_entropy = sum([(features[attribute] == val).mean() *
            self._entropy(labels[features[attribute] == val]) for val in unique_values])
        return parent_entropy - weighted_entropy

    def _majority_error(self, features, labels, attribute):
        unique_values = set(features[attribute])
        return sum([(features[attribute] == val).mean() *
            (1 - Counter(labels[features[attribute] == val]).most_common(1)[0][1] / len(labels[features[attribute] == val]))
                    for val in unique_values])

    def _gini_index(self, features, labels, attribute):
        unique_values = set(features[attribute])
        gini = 1
        for val in unique_values:
            p = (features[attribute] == val).mean()
            gini -= p**2
        return gini

    def _choose_attribute(self, features, labels):
        if self.splitting_criterion == 'information_gain':
            scoring_function = self._information_gain
        elif self.splitting_criterion == 'majority_error':
            scoring_function = self._majority_error
        elif self.splitting_criterion == 'gini_index':
            scoring_function = self._gini_index
        gain_values = {attr: scoring_function(features, labels, attr) for attr in features.columns}
        return max(gain_values, key=gain_values.get)

    def _build_tree(self, features, labels, depth):
        if depth == self.max_depth or len(labels) < min_samples_for_split:
            return TreeNode(label=Counter(labels).most_common(1)[0][0])

        best_attribute = self._choose_attribute(features, labels)
        unique_attribute_values = set(features[best_attribute])
        node = TreeNode(attribute=best_attribute)

        for value in unique_attribute_values:
            subset_features = features[features[best_attribute] == value]
            subset_labels = labels[features[best_attribute] == value]
            child_node = self._build_tree(subset_features, subset_labels, depth + 1)
            node.children[value] = child_node

        return node

    def predict(self, sample):
        current_node = self.root
        while current_node.children:
            attribute = current_node.attribute
            value = sample[attribute]
            current_node = current_node.children[value]
        return current_node.label
#%%
#QUESTION 2B

# Load the training data
training_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/train.csv")

# Split the training data into features and labels
features_train = training_data.drop('label', axis=1)
labels_train = training_data['label']

# Load the test data
test_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/test.csv")

# Split the test data into features and labels
features_test = test_data.drop('label', axis=1)
labels_test = test_data['label']

# Define a dictionary to store results
results = {'Depth': [], 'Criterion': [], 'Train Error': [], 'Test Error': []}

# Iterate over different max depths and splitting criteria
for max_depth in range(1, 7):
    for splitting_criterion in ['information_gain', 'majority_error', 'gini_index']:
        # Create an instance of the DecisionTree with specified parameters
        tree = DecisionTree(max_depth=max_depth, splitting_criterion=splitting_criterion)
        
        # Train the decision tree on the training data
        tree.train(features_train, labels_train)
        
        # Predict labels for the training and test data
        labels_train_pred = features_train.apply(tree.predict, axis=1)
        labels_test_pred = features_test.apply(tree.predict, axis=1)
        
        # Calculate training and test errors
        train_error = 1 - accuracy_score(labels_train, labels_train_pred)
        test_error = 1 - accuracy_score(labels_test, labels_test_pred)
        
        # Store the results in the dictionary
        results['Depth'].append(max_depth)
        results['Criterion'].append(splitting_criterion)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Print the results table
print(results_df.to_string(index=False))

#%%
#QUESTION 2 C
print('It can be concluded that the train_error and test_error were the same for all 3 criterias at depth one. But they decrease gradually for Information gain and attain the best values at the 4th depth. They remain unaffected after depth 4. For Majority error, they remain the same regardless of the depth. Gini index test and train errors increase with depth. In general, train error decreased with depth and test error increased with depth. This is as a result of overfitting on the data.')


#%%
#QUESTION 3 A
# Define a class for decision tree nodes
class TreeNode:
    def __init__(self, attribute=None, children=None, label=None):
        self.attribute = attribute
        self.children = children or {}
        self.label = label

# Minimum number of samples required for splitting a node
min_samples_for_split = 100

# Define a class for the Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None, splitting_criterion='information_gain'):
        self.max_depth = max_depth
        self.splitting_criterion = splitting_criterion

    def train(self, features, labels):
        self.root = self._build_tree(features, labels, depth=0)

    def _entropy(self, labels):
        label_counts = Counter(labels)
        label_probs = [count / len(labels) for count in label_counts.values()]
        return -sum([p * math.log(p, 2) for p in label_probs if p > 0])

    def _information_gain(self, features, labels, attribute):
        parent_entropy = self._entropy(labels)
        unique_values = set(features[attribute])
        weighted_entropy = sum([(features[attribute] == val).mean() *
            self._entropy(labels[features[attribute] == val]) for val in unique_values])
        return parent_entropy - weighted_entropy

    def _majority_error(self, features, labels, attribute):
        unique_values = set(features[attribute])
        return sum([(features[attribute] == val).mean() *
            (1 - Counter(labels[features[attribute] == val]).most_common(1)[0][1] / len(labels[features[attribute] == val]))
                    for val in unique_values])

    def _gini_index(self, features, labels, attribute):
        unique_values = set(features[attribute])
        gini = 1
        for val in unique_values:
            p = (features[attribute] == val).mean()
            gini -= p**2
        return gini

    def _choose_attribute(self, features, labels):
        if self.splitting_criterion == 'information_gain':
            scoring_function = self._information_gain
        elif self.splitting_criterion == 'majority_error':
            scoring_function = self._majority_error
        elif self.splitting_criterion == 'gini_index':
            scoring_function = self._gini_index
        gain_values = {attr: scoring_function(features, labels, attr) for attr in features.columns}
        return max(gain_values, key=gain_values.get)

    def _build_tree(self, features, labels, depth):
       if depth == self.max_depth or len(labels) < min_samples_for_split:
           return TreeNode(label=Counter(labels).most_common(1)[0][0])

       best_attribute = self._choose_attribute(features, labels)
       unique_attribute_values = set(features[best_attribute])
       node = TreeNode(attribute=best_attribute)

       for value in unique_attribute_values:
           subset_features = features[features[best_attribute] == value]
           subset_labels = labels[features[best_attribute] == value]
           child_node = self._build_tree(subset_features, subset_labels, depth + 1)
           node.children[value] = child_node

       return node

    def predict(self, sample):
        current_node = self.root
        while current_node.children:
            attribute = current_node.attribute
            value = sample.get(attribute)  # Use .get() to handle unseen values
            try:
                current_node = current_node.children[value]
            except KeyError:
                return 'No'
        return current_node.label


#%%

#QUESTION 3A

# Load the training data
training_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/bank/train.csv")

# Calculate the median for each of the specified columns
median_age = training_data['age'].median()
median_balance = training_data['Balance'].median()
median_day = training_data['day'].median()
median_duration = training_data['duration'].median()
median_campaign = training_data['campaign'].median()
median_pdays = training_data['pdays'].median()
median_previous = training_data['previous'].median()

# Replace values in the specified columns with 1 if above median, else 0
training_data['age'] = (training_data['age'] > median_age).astype(int)
training_data['Balance'] = (training_data['Balance'] > median_balance).astype(int)
training_data['day'] = (training_data['day'] > median_day).astype(int)
training_data['duration'] = (training_data['duration'] > median_duration).astype(int)
training_data['campaign'] = (training_data['campaign'] > median_campaign).astype(int)
training_data['pdays'] = (training_data['pdays'] > median_pdays).astype(int)
training_data['previous'] = (training_data['previous'] > median_previous).astype(int)

# Split the training data into features and labels
features_train = training_data.drop('y', axis=1)
labels_train = training_data['y']

# Load the test data
test_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/bank/test.csv")

# Calculate the median for the specified columns in the test data
median_age_test = test_data['age'].median()
median_balance_test = test_data['Balance'].median()
median_day_test = test_data['day'].median()
median_duration_test = test_data['duration'].median()
median_campaign_test = test_data['campaign'].median()
median_pdays_test = test_data['pdays'].median()
median_previous_test = test_data['previous'].median()

# Replace values in the specified columns of the test data with 1 if above median, else 0
test_data['age'] = (test_data['age'] > median_age_test).astype(int)
test_data['Balance'] = (test_data['Balance'] > median_balance_test).astype(int)
test_data['day'] = (test_data['day'] > median_day_test).astype(int)
test_data['duration'] = (test_data['duration'] > median_duration_test).astype(int)
test_data['campaign'] = (test_data['campaign'] > median_campaign_test).astype(int)
test_data['pdays'] = (test_data['pdays'] > median_pdays_test).astype(int)
test_data['previous'] = (test_data['previous'] > median_previous_test).astype(int)

# Split the test data into features and labels
features_test = test_data.drop('y', axis=1)
labels_test = test_data['y']

# Define a dictionary to store results
results = {'Depth': [], 'Criterion': [], 'Train Error': [], 'Test Error': []}

# Iterate over different max depths and splitting criteria
for max_depth in range(1, 16):
    for splitting_criterion in ['information_gain', 'majority_error', 'gini_index']:
        # Create an instance of the DecisionTree with specified parameters
        tree = DecisionTree(max_depth=max_depth, splitting_criterion=splitting_criterion)
        
        # Train the decision tree on the training data
        tree.train(features_train, labels_train)
        
        # Predict labels for the training and test data
        labels_train_pred = features_train.apply(tree.predict, axis=1)
        labels_test_pred = features_test.apply(tree.predict, axis=1)
        
        # Calculate training and test errors
        train_error = 1 - accuracy_score(labels_train, labels_train_pred)
        test_error = 1 - accuracy_score(labels_test, labels_test_pred)
        
        # Store the results in the dictionary
        results['Depth'].append(max_depth)
        results['Criterion'].append(splitting_criterion)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Print the results table
print(results_df.to_string(index=False))

#%%

# QUESTION 3B


# Load the training data
training_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/bank/train.csv")

# Replace "unknown" values in the "poutcome" column with the majority value
majority_poutcome = training_data['poutcome'].mode()[0]
training_data['poutcome'] = training_data['poutcome'].replace('unknown', majority_poutcome)

# Calculate the median for each of the specified columns
median_age = training_data['age'].median()
median_balance = training_data['Balance'].median()
median_day = training_data['day'].median()
median_duration = training_data['duration'].median()
median_campaign = training_data['campaign'].median()
median_pdays = training_data['pdays'].median()
median_previous = training_data['previous'].median()

# Replace values in the specified columns with 1 if above median, else 0
training_data['age'] = (training_data['age'] > median_age).astype(int)
training_data['Balance'] = (training_data['Balance'] > median_balance).astype(int)
training_data['day'] = (training_data['day'] > median_day).astype(int)
training_data['duration'] = (training_data['duration'] > median_duration).astype(int)
training_data['campaign'] = (training_data['campaign'] > median_campaign).astype(int)
training_data['pdays'] = (training_data['pdays'] > median_pdays).astype(int)
training_data['previous'] = (training_data['previous'] > median_previous).astype(int)

# Split the training data into features and labels
features_train = training_data.drop('y', axis=1)
labels_train = training_data['y']

# Load the test data
test_data = pd.read_csv("C:/Users/u1422952/OneDrive - University of Utah/Desktop/Fall 2023/Machine Learning/hw1/bank/test.csv")

# Replace "unknown" values in the "poutcome" column with the majority value
test_data['poutcome'] = test_data['poutcome'].replace('unknown', majority_poutcome)

# Calculate the median for the specified columns in the test data
median_age_test = test_data['age'].median()
median_balance_test = test_data['Balance'].median()
median_day_test = test_data['day'].median()
median_duration_test = test_data['duration'].median()
median_campaign_test = test_data['campaign'].median()
median_pdays_test = test_data['pdays'].median()
median_previous_test = test_data['previous'].median()

# Replace values in the specified columns of the test data with 1 if above median, else 0
test_data['age'] = (test_data['age'] > median_age_test).astype(int)
test_data['Balance'] = (test_data['Balance'] > median_balance_test).astype(int)
test_data['day'] = (test_data['day'] > median_day_test).astype(int)
test_data['duration'] = (test_data['duration'] > median_duration_test).astype(int)
test_data['campaign'] = (test_data['campaign'] > median_campaign_test).astype(int)
test_data['pdays'] = (test_data['pdays'] > median_pdays_test).astype(int)
test_data['previous'] = (test_data['previous'] > median_previous_test).astype(int)

# Split the test data into features and labels
features_test = test_data.drop('y', axis=1)
labels_test = test_data['y']

# Define a dictionary to store results
results = {'Depth': [], 'Criterion': [], 'Train Error': [], 'Test Error': []}

# Iterate over different max depths and splitting criteria
for max_depth in range(1, 16):
    for splitting_criterion in ['information_gain', 'majority_error', 'gini_index']:
        # Create an instance of the DecisionTree with specified parameters
        tree = DecisionTree(max_depth=max_depth, splitting_criterion=splitting_criterion)
        
        # Train the decision tree on the training data
        tree.train(features_train, labels_train)
        
        # Predict labels for the training and test data
        labels_train_pred = features_train.apply(tree.predict, axis=1)
        labels_test_pred = features_test.apply(tree.predict, axis=1)
        
        # Calculate training and test errors
        train_error = 1 - accuracy_score(labels_train, labels_train_pred)
        test_error = 1 - accuracy_score(labels_test, labels_test_pred)
        
        # Store the results in the dictionary
        results['Depth'].append(max_depth)
        results['Criterion'].append(splitting_criterion)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Print the results table
print(results_df.to_string(index=False))


#%%
print('It can be concluded that the train_error and test_error were the same for all 3 criterias at depth one. But they decrease gradually for Information gain and attain the best values at the 4th depth. They remain unaffected after depth 4. For Majority error, they remain the same regardless of the depth. Gini index test and train error decreases at dept 2 but stays the same throughout the depths. In general, train error decreased with depth and test error increased with depth. This is as a result of overfitting on the data.')

print('\nWays to deal with missing values include 1. Using the most common value in that attribute 2. Using fractional counts and 3. Using the attribute value that has thew same taget label as the missing values label')