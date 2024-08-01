# Logistic Regression Model

This document outlines the process of creating a logistic regression model using the Titanic dataset. The process is divided into several key sections: data loading, data processing, model definition, model training, and evaluation.

## Libraries

First, we import the necessary libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Loading

Load the dataset from a CSV file:

```python
# Load the dataset
data = pd.read_csv('/content/train.csv')
```

## Data Processing

Process the data by handling missing values, dropping unnecessary columns, and checking for duplicates:

```python


# Fill the missing values in Age with the mean of the Age column
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill the missing values in Embarked with the mode of the Embarked column
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the PassengerId and Name columns from the dataset
data.drop(columns=['PassengerId', 'Name'], inplace=True)

# Check for duplicates and drop them
data.drop_duplicates(inplace=True)
```

## Define Features and Target Variable

Define the feature columns and the target variable:

```python
# Define the feature columns and target variable
X = data.drop(columns=['Survived'])
y = data['Survived']
```

## Data Splitting

Split the data into training and testing sets:

```python
# Split the data into training data & testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Definition

Create a Logistic Regression model:

```python
# Create a Logistic Regression model
model = LogisticRegression()
```

## Model Training

Train the model using the training data:

```python
# Train it on the training data
model.fit(X_train, y_train)
```

## Evaluation

Predict the target variable on the test data and evaluate the model:

```python
# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')
```

