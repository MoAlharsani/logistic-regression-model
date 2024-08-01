# Logistic Regression Model

This project demonstrates the creation of a logistic regression model using the Titanic dataset. We will cover data loading, processing, model definition, training, and evaluation.

## Project Overview

The goal of this project is to introduce key steps in data analysis and machine learning using the Titanic dataset. This dataset, available on Kaggle, contains information about Titanic passengers and whether they survived. You can download the dataset and explore more details at the following links:

- [Titanic Dataset Overview](https://www.kaggle.com/competitions/titanic/overview)
- [Download Dataset](https://www.kaggle.com/competitions/titanic/data)

This repo is based on a Colab note prepared by [Majd Ahmad
](https://github.com/Mjd0001)

## Accuracy

The logistic regression model achieves an accuracy of **78.85%**.

## Libraries

We use the following libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

- `numpy` and `pandas` for data manipulation.
- `matplotlib.pyplot` and `seaborn` for visualization.
- `sklearn` for machine learning tasks.

## Data Loading

Load the dataset from a CSV file:

```python
data = pd.read_csv('https://github.com/MoAlharsani/logistic-regression-model/raw/main/data/data.csv')
```

## Data Processing

The dataset requires cleaning and preprocessing:

1. **Handle Missing Values:**
   - Fill missing values in the `Age` column with the mean age.
   - Fill missing values in the `Embarked` column with the most frequent value.
   - Drop the `Cabin` column due to excessive missing values.

2. **Drop Unnecessary Columns:**
   - Remove columns like `PassengerId`, `Name`, `Ticket`, and `Cabin` that are not useful for the model.

3. **Encode Categorical Variables:**
   - Replace `Sex` values with 0 for male and 1 for female.
   - Replace `Embarked` values with 0 for 'S', 1 for 'C', and 2 for 'Q'.

4. **Handle Duplicates:**
   - Drop any duplicate rows in the dataset.

```python
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
data.drop_duplicates(inplace=True)
data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
```

## Define Features and Target Variable

Separate the features from the target variable:

```python
X = data.drop(columns=['Survived'])
y = data['Survived']
```

## Data Splitting

Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- `test_size=0.2` means 20% of the data is used for testing.
- `random_state=42` ensures reproducibility.

## Model Definition

Create the logistic regression model:

```python
model = LogisticRegression()
```

## Model Training

Train the model on the training data:

```python
model.fit(X_train, y_train)
```

## Evaluation

Predict and evaluate the model:

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

- `accuracy_score` measures the proportion of correct predictions.

