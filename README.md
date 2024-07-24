# Titanic - Machine Learning from Disaster

This project aims to predict the survival of passengers on the Titanic using a machine learning model. It includes a graphical user interface (GUI) for users to input passenger details and obtain survival predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Technologies Used](#technologies-used)

## Overview

The Titanic - Machine Learning from Disaster project is a Kaggle competition aimed at predicting the survival of passengers on the Titanic. This repository contains the code for data preprocessing, model training, and a user-friendly GUI for predictions.

## Features

- Imputation of missing values
- Feature engineering (family size, title extraction, fare per person)
- Gradient Boosting Classifier for predictions
- CustomTkinter-based GUI for user input and prediction display

## Usage

- Data Preprocessing: Ensure the train.csv file is placed in the data/ directory. The data preprocessing steps are included in the script.
- Running the GUI: Navigate to the src/ directory and run the GUI_2.py script.
- Using the GUI
  Enter the passenger's details:
  Class
  Name
  Sex
  Age
  Number of siblings/spouses
  Number of parents/children
  Ticket fare
  Embarkation point
  Click the "PREDICT" button to see the survival prediction.
  Click the "CLEAR" button to reset the input fields.

### Prerequisites

- Python 3.x

## Data Preprocessing

1. Imputation of Missing Values

- Mean age for missing age values
- Most common port for missing embarkation values

2. Feature Engineering

- One-hot encoding for 'Sex' and 'Embarked' columns
- Title extraction from names and grouping into categories
- Family size calculation and 'IsAlone' feature creation
- Age grouping into bins
- Fare per person calculation

3. Model Training

- Gradient Boosting Classifier with selected features

## Technologies Used

- Python
- pandas
- scikit-learn
- customtkinter
