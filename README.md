
# Drug Consumption Classification

This project uses machine learning techniques to predict if an individual its prone to use different types of drugs based on several factors such as education, age and different personality tests. The goal is to create a tool to help fight drug abuse.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
My goal with this notebook is to interpret the use of drugs by the global population and to explore any relationship between substance use and the individual's psyche. Additionally, a study and classification model will be conducted to determine whether a given individual is prone to using certain substances. The idea is that this mini-project will serve as a foundation for anyone wishing to conduct future analyses for the prevention and treatment of people in this situation, a problem that persists in my home country, Brazil.


## Data
The dataset used in this project was cleaned and transformed to fit the model. You can explore the original data by downloading it from the following link:

- [Drug Consumption](https://www.kaggle.com/datasets/mexwell/drug-consumption-classification)

## Data Cleaning

The data cleaning process involves several steps to ensure the dataset is properly formatted for analysis:

1. **Loading the Data**:  
   The dataset is imported from a CSV file hosted on GitHub using `pd.read_csv()`. A delimiter of `','` is used to separate the columns.

2. **Dropping Unnecessary Columns**:  
   Several columns that are not relevant to the analysis, such as `ID`, `Gender`, `Ethnicity`, and some drug-related columns (`Amphet`, `Amyl`, etc.), are removed from the dataset using the `drop()` method.

3. **Age Transformation**:  
   The numeric values in the `Age` column are replaced with meaningful age ranges. This is done using a dictionary that maps the numerical codes to human-readable age ranges (e.g., `-0.95197` is replaced with `'18-24'`).

4. **Education Transformation**:  
   Similarly, the `Education` column, which contains numerical codes representing different education levels, is transformed into human-readable labels such as `'Some College'`, `'Masters'`, and `'Doctorate'`.

5. **Drug Usage Encoding**:  
   The drug consumption columns, which contain categorical values representing different levels of consumption (e.g., `CL0`, `CL1`), are recoded into binary values (`0` for non-users, `1` for users). This binary encoding simplifies the analysis.

6. **Rejoining Columns**:  
   The dataset is split into three parts during processing:
   - `df_object`: contains object-type columns (`Age`, `Education`).
   - `df_numbers`: contains the numeric columns.
   - `df_drugs`: contains drug consumption columns, after being encoded.  
   After processing, these components are merged back together using `join()`.

This cleaned dataset is then ready for analysis and visualization.


### Exploratory Data Analysis (EDA)

This section provides insights into the dataset with visualizations and statistical summaries.

1. **Dataframe Display:**
   - Users can toggle the full dataset view.
   
2. **Visualization Options:**
   - **Drug Frequency:** Shows the count of drug use per substance in a grid of subplots.
   - **Frequency per Age:** Compares drug usage across age groups.
   - **Frequency per Education:** Visualizes drug usage across different education levels.

3. **Statistical Insights:**
   - Provides the percentage of cocaine users by education and age, highlighting higher usage among university students and young adults (18-24). However, sample bias is noted due to uneven representation.

4. **Key Findings:**
   - University students show higher drug use frequency.
   - Pessimism, impulsiveness, and openness correlate with drug use tendencies.

These visualizations help uncover trends that can guide further analysis, such as feature selection for predictive modeling.

## Model Training
The dataset was split into training and test sets, and a RandomForestRegressor was trained. A GridSearchCV was applied to optimize the model's hyperparameters.

Steps:
1. Data was encoded using OneHotEncoding for categorical variables and normalized.
2. The best model was selected using cross-validation.
3. Final evaluation metrics were calculated on the test set.

## Results
The model achieved above 70% F1-score for all drugs except Nicotine and Alcohol, maintaining an accuracy above 50% for all drugs except Alcohol.
* [Project Video(PT-BR)](https://youtu.be/wKpKfJYoIjY)

## Conclusion
The model demonstrated good performance with an F1-score above 70% for most drugs, excluding Nicotine and Alcohol. It achieved over 50% accuracy across all substances except Alcohol. However, due to imbalanced classes, particularly with harder drugs where most respondents were non-users, the high precision scores may be misleading. For example, 96% of users did not use crack, so predicting all as non-users would still yield high accuracy.

The study highlights that false negatives (failing to identify drug users) are less problematic than false positives, which could fail to provide help to those in need. With more data, the model could become more robust, but the results so far show promising potential for this type of analysis.

### [App](https://drugconsumption.streamlit.app/)
