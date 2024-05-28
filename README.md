# Predicting House Prices

This project focuses on predicting house prices using the Ames, Iowa housing dataset. It involves a comprehensive workflow that starts with data analysis and visualization to understand the dataset's structure and key features. Feature engineering is performed to handle missing values, encode categorical variables, and create new features. Finally, various machine learning models are trained on the data alongside hyperparameter tuning and performance benchmarking to select the best approach.

## Installation

This script requires the use of Jupyter Notebook and the following dependencies:
- Matplotlib (plotting results)
- NumPy (manipulating arrays and apply mathematical operations)
- Pandas (store CSV data as a dataframe)
- Seaborn (statistical data visualisation)
- Scikit-learn (model creation and evaluating performance)
- XGBoost (model creation)

```bash
pip install matplotlib
pip install numpy
pip install pandas
pip install seaborn
pip install scikit-learn
pip install xgboost
```

## Usage

To use this script, follow these steps:
- Clone the repository or download it as a zip folder and extract all files.
- Ensure you have installed the required dependencies.
- Run the Price_Predictor.ipynb notebook and ensure the relative paths to the CSV files are correct.

## Methodology

**Data Collection**
- The Ames Housing dataset is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

**Data Checks**
- Ensure that all data points are assigned to the correct data type and looks consistent.

**Exploratory Data Analysis (EDA)**
- EDA was performed to understand the relationships and distributions of the features. Key steps included:
    - Plotting distributions of numerical features.
    - Using correlation heatmaps to identify relationships numerical features and the target variable, SalePrice.
    - Using box plots to visualise relationships between categorical features and the target variable, SalePrice.

**Transform Data**
- Data preprocessing is crucial to ensure the quality and usability of the dataset for modeling. The following steps were taken:
    - Missing values in numerical features were imputed using the mean of the respective feature.
    - Missing values in categorical features were imputed using the mean of the respective feature.
    - Any features that did not show signs of correlation with the target variable were removed
    - Categorical features were encoded using one-hot encoding to convert them into numerical format, suitable for machine learning algorithms.

**Feature Engineering**
- New features were created to enhance the model's predictive power.

**Model Selection**
- Several machine learning algorithms were considered and evaluated to ensure robust performance metrics
- The primary evaluation metric was Root Mean Squared Error (RMSE).
- Hyperparameter tuning was performed using GridSearchCV to find the optimal parameters for the best-performing models.
- The best model was selected based on the lowest RMSE on the training set.

## Results
The best performing model - **XGB** - achieved a RMSE of **25,913** on the training set, using 10-fold cross-validation. More information can be found in the notebook.

## References
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education. [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
