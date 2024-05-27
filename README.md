# Predicting House Price

This project focuses on predicting house prices using the Ames, Iowa housing dataset. It involves a comprehensive workflow that starts with data analysis and visualization to understand the dataset's structure and key features. Feature engineering is performed to handle missing values, encode categorical variables, and create new features. Finally, various machine learning models are trained on the data alongside hyperparameter tuning and performance benchmarking to select the best fitting approach.

## Installation

This script requires the use of Jupyter Notebook and the following dependencies:
- Matplotlib (plotting results)
- NumPy (manipulating arrays and apply mathematical operations)
- Pandas (store CSV data as a dataframe)
- Seaborn (statistical data visualisation)
- Scikit-learn (model creation and evaluating performance)

```bash
pip install matplotlib
pip install numpy
pip install pandas
pip install seaborn
pip install scikit-learn
```

## Usage

To use this script, follow these steps:
- Clone the repository or download it as a zip folder and extract all files.
- Ensure you have installed the required dependencies.
- Run the Price_Predictor.ipynb notebook and ensure the relative paths to the CSV files are correct.

## Methodology

**Data Collection**
The Ames Housing dataset is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

**Data Checks**
Ensure that all data points are assigned to the correct data type and looks consistent.

**Exploratory Data Analysis (EDA)**
EDA was performed to understand the relationships and distributions of the features. Key steps included:
- Plotting distributions of numerical features.
- Using correlation heatmaps to identify relationships numerical features and the target variable, SalePrice.
- Using box plots to visualise relationships between categorical features and the target variable, SalePrice.

**Transform Data**
Data preprocessing is crucial to ensure the quality and usability of the dataset for modeling. The following steps were taken:

#### 3.1 Handling Missing Values
- **Numerical Features**: Missing values were imputed using the median of the respective feature.
- **Categorical Features**: Missing values were imputed using the mode (most frequent value) of the respective feature.

#### 3.2 Encoding Categorical Features
Categorical features were encoded using one-hot encoding to convert them into numerical format, suitable for machine learning algorithms.

#### 3.3 Feature Scaling
Numerical features were scaled using StandardScaler to standardize the features by removing the mean and scaling to unit variance.

**Feature Engineering**
New features were created to enhance the model's predictive power.

**Model Selection**
Several machine learning algorithms were considered and evaluated to ensure robust performance metrics: The primary evaluation metric was Root Mean Squared Error (RMSE).

### 8. Hyperparameter Tuning
Hyperparameter tuning was performed using GridSearchCV to find the optimal parameters for the best-performing models.

### 9. Model Training and Validation
The dataset was split into training and validation sets using an 80-20 split. The selected models were trained on the training set and evaluated on the validation set.

### 10. Final Model Selection
The best model was selected based on the lowest RMSE on the validation set. Feature importances were also analyzed to understand the contribution of each feature to the model.



## Results
The best performing model - MODEL NAME - achieved an accuracy of **xx.xx%** on the test set, with a MSPE of **______**. More information can be found in the notebook.

## References
- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education. [Kaggle] (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).