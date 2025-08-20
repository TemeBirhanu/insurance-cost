# Medical Insurance Pricing Analysis Project

## Project Overview

This project focuses on analyzing the factors that influence medical insurance pricing. The key objective is to examine relationships between several demographic and lifestyle factors such as BMI, age, smoking status, region, and medical charges. Additionally, the project implements machine learning models to predict insurance charges using these attributes.

### Key Steps of the Project
1. **Data Preprocessing**:

Normalization of continuous variables (e.g., Age, BMI, Medical Charges).
Encoding categorical variables (e.g., Sex, Smoker, Region) using one-hot encoding.
Log transformation of skewed features (e.g., Medical Charges) to ensure normal distribution.
2. **Exploratory Data Analysis**:

Visualizing raw distributions of key variables.
Performing correlation analysis to identify relationships between features.
Using statistical tests like Kolmogorov-Smirnov, Mann-Whitney U, and ANOVA to test hypotheses.

3, **Machine Learning Models:**

Linear Regression, XGBoost and ANN models were trained and evaluated.

**A. Linear Regression:**
Linear Regression is a basic yet powerful technique used to estimate the relationship between
one target variable and one or more predictors by fitting a straight line that minimizes prediction
error. Its simplicity, speed, and interpretability make it a strong baseline model for evaluating other
machine learning algorithms in structured datasets. 

**B. Extreme Gradient Boosting (XGBoost) Regression Model:**
Extreme Gradient Boosting (XGBoost) is a highly efficient and scalable ensemble learning
algorithm based on the gradient boosting framework. It constructs decision trees in a sequential
manner, with each new tree attempting to correct the residual errors of the previous ones. XGBoost
enhances this process through techniques such as regularization (L1 and L2), column subsampling,
and parallel computation, which collectively contribute to improved model accuracy and reduced
risk of overfitting

**C. Deep Learning Model: Artificial Neural Network (ANN):**
An Artificial Neural Network (ANN) was developed using the Keras Sequential API to perform
regression analysis for predicting medical insurance charges. The network architecture included an
input layer with 64 neurons, followed by a hidden layer with 32 neurons utilizing the ReLU activation
function. The output layer contained a single neuron, designed to output continuous insurance
charge feature.
The model was compiled with the Adam optimizer, and Mean Squared Error (MSE) was used
as the loss function to measure prediction accuracy during training. This deep learning model was
trained on the processed dataset to learn complex, non-linear relationships between the input features
and target variable.

To conclude, a range of models were implemented with appropriate configurations to predict
insurance charges.

## Model Interpretation
In addition to being able to achieve a high level of predictive accuracy, knowing why a model
makes certain predictions is extremely important, particularly in healthcare, where you need to know
the rationale behind your recommendations. To address this issue, we employed the SHAP (SHapley
Additive exPlanations) as an Explainable AI (XAI) method to interpret the predictions made by the
best model
SHAP (SHapley Additive exPlanations) was used to interpret the models and identify the most important features for predicting medical charges.
SHAP is valuable for detailing how each input feature contributes to the output of the model by
giving importance values to the individual features, which are based on cooperative game theory. 

Moreover, the SHAP algorithm reveals the magnitude of each feature's significance and the direction
in which each one impacts predicted insurance charges.


### Dataset

The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) and  It is a publicly available dataset that reflects real-world scenarios and has been
widely used for predictive modelling tasks related to healthcare cost estimation. The dataset consists
of 1,338 records and contains six independent features along with one target variable, totally seven
columns. It was further processed using data preprocessing, exploratory data analysis (EDA), and
feature engineering techniques to prepare it for model development and evaluation.

The dataset contains 1,338 records and includes multiple independent features along with one
dependent (target) variable. The features are of two data types: numerical and categorical. The
attributes include age, sex, BMI, number of children, smoking status, and region, while the target
variable is medical insurance charges. These features are used to predict how much a person might
be charged for medical insurance.

 

### How to Run

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TemeBirhanu/insurance-cost.git

2. **Navigate to the project directory**:
   ```bash
   cd insurance-cost
   ```
3. **Install the required Python packages**:
   Make sure you have Python 3.x installed on your system. Run the following command to install all the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter notebook**:
   After installing the dependencies, you can run the Jupyter notebook to reproduce the analysis:
   ```bash
   jupyter notebook
   ```

   Open the `notebk.ipynb` and run all the cells to perform the analysis and generate results.