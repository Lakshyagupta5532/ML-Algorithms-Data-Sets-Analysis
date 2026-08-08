# Bank Marketing Campaign Prediction

## About the Project

In this project, I worked on the Bank Marketing dataset to predict whether a customer would subscribe to a term deposit.

I followed the complete machine learning workflow, starting from data cleaning and exploratory data analysis to model training and evaluation.

## Dataset

- 41,188 records
- 20 input features
- 1 target variable: `y`
- Target classes:
  - `no`
  - `yes`

The dataset contains information about customers, their previous campaign history, contact details, and different economic indicators.

## Machine Learning Workflow

The following steps were performed:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Categorical Encoding
6. Train-Test Split
7. Feature Scaling
8. Model Training
9. Model Evaluation

## Models Used

I trained and compared the following classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Decision Tree
- Random Forest

## Model Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 90.11% |
| KNN | 89.50% |
| SVM | 90.14% |
| Naive Bayes | 75.63% |
| Decision Tree | 84.10% |
| Random Forest | 89.48% |

SVM achieved the highest accuracy of approximately 90.14%.

## Model Evaluation

The target variable was imbalanced, with significantly more `no` cases than `yes` cases.

Because of this, I did not rely only on accuracy. I also evaluated the models using:

- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation showed that although the models achieved around 90% accuracy, the recall for the `yes` class was only around 21-22%.

This means that the model was able to correctly identify only a relatively small portion of the customers who actually subscribed.

## Key Learning

The main learning from this project was that accuracy alone is not always enough to evaluate a classification model.

For imbalanced datasets, metrics such as precision, recall, F1-score, and the confusion matrix provide a better understanding of model performance.

## Conclusion

This project helped me understand the complete workflow of a machine learning classification problem, from data preprocessing and encoding to model training and evaluation.

It also helped me understand the effect of class imbalance and why different evaluation metrics are important when judging a machine learning model.
