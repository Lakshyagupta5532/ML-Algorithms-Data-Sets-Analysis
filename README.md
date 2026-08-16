# ML Algorithms & Data Sets Analysis

This repository documents my hands-on Machine Learning journey through real-world datasets and end-to-end ML projects.

Each project focuses on understanding the reasoning behind the Machine Learning workflow rather than simply applying algorithms. The goal is to build practical ML problem-solving skills and gradually move from classical Machine Learning towards Deep Learning and Generative AI.

---

## Workflow

Depending on the problem, projects may include:

- Problem Understanding
- Data Loading & Basic Understanding
- Data Cleaning
- Handling Missing Values
- Duplicate Detection & Removal
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Categorical Encoding
- Feature Scaling
- Train-Test Split
- Baseline Model
- Machine Learning Model Training
- Model Comparison
- Cross-Validation
- Hyperparameter Tuning
- Ensemble Learning
- Boosting
- Model Evaluation
- Error Analysis
- Threshold Tuning
- Model Interpretation
- Final Observations & Insights

---

## Machine Learning Algorithms Covered

### Classical Supervised Learning

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

### Ensemble Learning

- Bagging
- Random Forest
- Stacking

### Boosting

- AdaBoost
- Gradient Boosting
- XGBoost

### Model Optimization

- Cross-Validation
- GridSearchCV
- RandomizedSearchCV

---

## Projects

### 1. Credit Card Fraud Detection

An end-to-end binary classification project focused on detecting fraudulent credit card transactions in a highly imbalanced dataset.

#### Highlights

- Data Cleaning & Duplicate Handling
- Exploratory Data Analysis
- Severe Class Imbalance Analysis
- Feature Engineering using `log1p` transformation
- Feature Selection
- Stratified Train-Test Split
- Feature Scaling
- Logistic Regression Baseline
- Random Forest
- XGBoost
- Stratified Cross-Validation
- RandomizedSearchCV
- Error Analysis
- Probability Threshold Tuning
- Random Forest Feature Importance

#### Final Model

**Tuned Random Forest Classifier**

Default threshold test performance:

- Precision: **0.97**
- Recall: **0.73**
- F1-score: **0.83**
- ROC-AUC: **0.950**
- PR-AUC: **0.820**

After threshold tuning at `0.2`:

- Precision: **0.915**
- Recall: **0.789**
- F1-score: **0.847**

This project served as my final end-to-end consolidation of classical supervised Machine Learning concepts.

---

### 2. Customer Churn Prediction

Classification project focused on predicting customers who are likely to leave a service.

The project covered data preprocessing, exploratory analysis, categorical feature handling, model training, and performance evaluation.

---

### 3. Loan Approval Prediction

Classification project focused on predicting whether a loan application should be approved based on applicant information.

The project included data preprocessing, exploratory analysis, multiple classification algorithms, and model comparison.

---

### 4. Bank Marketing Campaign Prediction

Classification project using the Bank Marketing dataset to predict whether a customer would subscribe to a term deposit.

The project included:

- Exploratory Data Analysis
- Categorical feature preprocessing
- Multiple classification algorithms
- Accuracy, Precision, Recall & F1-score
- Confusion Matrix
- Class imbalance analysis

This project also helped me understand why accuracy alone can be misleading when the target classes are imbalanced.

---

### More Projects Coming Soon...

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Google Colab

---

## Key Learnings

Through these projects, I have learned how to:

- Understand and clean real-world datasets
- Perform exploratory data analysis
- Handle categorical and numerical features
- Detect missing values and duplicate records
- Identify and investigate outliers
- Perform feature engineering and feature selection
- Prepare data for Machine Learning models
- Choose appropriate preprocessing and scaling techniques
- Build baseline models
- Compare different Machine Learning algorithms
- Use cross-validation for reliable model comparison
- Perform hyperparameter tuning
- Apply ensemble learning and boosting
- Handle highly imbalanced classification problems
- Evaluate models using multiple metrics
- Perform error analysis
- Tune classification thresholds
- Interpret model feature importance
- Make model decisions based on the problem rather than accuracy alone

---

## Current Learning Roadmap

### Completed

- Classical Machine Learning Fundamentals
- Supervised Learning
- Ensemble Learning
- Boosting
- Cross-Validation
- Hyperparameter Tuning
- Practical ML Project Workflow

### Next

- Unsupervised Learning
- Feature Selection & Model Interpretability
- Deep Learning
- PyTorch
- Neural Networks
- Computer Vision / NLP
- Transformers
- Large Language Models (LLMs)
- RAG
- Generative AI
- AI Agents

---

## Goal

This repository is part of my ongoing Machine Learning and AI journey.

My goal is to build strong practical ML skills through real-world projects, understand how and why models work, and gradually move from classical Machine Learning towards Deep Learning, Generative AI, and modern AI systems.

The focus is on **learning by building, experimenting, and understanding the reasoning behind each step.**

---

⭐ More projects, experiments, and AI/ML work will be added as I continue learning and improving.
