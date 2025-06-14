# Customer-Feedback-Classification

This project, "Can I Speak to the Manager? Machine Learning for Customer Feedback Classification", focuses on building a machine learning system to automatically classify customer comments into 28 distinct product categories. The aim is to streamline customer feedback management by routing each comment to the appropriate department, improving response efficiency and internal workflows.

## Dataset
Given a dataset of customer comments transformed into 300-dimensional NLP feature vectors, the objective is to develop a multiclass classification model that accurately directs each comment to the correct product department.
The dataset is provided in three subsets:
- Training Set: 10,000 labeled customer comments.
- Test Set 1: 1,000 unlabeled comments for initial evaluation.
- Test Set 2: 202 labeled + 1,818 unlabeled comments for investigating model performance under data distribution shifts.
Each customer comment is represented as a 300-dimensional feature vector, and the target variable includes 28 possible department categories.

## Key Challenges
- Class Imbalance: Some departments have significantly fewer feedback samples than others.
- Data Distribution Shift: Model performance degrades on newly collected data due to distributional changes.
- Model Generalisation: Ensuring the model can maintain performance under real-world deployment conditions.

## Project Components
- Exploratory Data Analysis (EDA): Cleaning and analysing the data to identify key features and class distribution patterns.
- Research: Reviewing state-of-the-art techniques for handling class imbalance and distribution shift in multiclass classification.
- Modeling: Building classification models (e.g., logistic regression, random forests, ensemble methods). Applying techniques like class weighting, oversampling (SMOTE), and hyperparameter tuning. Implementing ensemble approaches to improve classification robustness.
- Evaluation: Assessing model performance using weighted cross-entropy loss and class-specific metrics (precision, recall, F1-score) to ensure fair evaluation across imbalanced classes.
Distribution Shift Handling: Diagnosing performance degradation in new datasets and exploring adaptation techniques to handle shifts in data distribution.

## Tools and Technologies
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- NLP techniques for feature extraction

## Key Learning Outcomes
1. Building effective multiclass classifiers for imbalanced datasets.
2. Handling real-world challenges like distribution shift.
3. Designing evaluation strategies suitable for imbalanced multiclass problems.
4. Developing reproducible and well-documented machine learning pipelines.



