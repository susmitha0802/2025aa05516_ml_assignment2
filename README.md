# Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to address a **Binary Classification Problem** by developing supervised machine learning models to classify breast cancer as **Malignant (1) or Benign (0)**.

## b. Dataset Description
Dataset Name: Breast Cancer Wisconsin (Diagnostic) Data Set  
Source: Kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

Number of Instances: 569  
Number of Features:  33

Class distribution: 357 benign, 212 malignant

## Dataset Features Description

| Feature | Description |
|--------|------------|
| diagnosis | The diagnosis of breast tissues (M = malignant, B = benign) |
| id | ID number (removed during preprocessing as it is non-informative) |
| radius_mean | Mean of distances from center to points on the perimeter |
| texture_mean | Standard deviation of gray-scale values |
| perimeter_mean | Mean size of the core tumor |
| area_mean | Mean area of the core tumor |
| smoothness_mean | Mean of local variation in radius lengths |
| compactness_mean | Mean of (perimeter² / area − 1) |
| concavity_mean | Mean severity of concave portions of the contour |
| concave points_mean | Mean number of concave portions of the contour |
| symmetry_mean | Mean symmetry of the tumor |
| fractal_dimension_mean | Mean coastline approximation − 1 |
| radius_se | Standard error for radius |
| texture_se | Standard error for texture |
| perimeter_se | Standard error for perimeter |
| area_se | Standard error for area |
| smoothness_se | Standard error for smoothness |
| compactness_se | Standard error for (perimeter² / area − 1) |
| concavity_se | Standard error for severity of concave portions of the contour |
| concave points_se | Standard error for number of concave portions of the contour |
| symmetry_se | Standard error for symmetry of the tumor |
| fractal_dimension_se | Standard error for coastline approximation − 1 |
| radius_worst | Largest mean value of distances from center to points on the perimeter |
| texture_worst | Largest mean value of standard deviation of gray-scale values |
| perimeter_worst | Largest mean value of core tumor size |
| area_worst | Largest mean value of core tumor area |
| smoothness_worst | Largest mean value of local variation in radius lengths |
| compactness_worst | Largest mean value of (perimeter² / area − 1) |
| concavity_worst | Largest mean value of severity of concave portions |
| concave points_worst | Largest mean value of number of concave portions |
| symmetry_worst | Largest mean value of tumor symmetry |
| fractal_dimension_worst | Largest mean value of coastline approximation − 1 |

## c. Models Used

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian / Multinomial)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.982456 | 0.997024 | 0.976190 | 0.976190 | 0.976190 | 0.962302 |
| Decision Tree | 0.938596 | 0.931548 | 0.926829 | 0.904762 | 0.915663 | 0.867553 |
| kNN | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes | 0.921053 | 0.989087 | 0.923077 | 0.857143 | 0.888889 | 0.829162 |
| Random Forest (Ensemble) | 0.964912 | 0.995370 | 1.000000 | 0.904762 | 0.950000 | 0.925820 |
| XGBoost (Ensemble) | 0.973684 | 0.994048 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | The model exhibits very high accuracy with nearly identical precision, recall, and F1 score, indicating highly balanced and reliable identification of malignant and benign breast cancer cases. |
| Decision Tree | The model shows reasonably high precision but lower recall and MCC, suggesting some malignant cases are missed despite generally accurate predictions. |
| kNN | The model achieves strong precision and AUC with slightly reduced recall, indicating accurate malignant predictions while overlooking a subset of positive cases. |
| Naive Bayes | The model attains high AUC but noticeably lower recall and F1 score, implying effective class separation but weaker sensitivity in detecting malignant tumors. |
| Random Forest (Ensemble) | The model records perfect precision with high accuracy but lower recall, indicating no false malignant predictions while failing to detect some actual malignant cases. |
| XGBoost (Ensemble) | The model combines perfect precision with high recall and F1 score, reflecting confident and well-balanced detection of malignant and benign breast cancer instances. |