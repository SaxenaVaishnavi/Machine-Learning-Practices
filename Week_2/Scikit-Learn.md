# SciKit-Learn Structure
## Introduction
sklearn API is well designed and uses the following design principles:
- **Consistency**: All APIs share a simple and consistent interface.
- **Inspection**: The learnable parameters as well as hyperparameters of all estimator's are accessible directly via public instance variables.
- **Nonproliferation of classes**: Datasets are represented as Numpy arrays or Scipy sparse matrix instead of custom designed classes.
- **Composition**: Existing building blocks are reduced as much as possible.
- **Sensible defaults** values are used for parameters that enables quick baseline building.

## Types of sklearn objects
1. **Estimators** performs parameter estimations.
2. **Predictors** performs predictions.
3. **Transformers** performs dataset transformation

### Estimators
- Estimates model parameter based on the dataset.
- Uses `fit()` method that takes dataset as an argument along with any hyperparameters. This method learns parameters of the model and returns reference to the estimator itself.
- Examples
    - `Imputer`
    - `LinearRegression`
    - `DecisionTreeRegression`
 
### Predictors
- Some estimators are capable of making prediction on a given dataset.
- Uses `predict()` method that takes dataset as an input and returns predictions.
- Uses `score()` method to measure quality of predictions for a given test set.
- For example, `LinearRegression`

### Transformers
- Some estimators can transform datasets.
- Uses `transform()` method to tranform the dataset.
- Possess `fit_transform()` method that fits the parameters and uses it to `transform()` the dataset.

## sklearn API
sklearn API can be arranged in the following manner:
1. Data (Loading, generation, preprocessing - transforms, feature selection and extraction)
2. Models
3. Model evaluation
4. Model inspection and selection

### Data

| Modules | Functionality |
| ------- | ------------- |
| [`sklearn.datasets`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) | Includes utilities to load datasets, including methods to load and fetch popular reference datasets. It also features some artificial data generators. |
| **Preprocessing**| |
| [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) | Includes scaling, centering, normalization, binarization methods.|
| [`sklearn.impute`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) | Transformers for missing value imputation |
| **Feature selection and extraction**||
| [`sklearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection) | Implements feature selection algorithms.|
| [`sklearn.feature_extraction`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction) | Performs feature extraction from raw data.|

### Model

### Supervised Models

- #### **REGRESSION**

  | Module | Functionality |
  | ------ | ------------- |
  | [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) | Implements linear models of regression and classification. |
  
  The regression classes include:
  1. Classicial linear regression models: Linear, ridge, lasso
  2. Linear regression with feature selection
  3. Bayesian linear regression
  4. Outlier robust regression
  5. Multi-task linear regression models
  6. Generalized linear models of regressions
  7. [`sklearn.trees`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree) includes decision tree-based models for classification and regression.
  8. [`sklearn.multioutput`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput) implements multioutput classification and regression.

- #### **CLASSIFICATION**

  1. [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) implements classical algorithms like logistic regression, SVM etc.
  2. [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm) includes Support Vector Machine algorithms.
  3. [`sklearn.trees`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree) includes decision tree-based models for classification and regression.
  4. [`sklearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors) implements the k-nearest neighbors algorithm.
  5. [`sklearn.naive_bayes`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes) implements *Naive-bayes* classification.
  6. [`sklearn.multiclass`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass) implements multi-class classification models.
  7. [`sklearn.multioutput`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput) implements multioutput classification and regression.
 
### Unsupervised Models
 - [`CLUSTERING`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) implements popular unsupervised clustering algorithms.

### Model Evaluation
* [`sklearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) implements several different metrics for model evaluation.
  * Classification metrics
  * Regression metrics
  * Clustering metrics

### Model inspection and selection
*  Model selection module: [`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
*  Model inspection: [`sklearn.inspection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection)

## References
* [sklearn user guide](https://scikit-learn.org/stable/user_guide.html#user-guide)
