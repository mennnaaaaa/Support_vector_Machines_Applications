# Support Vector Machine Applications

A personal collection of machine learning implementations focused on **Support Vector Machines (SVM)** for supervised classification.

The notebooks in this repository were developed while learning through courses, tutorials, and different learning resources. They are maintained as practical implementations and applied exercises, not as a university project.

## Repository Overview

The repository contains two SVM classification workflows:

1. An applied SVM project using the Iris flower dataset, including exploratory data analysis, model training, evaluation, and hyperparameter tuning with GridSearchCV.
2. An SVM implementation using the Breast Cancer Wisconsin dataset provided through Scikit-learn.

Together, the notebooks demonstrate the basic SVM workflow from loading and exploring data to training, prediction, evaluation, and parameter optimization.

## Project Structure

```text
Support_vector_Machines_Applications/
├── 02-Support Vector Machines Project.ipynb
└── SVMwithPython.ipynb
```

## Files and Their Purpose

| File | Description |
|---|---|
| `02-Support Vector Machines Project.ipynb` | Complete SVM classification project based on the Iris dataset. Includes exploratory data analysis, train/test splitting, SVC training, model evaluation, and GridSearchCV hyperparameter tuning. |
| `SVMwithPython.ipynb` | SVM classification implementation using Scikit-learn's Breast Cancer dataset, including dataset inspection, train/test splitting, model training, prediction, and evaluation. |

## 1. Iris Flower Classification with SVM

### Notebook

`02-Support Vector Machines Project.ipynb`

This notebook applies a Support Vector Classifier to the well-known Iris dataset.

### Dataset

The notebook loads the dataset directly with Seaborn:

```python
iris = sns.load_dataset("iris")
```

The dataset contains 150 flower samples distributed across three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The four measured input features are:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is:

```text
species
```

### Exploratory Data Analysis

The notebook explores the dataset visually using Matplotlib and Seaborn.

#### Pair Plot

A pair plot is created using:

```python
sns.pairplot(iris, hue="species", palette="Dark2")
```

This compares the feature relationships across the three flower species.

#### KDE Plot

The notebook also isolates the Setosa samples and creates a two-dimensional KDE plot:

```python
sns.kdeplot(
    x=setosa["sepal_width"],
    y=setosa["sepal_length"],
    cmap="plasma",
    fill=True
)
```

This provides a visual representation of the distribution of sepal measurements for the Setosa class.

### Feature and Target Separation

The feature matrix and target vector are created with:

```python
x = iris.drop("species", axis=1)
y = iris["species"]
```

### Train/Test Split

The data is separated into training and testing subsets using:

```python
train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=101
)
```

This creates a training set and a test set, with 30% of the observations reserved for testing.

### Support Vector Classifier

The model is implemented with Scikit-learn's `SVC`:

```python
from sklearn.svm import SVC

model = SVC()
model.fit(x_train, y_train)
```

Predictions are then generated using:

```python
pred = model.predict(x_test)
```

### Model Evaluation

The notebook evaluates the predictions using:

- Confusion Matrix
- Classification Report

The required metrics are generated with:

```python
from sklearn.metrics import classification_report, confusion_matrix
```

This provides a direct evaluation of the classifier's performance on the unseen test data.

### Hyperparameter Tuning with GridSearchCV

The notebook then demonstrates hyperparameter optimization using `GridSearchCV`.

A parameter grid is defined for:

- `C`
- `gamma`

The values tested are:

```python
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001]
}
```

A grid-search model is created with:

```python
grid = GridSearchCV(
    SVC(),
    param_grid,
    verbose=2
)
```

It is then fitted on the training data:

```python
grid.fit(x_train, y_train)
```

Predictions from the tuned model are generated using:

```python
grid_predictions = grid.predict(x_test)
```

The tuned model is evaluated again with a confusion matrix and classification report.

### Key Learning Point

The notebook demonstrates that hyperparameter tuning can be used to systematically test combinations of SVM parameters. It also emphasizes that tuning should not be used simply to fit isolated noisy observations, because excessive model complexity can lead to overfitting.

## 2. Breast Cancer Classification with SVM

### Notebook

`SVMwithPython.ipynb`

This notebook demonstrates SVM classification on the Breast Cancer Wisconsin dataset included in Scikit-learn.

### Dataset Loading

The dataset is imported with:

```python
from sklearn.datasets import load_breast_cancer
```

It is then loaded with:

```python
cancer = load_breast_cancer()
```

The notebook inspects the dataset structure using:

```python
cancer.keys()
```

and displays the built-in dataset description using:

```python
print(cancer["DESCR"])
```

### Feature DataFrame

The numerical feature matrix is converted into a Pandas DataFrame:

```python
df_feat = pd.DataFrame(
    cancer["data"],
    columns=cancer["feature_names"]
)
```

The notebook inspects the first observations and the DataFrame structure with:

```python
df_feat.head(2)
df_feat.info()
```

The available target class names are also inspected using:

```python
cancer["target_names"]
```

### Feature and Target Definition

The features and labels are defined as:

```python
x = df_feat
y = cancer["target"]
```

### Train/Test Split

The notebook splits the data with:

```python
train_test_split(
    x,
    y,
    test_size=0.33,
    random_state=101
)
```

Here, 33% of the observations are reserved for testing.

### Support Vector Classifier

The SVM model is created and trained using:

```python
from sklearn.svm import SVC

model = SVC()
model.fit(x_train, y_train)
```

Predictions are generated with:

```python
predictions = model.predict(x_test)
```

### Model Evaluation

The notebook uses the same standard evaluation tools:

```python
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

This provides classification metrics and a confusion matrix for the Breast Cancer dataset.

## Algorithms and Concepts Covered

### Support Vector Machine

A Support Vector Machine is a supervised learning algorithm that identifies a decision boundary that separates classes in feature space.

The repository uses Scikit-learn's:

```python
SVC()
```

for classification.

### Support Vector Classifier (SVC)

`SVC` is the Scikit-learn implementation used in both notebooks for classification tasks.

### Hyperparameter Tuning

The Iris project demonstrates systematic parameter search using `GridSearchCV`.

The main parameters investigated are:

- `C`
- `gamma`

### GridSearchCV

`GridSearchCV` evaluates combinations of specified hyperparameters using cross-validation internally and selects the configuration according to the search procedure.

### Train/Test Split

Both notebooks separate model training data from testing data so predictions can be evaluated on observations not used during model fitting.

### Classification Report

The classification report provides standard classification metrics for the predicted labels.

### Confusion Matrix

The confusion matrix compares actual class labels with predicted class labels.

### Exploratory Data Analysis

The Iris project uses pair plots and KDE visualization to inspect relationships and distributions between features and classes before model training.

## Machine Learning Workflow

The notebooks follow a standard supervised learning workflow:

```text
Load Dataset
    ↓
Inspect Data
    ↓
Exploratory Data Analysis
    ↓
Separate Features and Target
    ↓
Train/Test Split
    ↓
Train SVM Classifier
    ↓
Generate Predictions
    ↓
Evaluate Model
    ↓
Tune Hyperparameters
    ↓
Re-Evaluate Tuned Model
```

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/mennnaaaaa/Support_vector_Machines_Applications.git
cd Support_vector_Machines_Applications
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks and run the cells sequentially.

## Learning Objectives

This repository was developed to build practical understanding of:

- Support Vector Machine classification
- SVC model implementation
- Exploratory data analysis
- Feature and target separation
- Train/test splitting
- Classification evaluation
- Confusion matrices
- Classification reports
- SVM hyperparameters
- GridSearchCV
- Basic model tuning and overfitting considerations

## Applications Represented

The examples in this repository demonstrate SVM classification in two datasets:

**Iris flower classification:** predicting one of three flower species using sepal and petal measurements.

**Breast cancer classification:** using the Scikit-learn Breast Cancer dataset for binary classification.

These implementations are educational applications of SVM rather than deployed production systems.

## Project Background

This repository is part of a personal machine learning learning collection.

The notebooks were developed while studying SVM through courses, tutorials, and different learning resources. They are maintained as practical implementations for understanding the algorithm, evaluation workflow, and hyperparameter tuning.

## Future Improvements

Possible extensions include:

- Experimenting with different SVM kernels.
- Comparing linear and nonlinear kernels.
- Performing explicit cross-validation analysis.
- Testing additional parameter ranges for `C` and `gamma`.
- Adding feature scaling pipelines where appropriate.
- Comparing SVM performance with other classification algorithms.
- Recording and comparing evaluation metrics across models.

## Author

**Menna Hany Abdelaziz**

GitHub: [mennnaaaaa](https://github.com/mennnaaaaa)
