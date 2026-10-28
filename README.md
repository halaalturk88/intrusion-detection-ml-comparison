# Network Intrusion Detection Using Machine Learning

### Comparative Analysis of NSL-KDD and UNSW-NB15 Datasets

**Project Type:** Individual Academic Project  
**Domain:** Cybersecurity | Machine Learning | Data Analysis  
**Language:** Python

## Project Overview

This project explores the application of Machine Learning to network intrusion detection by comparing two benchmark cybersecurity datasets: **NSL-KDD** and **UNSW-NB15**.

The objective is to investigate how different classification algorithms and feature selection techniques can be used to distinguish normal network traffic from malicious activity.

The project focuses on data preprocessing, feature selection, model training, hyperparameter optimization, and performance evaluation.

## Technologies & Tools

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Classification Models:** Support Vector Machine (SVM), Random Forest, XGBoost
- **Feature Selection:** ANOVA F-test, Mutual Information, RFE, RFECV
- **Model Optimization:** GridSearchCV
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Development Environment:** Jupyter Notebook, Python

## Datasets

The project uses two benchmark datasets:

**1. NSL-KDD**

An improved version of the KDD99 dataset with reduced redundant records, widely used in network intrusion detection research.

**2. UNSW-NB15**

A network intrusion detection dataset containing normal traffic and multiple attack categories.

Both datasets are processed for binary classification:

- `0` = Normal Traffic
- `1` = Attack Traffic

## Project Methodology

### 1. Data Preprocessing

- Handling missing values.
- Encoding categorical features using One-Hot Encoding.
- Standardizing numerical features.
- Preparing training and testing datasets.

### 2. Feature Selection

Four techniques are included to investigate the impact of feature selection on classification performance:

- ANOVA F-test
- Mutual Information
- Recursive Feature Elimination (RFE)
- Recursive Feature Elimination with Cross-Validation (RFECV)

A full-feature baseline is also included for comparison.

### 3. Machine Learning Models

The following classification algorithms are implemented:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Hyperparameter tuning is performed using GridSearchCV, with F1-Score as the optimization metric.

### 4. Model Evaluation

The implementation evaluates models using:

- **Accuracy:** Overall classification correctness.
- **Precision:** Reliability of predicted intrusion alerts.
- **Recall:** Ability to identify actual attacks.
- **F1-Score:** Balance between Precision and Recall.

F1-Score and Recall are particularly relevant because missed attacks can significantly affect intrusion detection effectiveness.

## Repository Structure

```text
intrusion-detection-ml-comparison/
|
|-- IDS_Assignment2_Implementation.ipynb
|-- ids_assignment2_implementation.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## Installation & Execution

**1. Install the required libraries**

```bash
pip install -r requirements.txt
```

**2. Prepare the datasets**

Create a folder named `data` and place the following files inside it:

NSL-KDD:
- `KDDTrain+.txt`
- `KDDTest+.txt`

UNSW-NB15:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

**3. Run the implementation**

```bash
python ids_assignment2_implementation.py --data-dir data --output-dir outputs --sample-size 15000
```

The implementation is designed to generate evaluation metrics, classification reports, selected feature lists, and optimized model parameters.

## Results & Discussion

The project provides a framework for comparing intrusion detection models across two datasets and multiple feature selection strategies.

The accompanying academic report discusses the expected impact of dataset complexity, feature redundancy, model selection, and hyperparameter tuning.

**Current status:** Verified numerical results are not included in this repository. Experimental performance comparisons will be documented after successfully running and validating the models.

## Project Limitations

- Model performance has not yet been independently verified in the repository.
- Some feature selection and model tuning combinations may require significant computational resources.
- Results depend on the selected datasets, preprocessing methods, and experimental configurations.

## Project Context

This project was independently developed as part of a university coursework assignment.

It provided practical exposure to applying machine learning techniques to cybersecurity problems, designing classification pipelines, and exploring the relationship between feature selection and model evaluation.
