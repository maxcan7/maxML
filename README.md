# maxML Machine Learning Framework
In its current state this is simply an sklearn demo_pipeline. Over time it will evolve into a robust ML Framework that can take yaml files and configure them into scikit-learn Pipelines, automatic the code through features such as a factory and plugin design and hierarchical YAML processing.

## Overview: demo_pipeline.py
This project explores the use of linear and logistic regression models to predict whether a customer will make a purchase based on various features.


## Dataset
The dataset (`/data/gemini_sample_data.csv`) was generated using Gemini and contains 1000 rows with the following columns:

* `Age` (numerical)
* `Gender` (categorical)
* `Education` (categorical, ordinal)
* `City` (categorical)
* `Income` (numerical)
* `Years_of_Experience` (numerical)
* `Purchased` (binary target: 0 or 1)

The dataset includes missing values and potential outliers, simulating real-world data challenges.


## Pipeline

The project utilizes scikit-learn pipelines to streamline the preprocessing and modeling steps. The pipeline includes:

* Imputation of missing values
* Encoding of categorical features
* Scaling of numerical features
* Linear Regression and Logistic Regression model


## Usage

1. **Install the package:**
   * `pip install -e .` from root directory (recommended with conda or virtual environment of choice).
   * `pip install -e .[dev]` if you want full dev features.

2. **Run the code:**
   * Execute the demo_pipeline after installing: `python ./pipelines/demo_pipeline.py`. It will:
     * Preprocess the data
     * Split the data into training and testing sets
     * Train the linear and logistic regression models
     * Evaluate the models on the test set
     * Print the evaluation metrics


## Evaluation Metrics
* Linear Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared
* Logistic Regression: Accuracy, Precision, Recall, F1-score, ROC AUC


## Next Steps
Demo Pipeline:
* Add configurable data handling e.g. config.yml for data paths, params, etc.
* Add pipeline abstraction e.g. a maxMLModel abc or protocol framework.
* Update demo_pipeline to leverage this flexibility.

Data Engineering and Modeling:
* Explore the data further to gain insights into the relationships between features and the target variable.
* Consider feature engineering to create new features or transform existing ones.
* Experiment with hyperparameter tuning to optimize model performance.
* Try other machine learning algorithms that might be better suited for this problem.

Model Framework:
* Add factory and plugin pattern.
* Add hierarchical YAML management.

Software Engineering:
* Add unit tests on these new features.
* Add CICD and define release strategy.
* Add model artifact support e.g. MLFlow.
* Add containerization.


## Important Notes
* This is currently a proof of concept, although I hope to develop it into something more broadly usable.
* The demo_pipeline in its most basic form, as well as the gemini_sample_data, were generated using Gemini.
