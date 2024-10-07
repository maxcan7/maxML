# maxML Machine Learning Framework
The maxML module allows Data Scientists to horizontally implement scikit-learn Pipelines through YAML configurations. The configs are parsed and validated, and the Pipelines are composed from Protocols, such that only the core code implementation requires testing, and everything else can be parameterized via YAML.


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

The project utilizes scikit-learn Pipelines to streamline the preprocessing and modeling steps. The pipeline includes:

* Imputation of missing values
* Encoding of categorical features
* Scaling of numerical features
* Linear Regression or Logistic Regression model


## Usage

1. **Install the package:**
   * `pip install -e .` from root directory (recommended with conda or virtual environment of choice).
   * `pip install -e .[dev]` if you want full dev features.

2. **Run the code:**
   * Execute the pipeline after installing: `python ./maxML/pipeline.py <path_to_yaml_config>`. It will:
     * Preprocess the data
     * Split the data into training and testing sets
     * Train the linear or logistic regression models
     * Evaluate the models on the test set
     * Print the evaluation metrics
   * Alternatively, the maxML module can be used within a script or notebook:
     ```
     import maxML


     maxML.pipeline.run("/path/to/config.yaml")
     ```


## Example Config

Several test configs exist in the `tests/test_configs/` subdirectory.

Using the `columntransformer_logistic.yaml` config as an example:
```
input_path: data/gemini_sample_data.csv

preprocessors:

  - type: ColumnTransformerPreprocessor

    pipelines:

    - name: numeric
      steps:
      - name: imputer
        sklearn_module: sklearn.impute.SimpleImputer
        args:
          strategy: median
      - name: scaler
        sklearn_module: sklearn.preprocessing.StandardScaler
      columns:
        - Age
        - Income
        - Years_of_Experience

    - name : nominal
      steps:
      - name: imputer
        sklearn_module: sklearn.impute.SimpleImputer
        args:
          strategy: most_frequent
      - name: onehot
        sklearn_module: sklearn.preprocessing.OneHotEncoder
        args:
          handle_unknown: ignore
      columns:
        - Gender
        - City

    - name : ordinal
      steps:
      - name: imputer
        sklearn_module: sklearn.impute.SimpleImputer
        args:
          strategy: most_frequent
      - name: ordinal
        sklearn_module: sklearn.preprocessing.OrdinalEncoder
        args:
          categories:
            -
              - High School
              - Bachelor's
              - Master's
              - PhD
          handle_unknown: use_encoded_value
          unknown_value: !!python/name:numpy.nan
      columns:
        - Education

train_test_split:
  test_size: 0.2
  random_state: 42

model:
  module: sklearn.linear_model.LogisticRegression
  target: Purchased

evaluators:
  - type: LogisticEvaluator
    metrics:
      - accuracy_score
      - classification_report
      - roc_auc_score
```

This config uses sklearn's Logistic Regression module as its model, the gemini sample dataset from the `data` subdirectory, and the column `Purchased` as the target.

The preprocessor is a ColumnTransformerPreprocessor consisting of three Pipelines, a numeric, categorical, and ordinal preprocessor, each consisting of two steps such as some kind of imputation, encoding, or scaling. Each Pipeline operates over the columns assigned by the columns key, and additional arguments can be passed through the args key. These fields correspond to the interfaces of the corresponding python modules.

train_test_split here sets a test_size and a random_state (since it's used for testing), but any other args for the train_test_split sklearn module can also be added as fields in this config.

The evaluator is for evaluation metrics associated with logistic regression.


## Evaluation Metrics
* Linear Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared
* Logistic Regression: Accuracy, Precision, Recall, F1-score, ROC AUC


## Next Steps
maxML Framework:
* Refactor classes and protocols (reduce code where feasible, currently feels anti-patterned)
* Add more Preprocessors, Evaluators, Models, etc.

Pipeline:
* Add write function or integrations
* Update printing

Software Engineering:
* Add release strategy.
* Add model artifact support e.g. MLFlow.
* Add containerization.
* Update versioning to use git tags.
* Add CD.

Data Engineering and Modeling:
* Explore the data further to gain insights into the relationships between features and the target variable.
* Consider feature engineering to create new features or transform existing ones.
* Experiment with hyperparameter tuning to optimize model performance.
* Try other machine learning algorithms that might be better suited for this problem.


## Important Notes
* This is currently a proof of concept, although I hope to develop it into something more broadly usable.
* The gemini_sample_data was generated using Gemini.
