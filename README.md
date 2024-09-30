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
* Linear Regression and Logistic Regression model


## Usage

1. **Install the package:**
   * `pip install -e .` from root directory (recommended with conda or virtual environment of choice).
   * `pip install -e .[dev]` if you want full dev features.

2. **Run the code:**
   * Execute the pipeline after installing: `python ./maxML/pipeline.py <path_to_yaml_config>`. It will:
     * Preprocess the data
     * Split the data into training and testing sets
     * Train the linear and logistic regression models
     * Evaluate the models on the test set
     * Print the evaluation metrics
   * Alternatively, the maxML module can be used within a script or notebook:
     ```
     import maxML


     maxML.pipeline.run("/path/to/config.yaml")
     ```


## Evaluation Metrics
* Linear Regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared
* Logistic Regression: Accuracy, Precision, Recall, F1-score, ROC AUC


## Next Steps
maxML Framework:
* Refactor classes and protocols (reduce code where feasible, currently feels anti-patterned)
* Add more Preprocessors, Evaluators, Models, etc.
* Add Model parameterization.

Pipeline:
* Add write function or integrations
* Update printing

Software Engineering:
* Add Evaluator unit tests.
* Add release strategy.
* Add model artifact support e.g. MLFlow.
* Add containerization.
* Update versioning to use git tags.

Data Engineering and Modeling:
* Explore the data further to gain insights into the relationships between features and the target variable.
* Consider feature engineering to create new features or transform existing ones.
* Experiment with hyperparameter tuning to optimize model performance.
* Try other machine learning algorithms that might be better suited for this problem.


## Important Notes
* This is currently a proof of concept, although I hope to develop it into something more broadly usable.
* The gemini_sample_data was generated using Gemini.
