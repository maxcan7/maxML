import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def logistic_preprocessor_fixture() -> ColumnTransformer:
    """Hard-coded ColumnTransformer fixture."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_columns = ["Age", "Income", "Years_of_Experience"]
    numeric = ("numeric", numeric_pipeline, numeric_columns)

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    nominal_columns = ["Gender", "City"]
    nominal = ("nominal", nominal_pipeline, nominal_columns)

    ed_categories = ["High School", "Bachelor's", "Master's", "PhD"]
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ed_categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )
    ordinal = ("ordinal", ordinal_pipeline, ["Education"])
    return ColumnTransformer(transformers=[numeric, nominal, ordinal])
