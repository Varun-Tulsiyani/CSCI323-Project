from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from transformers import DistanceTransformer


def create_nn_pipeline(numeric_cols, location_col='location'):
    numeric_pipeline = Pipeline([
        ("dist", DistanceTransformer(x_columns=numeric_cols[:16], y_columns=numeric_cols[16:32])),
        ("num", StandardScaler())
    ])

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [location_col]),
        ], remainder="passthrough"))
    ])
    return pipeline


def create_tree_pipeline(x_cols=range(16), y_cols=range(16, 32), location_col='location'):
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [location_col]),
            ("dist", DistanceTransformer(x_columns=x_cols, y_columns=y_cols), list(x_cols) + list(y_cols))
        ], remainder='passthrough'))
    ])
    return pipeline
