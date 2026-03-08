import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from pipelines import create_nn_pipeline, create_tree_pipeline
from nn_models import neural_network_training
from linear_models import ridge_training, lasso_training
from tree_models import random_forest_training


def data_loading(path, location):
    df = pd.read_csv(path, header=None)
    rename_cols = lambda col: (
        f'x{c + 1}' if (c := int(col)) < 16 else
        f'y{c % 16 + 1}' if c < 32 else
        f'p{c % 16 + 1}' if c < 48 else
        f'total_power'
    )
    df.rename(columns=rename_cols, inplace=True)
    df['location'] = location

    return df


def data_split(df):
    X = df.drop(df.columns[32:49], axis=1)
    y = df.iloc[:, 32:49]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test




def main():
    df = pd.concat([
        data_loading('../data/Adelaide_Data.csv', 'Adelaide'),
        data_loading('../data/Perth_Data.csv', 'Perth'),
        data_loading('../data/Sydney_Data.csv', 'Sydney'),
        data_loading('../data/Tasmania_Data.csv', 'Tasmania')
    ])

    # Raw data (best for Linear Regression, Ridge, Lasso)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(df)

    # Create pipelines
    nn_pipeline = create_nn_pipeline(numeric_cols=list(range(0, 32)))
    tree_pipeline = create_tree_pipeline(x_cols=range(16), y_cols=range(16, 32))

    # Data with one-hot encoding, scaling, and pairwise-distances (best for Neural Networks)
    nn_X_train = pd.DataFrame(nn_pipeline.fit_transform(X_train), columns=nn_pipeline.get_feature_names_out())
    nn_X_val = pd.DataFrame(nn_pipeline.transform(X_val), columns=nn_pipeline.get_feature_names_out())
    nn_X_test = pd.DataFrame(nn_pipeline.transform(X_test), columns=nn_pipeline.get_feature_names_out())

    # Data with only ordinal encoding (best for Decision Trees, Random Forest, Gradient Boosting models)
    tree_X_train = pd.DataFrame(tree_pipeline.fit_transform(X_train), columns=tree_pipeline.get_feature_names_out())
    tree_X_val = pd.DataFrame(tree_pipeline.transform(X_val), columns=tree_pipeline.get_feature_names_out())
    tree_X_test = pd.DataFrame(tree_pipeline.transform(X_test), columns=tree_pipeline.get_feature_names_out())

    print("Raw features shape:", X_train.shape)
    print("NN features shape:", nn_X_train.shape)
    print("Tree features shape:", tree_X_train.shape)

    ridge_results = ridge_training(nn_X_train, nn_X_test, y_train, y_test)
    lasso_results = lasso_training(nn_X_train, nn_X_test, y_train, y_test)
    random_forest_results = random_forest_training(tree_X_train, tree_X_test, y_train, y_test)
    neural_network_results = neural_network_training(nn_X_train, nn_X_val, nn_X_test, y_train, y_val, y_test)

    results_df = pd.DataFrame([ridge_results, lasso_results, random_forest_results, neural_network_results])

    print("\nFinal Comparison")
    print(results_df.sort_values("RMSE"))


if __name__ == "__main__":
    main()
