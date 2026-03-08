import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ridge_training(X_train, X_test, y_train, y_test):
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Ridge MAE:", mae)
    print("Ridge RMSE:", rmse)
    print("Ridge R2:", r2)

    param_grid = {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=ridge_model,
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring={
            "MSE": "neg_mean_squared_error",
            "R2": "r2"
        },
        refit="MSE",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    best_ridge = grid_search.best_estimator_
    y_pred_tuned = best_ridge.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_tuned)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2 = r2_score(y_test, y_pred_tuned)

    print("Tuned Ridge MAE:", mae)
    print("Tuned Ridge RMSE:", rmse)
    print("Tuned Ridge R2:", r2)

    cv_scores = cross_val_score(
        best_ridge,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = np.sqrt(-cv_scores)
    print("Cross-Validation RMSE Scores:", cv_rmse)
    print("Average CV RMSE:", cv_rmse.mean())

    return {
        "Model": "Random Forest Regressor",
        "RMSE": cv_rmse.mean(),
        "MAE": mae
    }


def lasso_training(X_train, X_test, y_train, y_test):
    lasso_model = Lasso(alpha=0.1, max_iter=20000)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Lasso MAE:", mae)
    print("Lasso RMSE:", rmse)
    print("Lasso R2:", r2)

    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10]
    }

    grid_search = GridSearchCV(
        estimator=Lasso(alpha=0.1, max_iter=20000),
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    best_lasso = grid_search.best_estimator_
    y_pred_tuned = best_lasso.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_tuned)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2 = r2_score(y_test, y_pred_tuned)

    print("Tuned Lasso MAE:", mae)
    print("Tuned Lasso RMSE:", rmse)
    print("Tuned Lasso R2:", r2)

    cv_scores = cross_val_score(
        best_lasso,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = np.sqrt(-cv_scores)
    print("Cross-Validation RMSE Scores:", cv_rmse)
    print("Average CV RMSE:", cv_rmse.mean())

    return {
        "Model": "Random Forest Regressor",
        "RMSE": cv_rmse.mean(),
        "MAE": mae
    }
