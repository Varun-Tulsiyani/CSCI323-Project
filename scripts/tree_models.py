import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


def random_forest_training(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    # Hyperparameter Optimization
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, None],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Evaluate tuned model
    best_rf = grid_search.best_estimator_

    y_pred_tuned = best_rf.predict(X_test)

    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

    print("Tuned MAE:", mae_tuned)
    print("Tuned RMSE:", rmse_tuned)

    # Cross validation
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(
        best_rf,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    cv_rmse = np.sqrt(-cv_scores)

    print("Cross-validation RMSE scores:", cv_rmse)
    print("Mean CV RMSE:", cv_rmse.mean())
    
    return {
        "Model": "Random Forest Regressor",
        "RMSE": cv_rmse.mean(),
        "MAE": mae_tuned
    }
