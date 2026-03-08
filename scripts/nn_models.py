import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from utils import train_model, predict_scaled, get_total_power, rmse_mae


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 128, 64), dropout=0.2,
                 batchnorm=True, activation="relu"):
        super().__init__()

        layers = []
        prev_dim = input_dim
        activation_layer = self.get_activation(activation) or None

        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))

            if batchnorm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(activation_layer)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def get_activation(name):
        name = name.lower()

        if name == "relu":
            return nn.ReLU()
        elif name == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif name == "gelu":
            return nn.GELU()


def prepare_dataloaders(X, y, power_cols, batch_size=2048):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def scale_targets(y_train, y_val, y_test, power_cols):
    y_mean = y_train[power_cols].mean(axis=0)
    y_std = y_train[power_cols].std(axis=0) + 1e-8

    y_train_scaled = (y_train[power_cols] - y_mean) / y_std
    y_val_scaled = (y_val[power_cols] - y_mean) / y_std
    y_test_scaled = (y_test[power_cols] - y_mean) / y_std

    return y_train_scaled, y_val_scaled, y_test_scaled


def train_and_evaluate(model, train_loader, val_loader, test_loader, settings):
    trained_model, history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        **settings
    )

    y_true_scaled, y_pred_scaled = predict_scaled(trained_model, test_loader)

    y_true_total = get_total_power(y_true_scaled)
    y_pred_total = get_total_power(y_pred_scaled)

    rmse, mae = rmse_mae(y_true_total, y_pred_total)

    return rmse, mae, best_epoch, history


def test_activations(input_dim, output_dim, train_loader, val_loader, test_loader):
    activations = ["relu", "leakyrelu", "gelu"]

    settings = {
        "lr": 0.0008,
        "weight_decay": 0.0001,
        "max_epochs": 60,
        "patience": 10
    }

    results = []

    for act in activations:
        print("Testing activation:", act)

        model = MLP(
            input_dim,
            output_dim,
            hidden=(256, 128, 64),
            dropout=0.2,
            batchnorm=True,
            activation=act
        )

        rmse, mae, best_epoch, _ = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            test_loader,
            settings
        )

        results.append({
            "Activation": act,
            "BestEpoch": best_epoch,
            "RMSE": rmse,
            "MAE": mae
        })

    return pd.DataFrame(results).sort_values("RMSE")


def cross_validation(X, y, input_dim, output_dim):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    settings = {
        "lr": 0.0005,
        "weight_decay": 0.0003,
        "max_epochs": 40,
        "patience": 12
    }

    hidden_layers = (512, 256, 128, 64)

    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")

        train_loader = prepare_dataloaders(X[train_idx], y[train_idx])
        val_loader = prepare_dataloaders(X[val_idx], y[val_idx])

        model = MLP(
            input_dim,
            output_dim,
            hidden=hidden_layers,
            dropout=0.25,
            batchnorm=True
        )

        trained_model, _, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            **settings
        )

        y_true_scaled, y_pred_scaled = predict_scaled(trained_model, val_loader)

        y_true_total = get_total_power(y_true_scaled)
        y_pred_total = get_total_power(y_pred_scaled)

        rmse, mae = rmse_mae(y_true_total, y_pred_total)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(
            f"Fold {fold + 1} | BestEpoch={best_epoch} "
            f"| RMSE={rmse:.4f} | MAE={mae:.4f}"
        )

    print("\nAverage RMSE:", np.mean(rmse_scores))
    print("Average MAE:", np.mean(mae_scores))


def neural_network_training(X_train, X_val, X_test, y_train, y_val, y_test):
    power_cols = [f"p{i}" for i in range(1, 17)]

    y_train_s, y_val_s, y_test_s = scale_targets(y_train, y_val, y_test, power_cols)

    train_loader = prepare_dataloaders(X_train, y_train_s)
    val_loader = prepare_dataloaders(X_val, y_val_s)
    test_loader = prepare_dataloaders(X_test, y_test_s)

    input_dim = X_train.shape[1]
    output_dim = len(power_cols)

    baseline_model = MLP(input_dim, output_dim, hidden=(128, 64), dropout=0.0, batchnorm=False)
    regularized_model = MLP(input_dim, output_dim, hidden=(256, 128, 64), dropout=0.2, batchnorm=True)
    tuned_model = MLP(input_dim, output_dim, hidden=(512, 256, 128, 64), dropout=0.25, batchnorm=True)

    models = {
        "MLP_Baseline": (
            baseline_model,
            {"lr": 0.001, "weight_decay": 0.0, "max_epochs": 40, "patience": 7}
        ),
        "MLP_Regularized": (
            regularized_model,
            {"lr": 0.0008, "weight_decay": 0.0001, "max_epochs": 60, "patience": 10}
        ),
        "MLP_Tuned": (
            tuned_model,
            {"lr": 0.0005, "weight_decay": 0.0003, "max_epochs": 40, "patience": 12}
        ),
    }

    results = []

    for name, (model, settings) in models.items():
        print("Training:", name)

        rmse, mae, best_epoch, _ = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            test_loader,
            settings
        )

        print(f"{name} | BestEpoch={best_epoch} | RMSE={rmse:.2f} | MAE={mae:.2f}")

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae
        })

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df)

    # Activation experiment
    act_df = test_activations(
        input_dim,
        output_dim,
        train_loader,
        val_loader,
        test_loader
    )

    print(act_df)

    # Cross validation
    X_full = np.concatenate([X_train, X_val, X_test], axis=0)
    y_full = np.concatenate([y_train_s, y_val_s, y_test_s], axis=0)

    cross_validation(X_full, y_full, input_dim, output_dim)

    return results_df
