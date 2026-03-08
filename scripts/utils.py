import numpy as np
import torch
import torch.nn as nn

def inverse_scale(y_scaled, std, mean):
    return y_scaled * std + mean


def rmse_mae(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def predict_scaled(model, loader, device="cpu"):
    model.eval()
    true_vals = []
    pred_vals = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            true_vals.append(y_batch.numpy())
            pred_vals.append(preds)

    return np.vstack(true_vals), np.vstack(pred_vals)


def get_total_power(y_scaled):
    power_values = inverse_scale(y_scaled)
    return power_values.sum(axis=1)


def train_model(model, train_loader, val_loader, lr=0.001, weight_decay=0.0,
                max_epochs=60, patience=10, device="cpu"):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_total += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)

        train_loss = train_total / train_count

        model.eval()
        val_total = 0.0
        val_count = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)

                val_total += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)

        val_loss = val_total / val_count

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        print(f"Epoch {epoch:02d} , train={train_loss:.5f} , val={val_loss:.5f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, best_epoch
