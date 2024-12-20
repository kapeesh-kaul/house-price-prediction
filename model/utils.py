import torch
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.metrics import r2_score

def train(model, train_loader, val_loader, lr=0.0001, num_epochs=50, device='cpu'):
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Move to device inside the loop
            inputs, targets = inputs.to(device), targets.to(device)

            # Check for NaNs in the inputs or targets
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                print("Found NaNs in the data!")
                continue

            # Forward pass
            predictions, reconstructed = model(inputs)

            # Prediction loss
            predictions = predictions.squeeze(-1)
            prediction_loss = criterion(predictions, targets)

            # Reconstruction loss
            inputs = inputs.squeeze(1) if len(inputs.size()) == 3 else inputs
            reconstruction_loss = criterion(reconstructed, inputs)

            loss = prediction_loss + 0.1 * reconstruction_loss  # Weighted loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Check for NaN loss
            if torch.isnan(loss):
                print("NaN detected in loss! Stopping training.")
                return

        r2, val_loss = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R2 Score: {r2:.4f}")


def evaluate(model, data_loader, device):
    criterion = MSELoss()
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Ensure targets match predictions
            targets = targets.unsqueeze(-1)  # Make targets (batch_size, 1)

            predictions, _ = model(inputs)
            predictions = predictions.squeeze(-1)  # Make predictions (batch_size)
            loss = criterion(predictions, targets.squeeze(-1))
            val_loss += loss.item()

            all_targets.extend(targets.squeeze(-1).cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    r2 = r2_score(all_targets, all_predictions)
    return r2, (val_loss / len(data_loader))

