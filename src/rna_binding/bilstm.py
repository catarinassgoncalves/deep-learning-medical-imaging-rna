import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import optuna

from utils import (
    load_rnacompete_data,
    masked_mse_loss,
    masked_spearman_correlation,
    plot,
    configure_seed,
    RNAConfig
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configure_seed(seed=42)

class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, device, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.device = device
    
    def __iter__(self):
        for batch in super().__iter__():
            yield self._move_to_device(batch)

    def _move_to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [x.to(self.device, non_blocking=True) for x in data]
        return data.to(self.device, non_blocking=True)

# Bi-LSTM Model
class RBFOX1_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=1, dropout_rate=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=4,       # (A, C, G, N)
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, x_mask):
        x, _ = self.lstm(x)
        
        mask_expanded = x_mask.unsqueeze(-1) 
        
        x = x.masked_fill(mask_expanded == 0, -1e9)
        
        x, _ = torch.max(x, dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Training
def train_val_model(model, model_name, train_loader, lr=1e-3, max_epochs=25):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []        
    val_losses = []
    start_time = time.time()

    for epoch in range(max_epochs):
        # TRAIN
        model.train()
        running_loss = 0
        for x, x_mask, y, mask in train_loader:
            optimizer.zero_grad()
            outputs = model(x, x_mask)
            loss = masked_mse_loss(outputs, y, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        val_batch_losses = []
        val_preds, val_targets, val_masks = [], [], []
    
        with torch.no_grad():
            for x, x_mask, y, mask in val_loader:
                preds = model(x, x_mask)
                loss = masked_mse_loss(preds, y, mask)
                val_batch_losses.append(loss.item())
                val_preds.append(preds)
                val_targets.append(y)
                val_masks.append(mask)
        
        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        # Spearman Metric
        full_preds = torch.cat(val_preds)
        full_targets = torch.cat(val_targets)
        full_masks = torch.cat(val_masks)
        spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Spearman: {spearman:.4f}")

    print(f"Training time: {time.time() - start_time:.2f}s")
    
    # Plot
    epoch_list = list(range(1, max_epochs + 1))
    plot_data = {'Train Loss': train_losses, 'Val Loss': val_losses}
    plot(epoch_list, plot_data, filename=f"{model_name}_loss_curve.png")
    
    return model

# Optuna objective 
def objective_lstm(trial):

    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Build Model
    model = RBFOX1_BiLSTM(
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Short Training Loop for Optimization
    max_epochs = 25 
    best_spearman = -1.0

    for epoch in range(max_epochs):
        model.train()
        for x, x_mask, y, mask in train_loader:
            optimizer.zero_grad()
            outputs = model(x, x_mask)
            loss = masked_mse_loss(outputs, y, mask)
            loss.backward()
            optimizer.step()

        # Validation Check
        model.eval()
        val_preds, val_targets, val_masks = [], [], []
        with torch.no_grad():
            for x, x_mask, y, mask in val_loader:
                preds = model(x, x_mask)
                val_preds.append(preds)
                val_targets.append(y)
                val_masks.append(mask)

        full_preds = torch.cat(val_preds)
        full_targets = torch.cat(val_targets)
        full_masks = torch.cat(val_masks)
        
        spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)
        
        # Track best score for user attributes
        if spearman > best_spearman:
            best_spearman = spearman
            trial.set_user_attr("best_epoch", epoch)

        # Report current score for pruning
        trial.report(spearman, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_spearman

if __name__ == "__main__":

    def get_rnacompete_loaders(PROTEIN, config, batch_size, device):
        train_dataset = load_rnacompete_data(PROTEIN, split='train', config=config)
        val_dataset   = load_rnacompete_data(PROTEIN, split='val', config=config)
        test_dataset  = load_rnacompete_data(PROTEIN, split='test', config=config)
        
        train_loader = DeviceDataLoader(train_dataset, device=device, batch_size=batch_size, shuffle=True)
        val_loader = DeviceDataLoader(val_dataset, device=device, batch_size=batch_size, shuffle=True)
        test_loader = DeviceDataLoader(test_dataset, device=device, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

    # Configuration
    config = RNAConfig()
    config.METADATA_PATH = "data/metadata.xlsx"
    config.DATA_PATH = "data/norm_data.txt"

    PROTEIN = 'RBFOX1'
    batch_size = 64
    
    # Load Data
    train_loader, val_loader, test_loader = get_rnacompete_loaders(PROTEIN, config, batch_size, device)

    ############################################################

    print("\n--- Starting Bi-LSTM Optimization ---")
    study_lstm = optuna.create_study(direction="maximize")
    study_lstm.optimize(objective_lstm, n_trials=30) # Set to 20-30 

    print("\nBest LSTM Params:", study_lstm.best_params)
    
    best_trial = study_lstm.best_trial
    results_lstm = {
        "best_params": best_trial.params,
        "best_epoch": best_trial.user_attrs["best_epoch"],
        "best_score": best_trial.value
    }
    
    # Save Results
    with open("OPTUNA_LSTM.json", "w") as f:
        json.dump(results_lstm, f, indent=4)

    ############################################################

    with open("OPTUNA_LSTM.json", "r") as f:
        results_LSTM = json.load(f)

    best_params = results_LSTM["best_params"]
    final_epochs = results_LSTM["best_epoch"] + 1
    
    print(f"\nRetraining Best LSTM for {final_epochs} epochs...")
    
    final_lstm = RBFOX1_BiLSTM(
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout_rate=best_params["dropout_rate"]
    ).to(device)
    
    final_lstm = train_val_model(
        final_lstm, 
        "Best_LSTM", 
        train_loader, 
        lr=best_params["lr"], 
        max_epochs=final_epochs
    )
    
    # Final Test Evaluation
    final_lstm.eval()
    test_preds, test_targets, test_masks = [], [], []
    
    with torch.no_grad():
        for x, x_mask, y, mask in test_loader:
             preds = final_lstm(x, x_mask)
             test_preds.append(preds)
             test_targets.append(y)
             test_masks.append(mask)

    full_preds = torch.cat(test_preds)
    full_targets = torch.cat(test_targets)
    full_masks = torch.cat(test_masks)
    
    test_spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)

    test_results = {"test_spearman": test_spearman.item()}

    with open("BEST_SPEARMAN_TEST_LSTM.json", "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"\nFINAL LSTM TEST SCORE (Spearman): {test_spearman:.4f}")