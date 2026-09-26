import torch
import torch.nn as nn
import torch.nn.functional as F
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
            return [x.to(self.device, non_blocking = True) for x in data]
        return data.to(self.device, non_blocking = True)
    
def get_rnacompete_loaders(PROTEIN, config, batch_size, device):

    train_dataset = load_rnacompete_data(PROTEIN, split='train', config=config)
    val_dataset   = load_rnacompete_data(PROTEIN, split='val', config=config)
    test_dataset  = load_rnacompete_data(PROTEIN, split='test', config=config)

    train_loader = DeviceDataLoader(train_dataset, device=device, 
        batch_size=batch_size, shuffle=True)
    
    val_loader = DeviceDataLoader(val_dataset, device=device, 
        batch_size=batch_size, shuffle=True)
    
    test_loader = DeviceDataLoader(test_dataset, device=device, 
        batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

class MaskedGlobalMaxPool1d(nn.Module):
    def __init__(self, kernel_size, k = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.k = k
    #n-k+1

    def forward(self, x, x_mask):
        """
        x: (Batch, Channels, Length)
        x_mask: (Batch, Length) - 1 for valid, 0 for padding
        """
        mask_float = x_mask.unsqueeze(1).float()
        inverted_mask = 1 - mask_float 

        filtered_mask = F.max_pool1d(
            inverted_mask,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = 0
        )
        filtered_mask = 1.0 - filtered_mask

        filtered_mask = filtered_mask.bool().expand_as(x)
        x_strictly_masked = x.masked_fill(~filtered_mask, -1)
        top_k_values, _ = torch.topk(x_strictly_masked, k=self.k, dim=2)

        return top_k_values.flatten(1)

class RBFOX1_CNN(nn.Module):
    def __init__(self, num_filters = 64, kernel_size = 9, k = 1, hidden_dim = 64, dropout_rate = 0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=4, 
                               out_channels=num_filters, 
                               kernel_size=kernel_size, 
                               padding=0)

        self.relu = nn.ReLU()
        self.max_pool = MaskedGlobalMaxPool1d(kernel_size=kernel_size, k=k)

        self.dropout = nn.Dropout(dropout_rate)
        FFN_input_size = num_filters * k

        self.fc1 = nn.Linear(FFN_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, x_mask):

        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x, x_mask)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, model_name="Model", opt_name="Adam"):
    model = model.to(device, non_blocking=True)
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        
        for x, x_mask, y, valid_mask in train_loader:
            
            optimizer.zero_grad()
            preds = model(x, x_mask)

            loss = masked_mse_loss(preds, y, valid_mask)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for x, x_mask, y, valid_mask in val_loader:
                
                preds = model(x, x_mask)
                loss = masked_mse_loss(preds, y, valid_mask)
                val_batch_losses.append(loss.item())
                
                all_preds.append(preds)
                all_targets.append(y)
                all_masks.append(valid_mask)

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        # Determine Spearman Rank Correlation
        full_preds = torch.cat(all_preds)
        full_targets = torch.cat(all_targets)
        full_masks = torch.cat(all_masks)
        spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Spearman: {spearman:.4f}")

    print(f"Training time: {time.time() - start_time:.2f}s")
    
    # Plots
    epoch_list = list(range(1, epochs + 1))
    plot_data = {'Train Loss': train_losses, 'Val Loss': val_losses}
    plot(epoch_list, plot_data, filename=f"{model_name}_loss_curve.png")
    
    return model

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


def objective(trial):

    kernel_size = trial.suggest_int("kernel_size", 5, 9)

    num_filters = trial.suggest_int("num_filters", 32, 128)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    k = trial.suggest_int("k", 1,3)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    model = RBFOX1_CNN(num_filters, kernel_size, k, hidden_dim,dropout_rate).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_epochs = 25
    best_spearman = -1

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0

        # TRAIN
        for x, x_mask, y, mask in train_loader:
            x, x_mask, y, mask = x.to(device), x_mask.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, x_mask)

            loss = masked_mse_loss(outputs, y, mask)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_preds = []
        val_targets = []
        val_masks = []
        val_batch_losses = []
        
        with torch.no_grad():
            for x, x_mask, y, mask in val_loader:
                x, x_mask, y, mask = x.to(device), x_mask.to(device), y.to(device), mask.to(device)
                
                preds = model(x, x_mask)
                val_preds.append(preds)
                val_targets.append(y)
                val_masks.append(mask)
                loss = masked_mse_loss(preds, y, mask)
                val_batch_losses.append(loss.item())

        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        full_preds = torch.cat(val_preds)
        full_targets = torch.cat(val_targets)
        full_masks = torch.cat(val_masks)

        spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)

        if spearman > best_spearman:
            best_spearman = spearman
            trial.set_user_attr("best_epoch", epoch)

        trial.report(best_spearman, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_spearman

if __name__ == "__main__":
    
    config = RNAConfig()
    config.METADATA_PATH = "data/metadata.xlsx"
    config.DATA_PATH = "data/norm_data.txt"

    PROTEIN = 'RBFOX1'
    batch_size = 64
    
    train_loader, val_loader, test_loader = get_rnacompete_loaders(PROTEIN, config, batch_size, device)

#############################################

    study_CNN = optuna.create_study(direction="maximize")

    print("Starting optimization...")
    study_CNN.optimize(objective, n_trials=30)

    print("\nBest Params found:")
    print(study_CNN.best_params)

    best_CNN = study_CNN.best_trial

    results_CNN = {
        "best_params": best_CNN.params,
        "best_epoch": best_CNN.user_attrs["best_epoch"],
        "best_score": best_CNN.value
    }

    params_CNN = results_CNN["best_params"]
    best_CNN = results_CNN["best_epoch"]

    with open("OPTUNA_CNN.json", "w") as f:
        json.dump(results_CNN, f, indent=4)

############################################

    with open("OPTUNA_CNN.json", "r") as f:
        results_CNN = json.load(f)

    best_params = results_CNN["best_params"]
    final_epochs = results_CNN["best_epoch"] + 1

    final_CNN = RBFOX1_CNN(
    num_filters=best_params["num_filters"],
    kernel_size=best_params["kernel_size"],
    k=best_params["k"],
    hidden_dim=best_params["hidden_dim"],
    dropout_rate=best_params["dropout_rate"]
    ).to(device)

    train_val_model(
        final_CNN, "best_CNN", 
        train_loader, 
        best_params["lr"], 
        final_epochs
    )
    # Final Test Evaluation
    final_CNN.eval()
    test_preds, test_targets, test_masks = [], [], []
    
    with torch.no_grad():
        for x, x_mask, y, mask in test_loader:
             preds = final_CNN(x, x_mask)
             test_preds.append(preds)
             test_targets.append(y)
             test_masks.append(mask)

    full_preds = torch.cat(test_preds)
    full_targets = torch.cat(test_targets)
    full_masks = torch.cat(test_masks)

    test_spearman = masked_spearman_correlation(full_preds, full_targets, full_masks)

    test_results = {"test_spearman": test_spearman.item()}

    with open("BEST_SPEARMAN_TEST_CNN.json", "w") as f:
        json.dump(test_results, f, indent=4)