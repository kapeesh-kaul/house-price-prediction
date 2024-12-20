import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd

from .load_rate import load_rate
from .load_cpi import merged_cpi_and_revisions
from .load_hpi import hpi_combined
import pickle
import os
from sklearn.preprocessing import StandardScaler

class HousePriceDataset(Dataset):
    def __init__(self, cityname, device='cpu'):
        self.device = device
        self.cityname = cityname
        self.target_variable = f"{cityname}_Single_Family_HPI"
        
        self.hpi = hpi_combined()
        self.cpi_and_revisions = merged_cpi_and_revisions()
        self.rate = load_rate()
        
        self.combined_data = self.hpi.merge(self.cpi_and_revisions, on='Date').merge(self.rate, on='Date')
        self.combined_data.fillna(0.0, inplace=True)  # Replace NaNs with 0

        # Remove columns that have high collinearity with the target variable
        correlation_matrix = self.combined_data.corr()
        high_correlation_columns = correlation_matrix.index[correlation_matrix[self.target_variable].abs() > 0.6].tolist()
        high_correlation_columns.remove(self.target_variable)  # Remove the target variable itself
        self.combined_data.drop(columns=high_correlation_columns, inplace=True)

        # Normalize the data to have zero mean and unit variance
        scaler = StandardScaler()
        self.combined_data = pd.DataFrame(scaler.fit_transform(self.combined_data), columns=self.combined_data.columns)

        
    def __len__(self):
        return len(self.combined_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.combined_data.iloc[idx]
        target = sample[self.target_variable]

        sample = sample.drop(self.target_variable)
        
        sample_tensor = torch.tensor(sample.values, dtype=torch.float32)  # Keep on CPU
        target_tensor = torch.tensor(target, dtype=torch.float32)  # Keep on CPU
        return sample_tensor, target_tensor

def generate_dataloader(cityname, device='cpu', batch_size=4, num_workers=2):
    dataset = HousePriceDataset(cityname, device=device)
    
    print(dataset.combined_data.isnull().sum()[dataset.combined_data.isnull().sum() > 0])

    print(f"Number of columns in the dataset: {dataset.combined_data.shape[1]}")

    # Split the dataset into train, validation, and test sets
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=2/3, random_state=42)  # 2/3 of 30% is 20%
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    
    return train_loader, val_loader, test_loader

def save_dataloaders(cityname, train_loader, val_loader, test_loader):
    os.makedirs('data/dataloader', exist_ok=True)
    
    train_loader_path = f"data/dataloader/{cityname}_train_loader.pkl"
    val_loader_path = f"data/dataloader/{cityname}_val_loader.pkl"
    test_loader_path = f"data/dataloader/{cityname}_test_loader.pkl"
    
    with open(train_loader_path, "wb") as f:
        pickle.dump(train_loader, f)

    with open(val_loader_path, "wb") as f:
        pickle.dump(val_loader, f)

    with open(test_loader_path, "wb") as f:
        pickle.dump(test_loader, f)
    
    return train_loader_path, val_loader_path, test_loader_path

def load_dataloaders(cityname):
    train_loader_path = f"data/dataloader/{cityname}_train_loader.pkl"
    val_loader_path = f"data/dataloader/{cityname}_val_loader.pkl"
    test_loader_path = f"data/dataloader/{cityname}_test_loader.pkl"
    
    with open(train_loader_path, "rb") as f:
        train_loader = pickle.load(f)
    
    with open(val_loader_path, "rb") as f:
        val_loader = pickle.load(f)
    
    with open(test_loader_path, "rb") as f:
        test_loader = pickle.load(f)
    
    return train_loader, val_loader, test_loader

    