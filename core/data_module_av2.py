import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional
import torch
import numpy as np
import pickle


def load_trajectories(path) -> np.ndarray:
    """
    To load the saved pickle file and ensure the return type is an np.ndarray.
    Args:
        path: path to the pickle file
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Ensure the returned data is a np.ndarray
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)

    return data

def load_traj_as_npy(file_path) -> np.ndarray:
    """
    If you directly have a np array then prefer this to pickle. Since pickle might have dependency issues if the
     `files.pkl` are created with np > 2.1.1.
    Args:
        file_path: Path to trajectories on the disk
    """

    with open(file_path, 'rb') as file:
        data = np.load(file)

    return data


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path: str = "./",
                 val_data_path: str = "./",
                 batch_size: int = 32,
                 num_workers: int = 12,
                 use_first_half_mask: bool = False):
        super().__init__()
        
        self.transform = None
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_trajectories = None
        self.val_trajectories = None
        self.mean = None
        self.std = None
        self.use_first_half_mask = use_first_half_mask
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        # Load trajectories here
        train_trajectories = load_trajectories(self.train_data_path)
        val_trajectories = load_trajectories(self.val_data_path)
        
        rotation_matrix_t_set = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        rotated_train_trajectories = train_trajectories @ rotation_matrix_t_set.T
        rotated_val_trajectories = val_trajectories @ rotation_matrix_t_set.T
        
        self.train_trajectories = rotated_train_trajectories
        self.val_trajectories = rotated_val_trajectories

    def setup(self, stage: Optional[str] = None):

        if (self.train_trajectories is None) or (self.val_trajectories is None):
            raise ValueError("Trajectories not loaded. Make sure to load them in prepare_data().")

        # Calculate mean and std for normalization
        self.mean = np.mean(self.train_trajectories, axis=(0, 1))
        self.std = np.std(self.train_trajectories, axis=(0, 1))

        # Apply transformation
        train_dataset = TrajectoryDataset(self.train_trajectories, transform=self.normalize_trajectory)
        val_dataset = TrajectoryDataset(self.val_trajectories, transform=self.normalize_trajectory)
        
        # Split dataset
        if stage == 'fit' or stage is None:
            print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
            train_size = int(len(train_dataset) * 0.99) # control the size of the train set
            val_size = len(val_dataset)                 # control the size of the val set
            print(f"Train size: {train_size}, Val size: {val_size}")
            
            self.train_dataset = train_dataset[:train_size]
            self.val_dataset = val_dataset[:12288] # 4096*3
            self.test_dataset = val_dataset[12288:]  # You might want to use a separate test set
        
        if stage == 'test' or stage is None:
            self.test_dataset = val_dataset[12288:]  # You might want to use a separate test set

        if stage == 'predict' or stage is None:
            # predict_size = int(len(full_dataset) * 0.99)
            self.predict_dataset = val_dataset[:100]

    def normalize_trajectory(self, trajectory, norm="raw_trajectory"):
        # min, max values are gathered from the dataset.
        self.max_value = [93.67529745, 157.84470083]
        self.min_value = [-69.88046041, -115.58759644]
    
        # Apply masking if enabled (keep only first 30 points)
        if self.use_first_half_mask:
            trajectory = trajectory[:,:30, :]
    
        if norm=="z_norm":
            normalized_trajectory = (trajectory - self.mean) / (self.std + 1e-8)

        elif norm=="raw_trajectory":
            normalized_trajectory = trajectory

        elif norm=="scaled_norm":
            normalized_trajectory = trajectory / self.max_value[1]

        else:
            raise ValueError("Normalization arg required")
        # print(f"Normalized trajectory shape: {normalized_trajectory.shape}")
        # print(self.use_first_half_mask)
        return torch.tensor(normalized_trajectory, dtype=torch.float32)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized. Run setup() first.")
        return DataLoader(self.train_dataset, 
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True,
                            persistent_workers=True, 
                            pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized. Run setup() first.")
        return DataLoader(self.val_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            persistent_workers=True,
                            pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized. Run setup() first.")
        return DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            num_workers=1,
                            shuffle=False,
                            persistent_workers=True,
                            pin_memory=True)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, transform=None):
        self.trajectories = trajectories
        self.transform = transform
        # print(f"Trajectory dataset shape: {self.trajectories.shape}")
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        if self.transform:
            trajectory = self.transform(trajectory)
        return trajectory