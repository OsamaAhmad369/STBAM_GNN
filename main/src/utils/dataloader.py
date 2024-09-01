import pickle
import torch
import io
from torch_geometric.data import DataLoader
from typing import Any


train_path = "../data/train_data_c2d2_BA.pkl"
val_path = "../data/val_data_c2d2_BA.pkl"
test_path = "../data/test_data_c2d2_BA.pkl"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, __module_name: str, __global_name: str) -> Any:
        if __module_name == 'torch.storage' and __global_name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(__module_name, __global_name)
        
class DataLoaderCreator:
    def __init__(self, train_path, val_path, test_path, batch_size=4):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size

        self.train_list = self._load_data(self.train_path)
        self.val_list = self._load_data(self.val_path)
        self.test_list = self._load_data(self.test_path)

    def _load_data(self, file_path):
        with open(file_path, "rb") as f:
            data_list = CPU_Unpickler(f).load()
        return data_list
    
    def get_loaders(self):
        train_loader = DataLoader(self.train_list, self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_list, self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_list, self.batch_size, shuffle=False)
        print(f"Training Dataset Size: {len(self.train_list)}")
        print(f"Testing Dataset Size: {len(self.test_list)}")
        print(f"Validation Dataset Size: {len(self.val_list)}")
        return train_loader, val_loader, test_loader
