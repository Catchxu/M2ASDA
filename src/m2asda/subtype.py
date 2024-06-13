import argparse
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List
from tqdm import tqdm

from .utils import seed_everything, update_configs_with_args
from .model import GeneratorWithMemory, Discriminator, Subtyper
from .configs import SubtypeConfigs


class PairedDataset(Dataset):
    def __init__(self, z, res):
        self.z = z
        self.res = res

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.res[idx]


class SubtypeModel:
    # List attributes
    n_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: torch.device
    random_state: int
    n_genes: int
    s_configs: Dict[str, int]

    def __init__(self, generator: GeneratorWithMemory, num_types: int, **kwargs):
        configs = SubtypeConfigs()

        # Update configs with kwargs
        for key, value in kwargs.items():
            if hasattr(configs, key):
                setattr(configs, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config")
        
        configs.build()

        # Initialize the attributes from configs
        for key, value in configs.__dict__.items():
            setattr(self, key, value)

        self._init_model(generator, num_types)

        seed_everything(self.random_state)
    
    def _init_model(self, generator: GeneratorWithMemory, num_types: int):
        self.S = Subtyper(generator, num_types, **self.s_configs).to(self.device)
        self.opt_S = optim.Adam(self.S.parameters(), lr=self.learning_rate, 
                                betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.sch_S = CosineAnnealingLR(self.opt_S, self.n_epochs)