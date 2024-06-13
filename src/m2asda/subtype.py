import argparse
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List
from tqdm import tqdm

from .utils import seed_everything, update_configs_with_args, PairDataset
from .model import GeneratorWithMemory, Subtyper
from .configs import SubtypeConfigs


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
        self.G = generator
        self.S = Subtyper(generator, num_types, **self.s_configs).to(self.device)
        self.opt_S = optim.Adam(self.S.parameters(), lr=self.learning_rate, 
                                betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.sch_S = CosineAnnealingLR(self.opt_S, self.n_epochs)
    
    def train(self, adata: ad.AnnData):
        data = torch.Tensor(adata.X).to(self.device)
        z, res = self.generate_z_res(data)

        self.S.mu_init(z.cpu().detach().numpy())
        dataset = PairDataset(z, res)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)

        self.S.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description('Training Epochs')

                for z_data, res_data in self.loader:
                    z_data = z_data.to(self.device)
                    res_data = res_data.to(self.device)


    @torch.no_grad()
    def generate_z_res(self, data: torch.Tensor):
        self.G.eval()
        fake_data, z = self.G(data)
        res = data - fake_data.detach()
        return z, res