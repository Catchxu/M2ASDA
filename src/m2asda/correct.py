import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List
from tqdm import tqdm

from .utils import seed_everything
from .model import GeneratorWithMemory, GeneratorWithPairs, Discriminator
from .configs import PairConfigs


class PairModel:
    # List attributes
    n_epochs: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, int]
    device: torch.device
    random_state: int
    n_genes: int
    d_configs: Dict[str, Any]

    def __init__(self, generator: GeneratorWithMemory, n_ref: int, n_tgt: int, **kwargs):
        configs = PairConfigs()

        # Update configs with kwargs
        for key, value in kwargs.items():
            if hasattr(configs, key):
                setattr(configs, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config")
        
        configs.update()

        # Initialize the attributes from configs
        for key, value in configs.__dict__.items():
            setattr(self, key, value)

        self._init_model(generator, n_ref, n_tgt)

        seed_everything(self.random_state)
    
    def _init_model(self, generator: GeneratorWithMemory, n_ref: int, n_tgt: int):
        self.G = GeneratorWithPairs(generator, n_ref, n_tgt).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

        self.loss = nn.L1Loss().to(self.device)
    
    def find(self, adata_list: List[ad.AnnData]):
        tqdm.write('Begin to find Kin Pairs between datasets...')
