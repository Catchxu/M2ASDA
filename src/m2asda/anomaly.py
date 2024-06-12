import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any

from .utils import seed_everything
from .configs import AnomalyConfigs
from .model import AutoEncoder, GeneratorWithMemory, Discriminator


class AnomalyModel:

    # Training
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, int]
    device: torch.device
    random_state: int

    n_genes: int

    # Model
    ae_configs: Dict[str, Any]
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]

    def __init__(self, configs: AnomalyConfigs, **kwargs):
        for key, value in configs.__dict__.items():
            setattr(self, key, value)

        # Update the attributes in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config")
        
        if 'n_genes' in kwargs:
            self.ae_configs['input_dim'] = kwargs['n_genes']
            self.d_configs['input_dim'] = kwargs['n_genes']

        self._init_model()

        seed_everything(self.random_state)
    
    def _init_model(self):
        model = AutoEncoder(**self.ae_configs)
        self.G = GeneratorWithMemory(model, **self.g_configs).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device) 

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)
        