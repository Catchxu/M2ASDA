import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict

from .utils import seed_everything
from .configs import AnomalyConfigs
from .model import GeneratorWithMemory, Discriminator


class AnomalyModel:
    # Attributes
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, int]
    device: torch.device
    random_state: int
    G_params: Dict[str, int]
    D_params: Dict[str, int]    

    def __init__(self, configs: AnomalyConfigs, **kwargs):
        for key, value in configs.__dict__.items():
            setattr(self, key, value)

        # Update the attributes in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of Config")

        self._init_model()

        seed_everything(self.random_state)
    
    def _init_model(self):
        self.G = GeneratorWithMemory(**self.G_params).to(self.device)
        self.D = Discriminator(**self.D_params).to(self.device) 

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)
        