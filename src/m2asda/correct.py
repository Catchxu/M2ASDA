import torch
from typing import Dict, Any

from .utils import seed_everything
from .model import GeneratorWithMemory
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
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]

    def __init__(self, generator: GeneratorWithMemory, **kwargs):
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

        self._init_model(generator)

        seed_everything(self.random_state)
    
    def _init_model(self, generator: GeneratorWithMemory):
        self.G = generator.to(self.device)
        for param in self.G.parameters():
            param.requires_grad = False
        
        