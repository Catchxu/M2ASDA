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

        self.n_ref = n_ref
        self.n_tgt = n_tgt
    
    def find(self, ref: ad.AnnData, tgt: ad.AnnData):
        self.check(ref, tgt)
        ref_data = torch.tensor(ref.X).to(self.device)
        tgt_data = torch.tensor(tgt.X).to(self.device)

        tqdm.write('Begin to find Kin Pairs between datasets...')

        self.G.train()
        self.D.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description('Training Epochs')

                for _ in range(self.n_critic):
                    self.UpdateD(ref_data, tgt_data)

                self.UpdateG(ref_data, tgt_data)       

    def check(self, ref: ad.AnnData, tgt: ad.AnnData):
        if self.n_ref != ref.n_obs:
            raise AttributeError(f"Number of cells in ref is different with n_ref")

        if self.n_tgt != tgt.n_obs:
            raise AttributeError(f"Number of cells in tgt is different with n_tgt")
        
        if ref.var_names != tgt.var_names:
            raise AttributeError(f"ref and tgt have different genes")
    
    def UpdateG(self, ref_data, tgt_data):
        fake_z_tgt, z_tgt, _ = self.G(ref_data, tgt_data)

        # Discriminator provides feedback
        d = self.D(fake_z_tgt)

        self.G_loss = self.loss_weight['alpha'] * self.loss(z_tgt, fake_z_tgt) \
                      - self.loss_weight['beta'] * torch.mean(d)
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

    def UpdateD(self, ref_data, tgt_data):
        fake_z_tgt, z_tgt, _ = self.G(ref_data, tgt_data)
        z_tgt = z_tgt.detach()
        fake_z_tgt = fake_z_tgt.detach()

        d1 = torch.mean(self.D(z_tgt))
        d2 = torch.mean(self.D(fake_z_tgt))
        gp = self.D.gradient_penalty(z_tgt, fake_z_tgt)
        self.D_loss = - d1 + d2 + gp * self.loss_weight['lambda']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()