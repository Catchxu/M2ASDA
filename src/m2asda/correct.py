import argparse
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List
from tqdm import tqdm

from .utils import seed_everything
from .model import GeneratorWithMemory, GeneratorWithPairs, Discriminator, GeneratorWithStyle
from .configs import PairConfigs, CorrectConfigs


class PairDataset(Dataset):
    def __init__(self, ref_data, tgt_data):
        self.ref_data = ref_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.ref_data)

    def __getitem__(self, index):
        ref_sample = self.ref_data[index]
        tgt_sample = self.tgt_data[index]

        return {'ref': ref_sample, 'tgt': tgt_sample}


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
        
        configs.build()

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
    
    def train(self, ref: ad.AnnData, tgt: ad.AnnData):
        self.check(ref, tgt)
        ref_data = torch.Tensor(ref.X).to(self.device)
        tgt_data = torch.Tensor(tgt.X).to(self.device)

        tqdm.write('Begin to find Kin Pairs between datasets...')

        self.G.train()
        self.D.train()

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description('Training Epochs')

                for _ in range(self.n_critic):
                    self.UpdateD(ref_data, tgt_data)

                self.UpdateG(ref_data, tgt_data)

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        P_matrix = F.relu(self.G.P).detach().cpu().numpy()
        idx = list(ref.obs_names[P_matrix.argmax(axis=1)])
        ref_pair = ref[idx]

        tqdm.write('Kin Pairs between datasets have been found.\n')
        return PairDataset(ref_pair, tgt)

    def check(self, ref: ad.AnnData, tgt: ad.AnnData):
        if self.n_ref != ref.n_obs:
            raise AttributeError(f"Number of cells in ref is different with n_ref")

        if self.n_tgt != tgt.n_obs:
            raise AttributeError(f"Number of cells in tgt is different with n_tgt")
        
        if ref.var_names != tgt.var_names:
            raise AttributeError(f"ref and tgt have different genes")
    
    def UpdateG(self, ref_data, tgt_data):
        fake_z_tgt, z_tgt = self.G(ref_data, tgt_data)

        # Discriminator provides feedback
        d = self.D(fake_z_tgt)

        self.G_loss = self.loss_weight['alpha'] * self.loss(z_tgt, fake_z_tgt) \
                      - self.loss_weight['beta'] * torch.mean(d)
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

    def UpdateD(self, ref_data, tgt_data):
        fake_z_tgt, z_tgt = self.G(ref_data, tgt_data)
        z_tgt = z_tgt.detach()
        fake_z_tgt = fake_z_tgt.detach()

        d1 = torch.mean(self.D(z_tgt))
        d2 = torch.mean(self.D(fake_z_tgt))
        gp = self.D.gradient_penalty(z_tgt, fake_z_tgt)
        self.D_loss = - d1 + d2 + gp * self.loss_weight['lambda']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()


class CorrectModel:
    # List attributes
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, int]
    device: torch.device
    random_state: int
    n_genes: int
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]

    def __init__(self, num_batches: int, **kwargs):
        configs = CorrectConfigs()

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

        self._init_model(num_batches)

        seed_everything(self.random_state)
    
    def _init_model(self, num_batches: int):
        self.G = GeneratorWithStyle(num_batches=num_batches, **self.g_configs).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))     
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

        self.loss = nn.L1Loss().to(self.device)
    
    def train(self, raw_data: List[ad.AnnData], dataset: Dataset):
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)
        
        tqdm.write('Begin to correct batch effects between datasets...')        
        
        self.G.train()
        self.D.train()

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description('Training Epochs')

                for data in self.loader:
                    ref_data = data['ref'].to(self.device)
                    tgt_data = data['tgt'].to(self.device)

                    for _ in range(self.n_critic):
                        self.UpdateD(ref_data, tgt_data)

                    self.UpdateG(ref_data, tgt_data)
        
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
        
        # Generate data without batch effects
        adata = ad.concat(raw_data)
        dataset = torch.Tensor(adata.X)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=False)
        
        self.G.eval()
        corrected = []
        with torch.no_grad():
            for data in self.loader:
                data = data.to(self.device)
                corrected_data = self.G(data)
                corrected.append(corrected_data.cpu().detach())

        corrected = torch.cat(corrected, dim=0).numpy()

        # Save the raw data
        adata.raw = adata
        adata.X = corrected
        tqdm.write('Batch effects have been corrected.\n')
        return adata

    def UpdateG(self, ref_data, tgt_data):
        fake_ref_data = self.G(tgt_data)

        # Discriminator provides feedback
        d = self.D(fake_ref_data)

        self.G_loss = self.loss_weight['alpha'] * self.loss(ref_data, fake_ref_data) \
                      - self.loss_weight['beta'] * torch.mean(d)
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

    def UpdateD(self, ref_data, tgt_data):
        fake_ref_data = self.G(tgt_data)
        fake_ref_data = fake_ref_data.detach()

        d1 = torch.mean(self.D(ref_data))
        d2 = torch.mean(self.D(fake_ref_data))
        gp = self.D.gradient_penalty(ref_data, fake_ref_data)
        self.D_loss = - d1 + d2 + gp * self.loss_weight['lambda']

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M2ASDA for batch correction.")
    p_configs = PairConfigs()
    c_configs = CorrectConfigs()


