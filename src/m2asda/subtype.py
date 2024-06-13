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


class KLLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def kld(self, target, pred):
        log_frac = torch.log(target/(pred + self.epsilon))
        return torch.mean(torch.sum(target*log_frac, dim=1))
    
    def forward(self, p, q):
        return self.kld(p, q)


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

        self.loss = KLLoss().to(self.device)
    
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
                    _, q = self.S(z_data, res_data)
                    p = self.S.target_distribution(q).data

                    self.opt_S.zero_grad()
                    loss = self.loss(p, q)
                    loss.backward()
                    self.opt_S.step()

                t.set_postfix(Loss = loss.item())
                t.update(1)
                self.sch_S.step()
        
        with torch.no_grad():
            self.S.eval()
            _, q = self.S(z, res)
            return q

    @torch.no_grad()
    def generate_z_res(self, data: torch.Tensor):
        self.G.eval()
        fake_data, z = self.G(data)
        res = data - fake_data.detach()
        return z, res




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M2ASDA for anomaly subtyping.")
    configs = SubtypeConfigs()

    # Data path arguments
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--read_path', type=str, help='Path to read the h5ad file')
    data_group.add_argument('--save_path', type=str, default='result.csv', help='Path to save output csv file')
    data_group.add_argument('--pth_path', type=str, required=True, help='Path to read the trained generator')

    # SubtypeModel arguments with defaults from SubtypeConfigs
    s_group = parser.add_argument_group('AnomalyModel Parameters')
    s_group.add_argument('--n_epochs', type=int, default=configs.n_epochs, help='Number of epochs')
    s_group.add_argument('--batch_size', type=int, default=configs.batch_size, help='Batch size')
    s_group.add_argument('--learning_rate', type=float, default=configs.learning_rate, help='Learning rate')
    s_group.add_argument('--weight_decay', type=float, default=configs.weight_decay, help='Weight decay rate')
    s_group.add_argument('--GPU', type=str, default=configs.GPU, help='GPU ID for training, e.g., cuda:0')
    s_group.add_argument('--random_state', type=int, default=configs.random_state, help='Random seed')
    s_group.add_argument('--n_genes', type=int, default=configs.n_genes, help='Number of genes')

    args = parser.parse_args()

    # Update the configs with command line argument
    args_dict = vars(args)
    update_configs_with_args(configs, args_dict, None)

    configs.build()
    configs.clear()

    # Print out all configurations to verify they are complete
    print("=============== SubtypeModel Parameters ===============")
    for key, value in configs.__dict__.items():
        print(f"{key} = {value}")