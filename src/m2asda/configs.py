from .utils import select_device


class AnomalyConfigs:
    def __init__(self):
        # Training
        self.n_epochs = 50
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.n_critic = 2
        self.loss_weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.device = select_device('cuda:0')
        self.random_state = 2024

        self.n_genes = 3000

        # Model
        self.g_configs = {
            'input_dim': self.n_genes,
            'hidden_dim': [1024, 512, 256],
            'latent_dim': 256,
            'memory_size': 512, 
            'threshold': 0.005,
            'temperature': 0.1,
            'normalization': True,
            'activation': True,
            'dropout': 0.1,
        }

        self.d_configs = {
            'input_dim': self.n_genes,
            'hidden_dim': [1024, 512, 256],
            'latent_dim': 256,
            'normalization': True,
            'activation': True,
            'dropout': 0.1            
        }
