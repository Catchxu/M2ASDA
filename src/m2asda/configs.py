from .utils import select_device


class AnomalyConfigs:
    def __init__(self):
        # Training
        self.n_epochs = 50
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.n_critic = 2

        self.alpha = 30
        self.beta = 10
        self.gamma = 1
        self.lamb = 10
        self.loss_weight = {
            'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma, 'lambda': self.lamb
        }

        self.GPU = 'cuda:0'
        self.device = select_device(self.GPU)
        self.random_state = 2024

        # Model
        self.n_genes = 3000
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

        self.gmm_configs = {
            'random_state': 1024,
            'max_iter': 100,
            'tol': 1e-5,
            'prior_beta': [1, 10]
        }
    
    def update(self):
        self.device = select_device(self.GPU)

        self.loss_weight = {
            'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma, 'lambda': self.lamb
        }

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

