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

        # Model

