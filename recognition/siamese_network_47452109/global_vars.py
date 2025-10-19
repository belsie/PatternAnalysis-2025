"""Global variables"""

import torch

SEED = 123
ROOT = "recognition\\siamese_network_47452109\\data\\"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')