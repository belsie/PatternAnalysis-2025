"""
This file contains the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import models

class Network(nn.Module):
    def __init__(self, embed_dim = 128, norm= True) -> None:
        super().__init__()

        # Pretrained model
        resnet50= models.resnet50(weights="ResNet50_Weights.DEFAULT")
        
        # remove classifier head
        modules = list(resnet50.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.proj = nn.Linear(2048, embed_dim)
        self.norm = norm

    def base_network(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        enc = torch.flatten(enc, 1)
        embed = self.proj(enc)

        if self.norm:
            embed = F.normalize(embed, p=2, dim=1)
        return embed

    def forward(self, x1: torch.Tensor, x2: torch.Tensor|None):
        e1 = self.base_network(x1)
        if x2 is None: 
            return e1
        e2 = self.base_network(x2)
        return e1, e2
    
class TripletLoss(nn.Module):
    def __init__(self, margin = 1) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """L(triplet) = max(0, distance(anchor,positive) - distance(anchor,negative) + margin)""" 
        def _distance(a, b):
            difference = a - b
            return (difference * difference).sum(dim=1).sqrt()   
        
        pos_dist = _distance(anchor, positive)
        neg_dist = _distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()
        
