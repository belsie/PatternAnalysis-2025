"""
This file contains the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Pretrained model
        resnet50= models.resnet50(weights="ResNet50_Weights.DEFAULT")
        a = 2

    def base_network(self):
        return

    def forward(self):
        return
    
class TripletLoss(nn.Module):
    def __init__(self, margin = 1) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """L(triplet) = max(0, d(a,p) - d(a,n) + m)""" 
        loss = torch.Tensor(0)
        return loss
        

        

        