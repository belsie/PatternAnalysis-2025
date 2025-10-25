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
    def __init__(self) -> None:
        super().__init__()

        # Pretrained model
        resnet50= models.resnet50(weights="ResNet50_Weights.DEFAULT")
        
        # remove classifier head
        modules = list(resnet50.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embedding = self.embed(features)
        return embedding
    
    class SiameseNetwork(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.network = Network()

        def forward(self, input1, input2): 
            output1 = self.network(input1)
            output2 = self.network(input2)

            return output1, output2 
    
class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0) -> None:
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
        
