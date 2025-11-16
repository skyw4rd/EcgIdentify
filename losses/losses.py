import torch
from torch.nn import functional as F

class TripletPlusCe(torch.nn.Module):
    def __init__(self, triplet_loss: torch.nn.Module, cross_entropy_loss: torch.nn.Module):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss
        self.triplet_loss = triplet_loss
    
    def forward(self, anchors, postives, negatives, outputs, labels):
        ce_loss = self.cross_entropy_loss(outputs, labels)
        triplet_loss = self.triplet_loss(anchors, postives, negatives)
        loss = ce_loss + triplet_loss
        return loss

def create_criterion():
    # TODD
    return None