import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: Generated text embeddings
            positive: Clean text embeddings
            negative: Toxic text embeddings
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)
        
        # Compute distances
        pos_dist = torch.norm(anchor - positive, dim=-1)
        neg_dist = torch.norm(anchor - negative, dim=-1)
        
        # Compute triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

class CombinedLoss(nn.Module):
    def __init__(self, triplet_weight=0.1, margin=1.0):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.triplet_weight = triplet_weight
    
    def forward(self, diffusion_loss, anchor, positive, negative):
        """
        Combine diffusion loss with triplet loss.
        
        Args:
            diffusion_loss: Standard diffusion loss
            anchor: Generated text embeddings
            positive: Clean text embeddings
            negative: Toxic text embeddings
        """
        trip_loss = self.triplet_loss(anchor, positive, negative)
        return diffusion_loss + self.triplet_weight * trip_loss, {
            'diffusion_loss': diffusion_loss.item(),
            'triplet_loss': trip_loss.item(),
            'total_loss': (diffusion_loss + self.triplet_weight * trip_loss).item()
        } 