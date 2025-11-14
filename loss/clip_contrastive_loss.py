"""
CLIP Contrastive Loss for Stage0 pretraining.

Implements the original CLIP loss functions:
L_i2t(i) = -log(exp(s(Vi, Ti)) / sum_a=1^B exp(s(Vi, Ta)))
L_t2i(i) = -log(exp(s(Vi, Ti)) / sum_a=1^B exp(s(Va, Ti)))

Where:
- Vi: image features for sample i
- Ti: text features for sample i  
- s(Vi, Ti): similarity score (cosine similarity * temperature)
- B: batch size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """
    CLIP contrastive loss for image-text pairs.
    
    This loss encourages matching image-text pairs to have high similarity
    while pushing non-matching pairs to have low similarity.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, image_features, text_features):
        batch_size = image_features.shape[0]

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity_matrix = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Ground truth: diagonal matrix (image_i matches text_i)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Image-to-Text loss: L_i2t(i) = -log(exp(s(Vi, Ti)) / sum_a exp(s(Vi, Ta)))
        i2t_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Text-to-Image loss: L_t2i(i) = -log(exp(s(Vi, Ti)) / sum_a exp(s(Va, Ti)))  
        t2i_loss = F.cross_entropy(similarity_matrix.t(), labels)

        total_loss = (i2t_loss + t2i_loss) / 2.0
        
        return total_loss, i2t_loss, t2i_loss
    
    def compute_accuracy(self, image_features, text_features):
        batch_size = image_features.shape[0]

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarity_matrix = torch.matmul(image_features, text_features.t())
        
        # Ground truth
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Image-to-text accuracy: for each image, is the most similar text the correct one?
        i2t_pred = similarity_matrix.argmax(dim=1)
        i2t_acc = (i2t_pred == labels).float().mean().item()
        
        # Text-to-image accuracy: for each text, is the most similar image the correct one?
        t2i_pred = similarity_matrix.t().argmax(dim=1)  
        t2i_acc = (t2i_pred == labels).float().mean().item()
        
        return i2t_acc, t2i_acc
