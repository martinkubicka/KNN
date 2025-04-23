import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    def __init__(self, embedding_size, class_num, margin=0.35, scale=30.0, device=None):
        """
        CosFace Loss: Additive Margin Softmax
        Args:
            embedding_dim (int): Dimensionality of the input embeddings.
            num_classes (int): Number of classes (classification labels).
            margin (float): Margin to add to the logits (default: 0.35).
            scale (float): Scaling factor for logits (default: 30.0).
        """
        super(CosFaceLoss, self).__init__()
        self.num_classes = class_num
        self.margin = margin
        self.scale = scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Correctly initialize the weight matrix
        self.weight = nn.Parameter(torch.randn( class_num, embedding_size))  # Corrected shape
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Forward pass for CosFace Loss.
        Args:
            embeddings (Tensor): Input embeddings of shape (batch_size, embedding_dim).
            labels (Tensor): Ground-truth labels of shape (batch_size,).
        Returns:
            loss (Tensor): CosFace loss value.
        """
        # Normalize the embeddings and weight matrix
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)  # Normalize along dim 1

        # Compute cosine similarity between embeddings and class weights
        logits = torch.matmul(embeddings, weight_norm.t())  # Removed .t()

        # Add margin to the logits for the true class
        target_logits = logits[torch.arange(logits.size(0)), labels]
        target_logits = target_logits - self.margin
        

        # Scale logits and replace target logits with margin-adjusted ones
        logits  *= self.scale
        logits[torch.arange(embeddings.size(0)), labels] = target_logits * self.scale

        # Compute softmax cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss
    
    
    def get_predictions(self, embeddings):
        """
        Get class predictions based on embeddings.

        Args:
            embeddings (Tensor): Input embeddings of shape (batch_size, embedding_dim).

        Returns:
            predictions (Tensor): Predicted class indices of shape (batch_size,).
        """
        # Normalize the embeddings and weight matrix
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)

        # Compute cosine similarity between embeddings and class weights
        logits = torch.matmul(embeddings, weight_norm.t())

        # Get the predicted class labels (index of the highest logit)
        predictions = torch.argmax(logits, dim=1)
        return predictions