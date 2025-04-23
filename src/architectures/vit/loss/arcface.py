import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, class_num, s=64.0, m=0.5, device=None):
        """
        ArcFace Loss
        :param embedding_size: Size of the feature embeddings
        :param class_num: Number of classes
        :param s: Feature scale (default=64.0)
        :param m: Angular margin (default=0.5 radians)
        :param device: Device to run the computations (default=None, will use CUDA if available)
        """
        super(ArcFaceLoss, self).__init__()
        self.class_num = class_num
        self.embedding_size = embedding_size
        self.s = s
        self.m = m
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learnable class weights (fully connected layer)
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Normalize embeddings and class weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)
        
        # Cosine similarity between embeddings and class weights
        cosine = torch.matmul(embeddings, weight_norm.t())
        
        # Clip logits to avoid numerical instability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute the angular margin
        theta = torch.acos(cosine)  # Inverse cosine to get angles

        # Create a mask for the target labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Calculate the modified cosine values for the target labels
        output = (one_hot * torch.cos(theta + self.m)) + ((1.0 - one_hot) * cosine)

        # Scale logits and calculate loss
        output = output * self.s
        loss = F.cross_entropy(output, labels)
        
        return loss

    def get_predictions(self, embeddings):
        """
        Predict class labels based on embeddings.
        :param embeddings: Feature embeddings from the model
        :return: Predicted class labels
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)
        logits = torch.matmul(embeddings, weight_norm.t())
        _, preds = torch.max(logits, 1)
        return preds