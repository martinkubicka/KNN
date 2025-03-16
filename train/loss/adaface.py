import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaFaceLoss(nn.Module):
    def __init__(self, embedding_size, class_num, m=0.4, h=0.333, s=64.0, device=None):
        """
        AdaFace Loss
        :param embedding_size: Size of the feature embeddings
        :param class_num: Number of classes
        :param m: Base margin value (default=0.4)
        :param h: Margin adaptation slope (default=0.333)
        :param s: Feature scale (default=64.0)
        :param device: Device to run the computations (default=None, will use CUDA if available)
        """
        super(AdaFaceLoss, self).__init__()
        self.class_num = class_num
        self.embedding_size = embedding_size
        self.m = m
        self.h = h
        self.s = s
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learnable class weights (fully connected layer)
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):

        
        # Normalize embeddings and class weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1) 
        
        #print(f"Embeddings shape: {embeddings.shape}")
        #print(f"Weight_norm shape: {weight_norm.shape}")
        
               
        # Cosine similarity between embeddings and class weights
        logits = torch.matmul(embeddings, weight_norm.t())
        
        # Clip logits to avoid numerical instability
        logits = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Calculate difficulty-adaptive margins
        with torch.no_grad():
            diff = 1.0 - logits
            adaptive_margin = self.m + self.h * diff

        # Get target logits and apply the margin
        target_logits = logits[torch.arange(0, logits.size(0)), labels]
        modified_target_logits = target_logits - adaptive_margin[torch.arange(0, logits.size(0)), labels]
        
        # Replace target logits with modified values
        logits[torch.arange(0, logits.size(0)), labels] = modified_target_logits
        
        # Scale logits and calculate loss
        logits *= self.s
        loss = F.cross_entropy(logits, labels)

        return loss

    def get_predictions(self, embeddings):

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)
        logits = torch.matmul(embeddings, weight_norm.t())
        _, preds = torch.max(logits, 1)
        return preds