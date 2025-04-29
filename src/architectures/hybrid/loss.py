# Inspired by: https://github.com/YirongMao/softmax_variants/blob/master/model_utils.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits
    
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


# https://github.com/GOKORURI007/pytorch_arcface/blob/main/arcface.py
class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is from https://github.com/deepinsight/insightface, which takes
        66.45489303627983 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        Please noted that, different with forward1&3, this implementation ignore the samples that
        caused \theta + m > \pi to happen if easy_margin is False. And if easy_margin is True,
        it will do nothing even if \theta + m > \pi.
        """
        embedding = F.normalize(embedding)
        w = F.normalize(self.weight)
        cos_theta = F.linear(embedding, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.easy_margin:
            mask = torch.ones_like(ground_truth)
        else:
            mask = torch.gather(cos_theta, 1, ground_truth.view(-1, 1)).view(-1)
            mask = torch.where(mask.acos_() + self.margin > math.pi, 0, 1)
        mask = torch.where(mask != 0)[0]
        m_hot = torch.zeros(mask.shape[0], cos_theta.shape[1], device=cos_theta.device)
        m_hot.scatter_(1, ground_truth[mask, None], self.margin)
        theta = cos_theta.acos()
        output = (theta + m_hot).cos()
        output.mul_(self.scale)
        loss = self.ce(output[mask], ground_truth[mask])
        return loss
    
    def get_predictions(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight.to(embeddings.device), p=2, dim=1)
        logits = torch.matmul(embeddings, weight_norm.t())
        _, preds = torch.max(logits, 1)
        return preds