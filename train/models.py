import torch.nn as nn
import timm
import torch
from transformers import ViTModel

class ViT_Wrapper(nn.Module):
    def __init__(self, vit_model):
        super(ViT_Wrapper, self).__init__()
        self.vit_model = vit_model

    def forward(self, x):
        outputs = self.vit_model(x)
     #   pooled_output = outputs.pooler_output
        return outputs

class ViT_Model(nn.Module):
    """
    Vision Transformer (ViT) Model.

    Args:
        model_name (str): Name of the ViT model to use. Default is 'vit_base_patch32_224'.
        num_classes (int): Number of output classes for the classification head. Default is 1000.
        pretrained (bool): Whether to use a pretrained model. Default is True.
        input_width (int): Width of the input image. Must be a multiple of 32. Default is 224.
        input_height (int): Height of the input image. Must be a multiple of 32. Default is 224.

    Raises:
        AttributeError: If input_width or input_height is not a multiple of 32. JKJ Keď to nebude štvorec tak sme v piči

    Attributes:
        vit (nn.Module): The Vision Transformer model.
    
    Methods:
        forward(x):
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after passing through the model.
    """
    def __init__(self, model_name='vit_base_patch32_224', num_classes=1000, pretrained=True, input_width = 224, input_height = 224):
        super().__init__()

        print(f"Using model: {model_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Pretrained: {pretrained}")
        print(f"Input width: {input_width}")
        print(f"Input height: {input_height}")

        if input_width % 32 or input_height % 32:
            raise AttributeError("Model sizes must add squeres")

        num_patches_h = input_height // 32  
        num_patches_w = input_width // 32   
        num_patches = num_patches_h * num_patches_w 

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head later
            img_size=(input_height, input_width), 
            patch_size=32,
        )
        #self.vit = ViTModel.from_pretrained(model_name)

        embed_dim = self.vit.embed_dim

        # We will add new positional embedings 
        self.vit.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # +1 for cls token
        self.vit.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.vit.forward_features(x)
        cls_token = features[:, 0]
        return cls_token
    

class CustomModel(nn.Module):
    def __init__(self, features, embedding_layer, classifier_layer):
        super(CustomModel, self).__init__()
        self.features = features
        self.embedding_layer = embedding_layer
        self.classifier_layer = classifier_layer

    def forward(self, x):
        x = self.features(x)

        x = self.embedding_layer(x)
        logits = self.classifier_layer(x)
        return logits
    
    def get_embedding(self, x):
        x = self.features(x)
        x = self.embedding_layer(x)
        return x


def get_model(config: dict):
    if config["architecture"]["name"] == "vit":
        vit_model = ViT_Model(num_classes = config["architecture"]["num_classes"],
                                  input_width = config["input_size"][0],
                                  input_height = config["input_size"][1])
        
        in_features = vit_model.vit.embed_dim
        model = ViT_Wrapper(vit_model)
    else:
        raise NotImplementedError(f"Model {config['model']} not implemented")

    embedding_layer = nn.Sequential(
        nn.Linear(in_features, config["embedding"]["dim"]),
        nn.ReLU(),
        nn.BatchNorm1d(config["embedding"]["dim"]),
        nn.Dropout(config["embedding"]["dropout"])
    )

    classifier_layer = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(config["embedding"]["dim"], config["architecture"]["num_classes"]),
        nn.Softmax(dim=1)
    )

    return CustomModel(model, embedding_layer, classifier_layer)