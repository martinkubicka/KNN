import torch.nn as nn
import timm
import torch
from transformers import ViTModel
import torch.nn.functional as F

class ViT_Wrapper(nn.Module):
    def __init__(self, vit_model):
        super(ViT_Wrapper, self).__init__()
        self.vit_model = vit_model

    def forward(self, x):
        outputs = self.vit_model(x)
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
    def __init__(self, model_name='vit_base_patch16_224', num_classes=1000, pretrained=True, input_width = 224, input_height = 224, patch_size = 8):
        super().__init__()
        print("Patch size: ", patch_size)
        print(f"Using model: {model_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Pretrained: {pretrained}")
        print(f"Input width: {input_width}")
        print(f"Input height: {input_height}")

        if input_width % patch_size or input_height % patch_size:
            raise AttributeError("Model sizes must add squeres")

        num_patches_h = input_height // patch_size  
        num_patches_w = input_width // patch_size   
        num_patches = num_patches_h * num_patches_w 

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0, 
            img_size=(input_height, input_width), 
            patch_size=patch_size,
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


class ConvNext(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000):
        super(ConvNext, self).__init__()
        self.convnext = timm.create_model(
            'convnext_base',
            pretrained=pretrained,
            num_classes=num_classes # or any dummy
        )
        # Replace the final classification head with Identity
        self.convnext.head = nn.Identity()

    def forward(self, x):
        x = self.convnext(x)
        print(x.shape)# shape is (B, 1024, H, W) after removing the head
        x = F.adaptive_avg_pool2d(x, 1)  # shape is now (B, 1024, 1, 1)
        x = x.view(x.size(0), -1)     # shape is now (B, 1024)
        return x

class MobileNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000):
        super(MobileNet, self).__init__()
        self.mobilenet = timm.create_model(
            'mobilenetv3_large_100',  # You can choose other MobileNet variants
            pretrained=pretrained,
            num_classes=num_classes # or any dummy
        )
        # Replace the final classification head with Identity
        self.mobilenet.classifier = nn.Identity()

    def forward(self, x):
        x = self.mobilenet.forward_features(x) # Use forward_features for feature extraction
        print(x.shape) # Shape will depend on the MobileNet variant (e.g., B, 960, H/32, W/32 for mobilenetv3_large_100)
        x = F.adaptive_avg_pool2d(x, 1)  # shape is now (B, C, 1, 1) where C is the number of channels
        x = x.view(x.size(0), -1)     # shape is now (B, C)
        return x

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
                                  input_height = config["input_size"][1],
                                  patch_size = config["architecture"]["patch_size"]
                                    )
        
        in_features = vit_model.vit.embed_dim
        model = ViT_Wrapper(vit_model)

    elif config["architecture"]["name"] == "convnext":
        model = ConvNext(num_classes=config["architecture"]["num_classes"], pretrained=config["architecture"]["pretrained"])
        in_features = model.convnext.num_features
        print("Features: ", in_features)

    elif config["architecture"]["name"] == "mobilenet":
        model = MobileNet(num_classes=config["architecture"]["num_classes"], pretrained=config["architecture"]["pretrained"])
        in_features = model.mobilenet.num_features
        print("Features: ", in_features)

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