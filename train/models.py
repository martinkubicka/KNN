import torch.nn as nn
import timm
import torch

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

        embed_dim = self.vit.embed_dim

        # We will add new positional embedings 
        self.vit.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim)) # +1 for cls token

        # Replace the classification head (if needed)
        if num_classes != 1000:  # Or whatever the original pretrained model had
            self.vit.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.vit(x)
    