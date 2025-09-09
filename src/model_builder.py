import torch
import torch.nn as nn
import torchvision.models


def create_model(model_name: str, num_classes: int):
    """
    Creates an image classification model based on the provided name.

    Args:
        model_name (str): The name of the model (e.g., 'efficientnet_b2').
        num_classes (int): The number of output classes.
        pretrained (bool): True to use weights pre-trained on ImageNet.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model = None

    if model_name == 'efficientnet_b2':
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.efficientnet_b2(weights=weights)
        # Freeze the feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace the classifier layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'convnext_tiny':
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.convnext_tiny(weights=weights)
        # Freeze the feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.mobilenet_v2(weights=weights)
        # Freeze the feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'vit_b_16':
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        auto_transforms = weights.transforms()
        model = torchvision.models.vit_b_16(weights=weights)
        # For ViT, the layers are not in 'features'
        for param in model.parameters():
            param.requires_grad = False
        # Replace the heads layer (ViT's classifier)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model name '{model_name}' is not supported.")

    return model, auto_transforms
