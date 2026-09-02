import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_mobilenet(num_classes=15, pretrained=True):
    """
    Loads a MobileNetV2 model and modifies the classification head 
    for our specific plant disease dataset.
    """
    # Load pre-trained weights from ImageNet to significantly speed up training
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    # MobileNetV2's classifier is a Sequential block. 
    # Index [1] is the final Linear layer that originally outputs 1000 classes.
    in_features = model.classifier[1].in_features
    
    # Replace it with a new Linear layer that outputs exactly 15 classes.
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    print("Building MobileNetV2 architecture...")
    
    try:
        model = get_mobilenet(num_classes=15)
        
        # Simulate passing a single 224x224 RGB image through the network
        # Shape: [Batch_Size, Channels, Height, Width]
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"Success! Output tensor shape: {output.shape} | Expected: torch.Size([1, 15])")
        print("The baseline model architecture is ready to go.")
        
    except Exception as e:
        print(f"An error occurred: {e}")