from src.generation.models.resnet1d import ResNet1D
import torch
# Test del modello
if __name__ == "__main__":
    # Test con parametri simili a quelli usati in approach1
    model = ResNet1D(
        in_channels=1, 
        base_filters=64, 
        kernel_size=16, 
        stride=2,  # Downsampling solo nel primo layer
        n_block=8, 
        n_classes=2048
    )
    
    # Input di esempio: 8s @ 64Hz = 512 campioni
    x = torch.randn(4, 1, 512)  # batch=4, channels=1, length=512
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Numero parametri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test con ACC (3 canali)
    model_acc = ResNet1D(
        in_channels=3, 
        base_filters=32, 
        kernel_size=8, 
        stride=2, 
        n_block=4, 
        n_classes=2048
    )
    x_acc = torch.randn(4, 3, 256)
    output_acc = model_acc(x_acc)
    print(f"\nACC Input shape: {x_acc.shape}")
    print(f"ACC Output shape: {output_acc.shape}")