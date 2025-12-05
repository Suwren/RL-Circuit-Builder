import torch
import torch.nn as nn
from utils.device_helper import get_device
from alphazero.model import AlphaZeroNet

def reproduce():
    device = get_device()
    print(f"Device: {device}")
    
    # Input shape: (Channels, H, W)
    # Channels = 9 (Types) + 5 (Inventory) + 4 (Feats) = 18 approx
    input_shape = (18, 12, 12) 
    num_actions = 100
    
    model = AlphaZeroNet(input_shape, num_actions).to(device)
    model.train() # Ensure training mode
    
    # Test with batch size causing the issue (maybe small batch?)
    # User had 90 samples, batch size 64. So batches were 64 and 26.
    batch_sizes = [64, 26, 1]
    
    for bs in batch_sizes:
        print(f"\nTesting Batch Size: {bs}")
        x = torch.randn(bs, *input_shape).to(device)
        
        # Forward
        pi, v = model(x)
        
        # Backward
        loss = pi.mean() + v.mean()
        
        try:
            model.zero_grad()
            loss.backward()
            print(f"Batch Size {bs}: Success")
        except RuntimeError as e:
            print(f"Batch Size {bs}: FAILED with error: {e}")
            break

if __name__ == "__main__":
    reproduce()
