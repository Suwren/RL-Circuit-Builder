import torch
import torch.nn as nn
from utils.device_helper import get_device
import warnings

def test_norm(name, layer_fn):
    device = get_device()
    print(f"\nTesting {name} on {device}...")
    
    # Setup
    N, C, H, W = 1, 64, 12, 12
    x = torch.randn(N, C, H, W).to(device)
    model = layer_fn(C).to(device)
    model.train()
    
    # Forward
    try:
        y = model(x)
        print(f"  Forward: Success")
    except Exception as e:
        print(f"  Forward: FAILED ({e})")
        return

    # Backward
    try:
        loss = y.mean()
        
        # Catch warnings to see if fallback happens
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss.backward()
            
            if len(w) > 0:
                print(f"  Backward: Success (WITH WARNINGS)")
                for warning in w:
                    print(f"    - {warning.message}")
            else:
                print(f"  Backward: Success (Clean)")
                
    except Exception as e:
        print(f"  Backward: FAILED ({e})")

def main():
    # 1. GroupNorm (Current)
    test_norm("GroupNorm (8 groups)", lambda c: nn.GroupNorm(8, c))
    
    # 2. InstanceNorm2d
    test_norm("InstanceNorm2d", lambda c: nn.InstanceNorm2d(c, affine=True))
    
    # 3. LayerNorm (Simulated with GroupNorm(1, c)) - likely same warning
    test_norm("LayerNorm (via GroupNorm(1))", lambda c: nn.GroupNorm(1, c))
    
    # 4. LayerNorm (Native) - requires reshaping usually, but let's see
    # LayerNorm expects specific normalized_shape
    # test_norm("LayerNorm (Native)", lambda c: nn.LayerNorm([c, 12, 12])) 
    # Skipping Native LayerNorm for 2D input for now as it's complex to setup generic test

if __name__ == "__main__":
    main()
