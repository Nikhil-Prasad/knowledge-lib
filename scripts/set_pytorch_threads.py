#!/usr/bin/env python3
"""Set PyTorch thread defaults for MPS to avoid oversubscription."""

import os
import torch

def setup_pytorch_for_mps():
    """Configure PyTorch for optimal MPS (Metal Performance Shaders) usage."""
    # Set PyTorch to use single thread to avoid oversubscription on MPS
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    else:
        print("MPS is not available on this system")

if __name__ == "__main__":
    setup_pytorch_for_mps()
