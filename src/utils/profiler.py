import os
import tempfile
import torch

try:
    from torchinfo import summary
except ImportError:
    summary = None
    print("Warning: torchinfo is not installed. FLOPs/MACs calculation will be skipped.")

def get_parameter_count(model):
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """
    Returns the disk footprint of the model's state_dict in KB and MB.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = os.path.getsize(tmp.name)
    os.remove(tmp.name)
    
    size_kb = size_bytes / 1024.0
    size_mb = size_kb / 1024.0
    return size_kb, size_mb

def get_flops_and_macs(model, input_size=(1, 3, 64, 64)):
    """
    Returns MACs and FLOPs for a given input size. 
    Requires torchinfo to be installed (`pip install torchinfo` or `uv add torchinfo`).
    """
    if summary is None:
        return "N/A", "N/A"
        
    model_stats = summary(model, input_size=input_size, verbose=0)
    macs = model_stats.total_mult_adds
    
    # FLOPs is conventionally approximated as 2x MACs for standard convolutions and linear layers
    flops = macs * 2 
    return macs, flops
