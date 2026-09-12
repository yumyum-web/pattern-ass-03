import os
import sys
import time
import torch
import torchvision.models as models

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.profiler import get_parameter_count, get_model_size_mb, get_flops_and_macs

def benchmark_model(model_name, model, input_size=(1, 3, 64, 64)):
    print(f"Benchmarking {model_name}...")
    
    # 1. Parameter count
    params = get_parameter_count(model)
    
    # 2. Disk Size
    size_kb, size_mb = get_model_size_mb(model)
    
    # 3. MACs / FLOPs
    macs, flops = get_flops_and_macs(model, input_size)
    
    # 4. Latency / dummy training time test
    # We do a quick dummy forward pass timing as a placeholder for actual training time
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end_time = time.time()
        
    avg_latency_ms = ((end_time - start_time) / 100.0) * 1000
    
    return {
        "Model": model_name,
        "Parameters": f"{params:,}",
        "Size (KB)": f"{size_kb:.2f}",
        "Size (MB)": f"{size_mb:.2f}",
        "MACs": f"{macs:,}" if isinstance(macs, int) else macs,
        "Latency (ms)": f"{avg_latency_ms:.2f}"
    }

if __name__ == "__main__":
    print("--- Starting Edge Architecture Profiler ---\n")
    
    # In the future, you will import Model A, Model B, MobileNet, and ShuffleNet here.
    # For now, we use dummy torchvision models to prove the generic profiler works!
    
    dummy_models = {
        "Dummy_CNN_Small (ResNet18)": models.resnet18(num_classes=9),
        "Dummy_CNN_Tiny (SqueezeNet)": models.squeezenet1_0(num_classes=9)
    }
    
    results = []
    for name, model in dummy_models.items():
        results.append(benchmark_model(name, model))
        
    # Output Markdown Table
    print("\n### Cross-Architecture Comparison (Markdown)\n")
    print("| Model | Parameters | Size (KB) | Size (MB) | MACs | Latency (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['Model']} | {r['Parameters']} | {r['Size (KB)']} | {r['Size (MB)']} | {r['MACs']} | {r['Latency (ms)']} |")
        
    # Output LaTeX Tabular Block
    print("\n### LaTeX Tabular Block\n")
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{|l|r|r|r|r|r|}")
    print("\\hline")
    print("Model & Parameters & Size (KB) & Size (MB) & MACs & Latency (ms) \\\\ \\hline")
    for r in results:
        print(f"{r['Model']} & {r['Parameters']} & {r['Size (KB)']} & {r['Size (MB)']} & {r['MACs']} & {r['Latency (ms)']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Hardware Profiling Comparison}")
    print("\\label{tab:profiling}")
    print("\\end{table}\n")
    print("Benchmarking Complete! When Model A and Model B are pushed, simply swap them into the `dummy_models` dictionary.")
