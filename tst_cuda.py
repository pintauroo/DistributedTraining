import torch

def single_gpu_test():
    print("===== Single GPU Test =====")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return
    
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # List all GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set device to first GPU
    device = torch.device("cuda:0")
    print(f"\nUsing device: {device}")
    
    # Create a tensor on CPU
    tensor_cpu = torch.randn(3, 3)
    print(f"Tensor on CPU:\n{tensor_cpu}")
    
    # Move tensor to GPU
    tensor_gpu = tensor_cpu.to(device)
    print(f"Tensor on GPU:\n{tensor_gpu}")
    
    # Perform a simple operation on GPU
    tensor_gpu = tensor_gpu * 2
    print(f"Tensor after multiplication on GPU:\n{tensor_gpu}")
    
    # Move tensor back to CPU
    tensor_cpu = tensor_gpu.to("cpu")
    print(f"Tensor moved back to CPU:\n{tensor_cpu}")

if __name__ == "__main__":
    single_gpu_test()

