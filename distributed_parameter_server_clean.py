import sys
import os
import argparse
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import datetime
import time
import json
import csv
import psutil  # For CPU and Memory metrics
import torch.cuda as cuda

# Define utility functions to flatten and unflatten tensors
def flatten_tensors(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten_tensors(flat_tensor, shapes):
    outputs = []
    offset = 0
    for shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        outputs.append(flat_tensor[offset : offset + numel].view(shape))
        offset += numel
    return outputs

# Define the neural network model (CNN for MNIST)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)                            # Output: 64x14x14
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))      # Apply first convolution and ReLU
        x = self.relu(self.conv2(x))      # Apply second convolution and ReLU
        x = self.pool(x)                  # Apply max pooling
        x = x.view(-1, 64 * 14 * 14)      # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))  # Fully connected layer with dropout
        x = self.fc2(x)                   # Output layer
        return x

# -------------------- Parameter Server Function --------------------
def parameter_server(rank, size, num_batches, device, backend: str):
    # Initialize process group
    dist.init_process_group(
        backend=backend,  # Use NCCL if GPU, Gloo if CPU
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    world_size_workers = size - 1  # Exclude parameter server

    # Prepare flattened parameter tensor and shapes
    param_tensors = [param.data for param in model.parameters()]
    param_shapes = [param.data.shape for param in model.parameters()]
    param_flat = flatten_tensors(param_tensors).to(device)

    # Synchronize all processes
    dist.barrier()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        batch_metrics = {'batch': batch_idx + 1}

        # Collect system metrics for PS
        ps_cpu_percent = psutil.cpu_percent()
        ps_memory_info = psutil.virtual_memory()
        batch_metrics['cpu_percent'] = ps_cpu_percent
        batch_metrics['memory_total'] = ps_memory_info.total
        batch_metrics['memory_available'] = ps_memory_info.available
        batch_metrics['memory_used'] = ps_memory_info.used
        batch_metrics['memory_percent'] = ps_memory_info.percent

        # If using GPU, collect GPU memory metrics
        if device.type == 'cuda':
            batch_metrics['gpu_memory_allocated'] = cuda.memory_allocated(device)
            batch_metrics['gpu_memory_reserved'] = cuda.memory_reserved(device)
            batch_metrics['gpu_memory_max_allocated'] = cuda.max_memory_allocated(device)
            batch_metrics['gpu_memory_max_reserved'] = cuda.max_memory_reserved(device)

        # Send parameters to all workers
        send_params_start = time.time()
        for worker_rank in range(1, size):
            dist.send(tensor=param_flat, dst=worker_rank)
        send_params_end = time.time()
        batch_metrics['send_params_time'] = send_params_end - send_params_start

        # Initialize a tensor to accumulate gradients
        init_grad_acc_start = time.time()
        grad_flat = torch.zeros_like(param_flat, device=device)
        init_grad_acc_end = time.time()
        batch_metrics['init_grad_accumulator_time'] = init_grad_acc_end - init_grad_acc_start

        # Receive gradients from all workers and sum them
        recv_grads_start = time.time()
        for worker_rank in range(1, size):
            worker_grad_flat = torch.zeros_like(param_flat, device=device)
            dist.recv(tensor=worker_grad_flat, src=worker_rank)
            grad_flat += worker_grad_flat
        recv_grads_end = time.time()
        batch_metrics['receive_gradients_time'] = recv_grads_end - recv_grads_start

        # Average gradients
        avg_grads_start = time.time()
        grad_flat /= world_size_workers
        avg_grads_end = time.time()
        batch_metrics['average_gradients_time'] = avg_grads_end - avg_grads_start

        # Unflatten gradients and set them in the model
        unflatten_grads_start = time.time()
        unflattened_grads = unflatten_tensors(grad_flat, param_shapes)
        for param, grad in zip(model.parameters(), unflattened_grads):
            param.grad = grad.clone()
        unflatten_grads_end = time.time()
        batch_metrics['unflatten_grads_time'] = unflatten_grads_end - unflatten_grads_start

        # Update parameters
        optimizer_step_start = time.time()
        optimizer.step()
        optimizer_step_end = time.time()
        batch_metrics['optimizer_step_time'] = optimizer_step_end - optimizer_step_start

        # Zero optimizer gradients
        zero_grad_start = time.time()
        optimizer.zero_grad()
        zero_grad_end = time.time()
        batch_metrics['optimizer_zero_grad_time'] = zero_grad_end - zero_grad_start

        # Update flattened parameter tensor with new parameters
        update_flat_params_start = time.time()
        param_tensors = [param.data for param in model.parameters()]
        param_flat = flatten_tensors(param_tensors).to(device)
        update_flat_params_end = time.time()
        batch_metrics['update_flattened_params_time'] = update_flat_params_end - update_flat_params_start

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_metrics['batch_duration'] = batch_duration

    # Destroy process group
    dist.barrier()
    dist.destroy_process_group()

# ------------------------ Worker Function ------------------------
def worker(rank, size, num_batches, batch_size, device, backend: str):
    # Initialize process group
    dist.init_process_group(
        backend=backend,  # Use NCCL if GPU, Gloo if CPU
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # Set up data loader with DistributedSampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Ensure dataset is pre-downloaded to avoid delays
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=size - 1, rank=rank - 1, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True if device.type == 'cuda' else False)
    data_iter = iter(data_loader)

    # Initialize model
    model = Net().to(device)

    # Prepare flattened parameter tensor and shapes
    param_tensors = [param.data for param in model.parameters()]
    param_shapes = [param.data.shape for param in model.parameters()]
    total_params = sum(p.numel() for p in model.parameters())
    param_flat = torch.zeros(total_params, device=device)

    # Synchronize all processes
    dist.barrier()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        batch_metrics = {'batch': batch_idx + 1}

        # Collect system metrics for Worker
        worker_cpu_percent = psutil.cpu_percent()
        worker_memory_info = psutil.virtual_memory()
        batch_metrics['cpu_percent'] = worker_cpu_percent
        batch_metrics['memory_total'] = worker_memory_info.total
        batch_metrics['memory_available'] = worker_memory_info.available
        batch_metrics['memory_used'] = worker_memory_info.used
        batch_metrics['memory_percent'] = worker_memory_info.percent

        # If using GPU, collect GPU memory metrics
        if device.type == 'cuda':
            batch_metrics['gpu_memory_allocated'] = cuda.memory_allocated(device)
            batch_metrics['gpu_memory_reserved'] = cuda.memory_reserved(device)
            batch_metrics['gpu_memory_max_allocated'] = cuda.max_memory_allocated(device)
            batch_metrics['gpu_memory_max_reserved'] = cuda.max_memory_reserved(device)

        # Receive parameters from parameter server
        recv_params_start = time.time()
        dist.recv(tensor=param_flat, src=0)
        recv_params_end = time.time()
        batch_metrics['receive_params_time'] = recv_params_end - recv_params_start

        # Unflatten parameters and set them in the model
        unflatten_params_start = time.time()
        unflattened_params = unflatten_tensors(param_flat, param_shapes)
        for param, new_data in zip(model.parameters(), unflattened_params):
            param.data.copy_(new_data)
        unflatten_params_end = time.time()
        batch_metrics['unflatten_params_time'] = unflatten_params_end - unflatten_params_start

        # Get next batch of data
        data_loading_start = time.time()
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data, target = next(data_iter)
        data_loading_end = time.time()
        batch_metrics['data_loading_time'] = data_loading_end - data_loading_start

        # Move data to device
        data_to_device_start = time.time()
        data, target = data.to(device), target.to(device)
        data_to_device_end = time.time()
        batch_metrics['data_to_device_time'] = data_to_device_end - data_to_device_start

        # Zero gradients
        zero_grad_start = time.time()
        model.zero_grad()
        zero_grad_end = time.time()
        batch_metrics['zero_grad_time'] = zero_grad_end - zero_grad_start

        # Forward pass
        forward_pass_start = time.time()
        output = model(data)
        forward_pass_end = time.time()
        batch_metrics['forward_pass_time'] = forward_pass_end - forward_pass_start

        # Compute loss
        compute_loss_start = time.time()
        loss = nn.functional.cross_entropy(output, target)
        loss_value = loss.item()
        compute_loss_end = time.time()
        batch_metrics['compute_loss_time'] = compute_loss_end - compute_loss_start
        batch_metrics['loss'] = loss_value

        # Backward pass
        backward_pass_start = time.time()
        loss.backward()
        backward_pass_end = time.time()
        batch_metrics['backward_pass_time'] = backward_pass_end - backward_pass_start

        # Flatten gradients
        flatten_grad_start = time.time()
        grad_tensors = [param.grad for param in model.parameters()]
        grad_flat = flatten_tensors(grad_tensors).to(device)
        flatten_grad_end = time.time()
        batch_metrics['flatten_grad_time'] = flatten_grad_end - flatten_grad_start

        # Send gradients to parameter server
        send_grads_start = time.time()
        dist.send(tensor=grad_flat, dst=0)
        send_grads_end = time.time()
        batch_metrics['send_gradients_time'] = send_grads_end - send_grads_start

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_metrics['batch_duration'] = batch_duration

    # Destroy process group
    dist.barrier()
    dist.destroy_process_group()

# ------------------------- Run Function -------------------------
def run(rank, size, num_batches, batch_size, device, backend: str):
    if rank == 0:
        parameter_server(rank, size, num_batches, device, backend)
    else:
        worker(rank, size, num_batches, batch_size, device, backend)

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Parameter Server Training")
    parser.add_argument("num_batches", type=int, help="Number of batches to process")
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None,
                        help="Device to use: 'cpu' or 'gpu'. If not specified, defaults to GPU if available, else CPU.")
    args = parser.parse_args()

    # torchrun sets WORLD_SIZE and RANK automatically
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    # Determine device
    if args.device:
        if args.device == "gpu":
            if torch.cuda.is_available():
                # All nodes have one GPU, always use GPU 0
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
            else:
                sys.exit("[Error] GPU requested but not available. Exiting.")
        else:
            device = torch.device("cpu")
    else:
        # Auto-select device: GPU if available, else CPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    backend = 'nccl' if device.type == 'cuda' else 'gloo'

    run(rank, world_size, args.num_batches, args.batch_size, device, backend)
