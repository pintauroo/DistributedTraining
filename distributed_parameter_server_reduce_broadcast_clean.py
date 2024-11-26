import sys
import os
import argparse
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import time
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
        timeout=torch.distributed.distributed_c10d.TIMEOUT
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

        # Broadcast parameters to all workers
        dist.broadcast(tensor=param_flat, src=0)

        # Initialize a tensor to accumulate gradients
        grad_flat = torch.zeros_like(param_flat, device=device)

        # Reduce gradients from workers
        dist.reduce(tensor=grad_flat, dst=0, op=dist.ReduceOp.SUM)

        # Average gradients
        grad_flat /= world_size_workers

        # Unflatten gradients and set them in the model
        unflattened_grads = unflatten_tensors(grad_flat, param_shapes)
        for param, grad in zip(model.parameters(), unflattened_grads):
            param.grad = grad.clone()

        # Update parameters
        optimizer.step()

        # Zero optimizer gradients
        optimizer.zero_grad()

        # Update flattened parameter tensor with new parameters
        param_tensors = [param.data for param in model.parameters()]
        param_flat = flatten_tensors(param_tensors).to(device)

        batch_end_time = time.time()

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
        timeout=torch.distributed.distributed_c10d.TIMEOUT
    )

    # Set up data loader with DistributedSampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
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

        # Receive parameters from parameter server via broadcast
        dist.broadcast(tensor=param_flat, src=0)

        # Unflatten parameters and set them in the model
        unflattened_params = unflatten_tensors(param_flat, param_shapes)
        for param, new_data in zip(model.parameters(), unflattened_params):
            param.data.copy_(new_data)

        # Get next batch of data
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data, target = next(data_iter)

        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = nn.functional.cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Flatten gradients
        grad_tensors = [param.grad for param in model.parameters()]
        grad_flat = flatten_tensors(grad_tensors).to(device)

        # Reduce gradients to parameter server
        dist.reduce(tensor=grad_flat, dst=0, op=dist.ReduceOp.SUM)

        batch_end_time = time.time()

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
    parser = argparse.ArgumentParser(description="Distributed Parameter Server Training with Broadcast and Reduce")
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
                # Assign specific GPU based on rank to avoid conflicts
                device_id = rank % torch.cuda.device_count()
                device = torch.device(f"cuda:{device_id}")
            else:
                sys.exit(1)
        else:
            device = torch.device("cpu")
    else:
        # Auto-select device: GPU if available, else CPU
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

    run(rank, world_size, args.num_batches, args.batch_size, device, backend='nccl' if device.type == 'cuda' else 'gloo')
