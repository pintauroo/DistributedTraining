import sys
import os
import argparse
import datetime
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def parameter_server(rank, size, num_batches, device, backend):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    world_size_workers = size - 1

    param_shapes = [p.data.shape for p in model.parameters()]
    param_flat = flatten_tensors([p.data for p in model.parameters()]).to(device)

    dist.barrier(device_ids=[device.index])

    for _ in range(num_batches):
        # Send parameters
        for worker_rank in range(1, size):
            dist.send(tensor=param_flat, dst=worker_rank)

        # Receive and aggregate gradients
        grad_flat = torch.zeros_like(param_flat, device=device)
        for worker_rank in range(1, size):
            worker_grad_flat = torch.zeros_like(param_flat, device=device)
            dist.recv(tensor=worker_grad_flat, src=worker_rank)
            grad_flat += worker_grad_flat

        grad_flat /= world_size_workers

        # Apply gradients
        unflattened_grads = unflatten_tensors(grad_flat, param_shapes)
        for param, grad in zip(model.parameters(), unflattened_grads):
            param.grad = grad
        optimizer.step()
        optimizer.zero_grad()

        param_flat = flatten_tensors([p.data for p in model.parameters()]).to(device)

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

def worker(rank, size, num_batches, batch_size, device, backend):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=size - 1, rank=rank - 1, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=(device.type=='cuda'))
    data_iter = iter(data_loader)

    model = Net().to(device)
    param_shapes = [p.data.shape for p in model.parameters()]
    total_params = sum(p.numel() for p in model.parameters())
    param_flat = torch.zeros(total_params, device=device)

    dist.barrier(device_ids=[device.index])

    for _ in range(num_batches):
        dist.recv(tensor=param_flat, src=0)

        # Set parameters
        unflattened_params = unflatten_tensors(param_flat, param_shapes)
        for param, new_data in zip(model.parameters(), unflattened_params):
            param.data.copy_(new_data)

        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data, target = next(data_iter)

        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        grad_tensors = [param.grad for param in model.parameters()]
        grad_flat = flatten_tensors(grad_tensors).to(device)
        dist.send(tensor=grad_flat, dst=0)

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

def run(rank, size, num_batches, batch_size):
    # Set device
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    backend = 'nccl' if device.type == 'cuda' else 'gloo'

    if rank == 0:
        parameter_server(rank, size, num_batches, device, backend)
    else:
        worker(rank, size, num_batches, batch_size, device, backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Distributed Parameter Server Example")
    parser.add_argument("num_batches", type=int, help="Number of batches to process")
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    run(rank, world_size, args.num_batches, args.batch_size)
