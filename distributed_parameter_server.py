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
import matplotlib.pyplot as plt
import pandas as pd
import psutil  # For CPU and Memory metrics
import torch.cuda as cuda

# ------------------------ Logger Class ------------------------
class Logger:
    """
    Logger class to handle logging based on verbosity flag.
    """
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

# --------------------- DataCollector Class ---------------------
class DataCollector:
    """
    DataCollector class to collect and store training metrics.
    """
    def __init__(self, role: str):
        self.role = role  # 'PS' or 'Worker'
        self.data = []

    def record(self, **kwargs):
        self.data.append(kwargs)

    def get_data(self):
        return self.data

    def serialize(self):
        """
        Serialize the collected data to a JSON string.
        """
        return json.dumps(self.data)

    @staticmethod
    def deserialize(json_str: str):
        """
        Deserialize JSON string back to data list.
        """
        return json.loads(json_str)

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
def parameter_server(rank, size, num_batches, device, logger: Logger, collector: DataCollector, backend: str):
    logger.log(f"[Parameter Server] Initializing on device {device} with backend {backend}.")

    # Initialize process group
    dist.init_process_group(
        backend=backend,  # Use NCCL if GPU, Gloo if CPU
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )
    logger.log(f"[Parameter Server] Process group initialized.")

    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    world_size_workers = size - 1  # Exclude parameter server

    # Prepare flattened parameter tensor and shapes
    param_tensors = [param.data for param in model.parameters()]
    param_shapes = [param.data.shape for param in model.parameters()]
    param_flat = flatten_tensors(param_tensors).to(device)
    logger.log(f"[Parameter Server] Flattened parameters size: {param_flat.size()}")

    # Synchronize all processes
    dist.barrier()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        logger.log(f"[Parameter Server] Starting batch {batch_idx + 1}/{num_batches}")
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
            logger.log(f"[Parameter Server] Sending parameters to worker {worker_rank}")
            dist.send(tensor=param_flat, dst=worker_rank)
            logger.log(f"[Parameter Server] Sent parameters to worker {worker_rank}")
        send_params_end = time.time()
        batch_metrics['send_params_time'] = send_params_end - send_params_start

        # Initialize a tensor to accumulate gradients
        init_grad_acc_start = time.time()
        grad_flat = torch.zeros_like(param_flat, device=device)
        init_grad_acc_end = time.time()
        batch_metrics['init_grad_accumulator_time'] = init_grad_acc_end - init_grad_acc_start
        logger.log(f"[Parameter Server] Initialized gradient accumulator.")

        # Receive gradients from all workers and sum them
        recv_grads_start = time.time()
        for worker_rank in range(1, size):
            logger.log(f"[Parameter Server] Receiving gradients from worker {worker_rank}")
            worker_grad_flat = torch.zeros_like(param_flat, device=device)
            dist.recv(tensor=worker_grad_flat, src=worker_rank)
            logger.log(f"[Parameter Server] Received gradients from worker {worker_rank}")
            grad_flat += worker_grad_flat
            logger.log(f"[Parameter Server] Accumulated gradients. Current grad_flat sum: {grad_flat.sum().item()}")
        recv_grads_end = time.time()
        batch_metrics['receive_gradients_time'] = recv_grads_end - recv_grads_start

        # Average gradients
        avg_grads_start = time.time()
        grad_flat /= world_size_workers
        avg_grads_end = time.time()
        batch_metrics['average_gradients_time'] = avg_grads_end - avg_grads_start
        logger.log(f"[Parameter Server] Averaged gradients. grad_flat mean: {grad_flat.mean().item()}")

        # Unflatten gradients and set them in the model
        unflatten_grads_start = time.time()
        unflattened_grads = unflatten_tensors(grad_flat, param_shapes)
        for param, grad in zip(model.parameters(), unflattened_grads):
            param.grad = grad.clone()
            logger.log(f"[Parameter Server] Set gradient for parameter {param.shape}")
        unflatten_grads_end = time.time()
        batch_metrics['unflatten_grads_time'] = unflatten_grads_end - unflatten_grads_start

        # Update parameters
        optimizer_step_start = time.time()
        optimizer.step()
        optimizer_step_end = time.time()
        batch_metrics['optimizer_step_time'] = optimizer_step_end - optimizer_step_start
        logger.log(f"[Parameter Server] Updated model parameters.")

        # Zero optimizer gradients
        zero_grad_start = time.time()
        optimizer.zero_grad()
        zero_grad_end = time.time()
        batch_metrics['optimizer_zero_grad_time'] = zero_grad_end - zero_grad_start
        logger.log(f"[Parameter Server] Zeroed optimizer gradients.")

        # Update flattened parameter tensor with new parameters
        update_flat_params_start = time.time()
        param_tensors = [param.data for param in model.parameters()]
        param_flat = flatten_tensors(param_tensors).to(device)
        update_flat_params_end = time.time()
        batch_metrics['update_flattened_params_time'] = update_flat_params_end - update_flat_params_start
        logger.log(f"[Parameter Server] Updated flattened parameters for next batch.")

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_metrics['batch_duration'] = batch_duration

        # Record batch metrics
        collector.record(**batch_metrics)

        logger.log(f"[Parameter Server] Batch {batch_idx + 1} completed.\n")

    logger.log(f"[Parameter Server] Training completed. Synchronizing before shutdown.")

    # Collect data from workers
    worker_data_collected = []
    for worker_rank in range(1, size):
        logger.log(f"[Parameter Server] Receiving collected data from worker {worker_rank}")
        # First receive the length of the JSON string
        length_tensor = torch.zeros(1, dtype=torch.int32, device=device)  # On GPU
        dist.recv(tensor=length_tensor, src=worker_rank)
        data_length = length_tensor.item()
        # Now receive the actual data
        data_tensor = torch.zeros(data_length, dtype=torch.uint8, device=device)  # On GPU
        dist.recv(tensor=data_tensor, src=worker_rank)
        json_str = data_tensor.cpu().numpy().tobytes().decode('utf-8')  # Move to CPU for processing
        worker_data = DataCollector.deserialize(json_str)
        # Add worker rank to each entry
        for entry in worker_data:
            entry['Role'] = 'Worker'
            entry['Rank'] = worker_rank
        worker_data_collected.extend(worker_data)
        logger.log(f"[Parameter Server] Received data from Worker {worker_rank}")

    # Combine PS data and worker data
    ps_data = collector.get_data()
    for entry in ps_data:
        entry['Role'] = 'PS'
        entry['Rank'] = rank  # PS rank is 0

    all_data = ps_data + worker_data_collected

    # Get all fieldnames
    fieldnames = set()
    for entry in all_data:
        fieldnames.update(entry.keys())

    fieldnames = sorted(fieldnames)

    csv_filename = 'training_metrics.csv'
    logger.log(f"[Parameter Server] Saving collected data to {csv_filename}")

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_data:
            writer.writerow(entry)

    logger.log(f"[Parameter Server] Data saved successfully.")

    # Plot metrics
    plot_metrics(csv_filename, logger)

    # Destroy process group
    dist.barrier()
    dist.destroy_process_group()
    logger.log(f"[Parameter Server] Process group destroyed.")

# ------------------------ Worker Function ------------------------
def worker(rank, size, num_batches, batch_size, device, logger: Logger, collector: DataCollector, backend: str):
    logger.log(f"[Worker {rank}] Initializing on device {device} with backend {backend}.")

    # Initialize process group
    dist.init_process_group(
        backend=backend,  # Use NCCL if GPU, Gloo if CPU
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )
    logger.log(f"[Worker {rank}] Process group initialized.")

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
    logger.log(f"[Worker {rank}] Data loader initialized with batch size {batch_size}.")

    # Initialize model
    model = Net().to(device)
    logger.log(f"[Worker {rank}] Model initialized and moved to device {device}.")

    # Prepare flattened parameter tensor and shapes
    param_tensors = [param.data for param in model.parameters()]
    param_shapes = [param.data.shape for param in model.parameters()]
    total_params = sum(p.numel() for p in model.parameters())
    param_flat = torch.zeros(total_params, device=device)
    logger.log(f"[Worker {rank}] Initialized parameter tensor of size {param_flat.size()}.")

    # Initialize loss collector
    loss_list = []

    # Synchronize all processes
    dist.barrier()

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        logger.log(f"[Worker {rank}] Starting batch {batch_idx + 1}/{num_batches}")
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
        logger.log(f"[Worker {rank}] Receiving parameters from parameter server.")
        dist.recv(tensor=param_flat, src=0)
        logger.log(f"[Worker {rank}] Received parameters. Parameter tensor size: {param_flat.size()}")
        recv_params_end = time.time()
        batch_metrics['receive_params_time'] = recv_params_end - recv_params_start

        # Unflatten parameters and set them in the model
        unflatten_params_start = time.time()
        unflattened_params = unflatten_tensors(param_flat, param_shapes)
        for param, new_data in zip(model.parameters(), unflattened_params):
            param.data.copy_(new_data)
            logger.log(f"[Worker {rank}] Updated parameter {param.shape} with received data.")
        unflatten_params_end = time.time()
        batch_metrics['unflatten_params_time'] = unflatten_params_end - unflatten_params_start

        # Get next batch of data
        data_loading_start = time.time()
        try:
            data, target = next(data_iter)
            logger.log(f"[Worker {rank}] Fetched data batch.")
        except StopIteration:
            logger.log(f"[Worker {rank}] Reinitializing data iterator.")
            data_iter = iter(data_loader)
            data, target = next(data_iter)
            logger.log(f"[Worker {rank}] Fetched data batch after reinitializing iterator.")
        data_loading_end = time.time()
        batch_metrics['data_loading_time'] = data_loading_end - data_loading_start

        # Move data to device
        data_to_device_start = time.time()
        data, target = data.to(device), target.to(device)
        logger.log(f"[Worker {rank}] Moved data to device {device}.")
        data_to_device_end = time.time()
        batch_metrics['data_to_device_time'] = data_to_device_end - data_to_device_start

        # Zero gradients
        zero_grad_start = time.time()
        model.zero_grad()
        logger.log(f"[Worker {rank}] Zeroed model gradients.")
        zero_grad_end = time.time()
        batch_metrics['zero_grad_time'] = zero_grad_end - zero_grad_start

        # Forward pass
        forward_pass_start = time.time()
        output = model(data)
        logger.log(f"[Worker {rank}] Completed forward pass.")
        forward_pass_end = time.time()
        batch_metrics['forward_pass_time'] = forward_pass_end - forward_pass_start

        # Compute loss
        compute_loss_start = time.time()
        loss = nn.functional.cross_entropy(output, target)
        loss_value = loss.item()
        logger.log(f"[Worker {rank}] Computed loss: {loss_value}.")
        compute_loss_end = time.time()
        batch_metrics['compute_loss_time'] = compute_loss_end - compute_loss_start
        batch_metrics['loss'] = loss_value

        # Backward pass
        backward_pass_start = time.time()
        loss.backward()
        logger.log(f"[Worker {rank}] Completed backward pass.")
        backward_pass_end = time.time()
        batch_metrics['backward_pass_time'] = backward_pass_end - backward_pass_start

        # Flatten gradients
        flatten_grad_start = time.time()
        grad_tensors = [param.grad for param in model.parameters()]
        grad_flat = flatten_tensors(grad_tensors).to(device)
        logger.log(f"[Worker {rank}] Flattened gradients. Gradient tensor size: {grad_flat.size()}.")
        flatten_grad_end = time.time()
        batch_metrics['flatten_grad_time'] = flatten_grad_end - flatten_grad_start

        # Send gradients to parameter server
        send_grads_start = time.time()
        logger.log(f"[Worker {rank}] Sending gradients to parameter server.")
        dist.send(tensor=grad_flat, dst=0)
        logger.log(f"[Worker {rank}] Sent gradients to parameter server.")
        send_grads_end = time.time()
        batch_metrics['send_gradients_time'] = send_grads_end - send_grads_start

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_metrics['batch_duration'] = batch_duration

        # Calculate throughput (samples per second)
        samples_processed = data.size(0)
        throughput = samples_processed / batch_duration if batch_duration > 0 else 0
        batch_metrics['throughput_samples_per_sec'] = throughput

        # Record batch metrics
        collector.record(**batch_metrics)

        logger.log(f"[Worker {rank}] Batch {batch_idx + 1} completed.\n")

    logger.log(f"[Worker {rank}] Training completed. Sending collected data to parameter server.")

    # Serialize collected data
    collected_data = collector.serialize()
    data_bytes = collected_data.encode('utf-8')
    data_length = len(data_bytes)
    data_tensor = torch.ByteTensor(list(data_bytes)).to(device)  # Move to GPU
    length_tensor = torch.tensor([data_length], dtype=torch.int32).to(device)  # Move to GPU

    # Send length first
    dist.send(tensor=length_tensor, dst=0)
    # Then send the actual data
    dist.send(tensor=data_tensor, dst=0)

    logger.log(f"[Worker {rank}] Data sent to parameter server.")

    # Destroy process group
    dist.barrier()
    dist.destroy_process_group()
    logger.log(f"[Worker {rank}] Process group destroyed.")

# ----------------------- Plotting Function -----------------------
def plot_metrics(csv_filename, logger: Logger):
    df = pd.read_csv(csv_filename)

    # Separate data for PS and Workers
    ps_data = df[df['Role'] == 'PS']
    worker_data = df[df['Role'] == 'Worker']

    # Define metrics for PS and Workers
    ps_metrics = ['send_params_time', 'init_grad_accumulator_time', 'receive_gradients_time',
                  'average_gradients_time', 'unflatten_grads_time', 'optimizer_step_time',
                  'optimizer_zero_grad_time', 'update_flattened_params_time', 'batch_duration',
                  'cpu_percent', 'memory_total', 'memory_available', 'memory_used', 'memory_percent']
    if 'gpu_memory_allocated' in ps_data.columns:
        ps_metrics += ['gpu_memory_allocated', 'gpu_memory_reserved', 'gpu_memory_max_allocated', 'gpu_memory_max_reserved']

    worker_metrics = ['receive_params_time', 'unflatten_params_time', 'data_loading_time',
                      'data_to_device_time', 'zero_grad_time', 'forward_pass_time',
                      'compute_loss_time', 'backward_pass_time', 'flatten_grad_time',
                      'send_gradients_time', 'batch_duration', 'loss', 'throughput_samples_per_sec',
                      'cpu_percent', 'memory_total', 'memory_available', 'memory_used', 'memory_percent']
    if 'gpu_memory_allocated' in worker_data.columns:
        worker_metrics += ['gpu_memory_allocated', 'gpu_memory_reserved', 'gpu_memory_max_allocated', 'gpu_memory_max_reserved']

    # Plot each PS metric separately
    for metric in ps_metrics:
        if metric in ps_data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(ps_data['batch'], ps_data[metric], marker='o', label='PS')
            plt.xlabel('Batch')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.title(f'Parameter Server - {metric.replace("_", " ").capitalize()} per Batch')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'ps_{metric}.png')
            plt.close()
            logger.log(f"[Plotting] Saved plot ps_{metric}.png")

    # Plot each Worker metric separately
    worker_ranks = worker_data['Rank'].unique()
    for metric in worker_metrics:
        if metric in worker_data.columns:
            plt.figure(figsize=(10, 6))
            for rank in worker_ranks:
                worker_rank_data = worker_data[worker_data['Rank'] == rank]
                plt.plot(worker_rank_data['batch'], worker_rank_data[metric], marker='o', label=f'Worker {rank}')
            plt.xlabel('Batch')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.title(f'Workers - {metric.replace("_", " ").capitalize()} per Batch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'workers_{metric}.png')
            plt.close()
            logger.log(f"[Plotting] Saved plot workers_{metric}.png")

    # Additionally, plot aggregated metrics across all workers
    # Ensure only numeric columns are included and exclude 'batch'
    numeric_cols = worker_data.select_dtypes(include=['number']).columns.tolist()
    if 'batch' in numeric_cols:
        numeric_cols.remove('batch')
    aggregated_worker_data = worker_data.groupby('batch')[numeric_cols].mean().reset_index()

    aggregated_metrics = ['receive_params_time', 'unflatten_params_time', 'data_loading_time',
                          'data_to_device_time', 'zero_grad_time', 'forward_pass_time',
                          'compute_loss_time', 'backward_pass_time', 'flatten_grad_time',
                          'send_gradients_time', 'batch_duration', 'loss', 'throughput_samples_per_sec',
                          'cpu_percent', 'memory_total', 'memory_available', 'memory_used', 'memory_percent']
    if 'gpu_memory_allocated' in aggregated_worker_data.columns:
        aggregated_metrics += ['gpu_memory_allocated', 'gpu_memory_reserved', 'gpu_memory_max_allocated', 'gpu_memory_max_reserved']

    for metric in aggregated_metrics:
        if metric in aggregated_worker_data.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(aggregated_worker_data['batch'], aggregated_worker_data[metric], marker='o', color='purple')
            plt.xlabel('Batch')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.title(f'Aggregated Workers - {metric.replace("_", " ").capitalize()} per Batch')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'aggregated_workers_{metric}.png')
            plt.close()
            logger.log(f"[Plotting] Saved plot aggregated_workers_{metric}.png")

    logger.log("All metrics have been plotted and saved as separate PNG files.")

# ------------------------- Run Function -------------------------
def run(rank, size, num_batches, batch_size, device, verbose: bool):
    logger = Logger(verbose=verbose)
    collector = DataCollector(role='PS' if rank == 0 else 'Worker')

    # Determine backend based on device
    backend = 'nccl' if device.type == 'cuda' else 'gloo'

    if rank == 0:
        # Parameter Server can run on CPU or GPU based on the device argument
        logger.log(f"[Run] Parameter server running on device {device}.")
        parameter_server(rank, size, num_batches, device, logger, collector, backend)
    else:
        # Workers can run on CPU or GPU based on the device argument
        logger.log(f"[Run] Worker {rank} running on device {device}.")
        worker(rank, size, num_batches, batch_size, device, logger, collector, backend)

# ------------------------ Main Execution ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Parameter Server Training")
    parser.add_argument("num_batches", type=int, help="Number of batches to process")
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default=None,
                        help="Device to use: 'cpu' or 'gpu'. If not specified, defaults to GPU if available, else CPU.")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable detailed logging.")
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
                print("[Error] GPU requested but not available. Exiting.")
                sys.exit(1)
        else:
            device = torch.device("cpu")
    else:
        # Auto-select device: GPU if available, else CPU
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            print(f"[Main] GPU detected. Using device {device}.")
        else:
            device = torch.device("cpu")
            print("[Main] GPU not available. Using CPU.")

    print(f"[Main] Starting process with rank {rank} out of {world_size}.")

    run(rank, world_size, args.num_batches, args.batch_size, device, verbose=args.verbose)

    print(f"[Main] Process {rank} finished execution.")
