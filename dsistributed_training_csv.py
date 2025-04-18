import os
import argparse
import datetime
import time
import csv

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

RESULTS_FILE = "results.csv"

def flatten_tensors(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten_tensors(flat_tensor, shapes):
    outputs, offset = [], 0
    for shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        outputs.append(flat_tensor[offset:offset+numel].view(shape))
        offset += numel
    return outputs

class Net(nn.Module):
    # … same as before …

def parameter_server(rank, size, num_batches, device, backend):
    # … same as before …

def worker(rank, size, num_batches, batch_size, device, backend):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=size - 1, rank=rank - 1, shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    data_iter = iter(loader)

    # model setup
    model        = Net().to(device)
    shapes       = [p.data.shape for p in model.parameters()]
    total_params = sum(p.numel() for p in model.parameters())
    param_flat   = torch.zeros(total_params, device=device)

    # prepare CSV header (only once per worker)
    if rank != 0:
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'rank','round','batch_size',
                'recv_bytes','recv_MBps',
                'send_bytes','send_MBps',
                'train_time',
                'loss'
            ])

    dist.barrier(device_ids=[device.index])

    # running totals for summary
    total_recv_bytes = 0
    total_send_bytes = 0
    sum_recv_time    = 0.0
    sum_send_time    = 0.0

    for round_idx in range(num_batches):
        # -- receive parameters --
        recv_start = time.perf_counter()
        dist.recv(tensor=param_flat, src=0)
        recv_end   = time.perf_counter()
        recv_time  = recv_end - recv_start

        # how many bytes?
        recv_bytes = param_flat.numel() * param_flat.element_size()
        recv_MBps  = (recv_bytes / recv_time) / (1024**2)

        total_recv_bytes += recv_bytes
        sum_recv_time    += recv_time

        # set model weights
        for p, new in zip(model.parameters(), unflatten_tensors(param_flat, shapes)):
            p.data.copy_(new)

        # -- training step --
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            data, target = next(data_iter)

        data, target = data.to(device), target.to(device)
        model.zero_grad()
        t0 = time.perf_counter()
        output = model(data)
        loss   = nn.functional.cross_entropy(output, target)
        loss.backward()
        t1 = time.perf_counter()
        train_time = t1 - t0

        # -- send gradients --
        send_start = time.perf_counter()
        grad_flat  = flatten_tensors([p.grad for p in model.parameters()]).to(device)
        dist.send(tensor=grad_flat, dst=0)
        send_end   = time.perf_counter()
        send_time  = send_end - send_start

        send_bytes = grad_flat.numel() * grad_flat.element_size()
        send_MBps  = (send_bytes / send_time) / (1024**2)

        total_send_bytes += send_bytes
        sum_send_time    += send_time

        # -- log this round --
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                rank,
                round_idx,
                batch_size,
                recv_bytes,
                f"{recv_MBps:.2f}",
                send_bytes,
                f"{send_MBps:.2f}",
                f"{train_time:.4f}",
                loss.item()
            ])

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

    # -- write summary (average throughput & totals) --
    avg_recv_MBps = (total_recv_bytes / sum_recv_time) / (1024**2)
    avg_send_MBps = (total_send_bytes / sum_send_time) / (1024**2)
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])  # blank line
        writer.writerow([
            f"worker {rank} SUMMARY",
            '',
            batch_size,
            f"total_recv={total_recv_bytes} bytes",
            f"avg_recv={avg_recv_MBps:.2f} MB/s",
            f"total_send={total_send_bytes} bytes",
            f"avg_send={avg_send_MBps:.2f} MB/s",
            '', ''
        ])

def run(rank, size, num_batches, batch_size):
    # device & backend
    if torch.cuda.is_available():
        dev_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(dev_id)
        device = torch.device(f"cuda:{dev_id}")
    else:
        device = torch.device("cpu")
    backend = 'nccl' if device.type=='cuda' else 'gloo'

    if rank == 0:
        parameter_server(rank, size, num_batches, device, backend)
    else:
        worker(rank, size, num_batches, batch_size, device, backend)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("num_batches", type=int)
    p.add_argument("batch_size",  type=int)
    args = p.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank       = int(os.environ.get('RANK',       0))

    run(rank, world_size, args.num_batches, args.batch_size)
