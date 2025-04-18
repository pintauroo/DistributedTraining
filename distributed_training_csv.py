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
        self.conv1    = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool     = nn.MaxPool2d(2, 2)
        self.fc1      = nn.Linear(64 * 14 * 14, 128)
        self.fc2      = nn.Linear(128, 10)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def log_row(row):
    with open(RESULTS_FILE, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def parameter_server(rank, size, num_batches, device, backend):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # write header once
    if rank == 0:
        with open(RESULTS_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'rank','round','phase',
                'ts_start','ts_end','duration_s',
                'bytes','throughput_MBps','loss'
            ])

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    workers = size - 1

    shapes     = [p.data.shape for p in model.parameters()]
    param_flat = flatten_tensors([p.data for p in model.parameters()]).to(device)

    dist.barrier(device_ids=[device.index])

    for rnd in range(num_batches):
        # PS → workers
        ts0 = datetime.datetime.now().isoformat()
        t0  = time.perf_counter()
        for w in range(1, size):
            dist.send(tensor=param_flat, dst=w)
        t1  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        bytes_sent = param_flat.numel() * param_flat.element_size() * workers
        thr_send   = (bytes_sent / (t1 - t0)) / (1024**2)
        log_row([0, rnd, 'ps_send', ts0, ts1, f"{t1-t0:.6f}", bytes_sent, f"{thr_send:.2f}", ''])

        # workers → PS
        ts0 = datetime.datetime.now().isoformat()
        t2  = time.perf_counter()
        agg_flat = torch.zeros_like(param_flat, device=device)
        for w in range(1, size):
            buf = torch.zeros_like(param_flat, device=device)
            dist.recv(tensor=buf, src=w)
            agg_flat += buf
        t3  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        bytes_recv = param_flat.numel() * param_flat.element_size() * workers
        thr_recv   = (bytes_recv / (t3 - t2)) / (1024**2)
        log_row([0, rnd, 'ps_recv', ts0, ts1, f"{t3-t2:.6f}", bytes_recv, f"{thr_recv:.2f}", ''])

        # update
        ts0 = datetime.datetime.now().isoformat()
        t4  = time.perf_counter()
        agg_flat /= workers
        grads = unflatten_tensors(agg_flat, shapes)
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        optimizer.step()
        optimizer.zero_grad()
        param_flat = flatten_tensors([p.data for p in model.parameters()]).to(device)
        t5  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        log_row([0, rnd, 'ps_update', ts0, ts1, f"{t5-t4:.6f}", '', '', ''])

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

    # dataset & loader
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST('./data', train=True, download=True, transform=tf)
    sampler = DistributedSampler(ds, num_replicas=size-1, rank=rank-1, shuffle=True)
    loader  = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    it = iter(loader)

    model        = Net().to(device)
    shapes       = [p.data.shape for p in model.parameters()]
    total_params = sum(p.numel() for p in model.parameters())
    param_flat   = torch.zeros(total_params, device=device)

    dist.barrier(device_ids=[device.index])

    for rnd in range(num_batches):
        # PS → worker
        ts0 = datetime.datetime.now().isoformat()
        t0  = time.perf_counter()
        dist.recv(tensor=param_flat, src=0)
        t1  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        bytes_recv = param_flat.numel() * param_flat.element_size()
        thr_recv   = (bytes_recv / (t1 - t0)) / (1024**2)
        log_row([rank, rnd, 'worker_recv', ts0, ts1, f"{t1-t0:.6f}", bytes_recv, f"{thr_recv:.2f}", ''])

        # load into model
        new_ws = unflatten_tensors(param_flat, shapes)
        for p, w in zip(model.parameters(), new_ws):
            p.data.copy_(w)

        # training
        try:
            data, target = next(it)
        except StopIteration:
            it = iter(loader)
            data, target = next(it)
        data, target = data.to(device), target.to(device)

        ts0 = datetime.datetime.now().isoformat()
        t2  = time.perf_counter()
        out  = model(data)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        t3  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        log_row([rank, rnd, 'worker_train', ts0, ts1, f"{t3-t2:.6f}", '', '', f"{loss.item():.6f}"])

        # worker → PS
        grad_flat = flatten_tensors([p.grad for p in model.parameters()]).to(device)
        ts0 = datetime.datetime.now().isoformat()
        t4  = time.perf_counter()
        dist.send(tensor=grad_flat, dst=0)
        t5  = time.perf_counter()
        ts1 = datetime.datetime.now().isoformat()
        bytes_sent = grad_flat.numel() * grad_flat.element_size()
        thr_send   = (bytes_sent / (t5 - t4)) / (1024**2)
        log_row([rank, rnd, 'worker_send', ts0, ts1, f"{t5-t4:.6f}", bytes_sent, f"{thr_send:.2f}", ''])

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

def run(rank, size, num_batches, batch_size):
    if torch.cuda.is_available():
        dev = rank % torch.cuda.device_count()
        torch.cuda.set_device(dev)
        device = torch.device(f"cuda:{dev}")
    else:
        device = torch.device("cpu")
    backend = 'nccl' if device.type=='cuda' else 'gloo'

    if rank == 0:
        parameter_server(rank, size, num_batches, device, backend)
    else:
        worker(rank, size, num_batches, batch_size, device, backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_batches", type=int, help="# of global rounds")
    parser.add_argument("batch_size",  type=int, help="mini‑batch size")
    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank       = int(os.environ.get('RANK',       0))

    run(rank, world_size, args.num_batches, args.batch_size)
