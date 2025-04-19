#!/usr/bin/env python3
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

PS_RESULTS   = "ps_results.csv"
RING_RESULTS = "ring_results.csv"

def flatten_tensors(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def unflatten_tensors(flat_tensor, shapes):
    outputs = []
    offset  = 0
    for shape in shapes:
        numel = 1
        for d in shape:
            numel *= d
        outputs.append(flat_tensor[offset:offset+numel].view(shape))
        offset += numel
    return outputs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1    = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool     = nn.MaxPool2d(2, 2)
        self.fc1      = nn.Linear(64*14*14, 128)
        self.fc2      = nn.Linear(128, 10)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*14*14)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

def log_row(fn, row):
    with open(fn, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def run_ps(rank, world_size, rounds, batch_size, device, backend):
    # initialize
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # write header
    if rank == 0:
        with open(PS_RESULTS, 'w', newline='') as f:
            csv.writer(f).writerow([
                'rank','round','phase',
                'ts_start','ts_end','duration_s',
                'bytes','throughput_MBps','loss'
            ])

    # common setup
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    shapes = [p.data.shape for p in model.parameters()]
    flat_params = flatten_tensors([p.data for p in model.parameters()]).to(device)
    workers = world_size - 1

    # data for workers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=workers, rank=rank-1 if rank>0 else 0, shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    data_iter = iter(loader)

    dist.barrier(device_ids=[device.index])

    for rnd in range(rounds):
        if rank == 0:
            # PS → workers
            ts0 = datetime.datetime.now().isoformat(); t0 = time.perf_counter()
            for w in range(1, world_size):
                dist.send(tensor=flat_params, dst=w)
            t1 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            sent_bytes = flat_params.numel()*flat_params.element_size()*workers
            thr = (sent_bytes/(t1-t0))/(1024**2)
            log_row(PS_RESULTS, [0, rnd, 'ps_send', ts0, ts1, f"{t1-t0:.6f}", sent_bytes, f"{thr:.2f}", ''])

            # workers → PS
            ts0 = datetime.datetime.now().isoformat(); t2 = time.perf_counter()
            agg = torch.zeros_like(flat_params, device=device)
            for w in range(1, world_size):
                buf = torch.zeros_like(flat_params, device=device)
                dist.recv(tensor=buf, src=w)
                agg += buf
            t3 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            recv_bytes = flat_params.numel()*flat_params.element_size()*workers
            thr = (recv_bytes/(t3-t2))/(1024**2)
            log_row(PS_RESULTS, [0, rnd, 'ps_recv', ts0, ts1, f"{t3-t2:.6f}", recv_bytes, f"{thr:.2f}", ''])

            # update
            ts0 = datetime.datetime.now().isoformat(); t4 = time.perf_counter()
            agg /= workers
            grads = unflatten_tensors(agg, shapes)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            optimizer.step(); optimizer.zero_grad()
            flat_params = flatten_tensors([p.data for p in model.parameters()]).to(device)
            t5 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            log_row(PS_RESULTS, [0, rnd, 'ps_update', ts0, ts1, f"{t5-t4:.6f}", '', '', ''])
        else:
            # worker: recv, train, send
            # recv
            ts0 = datetime.datetime.now().isoformat(); t0 = time.perf_counter()
            dist.recv(tensor=flat_params, src=0)
            t1 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            rbytes = flat_params.numel()*flat_params.element_size()
            thr = (rbytes/(t1-t0))/(1024**2)
            log_row(PS_RESULTS, [rank, rnd, 'worker_recv', ts0, ts1, f"{t1-t0:.6f}", rbytes, f"{thr:.2f}", ''])

            # load & train
            new_ws = unflatten_tensors(flat_params, shapes)
            for p, w in zip(model.parameters(), new_ws):
                p.data.copy_(w)
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                data, target = next(data_iter)
            data, target = data.to(device), target.to(device)

            ts0 = datetime.datetime.now().isoformat(); t2 = time.perf_counter()
            out  = model(data)
            loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            t3 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            log_row(PS_RESULTS, [rank, rnd, 'worker_train', ts0, ts1, f"{t3-t2:.6f}", '', '', f"{loss.item():.6f}"])

            # send
            grad_flat = flatten_tensors([p.grad for p in model.parameters()]).to(device)
            ts0 = datetime.datetime.now().isoformat(); t4 = time.perf_counter()
            dist.send(tensor=grad_flat, dst=0)
            t5 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
            sbytes = grad_flat.numel()*grad_flat.element_size()
            thr = (sbytes/(t5-t4))/(1024**2)
            log_row(PS_RESULTS, [rank, rnd, 'worker_send', ts0, ts1, f"{t5-t4:.6f}", sbytes, f"{thr:.2f}", ''])

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

def run_ring(rank, world_size, rounds, batch_size, device, backend):
    # re-init group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300)
    )

    # header
    if rank == 0:
        with open(RING_RESULTS, 'w', newline='') as f:
            csv.writer(f).writerow([
                'rank','round','phase',
                'ts_start','ts_end','duration_s',
                'bytes','throughput_MBps','loss'
            ])

    # setup same as PS
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    shapes = [p.data.shape for p in model.parameters()]
    flat_params = flatten_tensors([p.data for p in model.parameters()]).to(device)

    # loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    data_iter = iter(loader)

    dist.barrier(device_ids=[device.index])

    for rnd in range(rounds):
        # broadcast params
        ts0 = datetime.datetime.now().isoformat(); t0 = time.perf_counter()
        dist.broadcast(tensor=flat_params, src=0)
        t1 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
        bbytes = flat_params.numel()*flat_params.element_size()
        thr = (bbytes/(t1-t0))/(1024**2)
        log_row(RING_RESULTS, [rank, rnd, 'ring_broadcast', ts0, ts1, f"{t1-t0:.6f}", bbytes, f"{thr:.2f}", ''])

        # load
        new_ws = unflatten_tensors(flat_params, shapes)
        for p, w in zip(model.parameters(), new_ws):
            p.data.copy_(w)

        # train
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            data, target = next(data_iter)
        data, target = data.to(device), target.to(device)

        ts0 = datetime.datetime.now().isoformat(); t2 = time.perf_counter()
        out  = model(data)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        t3 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
        log_row(RING_RESULTS, [rank, rnd, 'ring_train', ts0, ts1, f"{t3-t2:.6f}", '', '', f"{loss.item():.6f}"])

        # all-reduce grads
        grad_flat = flatten_tensors([p.grad for p in model.parameters()]).to(device)
        ts0 = datetime.datetime.now().isoformat(); t4 = time.perf_counter()
        dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM)
        t5 = time.perf_counter(); ts1 = datetime.datetime.now().isoformat()
        abytes = grad_flat.numel()*grad_flat.element_size()
        thr = (abytes/(t5-t4))/(1024**2)
        log_row(RING_RESULTS, [rank, rnd, 'ring_allreduce', ts0, ts1, f"{t5-t4:.6f}", abytes, f"{thr:.2f}", ''])

        # update
        grad_flat /= world_size
        grads = unflatten_tensors(grad_flat, shapes)
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        optimizer.step(); optimizer.zero_grad()
        flat_params = flatten_tensors([p.data for p in model.parameters()]).to(device)

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("rounds",     type=int, help="Number of global rounds")
    p.add_argument("batch_size", type=int, help="Mini‑batch size")
    args = p.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank       = int(os.environ.get('RANK',       0))

    # choose device & backend
    if torch.cuda.is_available():
        dev = rank % torch.cuda.device_count()
        torch.cuda.set_device(dev)
        device = torch.device(f"cuda:{dev}")
    else:
        device = torch.device("cpu")
    backend = 'nccl' if device.type=='cuda' else 'gloo'

    # 1) Parameter‐Server run
    run_ps(rank, world_size, args.rounds, args.batch_size, device, backend)

    # 2) Ring All‐Reduce run
    run_ring(rank, world_size, args.rounds, args.batch_size, device, backend)

if __name__ == "__main__":
    main()
