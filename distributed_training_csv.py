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

def unflatten_tensors(flat, shapes):
    out, offset = [], 0
    for shape in shapes:
        numel = 1
        for d in shape:
            numel *= d
        out.append(flat[offset:offset+numel].view(shape))
        offset += numel
    return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.pool    = nn.MaxPool2d(2,2)
        self.fc1     = nn.Linear(64*14*14, 128)
        self.fc2     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*14*14)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

def log_row(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def run_ps(rank, world_size, rounds, batch_size, device, backend):
    # -- init group
    dist.init_process_group(backend=backend)

    # header
    if rank == 0:
        with open(PS_RESULTS, 'w', newline='') as f:
            csv.writer(f).writerow([
                'rank','round','phase',
                'ts_start','ts_end','duration_s',
                'bytes','throughput_MBps','loss'
            ])

    # model + optimizer
    model      = Net().to(device)
    optimizer  = optim.SGD(model.parameters(), lr=0.01)
    shapes     = [p.data.shape for p in model.parameters()]
    flat_model = flatten_tensors([p.data for p in model.parameters()]).to(device)
    workers    = world_size - 1

    # prepare data for workers
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=workers, rank=(rank-1 if rank>0 else 0), shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    data_it = iter(loader)

    dist.barrier()

    for rnd in range(rounds):
        if rank == 0:
            # PS → workers
            ts0, t0 = datetime.datetime.now().isoformat(), time.perf_counter()
            for w in range(1, world_size):
                dist.send(flat_model, dst=w)
            t1, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            sent_bytes = flat_model.numel()*flat_model.element_size()*workers
            thr = (sent_bytes/(t1-t0))/(1024**2)
            log_row(PS_RESULTS, [0, rnd, 'ps_send', ts0, ts1, f"{t1-t0:.6f}", sent_bytes, f"{thr:.2f}", ''])

            # workers → PS
            ts0, t2 = datetime.datetime.now().isoformat(), time.perf_counter()
            agg = torch.zeros_like(flat_model, device=device)
            for w in range(1, world_size):
                buf = torch.zeros_like(flat_model, device=device)
                dist.recv(buf, src=w)
                agg += buf
            t3, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            recv_bytes = flat_model.numel()*flat_model.element_size()*workers
            thr = (recv_bytes/(t3-t2))/(1024**2)
            log_row(PS_RESULTS, [0, rnd, 'ps_recv', ts0, ts1, f"{t3-t2:.6f}", recv_bytes, f"{thr:.2f}", ''])

            # update
            ts0, t4 = datetime.datetime.now().isoformat(), time.perf_counter()
            agg.div_(workers)
            grads = unflatten_tensors(agg, shapes)
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            optimizer.step(); optimizer.zero_grad()
            flat_model = flatten_tensors([p.data for p in model.parameters()]).to(device)
            t5, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            log_row(PS_RESULTS, [0, rnd, 'ps_update', ts0, ts1, f"{t5-t4:.6f}", '', '', ''])

        else:
            # worker recv
            ts0, t0 = datetime.datetime.now().isoformat(), time.perf_counter()
            dist.recv(flat_model, src=0)
            t1, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            rb = flat_model.numel()*flat_model.element_size()
            thr = (rb/(t1-t0))/(1024**2)
            log_row(PS_RESULTS, [rank, rnd, 'worker_recv', ts0, ts1, f"{t1-t0:.6f}", rb, f"{thr:.2f}", ''])

            # load + train
            new_w = unflatten_tensors(flat_model, shapes)
            for p, w in zip(model.parameters(), new_w):
                p.data.copy_(w)
            try:
                data, target = next(data_it)
            except StopIteration:
                data_it = iter(loader)
                data, target = next(data_it)
            data, target = data.to(device), target.to(device)

            ts0, t2 = datetime.datetime.now().isoformat(), time.perf_counter()
            out  = model(data)
            loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            t3, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            log_row(PS_RESULTS, [rank, rnd, 'worker_train', ts0, ts1, f"{t3-t2:.6f}", '', '', f"{loss.item():.6f}"])

            # send grads
            grad_flat = flatten_tensors([p.grad for p in model.parameters()]).to(device)
            ts0, t4 = datetime.datetime.now().isoformat(), time.perf_counter()
            dist.send(grad_flat, dst=0)
            t5, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
            sb = grad_flat.numel()*grad_flat.element_size()
            thr = (sb/(t5-t4))/(1024**2)
            log_row(PS_RESULTS, [rank, rnd, 'worker_send', ts0, ts1, f"{t5-t4:.6f}", sb, f"{thr:.2f}", ''])

    dist.barrier()
    dist.destroy_process_group()

def run_ring(rank, world_size, rounds, batch_size, device, backend):
    # re-init
    dist.init_process_group(backend=backend)

    # header
    if rank == 0:
        with open(RING_RESULTS, 'w', newline='') as f:
            csv.writer(f).writerow([
                'rank','round','phase',
                'ts_start','ts_end','duration_s',
                'bytes','throughput_MBps','loss'
            ])

    # same model setup
    model      = Net().to(device)
    optimizer  = optim.SGD(model.parameters(), lr=0.01)
    shapes     = [p.data.shape for p in model.parameters()]
    flat_model = flatten_tensors([p.data for p in model.parameters()]).to(device)

    # loader for *all* ranks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         pin_memory=(device.type=='cuda'))
    data_it = iter(loader)

    dist.barrier()

    for rnd in range(rounds):
        # broadcast
        ts0, t0 = datetime.datetime.now().isoformat(), time.perf_counter()
        dist.broadcast(flat_model, src=0)
        t1, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
        bb = flat_model.numel()*flat_model.element_size()
        thr = (bb/(t1-t0))/(1024**2)
        log_row(RING_RESULTS, [rank, rnd, 'ring_broadcast', ts0, ts1, f"{t1-t0:.6f}", bb, f"{thr:.2f}", ''])

        # load
        new_w = unflatten_tensors(flat_model, shapes)
        for p, w in zip(model.parameters(), new_w):
            p.data.copy_(w)

        # train
        try:
            data, target = next(data_it)
        except StopIteration:
            data_it = iter(loader)
            data, target = next(data_it)
        data, target = data.to(device), target.to(device)

        ts0, t2 = datetime.datetime.now().isoformat(), time.perf_counter()
        out  = model(data)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        t3, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
        log_row(RING_RESULTS, [rank, rnd, 'ring_train', ts0, ts1, f"{t3-t2:.6f}", '', '', f"{loss.item():.6f}"])

        # all‑reduce
        grad_flat = flatten_tensors([p.grad for p in model.parameters()]).to(device)
        ts0, t4 = datetime.datetime.now().isoformat(), time.perf_counter()
        dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM)
        t5, ts1 = time.perf_counter(), datetime.datetime.now().isoformat()
        ab = grad_flat.numel()*grad_flat.element_size()
        thr = (ab/(t5-t4))/(1024**2)
        log_row(RING_RESULTS, [rank, rnd, 'ring_allreduce', ts0, ts1, f"{t5-t4:.6f}", ab, f"{thr:.2f}", ''])

        # update
        grad_flat.div_(world_size)
        grads = unflatten_tensors(grad_flat, shapes)
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        optimizer.step(); optimizer.zero_grad()
        flat_model = flatten_tensors([p.data for p in model.parameters()]).to(device)

    dist.barrier()
    dist.destroy_process_group()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("rounds",     type=int, help="global rounds")
    p.add_argument("batch_size", type=int, help="mini‑batch size")
    args = p.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank       = int(os.environ.get('RANK',       0))

    # device / backend
    if torch.cuda.is_available():
        dev = rank % torch.cuda.device_count()
        torch.cuda.set_device(dev)
        device = torch.device(f"cuda:{dev}")
    else:
        device = torch.device("cpu")
    backend = 'nccl' if device.type=='cuda' else 'gloo'

    # 1) PS run → ps_results.csv
    run_ps(rank, world_size, args.rounds, args.batch_size, device, backend)
    # 2) Ring‑all‑reduce run → ring_results.csv
    run_ring(rank, world_size, args.rounds, args.batch_size, device, backend)

if __name__ == "__main__":
    main()
