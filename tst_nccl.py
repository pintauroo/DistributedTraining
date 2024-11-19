import os
import torch
import torch.distributed as dist

def main():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=300),
        device_id=local_rank
    )

    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {rank}: tensor value after all_reduce: {tensor.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
