# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch

def distributed(opt, is_nocuda):
    cluster = opt['cluster']
    world_size = 1
    local_size = 1
    rank = 0
    local_rank = 0
    is_master = True
    run = None

    if is_nocuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            is_master = rank == 0

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # the following assumes that all processes run on a single node
        if (torch.distributed.is_available() and world_size > 1):
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = opt['master_port'] if 'master_port' in opt else '35551'
            torch.distributed.init_process_group(backend='nccl') # using environment variable initialization
            print("Distributed package is available. Process group initialized.")

    return device, n_gpu, world_size, local_size, rank, local_rank, is_master, run
