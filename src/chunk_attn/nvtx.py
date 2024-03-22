import torch

def range_push(msg: str):
    pass

def range_pop():
    pass

if torch.cuda.is_available():
    import torch.cuda.nvtx as nvtx
    range_push = nvtx.range_push
    range_pop = nvtx.range_pop