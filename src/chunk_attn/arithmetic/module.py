from typing import List, Tuple
import inspect
from .metrics import Metrics

class Module:  
    def __init__(self, metrics=True):
        self.metrics_self = Metrics()
        self.metrics_total = Metrics()
        self.metrics = metrics
    
    def inc_memory(self, memory_bytes: int):
        if not self.metrics:
            return
        self.metrics_self.memory += memory_bytes
        stack = inspect.stack()
        visited = set()
        for frame in stack:
            local_vars = frame[0].f_locals
            if 'self' not in local_vars or not isinstance(local_vars['self'], Module):
                continue
            obj = local_vars['self']
            if obj in visited:
                continue
            visited.add(obj)
            obj.metrics_total.memory += memory_bytes
        return self.metrics_total.memory
    
    def inc_io(self, io_bytes: int):
        if not self.metrics:
            return
        self.metrics_self.io += io_bytes
        stack = inspect.stack()
        visited = set()
        for frame in stack:
            local_vars = frame[0].f_locals
            if 'self' not in local_vars or not isinstance(local_vars['self'], Module):
                continue
            obj = local_vars['self']
            if obj in visited:
                continue
            visited.add(obj)
            obj.metrics_total.io += io_bytes
        return self.metrics_total.io
    
    def inc_flops(self, flops: int):
        if not self.metrics:
            return
        self.metrics_self.flops += flops
        stack = inspect.stack()
        visited = set()
        for frame in stack:
            local_vars = frame[0].f_locals
            if 'self' not in local_vars or not isinstance(local_vars['self'], Module):
                continue
            obj = local_vars['self']
            if obj in visited:
                continue
            visited.add(obj)
            obj.metrics_total.flops += flops
        return self.metrics_total.flops
