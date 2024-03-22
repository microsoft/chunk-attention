from .dtype import DType
from .module import Module
from typing import List, Tuple


class Tensor(Module):  
    def __init__(self,
                 size:Tuple=None,
                 dtype=DType.float32,
                 metrics=False):
        super().__init__(metrics=metrics)
        if type(size) is tuple:
            self.shape = size
        elif type(size) is list:
            self.shape = tuple(size)
        elif type(size) is int:
            self.shape = (size,)
        else:
            raise Exception(f'unknown size {size}')
        
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = DType.float32
        
        self.inc_memory(self.calc_memory())
    
    def size(self):
        return self.shape
    
    def calc_memory(self):
        size_of_data = DType.sizeof(self.dtype)
        m = 1
        for d in self.shape:
            m *= d
        return m * size_of_data
