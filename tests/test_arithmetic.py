from chunk_attn.arithmetic import *

if __name__ == '__main__':
    l = Linear(3, 2)
    l.forward(Tensor((2, 2, 3)))
    print(f'total:{l.metrics_total}')
    print(f'self:{l.metrics_self}')