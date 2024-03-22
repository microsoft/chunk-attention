class Metrics:  
    def __init__(self):
        self.memory = 0
        self.io = 0
        self.flops = 0
    
    def __str__(self):
        return f'memory={self.memory:,}, io={self.io:,}, flops={self.flops:,}'