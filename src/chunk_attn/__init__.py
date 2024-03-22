__version__ = '1.0rc1'

import os
import platform

# add the lib folder to env
if platform.system() == 'Windows':
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib'))
    dll_paths = [dll_path, os.environ['PATH']]
    os.environ['PATH'] = ';'.join(dll_paths)
else:
    pass
    #raise NotImplementedError(f'platform {platform.system()} is not supported')

import torch
from .lib.chunk_attn_c import *