import os
from omegaconf import OmegaConf
import torch

FILEPATH = os.path.realpath(os.path.dirname(__file__))
ROOTDIR = os.path.split(FILEPATH)[0]
OmegaConf.register_new_resolver("ROOTDIR", lambda: ROOTDIR)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    OmegaConf.register_new_resolver("num_jobs", lambda first_gpu=0: num_gpus - first_gpu)
else:
    num_cores = os.cpu_count()
    OmegaConf.register_new_resolver("num_jobs", lambda first_gpu=0: num_cores)
