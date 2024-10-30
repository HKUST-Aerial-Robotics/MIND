import torch
import numpy as np

def from_numpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def gpu(data, device):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data

def padding_traj_nn(traj):
    n = len(traj)
    # forward
    buff = None
    for i in range(n):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    # backward
    buff = None
    for i in reversed(range(n)):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    return traj
