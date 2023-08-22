import torch

from .irt import IRT


def get_model(cfg, init_ckpt=None):
    model = IRT(cfg)
    if init_ckpt != None:
        model.load_state_dict(torch.load(init_ckpt))

    return model
