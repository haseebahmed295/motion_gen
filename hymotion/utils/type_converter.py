import torch
from torch import nn


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        p = next(module.parameters())
    except StopIteration:
        raise ValueError("The input module should contain parameters.")

    return p.device
