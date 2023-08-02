import torch
from botorch.models.transforms.input import AffineInputTransform

from base import BaseModule


def create_input_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an input transformer based on the given calibration module.

    Forward calls of the calibration module correspond to input_transformer(x).

    Args:
        module: The calibration module.

    Returns:
          The input transformer.
    """
    x_offset_size = 1
    if hasattr(module, "x_offset_size"):
        x_offset_size = module.x_offset_size
    x_offset = torch.zeros(x_offset_size)
    if hasattr(module, "x_offset"):
        x_offset = module.x_offset.detach()
    x_scale_size = 1
    if hasattr(module, "x_scale_size"):
        x_scale_size = module.x_scale_size
    x_scale = torch.ones(x_scale_size)
    if hasattr(module, "x_scale"):
        x_scale = module.x_scale.detach()
    return AffineInputTransform(d=len(x_offset), coefficient=x_scale, offset=x_offset)


def create_output_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an output transformer corresponding to the given calibration module.

    Forward calls of the calibration module correspond to output_transformer.untransform(x).

    Args:
        module: The calibration module.

    Returns:
        The output transformer.
    """
    y_offset_size = 1
    if hasattr(module, "y_offset_size"):
        y_offset_size = module.y_offset_size
    y_offset = torch.zeros(y_offset_size)
    if hasattr(module, "y_offset"):
        y_offset = module.y_offset.detach()
    y_scale_size = 1
    if hasattr(module, "y_scale_size"):
        y_scale_size = module.y_scale_size
    y_scale = torch.ones(y_scale_size)
    if hasattr(module, "y_scale"):
        y_scale = module.y_scale.detach()
    return AffineInputTransform(d=len(y_offset), coefficient=1/y_scale, offset=-y_offset)
