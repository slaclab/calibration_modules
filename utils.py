import torch
from botorch.models.transforms.input import AffineInputTransform

from base import BaseModule


def extract_input_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an input transformer based on the given calibration module.

    Forward calls of the calibration module correspond to forward calls of the transformer: transformer(x).

    Args:
        module: The calibration module.

    Returns:
          The input transformer.
    """
    x_offset = torch.zeros(1)
    if hasattr(module, "x_offset"):
        x_offset = module.x_offset.detach()
    x_scale = torch.ones(1)
    if hasattr(module, "x_scale"):
        x_scale = module.x_scale.detach()
    return AffineInputTransform(d=len(x_offset), coefficient=1/x_scale, offset=-x_offset)


def extract_output_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an output transformer corresponding to the given calibration module.

    Forward calls of the calibration module correspond to backward calls of the transformer:
    transformer.untransform(x).

    Args:
        module: The calibration module.

    Returns:
        The output transformer.
    """
    y_offset = torch.zeros(1)
    if hasattr(module, "y_offset"):
        y_offset = module.y_offset.detach()
    y_scale = torch.ones(1)
    if hasattr(module, "y_scale"):
        y_scale = module.y_scale.detach()
    return AffineInputTransform(d=len(y_offset), coefficient=y_scale, offset=y_offset)
