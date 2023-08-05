from typing import Optional, Union

import torch
from gpytorch.priors import Prior, NormalPrior, GammaPrior
from gpytorch.constraints import Interval, Positive

from base import ParameterModule


class InputOffset(ParameterModule):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Adds learnable input offset calibration.

        Inputs are offset by a learnable parameter: y = model(x + x_offset).

        Args:
            model: The model to be calibrated.

        Keyword Args:
            x_offset_size (Union[int, Tuple[int]]): Size of the x_offset parameter. Defaults to 1.
            x_offset_value (Union[float, torch.Tensor]): Initial value(s) of the x_offset parameter.
              Defaults to zero(s).
            x_offset_prior (Prior): Prior on x_offset parameter. Defaults to a Normal distribution.
            x_offset_constraint (Interval): Constraint on x_offset parameter. Defaults to None.

        Attributes:
            raw_x_offset (torch.nn.Parameter): Unconstrained parameter tensor.
            x_offset (Union[torch.Tensor, torch.nn.Parameter]): Constrained version of raw_x_offset.
        """
        name = "x_offset"
        kwargs[f"{name}_size"] = kwargs.get(f"{name}_size", 1)
        kwargs[f"{name}_value"] = kwargs.get(f"{name}_value", 0.0)
        tensor_size = kwargs[f"{name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{name}_prior"] = kwargs.get(
            f"{name}_prior", NormalPrior(loc=torch.zeros(tensor_size), scale=torch.ones(tensor_size))
        )
        self._add_parameter_name_to_kwargs(name, kwargs)
        super().__init__(model, **kwargs)

    @property
    def x_offset(self) -> Union[torch.Tensor, torch.nn.Parameter]:
        return self._param("x_offset", self)

    @x_offset.setter
    def x_offset(self, value: Union[float, torch.Tensor]):
        self._closure("x_offset", self, value)

    def input_offset(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.x_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.input_offset(x))


class InputScale(ParameterModule):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Adds learnable input scale calibration.

        Inputs are scaled by a learnable parameter: y = model(x_scale * x).

        Args:
            model: The model to be calibrated.

        Keyword Args:
            x_scale_size (Union[int, Tuple[int]]): Size of the x_scale parameter. Defaults to 1.
            x_scale_value (Union[float, torch.Tensor]): Initial value(s) of the x_scale parameter.
              Defaults to one(s).
            x_scale_prior (Prior): Prior on x_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            x_scale_constraint (Interval): Constraint on x_scale parameter. Defaults to Positive().

        Attributes:
            raw_x_scale (torch.nn.Parameter): Unconstrained parameter tensor.
            x_scale (Union[torch.Tensor, torch.nn.Parameter]): Constrained version of raw_x_scale.
        """
        name = "x_scale"
        kwargs[f"{name}_size"] = kwargs.get(f"{name}_size", 1)
        kwargs[f"{name}_value"] = kwargs.get(f"{name}_value", 1.0)
        tensor_size = kwargs[f"{name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{name}_prior"] = kwargs.get(
            # mean=1.0, std=0.5
            f"{name}_prior", GammaPrior(concentration=2.0 * torch.ones(tensor_size),
                                        rate=2.0 * torch.ones(tensor_size))
        )
        kwargs[f"{name}_constraint"] = kwargs.get(f"{name}_constraint", Positive())
        self._add_parameter_name_to_kwargs(name, kwargs)
        super().__init__(model, **kwargs)

    @property
    def x_scale(self) -> Union[torch.Tensor, torch.nn.Parameter]:
        return self._param("x_scale", self)

    @x_scale.setter
    def x_scale(self, value: Union[float, torch.Tensor]):
        self._closure("x_scale", self, value)

    def input_scale(self, x: torch.Tensor) -> torch.Tensor:
        return self.x_scale * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.input_scale(x))


class DecoupledLinearInput(InputOffset, InputScale):
    def __init__(
            self,
            model: torch.nn.Module,
            x_size: Optional[int] = None,
            x_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """Adds decoupled learnable linear input calibration.

        Inputs are passed through decoupled linear calibration nodes with learnable offset and scaling
        parameters: y = model(x_scale * (x + x_offset)).

        Args:
            model: The model to be calibrated.
            x_size: Overwrites x_offset_size and x_scale_size.
            x_mask: Overwrites x_offset_mask and x_scale_mask.

        Keyword Args:
            Inherited from InputOffset and InputScale.

        Attributes:
            Inherited from InputOffset and InputScale.
        """
        if x_size is not None:
            kwargs["x_offset_size"] = x_size
            kwargs["x_scale_size"] = x_size
        if x_mask is not None:
            kwargs["x_offset_mask"] = x_mask
            kwargs["x_scale_mask"] = x_mask
        super().__init__(model, **kwargs)

    def decoupled_linear_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_scale(self.input_offset(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.decoupled_linear_input(x))


class OutputOffset(ParameterModule):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Adds learnable output offset calibration.

        Outputs are offset by a learnable parameter: y = model(x) + y_offset.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            y_offset_size (Union[int, Tuple[int]]): Size of the y_offset parameter. Defaults to 1.
            y_offset_value (Union[float, torch.Tensor]): Initial value(s) of the y_offset parameter.
              Defaults to zero(s).
            y_offset_prior (Prior): Prior on y_offset parameter. Defaults to a Normal distribution.
            y_offset_constraint (Interval): Constraint on y_offset parameter. Defaults to None.

        Attributes:
            raw_y_offset (torch.nn.Parameter): Unconstrained parameter tensor.
            y_offset (Union[torch.Tensor, torch.nn.Parameter]): Constrained version of raw_y_offset.
        """
        name = "y_offset"
        kwargs[f"{name}_size"] = kwargs.get(f"{name}_size", 1)
        kwargs[f"{name}_value"] = kwargs.get(f"{name}_value", 0.0)
        tensor_size = kwargs[f"{name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{name}_prior"] = kwargs.get(
            f"{name}_prior", NormalPrior(loc=torch.zeros(tensor_size), scale=torch.ones(tensor_size))
        )
        self._add_parameter_name_to_kwargs(name, kwargs)
        super().__init__(model, **kwargs)

    @property
    def y_offset(self) -> Union[torch.Tensor, torch.nn.Parameter]:
        return self._param("y_offset", self)

    @y_offset.setter
    def y_offset(self, value: Union[float, torch.Tensor]):
        self._closure("y_offset", self, value)

    def output_offset(self, y: torch.Tensor) -> torch.Tensor:
        return y + self.y_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_offset(self.model(x))


class OutputScale(ParameterModule):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Adds learnable output scale calibration.

        Outputs are scaled by a learnable parameter: y = y_scale * model(x).

        Args:
            model: The model to be calibrated.

        Keyword Args:
            y_scale_size (Union[int, Tuple[int]]): Size of the y_scale parameter. Defaults to 1.
            y_scale_value (Union[float, torch.Tensor]): Initial value(s) of the y_scale parameter.
              Defaults to one(s).
            y_scale_prior (Prior): Prior on y_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            y_scale_constraint (Interval): Constraint on y_scale parameter. Defaults to Positive().

        Attributes:
            raw_y_scale (torch.nn.Parameter): Unconstrained parameter tensor.
            y_scale (Union[torch.Tensor, torch.nn.Parameter]): Constrained version of raw_y_scale.
        """
        name = "y_scale"
        kwargs[f"{name}_size"] = kwargs.get(f"{name}_size", 1)
        kwargs[f"{name}_value"] = kwargs.get(f"{name}_value", 1.0)
        tensor_size = kwargs[f"{name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{name}_prior"] = kwargs.get(
            # mean=1.0, std=0.5
            f"{name}_prior", GammaPrior(concentration=2.0 * torch.ones(tensor_size),
                                        rate=2.0 * torch.ones(tensor_size))
        )
        kwargs[f"{name}_constraint"] = kwargs.get(f"{name}_constraint", Positive())
        self._add_parameter_name_to_kwargs(name, kwargs)
        super().__init__(model, **kwargs)

    @property
    def y_scale(self) -> Union[torch.Tensor, torch.nn.Parameter]:
        return self._param("y_scale", self)

    @y_scale.setter
    def y_scale(self, value: Union[float, torch.Tensor]):
        self._closure("y_scale", self, value)

    def output_scale(self, y: torch.Tensor) -> torch.Tensor:
        return self.y_scale * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_scale(self.model(x))


class DecoupledLinearOutput(OutputOffset, OutputScale):
    def __init__(
            self,
            model: torch.nn.Module,
            y_size: Optional[int] = None,
            y_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """Adds decoupled learnable linear output calibration.

        Outputs are passed through decoupled linear calibration nodes with learnable offset and scaling
        parameters: y = y_scale * (model(x) + y_offset).

        Args:
            model: The model to be calibrated.
            y_size: Overwrites y_offset_size and y_scale_size.

        Keyword Args:
            Inherited from OutputOffset and OutputScale.

        Attributes:
            Inherited from OutputOffset and OutputScale.
        """
        if y_size is not None:
            kwargs["y_offset_size"] = y_size
            kwargs["y_scale_size"] = y_size
        if y_mask is not None:
            kwargs["y_offset_mask"] = y_mask
            kwargs["y_scale_mask"] = y_mask
        super().__init__(model, **kwargs)

    def decoupled_linear_output(self, y: torch.Tensor) -> torch.Tensor:
        return self.output_scale(self.output_offset(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoupled_linear_output(self.model(x))


class DecoupledLinear(DecoupledLinearInput, DecoupledLinearOutput):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Adds decoupled learnable linear in- and output calibration.

        In- and outputs are passed through decoupled linear calibration nodes with learnable offset and
        scaling parameters: y = y_scale * (model(x_scale * (x + x_offset)) + y_offset).

        Args:
            model: The model to be calibrated.

        Keyword Args:
            Inherited from DecoupledLinearInput and DecoupledLinearOutput.

        Attributes:
            Inherited from DecoupledLinearInput and DecoupledLinearOutput.
        """
        super().__init__(model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.decoupled_linear_input(x)
        return self.decoupled_linear_output(self.model(_x))
