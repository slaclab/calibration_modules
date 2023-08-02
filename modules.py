from typing import Optional, Union

import torch
from gpytorch.priors import Prior, NormalPrior, GammaPrior
from gpytorch.constraints import Interval, Positive

from base import BaseModule


class InputOffset(BaseModule):
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
            x_offset_size (int): Size of the x_offset parameter. Defaults to 1.
            x_offset_prior (Optional[Prior]): Prior on x_offset parameter. Defaults to a Normal distribution.
            x_offset_constraint (Optional[Interval]): Constraint on x_offset parameter. Defaults to None.
            x_offset_fixed (Union[float, torch.Tensor]): Provides the option to use a fixed parameter value.
              Defaults to None.

        Attributes:
            raw_x_offset (torch.nn.Parameter): Unconstrained parameter tensor.
            x_offset (torch.nn.Parameter): Constrained version of raw_x_offset.
        """
        super().__init__(model, **kwargs)
        name = "x_offset"
        initial_value = 0.0
        given_size = kwargs.get(f"{name}_size")
        fixed_value = kwargs.get(f"{name}_fixed")
        size = self._get_size(given_size, fixed_value)
        self._register_parameter(name, size, initial_value=initial_value)
        # prior
        prior = kwargs.get(
            f"{name}_prior", NormalPrior(loc=torch.zeros((1, size)), scale=torch.ones((1, size)))
        )
        self._register_prior(name, prior)
        # constraint
        constraint = kwargs.get(f"{name}_constraint")
        self._register_constraint(name, constraint)
        # fixed value
        if fixed_value is not None:
            self._closure(name, self, fixed_value)
            getattr(self, f"raw_{name}").requires_grad = False
        else:
            self._closure(name, self, initial_value)

    @property
    def x_offset(self):
        return self._x_offset_param(self)

    @x_offset.setter
    def x_offset(self, value):
        self._x_offset_closure(self, value)

    def _x_offset_param(self, m):
        return self._param("x_offset", m)

    def _x_offset_closure(self, m, value):
        self._closure("x_offset", m, value)

    def input_offset(self, x):
        return x + self.x_offset

    def forward(self, x):
        return self.model(self.input_offset(x))


class InputScale(BaseModule):
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
            x_scale_size (int): Size of the x_scale parameter. Defaults to 1.
            x_scale_prior (Optional[Prior]): Prior on x_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            x_scale_constraint (Optional[Interval]): Constraint on x_scale parameter. Defaults to Positive().
            x_scale_fixed (Union[float, torch.Tensor]): Provides the option to use a fixed parameter value.
              Defaults to None.

        Attributes:
            raw_x_scale (torch.nn.Parameter): Unconstrained parameter tensor.
            x_scale (torch.nn.Parameter): Constrained version of raw_x_scale.
        """
        super().__init__(model, **kwargs)
        name = "x_scale"
        initial_value = 1.0
        given_size = kwargs.get(f"{name}_size")
        fixed_value = kwargs.get(f"{name}_fixed")
        size = self._get_size(given_size, fixed_value)
        self._register_parameter(name, size, initial_value=initial_value)
        # prior
        prior = kwargs.get(
            f"{name}_prior",
            # mean=1.0, std=0.5
            GammaPrior(concentration=2.0 * torch.ones((1, size)), rate=2.0 * torch.ones((1, size))),
        )
        self._register_prior(name, prior)
        # constraint
        constraint = kwargs.get(f"{name}_constraint", Positive())
        self._register_constraint(name, constraint)
        # fixed value
        if fixed_value is not None:
            self._closure(name, self, fixed_value)
            getattr(self, f"raw_{name}").requires_grad = False
        else:
            self._closure(name, self, initial_value)

    @property
    def x_scale(self):
        return self._x_scale_param(self)

    @x_scale.setter
    def x_scale(self, value):
        self._x_scale_closure(self, value)

    def _x_scale_param(self, m):
        return self._param("x_scale", m)

    def _x_scale_closure(self, m, value):
        self._closure("x_scale", m, value)

    def input_scale(self, x):
        return self.x_scale * x

    def forward(self, x):
        return self.model(self.input_scale(x))


class DecoupledLinearInput(InputOffset, InputScale):
    def __init__(
            self,
            model: torch.nn.Module,
            x_size: Optional[int] = None,
            **kwargs,
    ):
        """Adds decoupled learnable linear input calibration.

        Inputs are passed through decoupled linear calibration nodes with learnable offset and scaling
        parameters: y = model(x_scale * (x + x_offset)).

        Args:
            model: The model to be calibrated.
            x_size: Overwrites x_offset_size and x_scale_size.

        Keyword Args:
            Inherited from InputOffset and InputScale.

        Attributes:
            Inherited from InputOffset and InputScale.
        """
        if x_size is not None:
            kwargs["x_offset_size"] = x_size
            kwargs["x_scale_size"] = x_size
        super().__init__(model, **kwargs)

    def decoupled_linear_input(self, x):
        return self.input_scale(self.input_offset(x))

    def forward(self, x):
        return self.model(self.decoupled_linear_input(x))


class OutputOffset(BaseModule):
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
            y_offset_size (int): Size of the y_offset parameter. Defaults to 1.
            y_offset_prior (Optional[Prior]): Prior on y_offset parameter. Defaults to a Normal distribution.
            y_offset_constraint (Optional[Interval]): Constraint on y_offset parameter. Defaults to None.
            y_offset_fixed (Union[float, torch.Tensor]): Provides the option to use a fixed parameter value.
              Defaults to None.

        Attributes:
            raw_y_offset (torch.nn.Parameter): Unconstrained parameter tensor.
            y_offset (torch.nn.Parameter): Constrained version of raw_y_offset.
        """
        super().__init__(model, **kwargs)
        name = "y_offset"
        initial_value = 0.0
        given_size = kwargs.get(f"{name}_size")
        fixed_value = kwargs.get(f"{name}_fixed")
        size = self._get_size(given_size, fixed_value)
        self._register_parameter(name, size, initial_value=initial_value)
        # prior
        prior = kwargs.get(
            f"{name}_prior", NormalPrior(loc=torch.zeros((1, size)), scale=torch.ones((1, size)))
        )
        self._register_prior(name, prior)
        # constraint
        constraint = kwargs.get(f"{name}_constraint")
        self._register_constraint(name, constraint)
        # fixed value
        if fixed_value is not None:
            self._closure(name, self, fixed_value)
            getattr(self, f"raw_{name}").requires_grad = False
        else:
            self._closure(name, self, initial_value)

    @property
    def y_offset(self):
        return self._y_offset_param(self)

    @y_offset.setter
    def y_offset(self, value):
        self._y_offset_closure(self, value)

    def _y_offset_param(self, m):
        return self._param("y_offset", m)

    def _y_offset_closure(self, m, value):
        self._closure("y_offset", m, value)

    def output_offset(self, y):
        return y + self.y_offset

    def forward(self, x):
        return self.output_offset(self.model(x))


class OutputScale(BaseModule):
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
            y_scale_size (int): Size of the y_scale parameter. Defaults to 1.
            y_scale_prior (Optional[Prior]): Prior on y_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            y_scale_constraint (Optional[Interval]): Constraint on y_scale parameter. Defaults to Positive().
            y_scale_fixed (Union[float, torch.Tensor]): Provides the option to use a fixed parameter value.
              Defaults to None.

        Attributes:
            raw_y_scale (torch.nn.Parameter): Unconstrained parameter tensor.
            y_scale (torch.nn.Parameter): Constrained version of raw_y_scale.
        """
        super().__init__(model, **kwargs)
        name = "y_scale"
        initial_value = 1.0
        given_size = kwargs.get(f"{name}_size")
        fixed_value = kwargs.get(f"{name}_fixed")
        size = self._get_size(given_size, fixed_value)
        self._register_parameter(name, size, initial_value=initial_value)
        # prior
        prior = kwargs.get(
            f"{name}_prior",
            # mean=1.0, std=0.5
            GammaPrior(concentration=2.0 * torch.ones((1, size)), rate=2.0 * torch.ones((1, size))),
        )
        self._register_prior(name, prior)
        # constraint
        constraint = kwargs.get(f"{name}_constraint", Positive())
        self._register_constraint(name, constraint)
        # fixed value
        if fixed_value is not None:
            self._closure(name, self, fixed_value)
            getattr(self, f"raw_{name}").requires_grad = False
        else:
            self._closure(name, self, initial_value)

    @property
    def y_scale(self):
        return self._y_scale_param(self)

    @y_scale.setter
    def y_scale(self, value):
        self._y_scale_closure(self, value)

    def _y_scale_param(self, m):
        return self._param("y_scale", m)

    def _y_scale_closure(self, m, value):
        self._closure("y_scale", m, value)

    def output_scale(self, y):
        return self.y_scale * y

    def forward(self, x):
        return self.output_scale(self.model(x))


class DecoupledLinearOutput(OutputOffset, OutputScale):
    def __init__(
            self,
            model: torch.nn.Module,
            y_size: Optional[int] = None,
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
        super().__init__(model, **kwargs)

    def decoupled_linear_output(self, y):
        return self.output_scale(self.output_offset(y))

    def forward(self, x):
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

    def forward(self, x):
        _x = self.decoupled_linear_input(x)
        return self.decoupled_linear_output(self.model(_x))
