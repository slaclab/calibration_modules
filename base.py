from abc import ABC
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from gpytorch.priors import Prior
from gpytorch.module import Module
from gpytorch.constraints import Interval


class BaseModule(Module, ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            **kwargs,
    ):
        """Abstract base module for calibration.

        Args:
            model: The model to be calibrated.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError()


class ParameterModule(BaseModule, ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            parameter_names: Optional[List[str]],
            **kwargs,
    ):
        """Abstract module providing the functionality to register parameters with a prior and constraint.

        Args:
            model: The model to be calibrated.
            parameter_names: A list of parameters to initialize.

        Keyword Args:
            {parameter_name}_size (Union[int, Tuple[int]]): Size of the named parameter. Defaults to 1.
            {parameter_name}_initial (Union[float, torch.Tensor]): Initial value(s) of the named parameter.
              Defaults to zero(s).
            {parameter_name}_default (Union[float, torch.Tensor]): Default value(s) of the named parameter,
              corresponding to zero value(s) of the raw parameter to support regularization during training.
              Defaults to zero(s).
            {parameter_name}_prior (Prior): Prior on named parameter. Defaults to None.
            {parameter_name}_constraint (Interval): Constraint on named parameter. Defaults to None.
            {parameter_name}_mask (Union[torch.Tensor, List]): Boolean mask matching the size of the parameter.
              Allows to select which entries of the unconstrained (raw) parameter are propagated to the constrained
              representation, other entries are set to their default values. This allows to exclude parts of the
              parameter tensor during training. Defaults to None.

        Attributes:
            raw_{parameter_name} (torch.nn.Parameter): Unconstrained parameter tensor.
            {parameter_name} (Union[torch.Tensor, torch.nn.Parameter]): Parameter tensor transformed according
              to constraint and default value.
        """
        super().__init__(model, **kwargs)
        self.model = model
        for name in parameter_names:
            self._initialize_parameter(
                name=name,
                size=kwargs.get(f"{name}_size", 1),
                initial=kwargs.get(f"{name}_initial", 0.0),
                default=kwargs.get(f"{name}_default", 0.0),
                prior=kwargs.get(f"{name}_prior"),
                constraint=kwargs.get(f"{name}_constraint"),
                mask=kwargs.get(f"{name}_mask", None),
            )

    def _register_parameter(self, name: str, initial_value: torch.Tensor):
        """Registers the named parameter with prefix "raw_" to allow for constraints.

        Args:
            name: Name of the parameter.
            initial_value: Initial value(s) of the parameter.
        """
        self.register_parameter(f"raw_{name}", torch.nn.Parameter(initial_value))

    def _register_prior(self, name: str, prior: Optional[Prior]):
        """Registers the prior for the named parameter.

        Args:
            name: Name of the parameter.
            prior: Prior to be placed on the parameter.
        """
        if prior is not None:
            self.register_prior(f"{name}_prior", prior, partial(self._param, name),
                                partial(self._closure, name))

    def _register_constraint(self, name: str, constraint: Optional[Interval]):
        """Registers the constraint for the named parameter.

        Args:
            name: Name of the parameter.
            constraint: Constraint to be placed on the parameter.
        """
        if constraint is not None:
            self.register_constraint(f"raw_{name}", constraint)

    @staticmethod
    def _param(name: str, m: Module) -> Union[torch.nn.Parameter, torch.Tensor]:
        """Returns the named parameter transformed according to constraint and default value.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be returned.

        Returns:
            The parameter transformed according to the constraint.
        """
        raw_parameter = getattr(m, f"raw_{name}").clone()
        mask = getattr(m, f"{name}_mask")
        if mask is not None:
            raw_parameter[~mask] = torch.zeros(torch.count_nonzero(~mask), dtype=raw_parameter.dtype)
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            default_offset = constraint.inverse_transform(getattr(m, f"_{name}_default"))
            return constraint.transform(raw_parameter + default_offset)
        else:
            default_offset = getattr(m, f"_{name}_default")
            return raw_parameter + default_offset

    @staticmethod
    def _closure(name: str, m: Module, value: Union[float, torch.Tensor]):
        """Sets the named parameter of the module to the given value considering constraint and default value.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be set to the given value.
            value: Value(s) the parameter shall be set to.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(getattr(m, f"raw_{name}"))
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            default_offset = constraint.inverse_transform(getattr(m, f"_{name}_default"))
            m.initialize(**{f"raw_{name}": constraint.inverse_transform(value) - default_offset})
        else:
            default_offset = getattr(m, f"_{name}_default")
            m.initialize(**{f"raw_{name}": value - default_offset})

    @staticmethod
    def _add_parameter_name_to_kwargs(name: str, kwargs: Dict):
        parameter_names = kwargs.get("parameter_names")
        if parameter_names is not None:
            if name in parameter_names:
                raise ValueError(f"Parameter {name} already exists!")
            parameter_names.append(name)
        else:
            parameter_names = [name]
        kwargs["parameter_names"] = parameter_names

    def _initialize_parameter(
            self,
            name: str,
            size: Union[int, Tuple[int]],
            initial: Union[float, torch.Tensor],
            default: Union[float, torch.Tensor],
            prior: Optional[Prior] = None,
            constraint: Optional[Interval] = None,
            mask: Optional[Union[torch.Tensor, List]] = None,
    ):
        """Initializes the named parameter.

        Args:
            name: Name of the parameter.
            size: Size of the named parameter.
            initial: Initial value(s) of the named parameter.
            default: Default value(s) of the named parameter.
            prior: Prior on the named parameter.
            constraint: Constraint on the named parameter.
            mask: Boolean mask applied to the transformation from unconstrained (raw) parameter to the
              constrained representation.
        """
        # define initial and default value(s)
        if not isinstance(initial, torch.Tensor):
            initial = float(initial) * torch.ones(size)
        if not isinstance(default, torch.Tensor):
            default = float(default) * torch.ones(size)
        initial_size, default_size = initial.shape, default.shape
        if initial.dim() == 1:
            initial_size = initial.shape[0]
        if default.dim() == 1:
            default_size = default.shape[0]
        if not size == initial_size:
            raise ValueError(f"Initial value tensor does not match given size of {size}!")
        if not size == default_size:
            raise ValueError(f"Default value tensor does not match given size of {size}!")
        setattr(self, f"_{name}_initial", initial)
        setattr(self, f"_{name}_default", default)
        # create parameter
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        setattr(self, f"{name}_mask", mask)
        self._register_parameter(name, initial)
        self._register_prior(name, prior)
        self._register_constraint(name, constraint)
        self._closure(name, self, initial)
        # define parameter property
        setattr(self.__class__, name, property(fget=partial(self._param, name),
                                               fset=partial(self._closure, name)))

    def forward(self, x):
        raise NotImplementedError()
