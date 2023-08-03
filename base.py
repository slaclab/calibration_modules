from abc import ABC
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
        """Abstract module providing the functionality to register parameters with a prior and/or constraint.

        Args:
            model: The model to be calibrated.
            parameter_names: A list of parameters to initialize.

        Keyword Args:
            {parameter_name}_size (Union[int, Tuple[int]]): Size of the named parameter. Defaults to 1.
            {parameter_name}_value (Optional[Union[[float, torch.Tensor]]): Initial value(s) of the named
              parameter. Defaults to zero(s).
            {parameter_name}_prior (Optional[Prior]): Prior on named parameter. Defaults to None.
            {parameter_name}_constraint (Optional[Interval]): Constraint on named parameter. Defaults to None.

        Attributes:
            raw_{parameter_name} (torch.nn.Parameter): Unconstrained parameter tensor.
        """
        super().__init__(model, **kwargs)
        self.model = model
        for name in parameter_names:
            self._initialize_parameter(
                name=name,
                size=kwargs.get(f"{name}_size", 1),
                value=kwargs.get(f"{name}_value", 0.0),
                prior=kwargs.get(f"{name}_prior"),
                constraint=kwargs.get(f"{name}_constraint"),
            )

    def _register_parameter(self, name: str, size: Union[int, Tuple[int]], value: Union[float, torch.Tensor]):
        """Registers the named parameter with prefix "raw_" to allow for constraints.

        Args:
            name: Name of the parameter.
            size: Size of the parameter.
            value: Initial value of the parameter.
        """
        if not isinstance(value, torch.Tensor):
            value = float(value) * torch.ones(size)
        value_size = value.shape
        if value.dim() == 1:
            value_size = value.shape[0]
        if not size == value_size:
            raise ValueError(f"Initial value tensor does not match given size of {size}!")
        self.register_parameter(f"raw_{name}", torch.nn.Parameter(value))

    def _register_prior(self, name: str, prior: Optional[Prior]):
        """Registers the prior for the named parameter.

        Args:
            name: Name of the parameter.
            prior: Prior to be placed on the parameter.
        """
        if prior is not None:
            self.register_prior(f"{name}_prior", prior, lambda m: self._param(name, m),
                                lambda m, value: self._closure(name, m, value))

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
        """Returns the named parameter transformed according to the constraint.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be returned.

        Returns:
            The parameter transformed according to the constraint.
        """
        raw_parameter = getattr(m, f"raw_{name}")
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            return constraint.transform(raw_parameter)
        return raw_parameter

    @staticmethod
    def _closure(name: str, m: Module, value: Union[float, torch.Tensor]):
        """Sets the named parameter of the module to the given value considering the constraint.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be set to the given value.
            value: Value the parameter shall be set to.
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(getattr(m, f"raw_{name}"))
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            m.initialize(**{f"raw_{name}": constraint.inverse_transform(value)})
        else:
            m.initialize(**{f"raw_{name}": value})

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
            value: Union[float, torch.Tensor],
            prior: Optional[Prior] = None,
            constraint: Optional[Interval] = None,
    ):
        """Initializes the named parameter.

        Args:
            name: Name of the parameter.
            size: Size of the named parameter.
            value: Initial value of the named parameter.
            prior: Prior on the named parameter.
            constraint: Constraint on the named parameter.
        """
        self._register_parameter(name, size, value=value)
        self._register_prior(name, prior)
        self._register_constraint(name, constraint)
        self._closure(name, self, value)

    def forward(self, x):
        raise NotImplementedError()
