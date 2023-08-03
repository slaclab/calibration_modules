from abc import ABC, abstractmethod
from typing import Optional, Union

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
        """Base module for calibration.

        Args:
            model: The model to be calibrated.
        """
        super().__init__()
        self.model = model

    @staticmethod
    def _get_size(name: str, **kwargs) -> int:
        """Infers parameter size from given keyword arguments.

        If no size but a fixed value is given, the length of the value is returned. If both are None, the
        returned size is 1.

        Args:
            name: Name of the parameter.

        Keyword Args:
            {name}_size: Size of the named parameter. Defaults to None.
            {name}_fixed: Optional fixed parameter value. Defaults to None.

        Returns:
            The parameter size to use.
        """
        size = kwargs.get(f"{name}_size")
        fixed_value = kwargs.get(f"{name}_fixed")
        if size is None:
            if fixed_value is not None:
                size = len(fixed_value)
            else:
                size = 1
        return size

    def _register_parameter(self, name: str, size: int, initial_value: Optional[float] = 0.0):
        """Registers the named parameter with prefix "raw_" to allow for constraints.

        Args:
            name: Name of the parameter.
            size: Size of the parameter.
            initial_value: Initial value of the parameter.
        """
        self.register_parameter(f"raw_{name}", torch.nn.Parameter(initial_value * torch.ones(size)))

    def _register_prior(self, name: str, prior: Optional[Prior]):
        """Registers the prior for the named parameter.

        Args:
            name: Name of the parameter.
            prior: Prior to be placed on the parameter.
        """
        if prior is not None:
            self.register_prior(f"{name}_prior", prior, lambda m: self._param(name, m),
                                lambda m, value: self._closure(name, m, value))

    def _register_constraint(self, name: str, constraint: Interval):
        """Registers the constraint for the named parameter.

        Args:
            name: Name of the parameter.
            constraint: Constraint to be placed on the parameter.
        """
        if constraint is not None:
            self.register_constraint(f"raw_{name}", constraint)

    @staticmethod
    def _param(name: str, m: Module) -> Union[torch.nn.Parameter, torch.Tensor]:
        """Returns the named parameter transformed according to existing constraints.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be returned.

        Returns:
            The parameter transformed according to existing constraints.
        """
        raw_parameter = getattr(m, f"raw_{name}")
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            return constraint.transform(raw_parameter)
        return raw_parameter

    @staticmethod
    def _closure(name: str, m: Module, value: Union[float, torch.Tensor]):
        """Sets the named parameter of the module to the given value considering existing constraints.

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

    def _init_parameter(
            self,
            name: str,
            size: int,
            initial_value: float,
            default_prior: Optional[Prior] = None,
            default_constraint: Optional[Interval] = None,
            **kwargs,
    ):
        """Initializes the named parameter.

        Args:
            name: Name of the parameter.
            size: Size of the named parameter.
            initial_value: Initial value of the named parameter.
            default_prior: Default prior on the named parameter.
            default_constraint: Default constraint on the named parameter.

        Keyword Args:
            {name}_fixed: Optional fixed value for the named parameter. Defaults to None.
            {name}_prior: Prior on the named parameter. Defaults to default_prior.
            {name}_constraint: Constraint on the named parameter. Defaults to default_constraint.
        """
        fixed_value = kwargs.get(f"{name}_fixed")
        self._register_parameter(name, size, initial_value=initial_value)
        # prior
        prior = kwargs.get(f"{name}_prior", default_prior)
        self._register_prior(name, prior)
        # constraint
        constraint = kwargs.get(f"{name}_constraint", default_constraint)
        self._register_constraint(name, constraint)
        # fixed value
        if fixed_value is not None:
            self._closure(name, self, fixed_value)
            getattr(self, f"raw_{name}").requires_grad = False
        else:
            self._closure(name, self, initial_value)

    @abstractmethod
    def forward(self, x):
        pass
