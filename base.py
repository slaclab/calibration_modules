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
        """Custom prior mean for a GP based on an arbitrary model.

        Args:
            model: The model to be calibrated.
        """
        super().__init__()
        self.model = model

    @staticmethod
    def _get_size(size: Optional[int], fixed_value: Optional[Union[float, torch.Tensor]]) -> int:
        """Infers parameter size from given size and fixed value.

        If size is None and fixed_value is not, the length of fixed_value is returned. If both arguments
        are None, the returned size is 1.

        Args:
            size: The given size.
            fixed_value: The given fixed value.

        Returns:
            The parameter size to use.
        """
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
    def _param(name: str, m: torch.nn.Module) -> Union[torch.nn.Parameter, torch.Tensor]:
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
    def _closure(name: str, m: torch.nn.Module, value: Union[float, torch.Tensor]):
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

    @abstractmethod
    def forward(self, x):
        pass
