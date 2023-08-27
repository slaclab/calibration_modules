import string
import random

import torch
import pytest
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval

from base import ParameterModule


def random_name() -> str:
    k = random.randint(1, 5)
    return ''.join(random.choices(string.ascii_lowercase, k=k))


@pytest.fixture(scope="session")
def linear_model() -> torch.nn.Module:
    return torch.nn.Linear(in_features=1, out_features=1)


@pytest.fixture(scope="function")
def ndim_size() -> tuple[int]:
    d = random.randint(1, 3)
    return tuple(torch.randint(1, 5, size=(d,)).tolist())


@pytest.fixture(scope="function")
def parameter_name() -> str:
    return random_name()


@pytest.fixture(scope="function")
def parameter_names() -> list[str]:
    n = random.randint(2, 5)
    names = []
    for i in range(n):
        name = ""
        while name in names:
            name = random_name()
        names.append(name)
    return names


@pytest.fixture(scope="function")
def extensive_parameter_module(linear_model, parameter_name, ndim_size) -> ParameterModule:
    kwargs = {
        f"{parameter_name}_size": ndim_size,
        f"{parameter_name}_default": torch.zeros(ndim_size),
        f"{parameter_name}_initial": torch.ones(ndim_size),
        f"{parameter_name}_prior": NormalPrior(loc=torch.zeros(ndim_size), scale=torch.ones(ndim_size)),
        f"{parameter_name}_constraint": Interval(lower_bound=-1.5, upper_bound=1.5),
        f"{parameter_name}_mask": torch.randint(0, 2, size=ndim_size, dtype=torch.bool),
    }
    m = ParameterModule(
        model=linear_model,
        parameter_names=[parameter_name],
        **kwargs,
    )
    return m
