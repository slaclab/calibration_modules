import string
import random

import torch
import pytest


@pytest.fixture(scope="session")
def linear_model() -> torch.nn.Module:
    return torch.nn.Linear(in_features=1, out_features=1)


@pytest.fixture(scope="function")
def ndim_size() -> tuple[int]:
    d = random.randint(1, 3)
    size = tuple(torch.randint(0, 5, size=(d,)).tolist())
    return size


@pytest.fixture(scope="function")
def parameter_name() -> str:
    k = random.randint(1, 10)
    name = ''.join(random.choices(string.ascii_lowercase + "_", k=k))
    return name


@pytest.fixture(scope="function")
def parameter_names() -> list[str]:
    n = random.randint(2, 5)
    names = []
    for i in range(n):
        k = random.randint(1, 10)
        names.append(''.join(random.choices(string.ascii_lowercase + "_", k=k)))
    return names
