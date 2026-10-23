import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(4)


@pytest.fixture(params=("cpu", "mps"))
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable; CPU is not a substitute")
    return request.param
