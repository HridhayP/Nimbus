import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """Engines held only by reference cycles (e.g. a FastAPI app) would keep
    their GPU memory until the next garbage collection; collect after every
    test so the next one sees the whole GPU."""
    yield
    if torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
