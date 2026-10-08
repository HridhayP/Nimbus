import pytest
import torch

import nimbus


def test_package_imports():
    assert nimbus.__version__


@pytest.mark.cuda
def test_triton_kernel_runs_on_gpu():
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 100_003  # not a multiple of BLOCK, so the mask matters
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.testing.assert_close(out, x + y)
