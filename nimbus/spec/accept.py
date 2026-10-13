"""Draft-token acceptance for speculative decoding.

Given k draft tokens d_1..d_k proposed from the draft model's
distributions q_1..q_k, and the target model's distributions p_1..p_{k+1}
computed in one verification pass:

- greedy: accept d_i while it equals argmax p_i; the first mismatch is
  replaced by argmax p_i; if all k match, argmax p_{k+1} is a free bonus.
- sampling (Leviathan et al. 2023, Chen et al. 2023): accept d_i with
  probability min(1, p_i(d_i) / q_i(d_i)); on the first rejection sample
  from normalize(max(0, p_i - q_i)); if all are accepted sample from
  p_{k+1}. The emitted tokens are distributed exactly as if sampled from
  the target alone.

Every round emits between 1 and k+1 tokens.
"""

from __future__ import annotations

import torch


def accept_greedy(draft: list[int], target_argmax: list[int]) -> tuple[list[int], int]:
    """draft: k tokens; target_argmax: k+1 tokens. Returns (emitted, accepted)."""
    out = []
    for i, d in enumerate(draft):
        if d != target_argmax[i]:
            out.append(target_argmax[i])
            return out, i
        out.append(d)
    out.append(target_argmax[len(draft)])
    return out, len(draft)


def accept_sampled(
    draft: list[int],
    q: torch.Tensor,  # [k, V] draft distributions the drafts were sampled from
    p: torch.Tensor,  # [k+1, V] target distributions
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    k = len(draft)
    idx = torch.arange(k, device=p.device)
    d = torch.tensor(draft, device=p.device)
    ratio = p[idx, d] / q[idx, d].clamp_min(1e-20)
    u = torch.rand(k, device=p.device, generator=generator)
    ok = (u < ratio).tolist()
    out = []
    for i in range(k):
        if not ok[i]:
            resid = (p[i] - q[i]).clamp_min(0)
            total = resid.sum()
            dist = resid / total if total > 0 else p[i]
            out.append(int(torch.multinomial(dist, 1, generator=generator)))
            return out, i
        out.append(draft[i])
    out.append(int(torch.multinomial(p[k], 1, generator=generator)))
    return out, k
