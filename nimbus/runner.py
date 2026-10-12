"""Turns a scheduled batch of sequences into one forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimbus.model.attention import AttnMetadata
from nimbus.sequence import Sequence


@dataclass
class StepInput:
    input_ids: torch.Tensor  # [T]
    positions: torch.Tensor  # [T]
    meta: AttnMetadata
    # Rows of the flat batch whose logits are needed, and for which sequence
    # (only sequences that finish their pending tokens this step sample).
    sample_rows: torch.Tensor
    sample_seqs: list[Sequence]


def _pad_tables(tables: list[list[int]], device) -> torch.Tensor:
    width = max((len(t) for t in tables), default=1) or 1
    out = torch.zeros((len(tables), width), dtype=torch.int32)
    for i, t in enumerate(tables):
        if t:
            out[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
    return out.to(device, non_blocking=True)


def prepare(batch: list[tuple[Sequence, int]], block_size: int, device) -> StepInput:
    """batch: (sequence, number of tokens to feed) pairs, prefills first.

    A sequence feeding more than one token is treated as a prefill (it may
    continue an already-cached prefix); a sequence feeding one token whose
    pending work it completes is a decode.
    """
    ids: list[int] = []
    pos: list[int] = []
    slots: list[int] = []
    pf_start = [0]
    pf_ctx: list[int] = []
    pf_tables: list[list[int]] = []
    dc_ctx: list[int] = []
    dc_tables: list[list[int]] = []
    sample_rows: list[int] = []
    sample_seqs: list[Sequence] = []
    seen_decode = False
    for seq, n in batch:
        start = seq.num_computed
        end = start + n
        if end > len(seq):
            raise ValueError(f"{seq} cannot feed {n} tokens")
        if len(seq.block_table) * block_size < end:
            raise ValueError(f"{seq} has too few blocks for {end} tokens")
        ids.extend(seq.token_ids[start:end])
        pos.extend(range(start, end))
        bt = seq.block_table
        slots.extend(bt[p // block_size] * block_size + p % block_size for p in range(start, end))
        is_decode = n == 1 and end == len(seq) and start > 0
        if is_decode:
            seen_decode = True
            dc_ctx.append(end)
            dc_tables.append(bt)
        else:
            if seen_decode:
                raise ValueError("prefill sequences must come before decodes")
            pf_start.append(pf_start[-1] + n)
            pf_ctx.append(end)
            pf_tables.append(bt)
        if end == len(seq):
            sample_rows.append(len(ids) - 1)
            sample_seqs.append(seq)

    t = lambda x, dt=torch.int32: torch.tensor(x, dtype=dt).to(device, non_blocking=True)  # noqa: E731
    meta = AttnMetadata(
        slot_mapping=t(slots, torch.int64),
        num_prefill_tokens=pf_start[-1],
        prefill_query_start=t(pf_start),
        prefill_context_lens=t(pf_ctx),
        prefill_block_tables=_pad_tables(pf_tables, device),
        decode_context_lens=t(dc_ctx),
        decode_block_tables=_pad_tables(dc_tables, device),
        max_decode_context=max(dc_ctx, default=0),
        prefill_query_start_host=pf_start,
        prefill_context_lens_host=pf_ctx,
    )
    return StepInput(
        input_ids=t(ids, torch.int64),
        positions=t(pos, torch.int64),
        meta=meta,
        sample_rows=t(sample_rows, torch.int64),
        sample_seqs=sample_seqs,
    )
