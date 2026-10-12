"""Token selection: greedy, temperature and nucleus (top-p) sampling."""

from __future__ import annotations

import torch

from nimbus.sequence import Sequence


def top_p_filter(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Zeroes the tail of each row outside the smallest set whose mass >= top_p."""
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    cum = sorted_p.cumsum(dim=-1)
    drop = cum - sorted_p > top_p[:, None]  # keep tokens until mass reaches top_p
    sorted_p = sorted_p.masked_fill(drop, 0.0)
    return torch.zeros_like(probs).scatter_(-1, idx, sorted_p)


class Sampler:
    def __init__(self, device):
        self.device = device
        self._generators: dict[int, torch.Generator] = {}

    def _generator(self, seq: Sequence) -> torch.Generator | None:
        if seq.params.seed is None:
            return None
        g = self._generators.get(seq.seq_id)
        if g is None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seq.params.seed)
            self._generators[seq.seq_id] = g
        return g

    def forget(self, seq: Sequence) -> None:
        self._generators.pop(seq.seq_id, None)

    def probs(self, logits: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        """Sampling distribution per row (temperature and top-p applied).
        Greedy rows get a one-hot distribution."""
        temps = torch.tensor([s.params.temperature for s in seqs], device=logits.device)
        greedy = temps <= 0
        scaled = logits / torch.where(greedy, 1.0, temps)[:, None]
        p = torch.softmax(scaled, dim=-1)
        top_p = torch.tensor([s.params.top_p for s in seqs], device=logits.device)
        if (top_p < 1).any():
            p = top_p_filter(p, top_p)
            p = p / p.sum(-1, keepdim=True)
        if greedy.any():
            one_hot = torch.zeros_like(p).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
            p = torch.where(greedy[:, None], one_hot, p)
        return p

    def __call__(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        if all(s.params.temperature <= 0 for s in seqs):
            return logits.argmax(-1).tolist()
        p = self.probs(logits, seqs)
        out = []
        for row, s in enumerate(seqs):
            if s.params.temperature <= 0:
                out.append(int(logits[row].argmax()))
            else:
                out.append(int(torch.multinomial(p[row], 1, generator=self._generator(s))))
        return out
