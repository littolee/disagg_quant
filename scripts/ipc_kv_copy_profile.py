"""CUDA IPC block-copy primitive for an experimental vLLM connector.

The caller must retain the producer's block lease until the returned ticket
has completed and must then notify the producer. This module only implements
the device copy and completion event.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


class CopyTicket:
    """Keep shared tensors and temporary buffers alive until CUDA work finishes."""

    def __init__(self, event: torch.cuda.Event, references: list[object]):
        self._event = event
        self._references = references

    def done(self) -> bool:
        return self._event.query()

    def wait(self) -> None:
        self._event.synchronize()
        self._references.clear()


def copy_kv_blocks(
    source_layers: Sequence[torch.Tensor],
    target_layers: Sequence[torch.Tensor],
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    stream: torch.cuda.Stream | None = None,
) -> CopyTicket:
    """Copy mapped blocks on one device and return a CUDA completion ticket.

    Each tensor has block ID as dimension zero. The caller is responsible for
    IPC mapping, producer synchronization, request cancellation and block leases.
    """
    if not source_layers or len(source_layers) != len(target_layers):
        raise ValueError("source and target must contain the same nonzero layer count")
    if not source_ids or len(source_ids) != len(target_ids):
        raise ValueError("source and target IDs must have the same nonzero length")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target block IDs must be unique")

    first = source_layers[0]
    if not first.is_cuda:
        raise ValueError("KV tensors must be CUDA tensors")
    device = first.device
    for source, target in zip(source_layers, target_layers):
        if source.device != device or target.device != device:
            raise ValueError("all tensors must use the same CUDA device")
        if source.dtype != target.dtype or source.shape[1:] != target.shape[1:]:
            raise ValueError("incompatible source and target block layouts")
        if min(source_ids) < 0 or max(source_ids) >= source.shape[0]:
            raise IndexError("source block ID out of range")
        if min(target_ids) < 0 or max(target_ids) >= target.shape[0]:
            raise IndexError("target block ID out of range")

    active_stream = stream or torch.cuda.current_stream(device)
    references: list[object] = [source_layers, target_layers]
    with torch.cuda.stream(active_stream):
        source_index = torch.tensor(source_ids, device=device, dtype=torch.long)
        target_index = torch.tensor(target_ids, device=device, dtype=torch.long)
        references.extend((source_index, target_index))
        for source, target in zip(source_layers, target_layers):
            temporary = source.index_select(0, source_index)
            target.index_copy_(0, target_index, temporary)
            references.append(temporary)
        event = torch.cuda.Event()
        event.record(active_stream)
    return CopyTicket(event, references)

def copy_kv_blocks_batch(
    source_layers: Sequence[torch.Tensor],
    target_layers: Sequence[torch.Tensor],
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    stream: torch.cuda.Stream | None = None,
) -> CopyTicket:
    """Copy block-contiguous KV tensors with vLLM's batched CUDA copy."""
    from vllm import _custom_ops as ops

    if not source_layers or len(source_layers) != len(target_layers):
        raise ValueError("source and target must contain the same nonzero layer count")
    if not source_ids or len(source_ids) != len(target_ids):
        raise ValueError("source and target IDs must have the same nonzero length")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target block IDs must be unique")
    device = source_layers[0].device
    if device.type != "cuda":
        raise ValueError("KV tensors must be CUDA tensors")
    source_pointers, target_pointers, sizes = [], [], []
    for source, target in zip(source_layers, target_layers):
        if source.device != device or target.device != device:
            raise ValueError("all tensors must use the same CUDA device")
        if source.dtype != target.dtype or source.shape[1:] != target.shape[1:]:
            raise ValueError("incompatible source and target block layouts")
        if not source.is_contiguous() or not target.is_contiguous():
            raise ValueError("KV layers must be block-contiguous")
        if min(source_ids) < 0 or max(source_ids) >= source.shape[0]:
            raise IndexError("source block ID out of range")
        if min(target_ids) < 0 or max(target_ids) >= target.shape[0]:
            raise IndexError("target block ID out of range")
        size = source.stride(0) * source.element_size()
        for source_id, target_id in zip(source_ids, target_ids):
            source_pointers.append(source.data_ptr() + source_id * size)
            target_pointers.append(target.data_ptr() + target_id * size)
            sizes.append(size)
    src = torch.tensor(source_pointers, dtype=torch.int64, device="cpu")
    dst = torch.tensor(target_pointers, dtype=torch.int64, device="cpu")
    count = torch.tensor(sizes, dtype=torch.int64, device="cpu")
    active_stream = stream or torch.cuda.current_stream(device)
    with torch.cuda.stream(active_stream):
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record(active_stream)
        ops.swap_blocks_batch(src, dst, count)
        gpu_end.record(active_stream)
        event = torch.cuda.Event()
        event.record(active_stream)
    ticket = CopyTicket(event, [source_layers, target_layers, src, dst, count])
    ticket.gpu_start = gpu_start
    ticket.gpu_end = gpu_end
    return ticket
