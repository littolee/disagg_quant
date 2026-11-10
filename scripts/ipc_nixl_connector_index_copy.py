"""Experimental same-device CUDA IPC reader with NIXL control plane.

Restricted to a single GPU, one KV group and identical block layouts. Unsupported
or failed cases fall back to the original NIXL reader. Not production-ready.
"""
import logging
import os
import pickle
from types import MethodType

import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPullConnector
from ipc_kv_copy import copy_kv_blocks

logger = logging.getLogger("ipc_kv_connector")


class IpcNixlConnector(NixlPullConnector):
    def register_kv_caches(self, kv_caches):
        result = super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.kv_role == "kv_consumer":
            worker = self.connector_worker
            worker._ipc_local_caches = kv_caches
            worker._ipc_remote_caches = None
            worker._ipc_remote_engine_id = None
            worker._ipc_original_read_blocks = worker._read_blocks
            worker._read_blocks = MethodType(_read_blocks, worker)
            logger.warning("IPC_COPY_READER enabled; limited experimental mode")
        return result


def _map_remote(worker, engine_id):
    if worker._ipc_remote_caches is not None:
        if worker._ipc_remote_engine_id != engine_id:
            raise RuntimeError("remote engine changed; refusing stale IPC mapping")
        return worker._ipc_remote_caches

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect("tcp://127.0.0.1:19041")
        socket.send(b"GET")
        payload = pickle.loads(socket.recv())
    finally:
        socket.close()
        context.term()
    if payload["engine_id"] != engine_id:
        raise RuntimeError("IPC publisher engine ID mismatch")
    handles = payload["layers"]
    caches = {name: rebuild(*args) for name, (rebuild, args) in handles.items()}
    if set(caches) != set(worker._ipc_local_caches):
        raise RuntimeError("IPC layer names do not match")
    worker._ipc_remote_caches = caches
    worker._ipc_remote_engine_id = engine_id
    logger.warning("IPC_COPY_MAPPED engine=%s layers=%d", engine_id, len(caches))
    return caches


def _read_blocks(worker, read_spec, dst_engine_id, request_id, remote_request_id,
                 local_xfer_side_handle, local_dram_handle, remote_xfer_side_handle,
                 expected_consumers, awaiting_kvs):
    original_args = (read_spec, dst_engine_id, request_id, remote_request_id,
                     local_xfer_side_handle, local_dram_handle, remote_xfer_side_handle,
                     expected_consumers, awaiting_kvs)
    local_ids = read_spec.local_block_ids
    remote_ids = read_spec.remote_block_ids
    remote_info = worker.transfer_topo.get_engine_info(dst_engine_id)
    supported = (not read_spec.block_ids_by_region and read_spec.remote_rank == 0
                 and len(local_ids) == len(remote_ids) == 1
                 and worker.dcp_size == remote_info.remote_dcp_size == 1
                 and worker.transfer_topo.block_size_ratio(remote_info.remote_block_size) == 1
                 and remote_info.remote_tp_size == worker.transfer_tp_size == 1)
    if not supported or not any(local_ids):
        return worker._ipc_original_read_blocks(*original_args)
    try:
        local_ids, remote_ids = worker._apply_prefix_caching(
            decode_block_ids=local_ids,
            prefill_block_ids=remote_ids,
            decode_physical_per_logical=worker._physical_blocks_per_logical_kv_block,
            prefill_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        if len(local_ids[0]) != len(remote_ids[0]):
            raise RuntimeError("IPC source and destination block counts differ")
        if not local_ids[0]:
            return worker._ipc_original_read_blocks(*original_args)
        remote_caches = _map_remote(worker, dst_engine_id)
        names = sorted(worker._ipc_local_caches)
        ticket = copy_kv_blocks(
            [remote_caches[name] for name in names],
            [worker._ipc_local_caches[name] for name in names],
            remote_ids[0], local_ids[0],
        )
        ticket.wait()
        if os.environ.get("IPC_VERIFY_KV") == "1":
            source_index = torch.tensor(remote_ids[0], device="cuda", dtype=torch.long)
            target_index = torch.tensor(local_ids[0], device="cuda", dtype=torch.long)
            for name in names:
                source = remote_caches[name].index_select(0, source_index)
                target = worker._ipc_local_caches[name].index_select(0, target_index)
                if not torch.equal(source, target):
                    raise RuntimeError(f"IPC KV mismatch in {name}")
            logger.warning("IPC_KV_VERIFIED request=%s layers=%d", request_id, len(names))
        # Keep the original scheduler's completion and lease-notification flow.
        worker._recving_transfers.setdefault(request_id, [])
        agent = worker._remote_agents[dst_engine_id][(0, read_spec.remote_rank)]
        notification = f"{remote_request_id}:{expected_consumers}".encode()
        worker._pending_recv_notifs.setdefault(request_id, []).append((agent, notification))
        logger.warning("IPC_COPY_COMPLETED request=%s blocks=%d", request_id, len(local_ids[0]))
        return True
    except Exception:
        logger.exception("IPC_COPY_FAILED request=%s, using NIXL", request_id)
        return worker._ipc_original_read_blocks(*original_args)