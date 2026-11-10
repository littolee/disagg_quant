"""Read-only CUDA IPC export of a test vLLM producer's real KV tensors."""
import logging
import pickle
import threading

import zmq
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPullConnector

logger = logging.getLogger("ipc_kv_inspect")


class InspectNixlConnector(NixlPullConnector):
    def register_kv_caches(self, kv_caches):
        result = super().register_kv_caches(kv_caches)
        for name, tensor in kv_caches.items():
            logger.warning("IPC_KV_TENSOR layer=%s shape=%s stride=%s dtype=%s device=%s offset=%s",
                           name, tuple(tensor.shape), tuple(tensor.stride()),
                           tensor.dtype, tensor.device, tensor.storage_offset())
        if self.kv_transfer_config.kv_role == "kv_producer":
            # Export only on loopback, for this isolated single-GPU experiment.
            self._ipc_export_refs = kv_caches
            payload = pickle.dumps({"engine_id": self.engine_id, "layers": {name: reduce_tensor(tensor)
                                    for name, tensor in kv_caches.items()}})
            def serve():
                context = zmq.Context()
                socket = context.socket(zmq.REP)
                socket.setsockopt(zmq.LINGER, 0)
                socket.bind("tcp://127.0.0.1:19041")
                logger.warning("IPC_EXPORT_READY layers=%d", len(kv_caches))
                while True:
                    request = socket.recv()
                    if request == b"GET":
                        socket.send(payload)
                    else:
                        socket.send(b"invalid request")
            threading.Thread(target=serve, daemon=True, name="ipc-kv-export").start()
        return result