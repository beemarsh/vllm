"""PyTorch-distributed adapter for MPI-backed CUDA collectives.

This class does not implement MPI. It deliberately routes vLLM's device
communicator API to ``torch.distributed`` so a PyTorch ``ProcessGroupMPI``
can provide the actual transport.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    DeviceCommunicatorBase,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _normal_empty_like(tensor: torch.Tensor) -> torch.Tensor:
    # vLLM warmup/profile can run under torch.inference_mode(). ProcessGroupMPI
    # mutates all_gather output tensors during work.wait(), so those destination
    # tensors must not be inference tensors.
    with torch.inference_mode(False), torch.no_grad():
        return torch.empty_like(tensor)


def _normal_empty(
    size: torch.Size | tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode(False), torch.no_grad():
        return torch.empty(size, dtype=dtype, device=device)


def _is_inference_tensor(tensor: torch.Tensor) -> bool | str:
    is_inference = getattr(tensor, "is_inference", None)
    if is_inference is None:
        return "unknown"
    return bool(is_inference())


def _tensor_summary(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"inference={_is_inference_tensor(tensor)}"
    )


class MPICommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
        )
        # Defensive: any code probing for these should see "not available".
        self.pynccl_comm = None
        self.ca_comm = None
        self._cuda_capture_debug_counter = 0

        if self.use_all2all:
            self.all2all_manager = self._init_all2all_manager()

    def _log_cuda_capture_collective(
        self,
        op_name: str,
        stage: str,
        *tensors: torch.Tensor,
    ) -> None:
        if not tensors or not any(tensor.is_cuda for tensor in tensors):
            return
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except RuntimeError:
            return
        if not capturing:
            return
        if stage == "before":
            self._cuda_capture_debug_counter += 1
        tensor_info = "; ".join(_tensor_summary(tensor) for tensor in tensors[:2])
        logger.error(
            "[MPI CG DEBUG] %s #%d %s rank=%d group=%s %s",
            op_name,
            self._cuda_capture_debug_counter,
            stage,
            self.rank_in_group,
            self.unique_name,
            tensor_info,
        )

    def _init_all2all_manager(self) -> All2AllManagerBase:
        if self.all2all_backend == "naive":
            from vllm.distributed.device_communicators.all2all import (
                NaiveAll2AllManager,
            )

            logger.info_once("Using naive all2all manager for MPI.", scope="global")
            return NaiveAll2AllManager(self.cpu_group)

        if self.all2all_backend not in (None, "allgather_reducescatter"):
            logger.warning_once(
                "`%s` all2all manager is not supported with MPI. "
                "Falling back to allgather_reducescatter.",
                self.all2all_backend,
            )

        from vllm.distributed.device_communicators.all2all import (
            AgRsAll2AllManager,
        )

        logger.info_once(
            "Using allgather_reducescatter all2all manager for MPI.",
            scope="global",
        )
        return AgRsAll2AllManager(self.cpu_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        output = input_.contiguous().clone()
        self._log_cuda_capture_collective("all_reduce", "before", output)
        dist.all_reduce(output, group=self.device_group)
        self._log_cuda_capture_collective("all_reduce", "after", output)
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()
        input_ = input_.contiguous()
        gather_list = [_normal_empty_like(input_) for _ in range(self.world_size)]
        self._log_cuda_capture_collective(
            "all_gather", "before", input_, gather_list[0]
        )
        dist.all_gather(gather_list, input_, group=self.device_group)
        self._log_cuda_capture_collective(
            "all_gather", "after", input_, gather_list[0]
        )
        return torch.cat(gather_list, dim=dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("MPI all_gatherv currently supports only dim 0")

        if sizes is not None:
            assert len(sizes) == self.world_size
        if sizes is not None and all(size == sizes[0] for size in sizes):
            sizes = None

        def _all_gather_single(
            tensor: torch.Tensor,
            sizes_: list[int] | None,
        ) -> torch.Tensor:
            if sizes_ is None:
                return self.all_gather(tensor, dim=0)

            assert len(sizes_) == self.world_size
            assert tensor.shape[0] == sizes_[self.rank_in_group], (
                f"{tensor.shape[0]} != {sizes_[self.rank_in_group]}"
            )
            gather_list = [
                _normal_empty(
                    (size,) + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for size in sizes_
            ]
            self._log_cuda_capture_collective(
                "all_gatherv", "before", tensor, gather_list[0]
            )
            dist.all_gather(gather_list, tensor.contiguous(), group=self.device_group)
            self._log_cuda_capture_collective(
                "all_gatherv", "after", tensor, gather_list[0]
            )
            return torch.cat(gather_list, dim=0).contiguous()

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        return [_all_gather_single(tensor, sizes) for tensor in input_]

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()
        assert input_tensor.shape[0] % self.world_size == 0
        chunk_size = input_tensor.shape[0] // self.world_size

        reduced = input_tensor.clone()
        self._log_cuda_capture_collective("reduce_scatter", "before", reduced)
        dist.all_reduce(reduced, group=self.device_group)
        self._log_cuda_capture_collective("reduce_scatter", "after", reduced)

        start = self.rank_in_group * chunk_size
        output = reduced.narrow(0, start, chunk_size).contiguous()
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()
        if sizes is None:
            assert input_tensor.shape[0] % self.world_size == 0
            chunk_size = input_tensor.shape[0] // self.world_size
            sizes = [chunk_size] * self.world_size
        else:
            assert len(sizes) == self.world_size
            assert input_tensor.shape[0] == sum(sizes)

        reduced = input_tensor.clone()
        self._log_cuda_capture_collective("reduce_scatterv", "before", reduced)
        dist.all_reduce(reduced, group=self.device_group)
        self._log_cuda_capture_collective("reduce_scatterv", "after", reduced)

        start = sum(sizes[: self.rank_in_group])
        output = reduced.narrow(0, start, sizes[self.rank_in_group]).contiguous()
        return output.movedim(0, dim).contiguous()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if self.all2all_manager is None:
            return super().dispatch_router_logits(
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if self.all2all_manager is None:
            return super().dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if self.all2all_manager is None:
            return super().combine(hidden_states, is_sequence_parallel)
        return self.all2all_manager.combine(hidden_states, is_sequence_parallel)

    def destroy(self):
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None
