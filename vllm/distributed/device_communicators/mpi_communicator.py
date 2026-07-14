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

        if self.use_all2all:
            self.all2all_manager = self._init_all2all_manager()

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
        output = input_.clone()
        dist.all_reduce(output, group=self.device_group)
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
        gather_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_, group=self.device_group)
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
                torch.empty(
                    (size,) + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                for size in sizes_
            ]
            dist.all_gather(gather_list, tensor.contiguous(), group=self.device_group)
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
        dist.all_reduce(reduced, group=self.device_group)

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
        dist.all_reduce(reduced, group=self.device_group)

        start = sum(sizes[: self.rank_in_group])
        output = reduced.narrow(0, start, sizes[self.rank_in_group]).contiguous()
        return output.movedim(0, dim).contiguous()

    def gather(
        self,
        input_: torch.Tensor,
        dst: int = 0,
        dim: int = -1,
    ) -> torch.Tensor | None:
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
        if self.rank_in_group == dst:
            return torch.cat(gather_list, dim=dim)
        return None

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        dist.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        output = tensor.clone()
        dist.broadcast(output, self.ranks[src], self.device_group)
        return output

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
