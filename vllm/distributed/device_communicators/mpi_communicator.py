"""Device communicator that routes ALL collectives through the PyTorch
ProcessGroup (MPI backend), bypassing pynccl and custom all-reduce.
MUST be paired with enforce_eager=True: MPI calls are not CUDA-graph
capturable, and GroupCoordinator.graph_capture() asserts the communicator
is a CudaCommunicator.
"""
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
class MPICommunicator(DeviceCommunicatorBase):
    def __init__(self, cpu_group, device=None, device_group=None,
                 unique_name="", global_ranks=None, global_world_size=None):
        super().__init__(cpu_group, device, device_group, unique_name,
                         global_ranks, global_world_size)
        # Defensive: any code probing for these should see "not available".
        self.pynccl_comm = None
        self.ca_comm = None
    # all_reduce / gather / send / recv / broadcast: inherited from
    # DeviceCommunicatorBase, already dist.*(group=self.device_group) -> MPI.
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        input_ = input_.contiguous()
        # list-form all_gather (MPI backend lacks all_gather_into_tensor)
        gather_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_, group=self.device_group)
        return torch.cat(gather_list, dim=dim).contiguous()
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        # all_reduce + slice: reduce-scatter == all-reduce then keep own shard.
        # Uses only all_reduce, which the MPI backend supports.
        reduced = input_.contiguous().clone()
        dist.all_reduce(reduced, group=self.device_group)
        assert reduced.shape[dim] % self.world_size == 0
        chunk = reduced.shape[dim] // self.world_size