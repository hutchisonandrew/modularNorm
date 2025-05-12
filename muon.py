import torch
from torch import Tensor
import torch.distributed as dist
def orthogonalize(M):
    # six step Newton-Schulz by @YouJiacheng
    # coefficients from: https://twitter.com/YouJiacheng/status/1893704552689303901
    # found by optimization: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae
    # the idea of stability loss was from @leloykun

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / torch.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = torch.eye(A.shape[0], device=M.device, dtype=M.dtype)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

# class Muon(torch.optim.Optimizer):
#     """
#     Muon - MomentUm Orthogonalized by Newton-schulz

#     https://kellerjordan.github.io/posts/muon/

#     Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
#     processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
#     matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
#     the advantage that it can be stably run in bfloat16 on the GPU.

#     Some warnings:
#     - This optimizer should not be used for the embedding layer, the final fully connected layer,
#     or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
#     - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

#     Arguments:
#         lr: The learning rate used by the internal SGD.
#         momentum: The momentum used by the internal SGD.
#         nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
#         ns_steps: The number of Newton-Schulz iteration steps to use.
#     """
#     def __init__(self, params, lr=0.02, weight_decay=None, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
#         if (rank is None) or (world_size is None):
#             raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        #   if weight_decay is None:
        #       raise ValueError("weight_decay must be provided")
#         self.rank = rank
#         self.world_size = world_size
#         defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
#         params: list[Tensor] = [*params]
#         param_groups = []
#         for size in {p.numel() for p in params}:
#             b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
#             group = dict(params=[p for p in params if p.numel() == size],
#                          update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
#             param_groups.append(group)
#         super().__init__(param_groups, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         for group in self.param_groups:
#             update_buffer: Tensor = group["update_buffer"]
#             update_buffer_views: list[Tensor] = group["update_buffer_views"]
#             params: list[Tensor] = group["params"]
            
#             _comm_handle = None 
#             _params_world_for_update = None 

#             def _apply_updates_to_previous_chunk(comm_handle_to_wait, params_to_update_chunk):
#                 if self.world_size > 1 and comm_handle_to_wait is not None:
#                     comm_handle_to_wait.wait()
                
#                 if params_to_update_chunk is not None:
#                     for p_world, g_world_view in zip(params_to_update_chunk, update_buffer_views):
#                         p_world.mul_(1 - group["lr"] * group["weight_decay"])
#                         p_world.add_(g_world_view.view_as(p_world),
#                                      alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

#             for base_i in range(0, len(params), self.world_size):
#                 if base_i > 0:
#                     _apply_updates_to_previous_chunk(_comm_handle, _params_world_for_update)

#                 g_for_communication = None
                
#                 current_param_idx_for_this_rank = base_i + self.rank
#                 if current_param_idx_for_this_rank < len(params):
#                     p = params[current_param_idx_for_this_rank]
#                     raw_grad = p.grad
#                     assert raw_grad is not None, f"Gradient for param {p} is None."
                    
#                     state = self.state[p]
#                     if "momentum_buffer" not in state:
#                         state["momentum_buffer"] = torch.zeros_like(raw_grad)
                    
#                     momentum_buf: Tensor = state["momentum_buffer"]
#                     momentum_buf.lerp_(raw_grad, 1 - group["momentum"]) 

#                     grad_to_process = raw_grad.lerp(momentum_buf, group["momentum"]) if group["nesterov"] else momentum_buf
                    
#                     if grad_to_process.ndim == 4: # for conv filters
#                         grad_to_process = grad_to_process.view(len(grad_to_process), -1)
                    
#                     g_for_communication = orthogonalize(grad_to_process).flatten()
#                 else:
#                     if self.world_size > 1:
#                         g_for_communication = update_buffer_views[self.rank]

#                 if self.world_size > 1:
#                     assert g_for_communication is not None, "g_for_communication is None in distributed mode"
#                     _comm_handle = dist.all_gather_into_tensor(update_buffer, g_for_communication, async_op=True)
#                 else:
#                     if g_for_communication is not None:
#                         update_buffer_views[0].copy_(g_for_communication)
#                     _comm_handle = None

#                 _params_world_for_update = params[base_i : min(base_i + self.world_size, len(params))]

#             if len(params) > 0 :
#                 _apply_updates_to_previous_chunk(_comm_handle, _params_world_for_update)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = orthogonalize(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step
                
        return loss