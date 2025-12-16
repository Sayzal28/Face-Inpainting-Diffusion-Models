"""
Utilities for mixed precision training and FP16 operations.
"""

import torch
import torch.nn as nn
from abc import abstractmethod


def convert_module_to_f16(module):
    """
    Convert primitive modules to float16.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module):
    """
    Convert primitive modules to float32.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(torch.empty(shape, dtype=torch.float32))
        master_param.data.copy_(param_group.data.view(-1))
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        if param_group.grad is not None:
            master_param.grad = torch.empty(shape, dtype=torch.float32)
            master_param.grad.data.copy_(param_group.grad.data.view(-1))
        else:
            master_param.grad = None


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        param_group.data.copy_(master_param.data.view(param_group.data.shape))


def unflatten_master_params(param_groups_and_shapes, master_params):
    """
    Unflatten the master parameters to look like param_groups_and_shapes.
    """
    return [
        master_param.view(shape)
        for master_param, (_, shape) in zip(master_params, param_groups_and_shapes)
    ]


def get_param_groups_and_shapes(named_model_params):
    """
    Get the shapes needed to make_master_params and the param groups.
    """
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        [-1],
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        [-1],
    )
    return [scalar_vector_named_params, matrix_named_params]


class MixedPrecisionTrainer:
    """
    A helper class for mixed precision training.
    """

    def __init__(self, *, model, use_fp16=False, fp16_scale_growth=1e-3, 
                 initial_lg_loss_scale=16):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        
        if self.use_fp16:
            self.lg_loss_scale = initial_lg_loss_scale
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
        else:
            self.master_params = list(self.model.parameters())

    def zero_grad(self):
        for param in self.master_params:
            param.grad = None

    def backward(self, loss):
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms()
        
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        for rate, params in zip(self.sync_cuda(), self.master_params):
            params.mul_(rate)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt):
        grad_norm, param_norm = self._compute_norms()
        opt.step()
        return True

    def _compute_norms(self):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with torch.no_grad():
                param_norm += torch.norm(p).item() ** 2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad).item() ** 2
        return grad_norm ** 0.5, param_norm ** 0.5

    def master_params_to_state_dict(self, state_dict):
        if self.use_fp16:
            state_dict = unflatten_master_params(
                self.param_groups_and_shapes, state_dict
            )
        return state_dict

    def state_dict_to_master_params(self, state_dict):
        if self.use_fp16:
            state_dict = [
                state_dict[name].view(-1)
                for name, _ in self.param_groups_and_shapes
            ]
        return state_dict


def check_overflow(value):
    """
    Check if a tensor contains NaN or infinity values.
    """
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
