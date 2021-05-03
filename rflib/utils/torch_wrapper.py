from functools import partial

import torch

TORCH_VERSION = torch.__version__


def _get_cuda_home():
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


def get_build_config():
    return torch.__config__.show()


def _get_conv():
    from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_dataloader():
    from torch.utils.data import DataLoader
    PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_extension():
    from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                               CUDAExtension)
    return BuildExtension, CppExtension, CUDAExtension


def _get_pool():
    from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                              _AdaptiveMaxPoolNd, _AvgPoolNd,
                                              _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


def _get_norm():
    from torch.nn.modules.instancenorm import _InstanceNorm
    from torch.nn.modules.batchnorm import _BatchNorm
    SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


CUDA_HOME = _get_cuda_home()
_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
BuildExtension, CppExtension, CUDAExtension = _get_extension()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


class SyncBatchNorm(SyncBatchNorm_):

    def _specify_ddp_gpu_num(self, gpu_size):
        super()._specify_ddp_gpu_num(gpu_size)

    def _check_input_dim(self, input):
        super()._check_input_dim(input)

from rflib.utils import ext_loader
ext_module = ext_loader.load_ext(
        '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])

def get_compiler_version():
    return ext_module.get_compiler_version()

def get_compiling_cuda_version():
    return ext_module.get_compiling_cuda_version()