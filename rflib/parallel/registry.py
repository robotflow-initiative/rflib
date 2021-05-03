from torch.nn.parallel import DataParallel, DistributedDataParallel

from rfvision.rvtools.utils import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
