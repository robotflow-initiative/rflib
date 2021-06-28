from unittest.mock import MagicMock, patch

import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from rflib.parallel import (MODULE_WRAPPERS, RFDataParallel,
                           RFDistributedDataParallel, is_module_wrapper)


def mock(*args, **kwargs):
    pass


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', MagicMock)
def test_is_module_wrapper():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    assert not is_module_wrapper(model)

    dp = DataParallel(model)
    assert is_module_wrapper(dp)

    mmdp = RFDataParallel(model)
    assert is_module_wrapper(mmdp)

    ddp = DistributedDataParallel(model, process_group=MagicMock())
    assert is_module_wrapper(ddp)

    mmddp = RFDistributedDataParallel(model, process_group=MagicMock())
    assert is_module_wrapper(mmddp)

    
    # test module wrapper registry
    @MODULE_WRAPPERS.register_module()
    class ModuleWrapper(object):

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    module_wraper = ModuleWrapper(model)
    assert is_module_wrapper(module_wraper)
