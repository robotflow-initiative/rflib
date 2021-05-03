import torch
from torch.autograd import Function
from rflib.utils import ext_loader


ext_module = ext_loader.load_ext('_ext',['knn'])


class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  
  @staticmethod
  def forward(ref, query, k):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.empty(query.shape[0], k, query.shape[2]).long().cuda()

    ext_module.knn(ref, query, inds)

    return inds.cuda()