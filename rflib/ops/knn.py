from rflib.utils import ext_loader
import torch
ext_module = ext_loader.load_ext(
    '_ext', ['knn'])

class KNN:
    '''
    knn-search used in densefusion
    https://github.com/unlimblue/KNN_CUDA
    '''

    def __call__(self,
                 ref:torch.Tensor,
                 query:torch.Tensor,
                 k: int):
        return ext_module.knn(ref, query, k)
