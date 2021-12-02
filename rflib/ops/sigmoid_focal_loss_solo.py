import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ..utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['sigmoid_focal_loss_solo_forward', 'sigmoid_focal_loss_solo_backward'])


class SigmoidFocalLossSOLOFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        loss = ext_module.sigmoid_focal_loss_solo_forward(input, target, num_classes,
                                               gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = ext_module.sigmoid_focal_loss_solo_backward(input, target, d_loss,
                                                   num_classes, gamma, alpha)
        return d_input, None, None, None, None


sigmoid_focal_loss_solo_function = SigmoidFocalLossSOLOFunction.apply


# TODO: remove this module
class SigmoidFocalLossSOLO(nn.Module):

    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss_solo_function(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(gamma={}, alpha={})'.format(
            self.gamma, self.alpha)
        return tmpstr
